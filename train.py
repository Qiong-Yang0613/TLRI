# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
#import seaborn as sns
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import AttentiveFP
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, Fragments
from rdkit.ML.Descriptors import MoleculeDescriptors
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from rdkit import RDLogger
import gc

RDLogger.DisableLog('rdApp.*')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False


base_desc_names = [d[0] for d in Descriptors._descList]
base_calc = MoleculeDescriptors.MolecularDescriptorCalculator(base_desc_names)
frag_funcs = [func_name for func_name in dir(Fragments) if func_name.startswith('fr_')]

def get_ultra_global_feat(mol):
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=2048)
    fp_arr = np.zeros((2048,), dtype=np.float32)
    Chem.DataStructs.ConvertToNumpyArray(fp, fp_arr)
    base_feats = list(base_calc.CalcDescriptors(mol))
    base_feats = np.nan_to_num(np.array(base_feats, dtype=np.float32), nan=0.0)
    base_feats = np.clip(base_feats, -1e6, 1e6)
    frag_feats = [float(getattr(Fragments, f_name)(mol)) for f_name in frag_funcs]
    return np.concatenate([fp_arr, base_feats, frag_feats])

def atom_to_feature(atom, mol):
    try:
        chg = float(atom.GetProp('_GasteigerCharge'))
        if np.isnan(chg) or np.isinf(chg): chg = 0.0
    except: chg = 0.0
    syms = ['As', 'B', 'Br', 'C', 'Cl', 'F', 'Fe', 'Ge', 'H', 'Hg', 'I', 'N', 'Ni', 'O', 'P', 'S', 'Se', 'Si', 'Sn', 'V', 'Zn']
    sym_feat = [float(atom.GetSymbol() == s) for s in syms]
    return sym_feat + [atom.GetAtomicNum() / 100.0, atom.GetDegree() / 6.0, atom.GetTotalNumHs() / 4.0, float(atom.GetIsAromatic()), float(atom.IsInRing()), chg]

def smiles_to_graph(smiles, y_raw, mean_val, std_val):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None: return None
        mol = Chem.AddHs(mol)
        AllChem.ComputeGasteigerCharges(mol)
        x = torch.tensor([atom_to_feature(a, mol) for a in mol.GetAtoms()], dtype=torch.float)
        edge_indices, edge_attrs = [], []
        for b in mol.GetBonds():
            f = [float(b.GetBondTypeAsDouble()), float(b.GetIsConjugated()), float(b.IsInRing())]
            edge_indices += [[b.GetBeginAtomIdx(), b.GetEndAtomIdx()], [b.GetEndAtomIdx(), b.GetBeginAtomIdx()]]
            edge_attrs += [f, f]
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
        g_feat = get_ultra_global_feat(mol)
        y = torch.tensor([(y_raw - mean_val) / std_val], dtype=torch.float)
        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, 
                    global_feat=torch.tensor(g_feat, dtype=torch.float).unsqueeze(0), smiles=smiles)
    except: return None


class UltimateHybridModel(nn.Module):
    def __init__(self, node_in, edge_in, global_in):
        super().__init__()
        self.gnn = AttentiveFP(node_in, 320, 320, edge_in, num_layers=4, num_timesteps=3, dropout=0.1)
        self.global_proj = nn.Sequential(
            nn.Linear(global_in, 1024), nn.LayerNorm(1024), nn.SiLU(), nn.Dropout(0.2),
            nn.Linear(1024, 320), nn.SiLU()
        )
        self.head = nn.Sequential(nn.Linear(640, 512), nn.SiLU(), nn.Dropout(0.2), nn.Linear(512, 1))

    def forward(self, x, edge_index, edge_attr, batch, g_feat):
        g_emb = self.gnn(x, edge_index, edge_attr, batch)
        f_emb = self.global_proj(g_feat)
        combined = torch.cat([g_emb, f_emb], dim=1)
        return self.head(combined), combined


def prepare_dataloader(df, mean_val, std_val, ri_column='RI', batch_size=64, shuffle=False):
    data_list = []
    for _, row in df.iterrows():
        g = smiles_to_graph(row['SMILES'], row[ri_column], mean_val, std_val)
        if g: data_list.append(g)
    return DataLoader(data_list, batch_size=batch_size, shuffle=shuffle)

def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    for d in loader:
        d = d.to(device)
        optimizer.zero_grad()
        pred, _ = model(d.x, d.edge_index, d.edge_attr, d.batch, d.global_feat)
        loss = criterion(pred.view(-1), d.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item() * d.num_graphs
    return total_loss / len(loader.dataset)

def evaluate_mae(model, loader, device, mean_val, std_val):
    model.eval()
    all_preds, all_ys = [], []
    with torch.no_grad():
        for d in loader:
            d = d.to(device)
            pred, _ = model(d.x, d.edge_index, d.edge_attr, d.batch, d.global_feat)
            all_preds.append(pred.cpu().numpy() * std_val + mean_val)
            all_ys.append(d.y.cpu().numpy() * std_val + mean_val)
    return mean_absolute_error(np.concatenate(all_ys), np.concatenate(all_preds))

def get_gnn_predictions(loader, model, device, mean_val, std_val):
    model.eval()
    preds, ys, smiles_list = [], [], []
    with torch.no_grad():
        for d in loader:
            d = d.to(device)
            pred, _ = model(d.x, d.edge_index, d.edge_attr, d.batch, d.global_feat)
            preds.append(pred.cpu().numpy() * std_val + mean_val)
            ys.append(d.y.cpu().numpy() * std_val + mean_val)
            smiles_list.extend(d.smiles)
    return np.concatenate(preds).flatten(), np.concatenate(ys).flatten(), smiles_list


SMISTDNP_CSV = '/data/semistdnp_ri.csv'
STDNP_CSV = 'data/stdnp_ri.csv'
STDNP_CSV = 'data/stdpolar_ri.csv'
PRETRAIN_WEIGHT_PATH = 'model/pretrained_model_ssnp.pt'
SAVE_DIR = 'results/'
BEST_MODEL_PATH=SAVE_DIR+'transfer_stdnp_model.pth'

os.makedirs(SAVE_DIR, exist_ok=True)


df_target = pd.read_csv(STDNP_CSV).dropna().reset_index(drop=True)

train_val_df, test_df = train_test_split(df_target, test_size=0.10, random_state=42)
train_df, val_df = train_test_split(train_val_df, test_size=0.111, random_state=42) # 0.111 * 0.9 ≈ 0.1


t_mean, t_std = train_df['RI'].mean(), train_df['RI'].std()


temp_smi = train_df['SMILES'].iloc[0]
sample_g = smiles_to_graph(temp_smi, 1000, 0, 1)
NODE_IN, EDGE_IN, GLOBAL_IN = sample_g.x.shape[1], sample_g.edge_attr.shape[1], sample_g.global_feat.shape[1]


train_loader = prepare_dataloader(train_df, t_mean, t_std, shuffle=True)
val_loader = prepare_dataloader(val_df, t_mean, t_std)
test_loader = prepare_dataloader(test_df, t_mean, t_std)


model = UltimateHybridModel(NODE_IN, EDGE_IN, GLOBAL_IN).to(device)
if os.path.exists(PRETRAIN_WEIGHT_PATH):
    model.load_state_dict(torch.load(PRETRAIN_WEIGHT_PATH, map_location=device))
   
else:
   
    df_pretrain = pd.read_csv(SMISTDNP_CSV).dropna()
    pretrain_mean, pretrain_std = df_pretrain['RI'].mean(), df_pretrain['RI'].std()
    train_df_pre, val_df_pre = train_test_split(df_pretrain, test_size=0.1, random_state=42)
    train_loader_pre = prepare_dataloader(train_df_pre, pretrain_mean, pretrain_std, batch_size=64, shuffle=True)
    val_loader_pre = prepare_dataloader(val_df_pre, pretrain_mean, pretrain_std, batch_size=64, shuffle=False)

    model_pretrain = UltimateHybridModel(NODE_IN, EDGE_IN, GLOBAL_IN).to(device)
    optimizer = torch.optim.AdamW(model_pretrain.parameters(), lr=0.0005, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.6, patience=12)
    criterion = nn.L1Loss()

    best_val_mae, patience_counter = float('inf'), 0
    for epoch in range(1, 250):
        train_loss = train_one_epoch(model_pretrain, train_loader_pre, optimizer, criterion, device)
        val_mae = evaluate_mae(model_pretrain, val_loader_pre, device, pretrain_mean, pretrain_std)
        scheduler.step(val_mae)
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            torch.save(model_pretrain.state_dict(), PRETRAIN_WEIGHT_PATH)
            patience_counter = 0
        else:
            patience_counter += 1
        if epoch % 20 == 0 or epoch == 1:
            print(f"Pretrain Epoch {epoch:3d} | Loss: {train_loss:.4f} | Val MAE: {val_mae:.2f}")
        if patience_counter >= 40:
            
            break
   

optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.6, patience=10)
criterion = nn.L1Loss()

best_v_mae = float('inf')
best_weights = None


for epoch in range(1, 301):
    train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
    v_mae = evaluate_mae(model, val_loader, device, t_mean, t_std)
    scheduler.step(v_mae)
    
    if v_mae < best_v_mae:
        best_v_mae = v_mae
        best_weights = copy.deepcopy(model.state_dict())
        patience_counter = 0
    else:
        patience_counter += 1
    
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Train Loss: {train_loss:.4f} | Val MAE: {v_mae:.2f}")
 
    if patience_counter >= 35:
        
        break


torch.save({
    'model_state_dict': best_weights,
    'mean': t_mean,
    'std': t_std,
    'dims': (NODE_IN, EDGE_IN, GLOBAL_IN)
}, BEST_MODEL_PATH)


model.load_state_dict(best_weights)
y_pred, y_true, _ = get_gnn_predictions(test_loader, model, device, t_mean, t_std)

print(f"   MAE: {mean_absolute_error(y_true, y_pred):.2f} | R²: {r2_score(y_true, y_pred):.4f}")
