# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
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


def get_model_and_predict(test_csv, model_path,  ri_column):
    
    checkpoint = torch.load(model_path, map_location=device)
    n_in, e_in, g_in = checkpoint['dims']
    m_val, s_val = checkpoint['mean'], checkpoint['std']
    
    
    model = UltimateHybridModel(n_in, e_in, g_in).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    
    df = pd.read_csv(test_csv)
    
    loader = prepare_dataloader(df, m_val, s_val, ri_column=ri_column)
    
    
    y_pred, _, _ = get_gnn_predictions(loader, model, device, m_val, s_val)
    return y_pred

def run_dual_phase_prediction():
    test_csv = 'your_file.csv'
    df = pd.read_csv(test_csv)
    
    
    df['TLRI_SNP'] = get_model_and_predict(
        test_csv, 
        '/results/transfer_stdnp_model.pth', 
        'RI_SNP'
    )
    
    df['TLRI_SP'] = get_model_and_predict(
        test_csv, 
        '/results/transfer_stdpolar_model.pth', 
        'RI_SP'
    )
    
    
    output_path = '/data/Merged_Predictions_Combined.csv'
    df.to_csv(output_path, index=False, encoding='utf-8-sig')
  

run_dual_phase_prediction()