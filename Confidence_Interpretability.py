# -*- coding: utf-8 -*-
import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
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


try:
    from rdkit.Chem.Draw import SimilarityMaps
    HAS_DRAWING_MODULE = True
except ImportError:
    HAS_DRAWING_MODULE = False

try:
    from captum.attr import IntegratedGradients
except ImportError:
    raise ImportError(" captum: pip install captum")


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


def predict_with_uncertainty(model, data, device, mean_val, std_val, n_iters=20):
    model.train() 
    data = data.to(device)
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    preds = []
    with torch.no_grad():
        for _ in range(n_iters):
            pred, _ = model(data.x, data.edge_index, data.edge_attr, batch, data.global_feat)
            preds.append(pred.item() * std_val + mean_val)
    preds = np.array(preds)
    return np.mean(preds), np.std(preds)

def explain_with_integrated_gradients(model, data, device):
    model.eval() 
    data = data.to(device)
    batch = torch.zeros(data.x.size(0), dtype=torch.long, device=device)
    def model_forward(node_x, edge_index, edge_attr, batch_vec, g_feat):
        pred, _ = model(node_x, edge_index, edge_attr, batch_vec, g_feat)
        return pred
    ig = IntegratedGradients(model_forward)
    input_x = data.x.clone().detach().requires_grad_(True).to(device)
    attributions = ig.attribute(
        inputs=input_x, additional_forward_args=(data.edge_index, data.edge_attr, batch, data.global_feat),
        target=0, n_steps=50, internal_batch_size=1
    )
    return attributions.abs().sum(dim=1).cpu().detach().numpy()

def run_visualization_analysis(model, test_df, t_mean, t_std, device, num_samples=100, save_name='sp'):
    print("\n" + "="*50)
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    results = []
    model.eval()
    for i in range(min(num_samples, len(test_df))):
        row = test_df.iloc[i]
        data = smiles_to_graph(row['SMILES'], row['RI'], t_mean, t_std)
        if data:
            mean_p, std_p = predict_with_uncertainty(model, data, device, t_mean, t_std, n_iters=20)
            results.append({
                'SMILES': row['SMILES'], 'True_RI': row['RI'], 'Pred_RI': mean_p,
                'Uncertainty_Std': std_p, 'Abs_Error': abs(mean_p - row['RI']),
                'Importances': explain_with_integrated_gradients(model, data, device)
            })
    viz_df = pd.DataFrame(results)
    fig, axes = plt.subplots(1, 3, figsize=(22, 7), dpi=300)
    
    ax1 = axes[0]
    ax1.scatter(viz_df['Uncertainty_Std'], viz_df['Abs_Error'], alpha=0.7, c='#445a64', edgecolors='w', s=70, label='Data points')
    if len(viz_df) > 1:
        z = np.polyfit(viz_df['Uncertainty_Std'], viz_df['Abs_Error'], 1)
        p = np.poly1d(z)
        ax1.plot(viz_df['Uncertainty_Std'], p(viz_df['Uncertainty_Std']), "r--", lw=2.5, label=f'Trend (Slope: {z[0]:.2f})')
    ax1.set_title('(a) Uncertainty vs. Error', fontsize=18, fontweight='bold', pad=15)
    ax1.set_xlabel('Predictive Uncertainty ($\sigma$)', fontsize=16)
    ax1.set_ylabel('Absolute Error (RI)', fontsize=16)
    ax1.set_ylim(bottom=-10, top=viz_df['Abs_Error'].max() * 1.35)
    ax1.legend(fontsize=15, loc='upper left')

    ax2 = axes[1]
    ax2.errorbar(viz_df['True_RI'], viz_df['Pred_RI'], yerr=viz_df['Uncertainty_Std'], fmt='o', ecolor='#90a4ae', alpha=0.6, mfc='#1976d2', mec='white', ms=8, label='Predicted RI $\pm \sigma$')
    l, h = np.concatenate([viz_df['True_RI'], viz_df['Pred_RI']]).min()-50, np.concatenate([viz_df['True_RI'], viz_df['Pred_RI']]).max()+150
    ax2.plot([l, h], [l, h], 'k--', lw=1.5, label='Ideal Reference')
    ax2.set_title('(b) Confidence Parity Plot', fontsize=18, fontweight='bold', pad=15)
    ax2.set_xlabel('Experimental RI', fontsize=16)
    ax2.set_ylabel('Predicted RI (Mean)', fontsize=16)
    ax2.set_aspect('equal')
    ax2.legend(fontsize=15, loc='upper left')

    ax3 = axes[2]
    case_study = results[0]
    mol = Chem.AddHs(Chem.MolFromSmiles(case_study['SMILES']))
    atom_labels = [f"{a.GetSymbol()}{i}" for i, a in enumerate(mol.GetAtoms())]
    imp_data = pd.Series(case_study['Importances'], index=atom_labels).sort_values(ascending=False).head(12)
    ax3.bar(imp_data.index, imp_data.values, color=plt.cm.GnBu(np.linspace(0.4, 0.8, 12)[::-1]), edgecolor='#37474f')
    
    ax3.set_title(f"(c) Atomic Importance (Top 12)\nSMILES: {case_study['SMILES']}", fontsize=14, fontweight='bold', pad=10)
    ax3.set_ylabel('Contribution Score', fontsize=16)
    plt.setp(ax3.get_xticklabels(), rotation=45)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)
    
    save_path = f'/plt/Figure_Scientific_Final_{save_name}.png'
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    plt.show()
    

def process_sp():
    STD_CSV = '/data/stdpolar_ri.csv'
    BEST_MODEL_PATH = '/results/transfer_stdpolar_model.pth'
  

    if not os.path.exists(BEST_MODEL_PATH): return
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    t_mean, t_std = checkpoint['mean'], checkpoint['std']

    dims = checkpoint['dims']

    model = UltimateHybridModel(dims[0], dims[1], dims[2]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    df_full = pd.read_csv(STD_CSV).dropna().reset_index(drop=True)
    _, test_df = train_test_split(df_full, test_size=0.10, random_state=2)
    run_visualization_analysis(model, test_df, t_mean, t_std, device, num_samples=100, save_name='sp')


def process_snp():
    STD_CSV = 'data/stdnp_ri.csv'
    BEST_MODEL_PATH = '/results/transfer_stdnp_model.pth'
   
    if not os.path.exists(BEST_MODEL_PATH): return
    checkpoint = torch.load(BEST_MODEL_PATH, map_location=device, weights_only=False)
    t_mean, t_std = checkpoint['mean'], checkpoint['std']
    dims = checkpoint['dims']

    model = UltimateHybridModel(dims[0], dims[1], dims[2]).to(device)
    model.load_state_dict(checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint)
    model.eval()

    df_full = pd.read_csv(STD_CSV).dropna().reset_index(drop=True)
    _, test_df = train_test_split(df_full, test_size=0.10, random_state=42)
    run_visualization_analysis(model, test_df, t_mean, t_std, device, num_samples=100, save_name='snp')


process_sp()
process_snp()