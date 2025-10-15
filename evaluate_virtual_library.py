import numpy as np
import pandas as pd
import joblib
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys, DataStructs
import argparse
def compute_ecfp6(smiles, radius=3, nBits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits))

def compute_maccs(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return np.array(MACCSkeys.GenMACCSKeys(mol))

# Add more descriptor functions if needed

def get_features(smiles_list, descriptor_type):
    features = []
    valid_indices = []
    for i, smi in enumerate(smiles_list):
        if descriptor_type == 'ECFP6':
            fp = compute_ecfp6(smi)
        elif descriptor_type == 'MACCS':
            fp = compute_maccs(smi)
        elif descriptor_type == 'ECFP6+MACCS':
            fp1 = compute_ecfp6(smi)
            fp2 = compute_maccs(smi)
            fp = np.concatenate([fp1, fp2]) if fp1 is not None and fp2 is not None else None
        # Add more elif branch for your other combinations
        else:
            fp = compute_ecfp6(smi)
        if fp is not None:
            features.append(fp)
            valid_indices.append(i)
    return np.array(features), valid_indices

def load_smiles(smiles_file):
    with open(smiles_file, 'r') as f:
        return [line.strip() for line in f if line.strip()]

# Load model and descriptor info
model_info = joblib.load('models/best_qsar_model.pkl')
if isinstance(model_info, dict):
    best_model = model_info['model']
    descriptor_type = model_info['descriptor']
else:
    best_model = model_info
    descriptor_type = 'ECFP6'  # fallback, update if needed


parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, default='data/improved_positive_transformed_smiles.txt', help='Virtual library smiles file')
args = parser.parse_args()
virtual_smiles = load_smiles(args.input)
X_virtual, valid_indices = get_features(virtual_smiles, descriptor_type)
virtual_smiles_valid = [virtual_smiles[i] for i in valid_indices]

if hasattr(best_model, 'predict_proba'):
    y_pred_prob = best_model.predict_proba(X_virtual)[:, 1]
    y_pred = (y_pred_prob > 0.5).astype(int)
else:
    y_pred = best_model.predict(X_virtual)

active_count = np.sum(y_pred == 1)
total_count = len(y_pred)
active_ratio = active_count / total_count if total_count > 0 else 0

print(f"Virtual library total molecules: {total_count}")
print(f"Predicted active molecules: {active_count}")
print(f"Active molecule ratio: {active_ratio:.4f}")

pd.DataFrame({
    'smiles': virtual_smiles_valid,
    'predicted_activity': y_pred
}).to_csv('data/virtual_library_qsar_prediction.csv', index=False)