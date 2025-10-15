import pandas as pd
import sys

def csv_to_smi(csv_file):
    nameBase = csv_file.replace('.csv', '')
    df = pd.read_csv(csv_file)
    smiles_col = None
    for col_name in ['SMILES', 'Smiles', 'smiles']:
        if col_name in df.columns:
            smiles_col = col_name
            break
    if smiles_col is None:
        print("No column named SMILES found.")
        return
    df.insert(0, "ID", range(1, len(df) + 1))
    df[[smiles_col, 'ID']].to_csv(f'{nameBase}_smiles.smi', sep='\t', index=False, header=False)
    df.drop(columns=[smiles_col]).to_csv(f'{nameBase}_prop.smi', sep='\t', index=False)
    print("SMILES and property files have been generated.")

if __name__ == "__main__":
    csv_file = sys.argv[1] if len(sys.argv) > 1 else 'data/input.csv'
    csv_to_smi(csv_file)