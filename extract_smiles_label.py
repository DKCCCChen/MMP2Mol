import pandas as pd
import sys

def extract_smiles_label(input_csv, output_csv):
    df = pd.read_csv(input_csv)
    # Find the SMILES column
    smiles_col = None
    for name in ['SMILES', 'Smiles', 'smiles']:
        if name in df.columns:
            smiles_col = name
            break
    if smiles_col is None:
        raise ValueError("No column named SMILES found in input CSV.")
    # Find the label column
    label_col = None
    for name in ['label', 'Label', 'LABEL', 'activity', 'Activity']:
        if name in df.columns:
            label_col = name
            break
    if label_col is None:
        raise ValueError("No column named label/activity found in input CSV.")
    # Extract and save
    df[[smiles_col, label_col]].to_csv(output_csv, index=False)
    print(f"Extracted [{smiles_col}, {label_col}] to {output_csv}")

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_smiles_label.py <input_csv> <output_csv>")
        print("Example: python extract_smiles_label.py data/input.csv data/input_label.csv")
        sys.exit(1)
    
    input_csv = sys.argv[1] if len(sys.argv) > 1 else 'data/input.csv'
    output_csv = sys.argv[2] if len(sys.argv) > 1 else 'data/input_label.csv'
    extract_smiles_label(input_csv, output_csv)