import os
import subprocess
import sys

def mmpdb_flow(file_base):
    smiles_file = f'{file_base}_smiles.smi'
    prop_file = f'{file_base}_prop.smi'
    fragments_file = f'{file_base}_smiles.fragments'
    mmpdb_file = f'{file_base}_smiles.mmpdb'
    trans_results_file = f'{file_base}_trans_results.txt'

    print("Fragmentation...")
    subprocess.run(['mmpdb', 'fragment', smiles_file, '-o', fragments_file, '--num-cuts', '1'], check=True)
    print("Indexing...")
    subprocess.run(['mmpdb', 'index', fragments_file, '-o', mmpdb_file, '--properties', prop_file], check=True)
    print("Loading properties...")
    subprocess.run(['mmpdb', 'loadprops', '-p', prop_file, mmpdb_file], check=True)
    print("Transformation...")
    if os.path.exists(trans_results_file):
        os.remove(trans_results_file)
    with open(smiles_file, 'r') as fin, open(trans_results_file, 'w') as fout:
        for line in fin:
            smiles = line.strip().split('\t')[0]
            if smiles:
                result = subprocess.run([
                    'mmpdb', 'transform', '--smiles', smiles, mmpdb_file, '--property', 'Pa'
                ], capture_output=True, text=True)
                fout.write(result.stdout)
    print("Transformation results saved.")

if __name__ == "__main__":
    file_base = sys.argv[1] if len(sys.argv) > 1 else 'data/input'
    mmpdb_flow(file_base)