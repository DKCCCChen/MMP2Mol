import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, QED
import re
import csv
import os
from scipy.stats import wilcoxon

def extract_smiles_with_label(csv_file_path):
    smiles_label_dict = {}
    with open(csv_file_path, 'r') as csvfile:
        csvreader = csv.reader(csvfile)
        next(csvreader)
        for row in csvreader:
            smiles_label_dict[row[0]] = row[1]
    return smiles_label_dict

def wilcoxon_validation(pairs):
    # pairs: list of (original_activity, transformed_activity)
    if not pairs or len(pairs) < 5:
        print("Not enough valid pairs for Wilcoxon test.")
        return None
    orig = [p[0] for p in pairs if p[0] is not None and p[1] is not None]
    trans = [p[1] for p in pairs if p[0] is not None and p[1] is not None]
    if len(orig) != len(trans) or len(orig) < 5:
        print("Wilcoxon test skipped: insufficient valid pairs.")
        return None
    try:
        stat, p_value = wilcoxon(orig, trans)
        print(f"\nWilcoxon signed-rank test result for transformation activity change:")
        print(f"Statistic: {stat:.4f}, p-value: {p_value:.4e}")
        if p_value < 0.05:
            print("Result is statistically significant (p < 0.05).")
        else:
            print("Result is NOT statistically significant (p >= 0.05).")
        return p_value
    except Exception as e:
        print(f"Error in Wilcoxon test: {e}")
        return None

def filter_and_transform_molecules(input_file, smiles_label_dict):
    with open(input_file, 'r') as file:
        lines = file.readlines()
    
    positive_label_avg_rows = []
    transformation_pairs = []  # For Wilcoxon validation
    
    for line in lines[1:]:
        if not line.strip():
            continue
        parts = line.strip().split('\t')
        if len(parts) >= 10:
            try:
                label_avg = float(parts[9])
                source_smiles = parts[2].replace("[*:1]", "[1*]").replace("[*:2]", "[2*]")
                merged_content = source_smiles + ">>" + parts[3].replace("[*:1]", "[1*]").replace("[*:2]", "[2*]")
                smiles = parts[1]
                
                # Collect activity values for Wilcoxon
                original_activity = None
                transformed_activity = None
                
                if len(parts) > 11:
                    try:
                        original_activity = float(parts[10])
                        transformed_activity = float(parts[11])
                    except (ValueError, IndexError):
                        pass
                
                if label_avg > 0 and smiles_label_dict.get(smiles) == '1':
                    positive_label_avg_rows.append(merged_content)
                    if original_activity is not None and transformed_activity is not None:
                        transformation_pairs.append((original_activity, transformed_activity))
                        
            except (ValueError, IndexError):
                pass
    
    # Perform Wilcoxon test on all pairs
    p_value = wilcoxon_validation(transformation_pairs)
    
    filter_transformations = []
    if p_value is not None and p_value < 0.05:
        # If significant, keep all transformations
        filter_transformations = positive_label_avg_rows
    else:
        # If not significant, handle this case differently
        print("No significant transformations found based on Wilcoxon test.")
        filter_transformations = []
    
    input_dir = os.path.dirname(input_file)
    
    with open(os.path.join(input_dir, "positive_label_avg_output.txt"), 'w') as file:
        for row in filter_transformations:
            file.write(row + '\n')

    return filter_transformations

def convert_fragment_smart(transform):
    left_fragment, right_fragment = transform.split('>>')
    left_fragment_smi_H = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(left_fragment)))
    left_fragment_H = Chem.MolFromSmiles(left_fragment_smi_H)
    left_smarts = Chem.MolToSmarts(left_fragment_H)
    if '[2#0]' in left_smarts:
        left_smarts = left_smarts.replace('[2#0]', '[*:2]')
    left_smarts = left_smarts.replace('[1#0]', '[*:1]')

    right_fragment_smi_H = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles(right_fragment)))
    right_fragment_H = Chem.MolFromSmiles(right_fragment_smi_H)
    right_smart = Chem.MolToSmarts(right_fragment_H)
    right_smart = right_smart.replace('[1#0]', '[*:1]')
    if '[2#0]' in right_smart:
        right_smart = right_smart.replace('[2#0]', '[*:2]')

    return left_smarts + '>>' + right_smart

def apply_transform_molecule(transf_smarts, smi):
    mol = Chem.AddHs(Chem.MolFromSmiles(smi))
    rxn = AllChem.ReactionFromSmarts(transf_smarts)
    ps = rxn.RunReactants((mol,))

    gen_smiles = []
    left_fragment = transf_smarts.split('>>')[1]
    pattern = r'\[\d+\*\]'
    left_str = re.sub(pattern, '', Chem.MolToSmiles(Chem.MolFromSmarts(left_fragment)))
    left_stand = ''.join(c.lower() for c in left_str if c.isalpha())
    if left_stand == 'h':
        left_stand = left_stand + left_stand
    for p in ps:
        smi = Chem.MolToSmiles(Chem.RemoveHs(p[0]))
        if ''.join(c.lower() for c in smi if c.isalpha()) != left_stand:
            gen_smiles.append(smi)
    return gen_smiles

def save_transformed_smiles(smiles_list, output_file):
    # Flatten the list if it contains nested lists
    flat_smiles = []
    for item in smiles_list:
        if isinstance(item, list):
            flat_smiles.extend(item)
        else:
            flat_smiles.append(item)
    
    # Remove duplicates
    unique_smiles = list(set(flat_smiles))
    
    with open(output_file, 'w') as file:
        for smi in unique_smiles:
            file.write(smi + '\n')
    print(f"Saved {len(unique_smiles)} unique SMILES to {output_file}")

def calculate_average_qed(smiles_list):
    valid_mols = [Chem.MolFromSmiles(smi) for smi in smiles_list]
    valid_mols = [mol for mol in valid_mols if mol is not None]
    if not valid_mols:
        return 0.0
    qed_values = [QED.qed(mol) for mol in valid_mols]
    average_qed = sum(qed_values) / len(qed_values)
    return average_qed

# Main execution
input_file = "data/input_trans_results.txt"
target_file = "data/input_label.csv"

smiles_label_dict = extract_smiles_with_label(target_file)
positive_trans = filter_and_transform_molecules(input_file, smiles_label_dict)

print(f"Found {len(positive_trans)} positive transformation rules")

positive_transformed_smiles = []
positive_transformed_rule = []

# Apply positive transformation rules
for transform in positive_trans:
    discard_rule = False
    transform_smarts = convert_fragment_smart(transform)
    
    # Test if the rule improves QED
    for smi, label in smiles_label_dict.items():
        if label == '1':
            new_mols = apply_transform_molecule(transform_smarts, smi)
            if new_mols:
                qed_before = QED.qed(Chem.MolFromSmiles(smi))
                qed_after_avg = calculate_average_qed(new_mols)
                
                print(f"Rule: {transform[:50]}...")
                print(f"  Original QED: {qed_before:.4f}, Transformed avg QED: {qed_after_avg:.4f}")
                
                if qed_after_avg < qed_before:
                    print(f"  Rule discarded: QED decreased")
                    discard_rule = True
                    break
                else:
                    print(f"  Rule accepted: QED improved or maintained")
    
    # If rule is not discarded, apply it to generate molecules
    if not discard_rule:
        positive_transformed_rule.append(transform)
        print(f"Applying rule: {transform[:50]}...")
        
        for smi, label in smiles_label_dict.items():
            if label == '1':
                new_mols = apply_transform_molecule(transform_smarts, smi)
                if new_mols:
                    positive_transformed_smiles.extend(new_mols)

# Save results
positive_output_file = "data/positive_transformed_smiles.txt"
positive_output_rule = "data/positive_transformed_rules.txt"

save_transformed_smiles(positive_transformed_smiles, positive_output_file)

# Save rules (rules are strings, can be saved directly)
with open(positive_output_rule, 'w') as file:
    for rule in positive_transformed_rule:
        file.write(rule + '\n')

print(f"Positive label_avg rules saved to: {positive_output_rule}")
print(f"Positive transformed SMILES saved to: {positive_output_file}")
print(f"Generated {len(set(positive_transformed_smiles))} unique transformed molecules")
print(f"Used {len(positive_transformed_rule)} transformation rules")