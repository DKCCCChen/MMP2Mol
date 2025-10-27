# Project Name

## Description



Abstract of the paper: Rationally designing novel molecules with desired biological activities is a crucial but challenging task in drug discovery, especially when biological activity data are limited or the crystal structures of targets have not yet been determined. Additionally, the reliance of deep learning models on large-scale structural data hinders their adaptability to diverse targets. To overcome these challenges, we propose MMP2Mol, a generative framework that integrates matched molecular pair analysis (MMPA) with a pre-training chemical language models (CLM) for ligand-based de novo drug design. Guided by MMP analysis, MMP2Mol efficiently extracts informative signals from limited bioactivity data and expands the fine-tuning datasets, enabling deep generative models to focus more accurately on the active subspace without compromising their robust spatial exploration capabilities. MMP2Mol offers a flexible strategy for generating bioactive molecules, resulting in compounds with stronger docking affinities and higher success rates in predicting activity, drug-likeness, and synthetic accessibility compared to existing methods. In our case studies, we employed MMP2Mol for ligand-based de novo drug design and analysis across multiple drug targets. Overall, its flexibility and effectiveness make MMP2Mol a valuable tool to accelerate the drug discovery process.

## Citation & Credits

This project references and uses code from [HybridCLM](https://github.com/michael1788/hybridCLMs). We thank the original authors for their contributions.

For more details, please refer to the [HybridCLM GitHub repository](https://github.com/michael1788/hybridCLMs).

This project also depends on [mmpdb 3.1](https://github.com/rdkit/mmpdb). Please ensure the package is installed for proper functionality.

## Installation

Clone this repository:

```bash
git clone https://github.com/DKCCCChen/MMP2Mol.git
```

Requirements:

```bash
cd MMP2Mol/
conda env create -f environment.yml
# Or pip install -r requirements.txt

```
Once the installation is done, you can activate the virtual conda environment for this project:

```bash
conda activate MMP2Mol
```
Please note that you will need to activate this virtual conda environment every time you want to use this project.

## Usage

1. Prepare your CSV data file. (You could refer to the "data/input.csv" file format)

2. Run the pipeline

```bash
python csv2smi.py data/your_data.csv
python mmpdb_batch.py data/your_data
python extract_smiles_label.py
python mmp_rule_extract.py
python QSAR_Model_Construction_and_Evaluation.py
python evaluate_virtual_library.py
```
You can modify the path and name of the file according to your own file name

3. Molecular generation

First, we recommend using the Linux environment for molecular generation and design.

You can run the following command, which will create a conda virtual environment and install all the needed packages:

```bash
cd Molecular generation and design/
conda env create -f environment.yml
```

Once the installation is done, you can activate the virtual conda environment for the next step:

```bash
conda activate MMP2Mol_MG
```

You can run the example experiment based on the data used in the paper by following the procedure.

A. Process the data and pretrain the chemical language model(CLM):

```bash
cd experiments/
sh run_processing.sh configfiles/clm/A01_clm.ini

sh run_training.sh configfiles/clm/A01_clm.ini
#You can modify the content in A01_clm.ini according to your needs to meet your personalized requirements.
```

You can skip this step, as we provide the processd pretraining data.

B. Process the data and fine-tune the CLM
Then, you can fine-tune the pre-trained model to generate molecules.

```bash
sh run_processing.sh configfiles/ft_clm_generation/A01_clm_ft.ini
sh run_training.sh configfiles/ft_clm_generation/A01_clm_ft.ini

#You can also modify the content in A01_clm_ft.ini according to your needs to meet your personalized requirements.
```

C. Generate the SMILES strings with the fine-tuned CLM:
```bash
sh run_generation.sh configfiles/ft_clm_generation/A01_clm_ft.ini
```

D. Process the generated SMILES strings to get the new molecules to constitute the focused virtual chemical library
```bash
sh run_novo.sh configfiles/ft_clm_generation/A01_clm_ft.ini
```



## Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check [issues page](https://github.com/DKCCCChen/MMP2Mol) if you want to contribute.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Some code in this repository is adapted from [HybridCLM](https://github.com/HybridCLM/HybridCLM) and uses [mmpdb 3.1](https://github.com/mmpdb/mmpdb). Please respect their licenses as described in their repositories.

## Contact

For questions or suggestions, please open an issue or contact [228101065@csu.edu.cn].
