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
conda activate hybrid
```
Please note that you will need to activate this virtual conda environment every time you want to use this project.
## Contributing

Contributions, issues, and feature requests are welcome!

Feel free to check [issues page](https://github.com/your-username/your-repo-name/issues) if you want to contribute.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Some code in this repository is adapted from [HybridCLM](https://github.com/HybridCLM/HybridCLM) and uses [mmpdb 3.1](https://github.com/mmpdb/mmpdb). Please respect their licenses as described in their repositories.

## Contact

For questions or suggestions, please open an issue or contact [your-email@example.com].
