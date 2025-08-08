# PocketXMol Installation Guide

## Quick Installation

### Install from source (recommended for development)

```bash
# Clone the repository
git clone https://github.com/pengxingang/PocketXMol.git
cd PocketXMol

# Install in editable mode
pip install -e .
```

### Alternative: Install dependencies separately

If you encounter issues with the automatic dependency installation, you can install dependencies manually:

```bash
# Install PyTorch (adjust CUDA version as needed)
pip install torch>=2.0.0 --index-url https://download.pytorch.org/whl/cu118

# Install PyTorch Geometric
pip install torch-geometric>=2.3.0

# Install additional PyTorch packages
pip install torch-scatter torch-sparse torch-cluster

# Install chemistry packages
pip install rdkit>=2023.9.3 biopython>=1.83 openbabel-wheel

# Install other dependencies
pip install -r requirements.txt

# Install the package
pip install -e .
```

## Conda Environment (Alternative)

You can also use the provided conda environment file:

```bash
# Create conda environment
conda env create -f environment.yml
conda activate pxm

# Install the package in the conda environment
pip install -e .
```

## Verify Installation

After installation, you can verify it works by running:

```bash
# Check if scripts are available
pxm-sample --help
pxm-train --help

# Test import in Python
python -c "import pocketxmol; print('PocketXMol installed successfully!')"
```

## Available Command Line Tools

After installation, the following command-line tools will be available:

- `pxm-sample`: Main sampling script
- `pxm-sample-drug3d`: 3D drug sampling 
- `pxm-sample-pdb`: PDB-based sampling
- `pxm-train`: Training script
- `pxm-believe`: Confidence scoring
- `pxm-believe-use-pdb`: PDB-based confidence scoring
- `pxm-rank-pose`: Pose ranking

## Troubleshooting

### OpenBabel Installation

If you have issues with OpenBabel, try:
```bash
pip install openbabel-wheel
```

Or use conda:
```bash
conda install -c conda-forge openbabel
```

### PyTorch Geometric Issues

If PyTorch Geometric installation fails, check the [official installation guide](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html) and install it separately before installing PocketXMol.

### CUDA Compatibility

Make sure your PyTorch installation matches your CUDA version. You can check compatible versions at the [PyTorch website](https://pytorch.org/get-started/locally/). 