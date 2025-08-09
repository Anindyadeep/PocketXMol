#!/usr/bin/env python3

from setuptools import setup, find_packages
import os

# Read the README file
def read_readme():
    with open("README.md", "r", encoding="utf-8") as fh:
        return fh.read()

# Read requirements from environment.yml and convert to pip format
def get_requirements():
    requirements = [
        # Core ML/Scientific packages
        "torch>=2.0.0",
        "pytorch-lightning",
        "torch-geometric>=2.3.0",
        "torch-scatter",
        "torch-sparse", 
        "torch-cluster"
        
        # Scientific computing
        "numpy>=1.24,<2.0",
        "pandas>=1.5.2",
        "scipy>=1.8.0",
        "scikit-learn>=1.1.0",
        "networkx>=2.8",
        
        # Chemistry/Biology
        "rdkit==2023.9.3",
        "biopython>=1.83",
        "peptidebuilder==1.1.0",
        "openbabel-wheel",  # Use openbabel-wheel for pip compatibility
        
        # Utilities
        "easydict>=1.9",
        "pyyaml>=5.4.1",
        "tqdm>=4.64.0",
        "lmdb>=1.2.1",
        "tensorboard",
        "seaborn>=0.12.1",
        
        # Additional dependencies from utils/requirements.txt
        "meeko>=0.1.0",
        "openmm>=7.0.0",  # Updated version for compatibility
    ]
    return requirements

setup(
    name="pocketxmol",
    version="0.1.0",
    author="PocketXMol Team", 
    author_email="",
    description="PocketXMol: A pocket-interacting foundation model for molecular generation",
    long_description=read_readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/pengxingang/PocketXMol",
    packages=[
        "pocketxmol",
        "pocketxmol.models",
        "pocketxmol.models.encoders", 
        "pocketxmol.models.fields",
        "pocketxmol.utils",
        "pocketxmol.utils.linker",
        "pocketxmol.scripts",
        "pocketxmol.evaluate",
        "pocketxmol.process",
    ],
    package_dir={
        "pocketxmol": ".",
        "pocketxmol.models": "models",
        "pocketxmol.utils": "utils", 
        "pocketxmol.scripts": "scripts",
        "pocketxmol.evaluate": "evaluate",
        "pocketxmol.process": "process",
        "pocketxmol.models.encoders": "models/encoders",
        "pocketxmol.models.fields": "models/fields", 
        "pocketxmol.utils.linker": "utils/linker",
    },
    package_data={
        "pocketxmol.utils": ["*.pkl.gz", "*.pkl"],
        "pocketxmol": ["*.yml", "*.yaml", "*.pdb", "*.sdf", "*.md"],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9", 
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=get_requirements(),
    dependency_links=[
        "https://data.pyg.org/whl/torch-2.6.0+cu126.html",
    ],
    extras_require={
        "dev": [
            "pytest",
            "black", 
            "flake8",
            "isort",
        ],
        "notebook": [
            "jupyter",
            "matplotlib",
            "ipywidgets",
        ],
    },
    entry_points={
        "console_scripts": [
            "pxm-sample=scripts.sample_use:main",
            "pxm-sample-drug3d=scripts.sample_drug3d:main", 
            "pxm-sample-pdb=scripts.sample_pdb:main",
            "pxm-train=scripts.train_pl:main",
            "pxm-believe=scripts.believe:main",
            "pxm-believe-use-pdb=scripts.believe_use_pdb:main",
            "pxm-rank-pose=scripts.rank_pose:main",
        ],
    },
    zip_safe=False,
) 