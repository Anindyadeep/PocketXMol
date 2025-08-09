"""
PocketXMol Server - A modular Pydantic-based interface for molecular generation tasks
"""

import os
import shutil
import gc
import torch
import torch.utils.tensorboard
import numpy as np
import pandas as pd
from typing import Optional, Union, List, Dict, Any, Literal, Tuple
from pathlib import Path
from datetime import datetime
from enum import Enum
from pydantic import BaseModel, Field, validator, root_validator
from easydict import EasyDict
from tqdm.auto import tqdm
from rdkit import Chem
from torch_geometric.loader import DataLoader
from Bio.SeqUtils import seq1
from Bio import PDB
from Bio.PDB import PDBIO

# Import from installed pocketxmol package
from pocketxmol.scripts.train_pl import DataModule
from pocketxmol.models.maskfill import PMAsymDenoiser
from pocketxmol.models.sample import seperate_outputs2, sample_loop3, get_cfd_traj
from pocketxmol.utils.transforms import *
from pocketxmol.utils.misc import *
from pocketxmol.utils.reconstruct import *
from pocketxmol.utils.dataset import UseDataset
from pocketxmol.utils.sample_noise import get_sample_noiser
from pocketxmol.process.utils_process import (
    extract_pocket, add_pep_bb_data, get_peptide_info, 
    get_input_from_file, make_dummy_mol_with_coordinate
)


# ============================================================================
# ENUMS
# ============================================================================

class TaskType(str, Enum):
    """Supported molecular generation tasks"""
    DOCK = "dock"
    SBDD = "sbdd"
    PEPDESIGN = "pepdesign"
    MASKFILL = "maskfill"
    CUSTOM = "custom"


class NoiseType(str, Enum):
    """Noise model types"""
    DOCK = "dock"
    SBDD = "sbdd"
    PEPDESIGN = "pepdesign"
    MASKFILL = "maskfill"
    CUSTOM = "custom"


class LevelStrategy(str, Enum):
    """Noise level strategies"""
    UNIFORM = "uniform"
    ADVANCE = "advance"


# ============================================================================
# BASE CONFIGURATION MODELS
# ============================================================================

class SampleConfig(BaseModel):
    """Sampling configuration parameters"""
    seed: int = Field(default=2024, description="Random seed for reproducibility")
    batch_size: int = Field(default=50, ge=1, description="Batch size for generation")
    num_mols: int = Field(default=100, ge=1, description="Total number of molecules to generate")
    num_repeats: int = Field(default=1, ge=1, description="Number of generation rounds")
    save_traj_prob: float = Field(default=0.02, ge=0, le=1, description="Probability of saving trajectory")
    save_output: List[str] = Field(default_factory=list, description="Output keys to save")
    
    class Config:
        extra = "allow"


class PocketDefinition(BaseModel):
    """Pocket definition parameters"""
    pocket_coord: Optional[List[float]] = Field(None, min_items=3, max_items=3, description="3D coordinates of pocket center")
    ref_ligand_path: Optional[Path] = Field(None, description="Path to reference ligand for pocket definition")
    radius: float = Field(default=15.0, gt=0, description="Pocket radius in Angstroms")
    criterion: Literal["center_of_mass", "geometric_center"] = Field(default="center_of_mass")
    
    @root_validator
    def validate_pocket_definition(cls, values):
        coord = values.get('pocket_coord')
        ref_path = values.get('ref_ligand_path')
        if coord is None and ref_path is None:
            # Will use input_ligand as reference
            pass
        return values


class PocMolMetadata(BaseModel):
    """Metadata for pocket-molecule pairs"""
    data_id: str = Field(default="experiment", description="Experiment identifier")
    pdbid: str = Field(default="protein", description="Protein PDB ID")
    
    class Config:
        extra = "allow"


class DataConfig(BaseModel):
    """Data configuration for molecular generation"""
    protein_path: Path = Field(..., description="Path to protein PDB file")
    input_ligand: Optional[Union[Path, str]] = Field(None, description="Input ligand (path, SMILES, or special format)")
    is_pep: Optional[bool] = Field(None, description="Whether ligand is a peptide")
    pocket_args: PocketDefinition = Field(default_factory=PocketDefinition)
    pocmol_args: PocMolMetadata = Field(default_factory=PocMolMetadata)
    transforms: List[Dict[str, Any]] = Field(default_factory=list, description="Additional transforms")
    
    @validator('protein_path')
    def validate_protein_path(cls, v):
        if not v.exists():
            raise ValueError(f"Protein file not found: {v}")
        if not str(v).endswith('.pdb'):
            raise ValueError(f"Protein file must be PDB format: {v}")
        return v
    
    @validator('input_ligand')
    def validate_input_ligand(cls, v):
        if v is None:
            return v
        if isinstance(v, str):
            # Check for special formats
            if v.startswith('pepseq_') or v.startswith('peplen_'):
                return v
            # Check if it's a SMILES string (contains typical SMILES characters)
            if any(c in v for c in ['(', ')', '[', ']', '=', '#', '@']):
                return v
            # Otherwise treat as path
            v = Path(v)
        if isinstance(v, Path) and not v.exists():
            raise ValueError(f"Ligand file not found: {v}")
        return v
    
    @root_validator
    def infer_is_pep(cls, values):
        if values.get('is_pep') is None:
            input_lig = values.get('input_ligand')
            if input_lig:
                if isinstance(input_lig, str):
                    values['is_pep'] = input_lig.startswith('pep')
                elif isinstance(input_lig, Path):
                    values['is_pep'] = str(input_lig).endswith('.pdb')
                else:
                    values['is_pep'] = False
            else:
                values['is_pep'] = False
        return values


class ModelConfig(BaseModel):
    """Model configuration"""
    checkpoint: Path = Field(default=Path("checkpoints/pocketxmol.ckpt"), description="Model checkpoint path")
    config_path: Optional[Path] = Field(None, description="Optional model config override")
    
    @validator('checkpoint')
    def validate_checkpoint(cls, v):
        if not v.exists():
            raise ValueError(f"Checkpoint not found: {v}")
        return v


# ============================================================================
# NOISE CONFIGURATION MODELS  
# ============================================================================

class NoiseLevel(BaseModel):
    """Noise level configuration"""
    name: LevelStrategy = Field(default=LevelStrategy.ADVANCE)
    min: float = Field(default=0.0, ge=0, le=1)
    max: float = Field(default=1.0, ge=0, le=1)
    step2level: Optional[Dict[str, float]] = Field(default_factory=lambda: {
        "scale_start": 0.99999,
        "scale_end": 0.00001, 
        "width": 3
    })
    
    @validator('max')
    def validate_max_greater_than_min(cls, v, values):
        if 'min' in values and v < values['min']:
            raise ValueError("max must be greater than or equal to min")
        return v


class NoiseConfig(BaseModel):
    """Noise configuration for denoising process"""
    name: NoiseType = Field(default=NoiseType.DOCK)
    num_steps: int = Field(default=100, ge=1, description="Number of denoising steps")
    prior: Union[Literal["from_train"], Dict[str, Any]] = Field(default="from_train")
    level: Union[NoiseLevel, Dict[str, Any]] = Field(default_factory=NoiseLevel)
    init_step: Optional[float] = Field(None, ge=0, le=1, description="Initial noise step for custom tasks")
    mapper: Optional[Dict[str, Any]] = Field(None, description="Custom noise mapping")
    
    class Config:
        extra = "allow"


# ============================================================================
# TRANSFORM CONFIGURATION MODELS
# ============================================================================

class AtomDistribution(BaseModel):
    """Atom number distribution for variable size molecules"""
    strategy: Literal["uniform", "mol_atoms_based", "gaussian"] = Field(default="mol_atoms_based")
    mean: Union[float, Dict[str, float]] = Field(default={"coef": 0, "bias": 28})
    std: Union[float, Dict[str, float]] = Field(default={"coef": 0, "bias": 2})
    min: int = Field(default=5, ge=1)
    max: Optional[int] = Field(None, ge=1)


class VariableMolSizeTransform(BaseModel):
    """Transform for variable molecule size generation"""
    name: Literal["variable_mol_size"] = Field(default="variable_mol_size")
    num_atoms_distri: AtomDistribution = Field(default_factory=AtomDistribution)


class VariableSCSizeTransform(BaseModel):
    """Transform for variable peptide side-chain size"""
    name: Literal["variable_sc_size"] = Field(default="variable_sc_size")
    applicable_tasks: List[str] = Field(default=["pepdesign"])
    num_atoms_distri: Dict[str, Any] = Field(default_factory=lambda: {
        "mean": 8,
        "std": {"coef": 0.3817, "bias": 1.8727}
    })
    not_remove: List[int] = Field(default_factory=list)


class FeaturizerPocket(BaseModel):
    """Pocket featurizer configuration"""
    center: Optional[List[float]] = Field(None, min_items=3, max_items=3)


class TransformConfig(BaseModel):
    """Transform configurations"""
    featurizer_pocket: Optional[FeaturizerPocket] = Field(None)
    variable_mol_size: Optional[VariableMolSizeTransform] = Field(None)
    variable_sc_size: Optional[VariableSCSizeTransform] = Field(None)
    
    class Config:
        extra = "allow"


# ============================================================================
# TASK-SPECIFIC CONFIGURATION MODELS
# ============================================================================

class DockingTaskConfig(BaseModel):
    """Configuration for molecular docking tasks"""
    name: Literal["dock"] = Field(default="dock")
    transform: Dict[str, Any] = Field(default_factory=lambda: {
        "name": "dock",
        "settings": {
            "free": 1,
            "flexible": 0
        }
    })
    
    def set_flexible_docking(self, flexible_prob: float = 1.0):
        """Enable flexible docking"""
        self.transform["settings"]["free"] = 1 - flexible_prob
        self.transform["settings"]["flexible"] = flexible_prob
        return self
    
    def set_fixed_atoms(self, atom_indices: List[int] = None, 
                       res_bb: List[int] = None, res_sc: List[int] = None):
        """Set fixed atoms for constrained docking"""
        if not hasattr(self.transform, "fix_some"):
            self.transform["fix_some"] = {}
        if atom_indices:
            self.transform["fix_some"]["atom"] = atom_indices
        if res_bb:
            self.transform["fix_some"]["res_bb"] = res_bb
        if res_sc:
            self.transform["fix_some"]["res_sc"] = res_sc
        return self


class SBDDTaskConfig(BaseModel):
    """Configuration for structure-based drug design"""
    name: Literal["sbdd"] = Field(default="sbdd")
    transform: Dict[str, Any] = Field(default_factory=lambda: {
        "name": "sbdd",
        "ar": False
    })
    
    def enable_autoregressive(self):
        """Enable auto-regressive refinement"""
        self.transform["ar"] = True
        return self


class PeptideDesignTaskConfig(BaseModel):
    """Configuration for peptide design"""
    name: Literal["pepdesign"] = Field(default="pepdesign")
    transform: Dict[str, Any] = Field(default_factory=lambda: {
        "name": "pepdesign",
        "mode": "full"
    })
    
    def set_mode(self, mode: Literal["full", "sc", "packing"]):
        """Set peptide design mode"""
        self.transform["mode"] = mode
        return self


class FragmentGrowingTaskConfig(BaseModel):
    """Configuration for fragment growing/linking"""
    name: Literal["maskfill"] = Field(default="maskfill")
    transform: Dict[str, Any] = Field(default_factory=lambda: {
        "name": "maskfill",
        "part1_pert": {
            "pos_pert": "gaussian",
            "pos_pert_kwargs": {"sigma": 0.3}
        }
    })
    
    def set_fragment_perturbation(self, sigma: float = 0.3):
        """Set fragment perturbation level"""
        self.transform["part1_pert"]["pos_pert_kwargs"]["sigma"] = sigma
        return self


class CustomTaskConfig(BaseModel):
    """Configuration for custom advanced tasks"""
    name: Literal["custom"] = Field(default="custom")
    transform: Dict[str, Any] = Field(...)
    
    @validator('transform')
    def validate_custom_transform(cls, v):
        required_keys = ["name", "is_peptide", "partition"]
        for key in required_keys:
            if key not in v:
                raise ValueError(f"Custom task transform must have '{key}' field")
        return v


# ============================================================================
# MAIN POCKETXMOL CONFIGURATION
# ============================================================================

class PocketXMolConfig(BaseModel):
    """Complete configuration for PocketXMol tasks"""
    sample: SampleConfig
    data: DataConfig
    task: Union[DockingTaskConfig, SBDDTaskConfig, PeptideDesignTaskConfig, 
                FragmentGrowingTaskConfig, CustomTaskConfig]
    noise: NoiseConfig
    transforms: Optional[TransformConfig] = Field(None)
    model: ModelConfig = Field(default_factory=ModelConfig)
    
    @root_validator
    def validate_config_consistency(cls, values):
        """Ensure configuration consistency across components"""
        task = values.get('task')
        noise = values.get('noise')
        
        # Match noise type to task type (with some flexibility)
        task_to_noise = {
            'dock': ['dock'],
            'sbdd': ['sbdd', 'dock'],
            'pepdesign': ['pepdesign'],
            'maskfill': ['maskfill'],
            'custom': ['custom']
        }
        
        if task and noise:
            task_name = task.name
            noise_name = noise.name
            if noise_name not in task_to_noise.get(task_name, [noise_name]):
                # Allow but warn about mismatch
                print(f"Warning: Noise type '{noise_name}' may not be optimal for task '{task_name}'")
        
        return values
    
    class Config:
        extra = "allow"


# ============================================================================
# PRESET CONFIGURATIONS
# ============================================================================

class PresetConfigs:
    """Factory for common configuration presets"""
    
    @staticmethod
    def small_molecule_docking(protein_path: Path, ligand: Union[Path, str],
                              pocket_center: List[float] = None,
                              ref_ligand: Path = None,
                              num_mols: int = 100) -> PocketXMolConfig:
        """Create configuration for small molecule docking"""
        pocket_args = PocketDefinition(
            pocket_coord=pocket_center,
            ref_ligand_path=ref_ligand,
            radius=15.0
        )
        
        return PocketXMolConfig(
            sample=SampleConfig(num_mols=num_mols, batch_size=50),
            data=DataConfig(
                protein_path=protein_path,
                input_ligand=ligand,
                pocket_args=pocket_args
            ),
            task=DockingTaskConfig(),
            noise=NoiseConfig(name=NoiseType.DOCK, num_steps=100)
        )
    
    @staticmethod
    def structure_based_drug_design(protein_path: Path,
                                   pocket_center: List[float] = None,
                                   ref_ligand: Path = None,
                                   num_mols: int = 100,
                                   use_refinement: bool = True) -> PocketXMolConfig:
        """Create configuration for SBDD"""
        pocket_args = PocketDefinition(
            pocket_coord=pocket_center,
            ref_ligand_path=ref_ligand,
            radius=15.0
        )
        
        task = SBDDTaskConfig()
        if use_refinement:
            task.enable_autoregressive()
        
        return PocketXMolConfig(
            sample=SampleConfig(num_mols=num_mols, batch_size=30),
            data=DataConfig(
                protein_path=protein_path,
                input_ligand=None,  # De novo design
                pocket_args=pocket_args
            ),
            task=task,
            noise=NoiseConfig(name=NoiseType.SBDD, num_steps=100),
            transforms=TransformConfig(
                variable_mol_size=VariableMolSizeTransform()
            )
        )
    
    @staticmethod
    def peptide_design(protein_path: Path,
                      peptide_length: int = 10,
                      pocket_center: List[float] = None,
                      ref_ligand: Path = None,
                      num_mols: int = 50) -> PocketXMolConfig:
        """Create configuration for peptide design"""
        pocket_args = PocketDefinition(
            pocket_coord=pocket_center,
            ref_ligand_path=ref_ligand,
            radius=15.0
        )
        
        return PocketXMolConfig(
            sample=SampleConfig(num_mols=num_mols, batch_size=20),
            data=DataConfig(
                protein_path=protein_path,
                input_ligand=f"peplen_{peptide_length}",
                is_pep=True,
                pocket_args=pocket_args
            ),
            task=PeptideDesignTaskConfig(),
            noise=NoiseConfig(name=NoiseType.PEPDESIGN, num_steps=100),
            transforms=TransformConfig(
                variable_sc_size=VariableSCSizeTransform()
            )
        )
    
    @staticmethod
    def peptide_docking(protein_path: Path,
                       peptide: Union[Path, str],
                       pocket_center: List[float] = None,
                       ref_ligand: Path = None,
                       num_mols: int = 50) -> PocketXMolConfig:
        """Create configuration for peptide docking"""
        pocket_args = PocketDefinition(
            pocket_coord=pocket_center,
            ref_ligand_path=ref_ligand,
            radius=15.0
        )
        
        # Handle peptide input format
        if isinstance(peptide, str) and not peptide.startswith('pepseq_'):
            # Assume it's a sequence, prepend pepseq_
            if peptide.isalpha() and peptide.isupper():
                peptide = f"pepseq_{peptide}"
        
        return PocketXMolConfig(
            sample=SampleConfig(num_mols=num_mols, batch_size=20),
            data=DataConfig(
                protein_path=protein_path,
                input_ligand=peptide,
                is_pep=True,
                pocket_args=pocket_args
            ),
            task=DockingTaskConfig(),
            noise=NoiseConfig(name=NoiseType.DOCK, num_steps=100)
        )


# ============================================================================
# MAIN SERVER CLASS
# ============================================================================

class PocketXMolServer:
    """Main server class for PocketXMol molecular generation"""
    
    def __init__(self, device: str = "cuda:0", num_workers: int = 4):
        """
        Initialize the PocketXMol server
        
        Args:
            device: PyTorch device to use
            num_workers: Number of workers for data loading
        """
        self.device = device
        self.num_workers = num_workers
        self.model = None
        self.train_config = None
        self.logger = None
        
    def _setup_logging(self, output_dir: Path, config_name: str) -> Tuple[Any, Path]:
        """Setup logging and output directories"""
        log_dir = get_new_log_dir(str(output_dir), prefix=config_name)
        logger = get_logger('sample', log_dir)
        writer = torch.utils.tensorboard.SummaryWriter(log_dir)
        
        # Create output directories
        sdf_dir = Path(log_dir) / 'SDF'
        pure_sdf_dir = Path(log_dir) / f'{Path(log_dir).name}_SDF'
        sdf_dir.mkdir(exist_ok=True, parents=True)
        pure_sdf_dir.mkdir(exist_ok=True, parents=True)
        
        return logger, Path(log_dir)
    
    def _load_model(self, config: PocketXMolConfig):
        """Load the model and training configuration"""
        # Load checkpoint
        ckpt = torch.load(config.model.checkpoint, map_location=self.device, weights_only=False)
        
        # Load training config
        cfg_dir = Path(str(config.model.checkpoint).replace('checkpoints', 'train_config'))
        train_config_files = list(cfg_dir.glob('*.yml'))
        if not train_config_files:
            raise ValueError(f"No training config found in {cfg_dir}")
        
        self.train_config = make_config(str(train_config_files[0]))
        
        # Setup data module for transforms
        dm = DataModule(self.train_config)
        in_dims = dm.get_in_dims()
        
        # Load model
        if self.train_config.model.name == 'pm_asym_denoiser':
            self.model = PMAsymDenoiser(config=self.train_config.model, **in_dims).to(self.device)
        
        # Load state dict
        state_dict = {k[6:]: v for k, v in ckpt['state_dict'].items() if k.startswith('model.')}
        self.model.load_state_dict(state_dict)
        self.model.eval()
        
        return dm
    
    def _prepare_data(self, config: PocketXMolConfig) -> Tuple[Dict, str, Any]:
        """Prepare input data for generation"""
        # Get pocket and input data
        ref_ligand = config.data.pocket_args.ref_ligand_path
        pocket_coord = config.data.pocket_args.pocket_coord
        
        if ref_ligand is not None:
            pass  # Use ref_ligand_path
        elif pocket_coord is not None:
            ref_ligand = make_dummy_mol_with_coordinate(pocket_coord)
        else:
            # Use input_ligand as reference
            if config.data.input_ligand is None:
                raise ValueError("No pocket definition provided")
            ref_ligand = config.data.input_ligand
        
        # Extract pocket
        pocket_pdb = extract_pocket(
            str(config.data.protein_path),
            str(ref_ligand) if isinstance(ref_ligand, Path) else ref_ligand,
            radius=config.data.pocket_args.radius,
            criterion=config.data.pocket_args.criterion
        )
        
        # Process input ligand
        pocmol_data, mol = get_input_from_file(
            str(config.data.input_ligand) if config.data.input_ligand else None,
            pocket_pdb, 
            return_mol=True,
            **config.data.pocmol_args.dict()
        )
        
        # Add peptide info if needed
        if config.data.is_pep:
            if config.data.input_ligand:
                if isinstance(config.data.input_ligand, Path) and str(config.data.input_ligand).endswith('.pdb'):
                    pep_info = get_peptide_info(str(config.data.input_ligand))
                elif isinstance(config.data.input_ligand, str) and config.data.input_ligand.startswith('peplen_'):
                    pep_info = add_pep_bb_data(pocmol_data)
                else:
                    pep_info = {}
                pocmol_data.update(pep_info)
        
        return pocmol_data, pocket_pdb, mol
    
    def _create_transforms(self, config: PocketXMolConfig, dm: DataModule) -> Any:
        """Create transformation pipeline"""
        # Update training config with sample transforms
        if config.transforms:
            for trans_name, trans_config in config.transforms.dict(exclude_none=True).items():
                if trans_name in self.train_config.transforms:
                    self.train_config.transforms[trans_name].update(trans_config)
        
        # Get base transforms
        featurizer_list = dm.get_featurizers()
        featurizer = featurizer_list[-1]
        
        # Get task transform
        task_transform = get_transforms(config.task.transform, mode='use')
        
        # Build transform pipeline
        transforms = featurizer_list + [task_transform]
        
        # Add variable size transforms if specified
        if config.transforms:
            if config.transforms.variable_mol_size:
                transforms.insert(-1, get_transforms(config.transforms.variable_mol_size.dict()))
            elif config.transforms.variable_sc_size:
                transforms.insert(-1, get_transforms(config.transforms.variable_sc_size.dict()))
        
        # Add additional data transforms
        addition_transforms = [get_transforms(tr) for tr in config.data.transforms]
        transforms = Compose(transforms + addition_transforms)
        
        return transforms, featurizer
    
    def _save_results(self, mol_info: Dict, output_dir: Path, index: int, is_pep: bool) -> Dict:
        """Save generated molecule to files"""
        rdmol = mol_info['rdmol']
        tag = mol_info.get('tag', '')
        filename_base = str(index) + (f'-{tag}' if tag else '')
        
        pure_sdf_dir = output_dir / f'{output_dir.name}_SDF'
        sdf_dir = output_dir / 'SDF'
        
        # Save PDB if peptide
        if is_pep and 'pdb_struc' in mol_info:
            pdb_struc = mol_info['pdb_struc']
            filename_pdb = filename_base + '.pdb'
            pdb_io = PDBIO()
            pdb_io.set_structure(pdb_struc)
            pdb_io.save(str(pure_sdf_dir / filename_pdb))
        
        # Save SDF
        filename_sdf = filename_base + ('.sdf' if not is_pep else '_mol.sdf')
        if tag != 'bad':
            Chem.MolToMolFile(rdmol, str(pure_sdf_dir / filename_sdf))
        else:
            with open(pure_sdf_dir / filename_sdf, 'w+') as f:
                f.write(rdmol)
        
        # Calculate confidence scores
        output = mol_info['output']
        cfd_traj = get_cfd_traj(output['confidence_pos_traj'])
        cfd_pos = output['confidence_pos'].detach().cpu().numpy().mean()
        cfd_node = output['confidence_node'].detach().cpu().numpy().mean()
        cfd_edge = output['confidence_halfedge'].detach().cpu().numpy().mean()
        
        # Return info dict
        return {
            'filename': filename_sdf if not is_pep else filename_pdb,
            'smiles': mol_info.get('smiles', ''),
            'tag': tag,
            'cfd_traj': cfd_traj,
            'cfd_pos': cfd_pos,
            'cfd_node': cfd_node,
            'cfd_edge': cfd_edge,
            **({"aaseq": mol_info.get('aaseq', '')} if is_pep else {})
        }
    
    def generate(self, config: PocketXMolConfig, output_dir: Path = Path("./outputs")) -> pd.DataFrame:
        """
        Generate molecules using the provided configuration
        
        Args:
            config: PocketXMol configuration
            output_dir: Directory to save outputs
            
        Returns:
            DataFrame with generation results
        """
        # Setup
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set seed
        seed = config.sample.seed + np.sum([ord(s) for s in str(output_dir)])
        seed_all(seed)
        
        # Setup logging
        config_name = f"{config.task.name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.logger, log_dir = self._setup_logging(output_dir, config_name)
        
        self.logger.info(f"Configuration: {config.dict()}")
        
        # Save configuration
        config_path = log_dir / "config.json"
        with open(config_path, 'w') as f:
            f.write(config.json(indent=2))
        
        # Load model if not loaded
        if self.model is None:
            self.logger.info("Loading model...")
            dm = self._load_model(config)
        else:
            dm = DataModule(self.train_config)
        
        # Prepare data
        self.logger.info("Preparing data...")
        pocmol_data, pocket_pdb, in_mol = self._prepare_data(config)
        
        # Save input structures
        input_dir = log_dir / f'{log_dir.name}_SDF' / '0_inputs'
        input_dir.mkdir(exist_ok=True, parents=True)
        
        with open(input_dir / 'pocket_block.pdb', 'w') as f:
            f.write(pocket_pdb)
        if in_mol:
            Chem.MolToMolFile(in_mol, str(input_dir / 'input_mol.sdf'))
        
        # Create transforms
        self.logger.info("Setting up transforms...")
        transforms, featurizer = self._create_transforms(config, dm)
        
        # Create dataset and dataloader
        test_set = UseDataset(pocmol_data, n=config.sample.num_mols, 
                             task=config.task.name, transforms=transforms)
        
        follow_batch = sum([getattr(t, 'follow_batch', []) for t in transforms.transforms], [])
        exclude_keys = sum([getattr(t, 'exclude_keys', []) for t in transforms.transforms], [])
        
        test_loader = DataLoader(
            test_set, 
            config.sample.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            follow_batch=follow_batch,
            exclude_keys=exclude_keys
        )
        
        # Setup noiser
        in_dims = dm.get_in_dims()
        noiser = get_sample_noiser(
            config.noise.dict(), 
            in_dims['num_node_types'],
            in_dims['num_edge_types'],
            mode='sample',
            device=self.device,
            ref_config=self.train_config.noise
        )
        
        # Generation loop
        self.logger.info(f"Starting generation (num_mols={config.sample.num_mols})...")
        
        results = []
        i_saved = 0
        is_ar = config.task.transform.get('ar', False) if hasattr(config.task.transform, 'get') else False
        
        pool = EasyDict({
            'succ': [],
            'bad': [],
            'incomp': [],
            **({'nonstd': []} if config.data.is_pep else {})
        })
        
        for batch in tqdm(test_loader, desc="Generating molecules"):
            if i_saved >= config.sample.num_mols:
                break
            
            # Generate molecules
            batch = batch.to(self.device)
            batch, outputs, trajs = sample_loop3(batch, self.model, noiser, self.device, is_ar=is_ar)
            
            # Decode outputs
            info_keys = ['data_id', 'db', 'task', 'key']
            data_list = [{key: batch[key][i] for key in info_keys if key in batch} 
                        for i in range(len(batch))]
            generated_list, outputs_list, traj_list_dict = seperate_outputs2(batch, outputs, trajs)
            
            # Process each molecule
            for i_mol in range(len(generated_list)):
                if i_saved >= config.sample.num_mols:
                    break
                
                # Decode molecule
                mol_info = featurizer.decode_output(**generated_list[i_mol])
                mol_info.update(data_list[i_mol] if i_mol < len(data_list) else {})
                
                # Reconstruct molecule
                try:
                    if not config.data.is_pep:
                        with CaptureLogger():
                            rdmol = reconstruct_from_generated_with_edges(mol_info, in_mol=in_mol)
                        smiles = Chem.MolToSmiles(rdmol)
                        
                        if '.' in smiles:
                            tag = 'incomp'
                            pool.incomp.append(mol_info)
                        else:
                            tag = ''
                            pool.succ.append(mol_info)
                    else:
                        with CaptureLogger():
                            pdb_struc, rdmol = reconstruct_pdb_from_generated(
                                mol_info, 
                                gt_path=str(config.data.input_ligand) if config.data.input_ligand else None
                            )
                        aaseq = seq1(''.join(res.resname for res in pdb_struc.get_residues()))
                        
                        if rdmol is None:
                            rdmol = Chem.MolFromSmiles('')
                        smiles = Chem.MolToSmiles(rdmol)
                        
                        if '.' in smiles:
                            tag = 'incomp'
                            pool.incomp.append(mol_info)
                        elif 'X' in aaseq:
                            tag = 'nonstd'
                            pool.nonstd.append(mol_info)
                        else:
                            tag = ''
                            pool.succ.append(mol_info)
                        
                        mol_info['pdb_struc'] = pdb_struc
                        mol_info['aaseq'] = aaseq
                        
                except MolReconsError:
                    pool.bad.append(mol_info)
                    smiles = ''
                    tag = 'bad'
                    rdmol = create_sdf_string(mol_info)
                    if config.data.is_pep:
                        aaseq = ''
                        pdb_struc = PDB.Structure.Structure('bad')
                        mol_info['pdb_struc'] = pdb_struc
                        mol_info['aaseq'] = aaseq
                
                mol_info.update({
                    'rdmol': rdmol,
                    'smiles': smiles,
                    'tag': tag,
                    'output': outputs_list[i_mol]
                })
                
                # Save results
                result_info = self._save_results(mol_info, log_dir, i_saved, config.data.is_pep)
                result_info.update({k: mol_info.get(k, '') for k in info_keys if k in mol_info})
                results.append(result_info)
                i_saved += 1
            
            # Clean up
            del batch, outputs, trajs
            if self.device != 'cpu':
                with torch.cuda.device(self.device):
                    torch.cuda.empty_cache()
            gc.collect()
        
        # Print statistics
        self.logger.info(f"Generation complete. Success: {len(pool.succ)}, "
                        f"Incomplete: {len(pool.incomp)}, Bad: {len(pool.bad)}"
                        + (f", Non-standard: {len(pool.nonstd)}" if config.data.is_pep else ""))
        
        # Save results DataFrame
        df = pd.DataFrame(results)
        df.to_csv(log_dir / 'gen_info.csv', index=False)
        
        return df
    
    def generate_from_dict(self, config_dict: Dict[str, Any], output_dir: Path = Path("./outputs")) -> pd.DataFrame:
        """
        Generate molecules from a dictionary configuration
        
        Args:
            config_dict: Configuration dictionary
            output_dir: Output directory
            
        Returns:
            DataFrame with generation results
        """
        config = PocketXMolConfig(**config_dict)
        return self.generate(config, output_dir)


# ============================================================================
# CONVENIENCE FUNCTIONS
# ============================================================================

def quick_dock(protein_path: str, ligand: str, 
               pocket_center: List[float] = None,
               num_mols: int = 100,
               device: str = "cuda:0") -> pd.DataFrame:
    """
    Quick docking function
    
    Args:
        protein_path: Path to protein PDB
        ligand: Ligand (SMILES or path)
        pocket_center: Pocket center coordinates
        num_mols: Number of molecules to generate
        device: PyTorch device
        
    Returns:
        DataFrame with results
    """
    config = PresetConfigs.small_molecule_docking(
        Path(protein_path), ligand, pocket_center, num_mols=num_mols
    )
    server = PocketXMolServer(device=device)
    return server.generate(config)


def quick_design(protein_path: str,
                pocket_center: List[float] = None, 
                num_mols: int = 100,
                device: str = "cuda:0") -> pd.DataFrame:
    """
    Quick drug design function
    
    Args:
        protein_path: Path to protein PDB
        pocket_center: Pocket center coordinates  
        num_mols: Number of molecules to generate
        device: PyTorch device
        
    Returns:
        DataFrame with results
    """
    config = PresetConfigs.structure_based_drug_design(
        Path(protein_path), pocket_center, num_mols=num_mols
    )
    server = PocketXMolServer(device=device)
    return server.generate(config)


if __name__ == "__main__":
    # Example usage
    print("PocketXMol Server ready for use")
    print("Use PresetConfigs for quick configurations or create custom PocketXMolConfig objects")