# PocketXMol Server

A modular, Pydantic-based Python interface for the PocketXMol molecular generation foundation model. This server provides a clean programmatic API that replaces CLI-based workflows with type-safe, validated configuration objects.

## 🚀 Quick Start

```python
from pocketxmol_server import quick_dock, quick_design

# Quick molecule docking
results = quick_dock(
    protein_path="protein.pdb",
    ligand="CC(C)(C)c1ccccc1",  # SMILES string
    pocket_center=[10.0, 15.0, 20.0],
    num_mols=100
)

# Quick drug design
results = quick_design(
    protein_path="protein.pdb", 
    pocket_center=[10.0, 15.0, 20.0],
    num_mols=100
)
```

## 📋 Features

- **Type-Safe Configuration**: Pydantic models with automatic validation
- **Modular Task System**: Separate configs for docking, SBDD, peptide design, etc.
- **Preset Configurations**: Ready-to-use configs for common tasks
- **Programmatic Interface**: No more YAML files or CLI arguments
- **Batch Processing**: Handle multiple targets programmatically
- **Extensible Architecture**: Easy to add custom tasks and constraints

## 🏗️ Architecture

The server is built around these core components:

### Configuration Models

```python
# Base configuration
class PocketXMolConfig(BaseModel):
    sample: SampleConfig      # Generation parameters
    data: DataConfig          # Input data and pocket definition  
    task: TaskConfig          # Task-specific settings
    noise: NoiseConfig        # Denoising process parameters
    transforms: TransformConfig  # Optional molecular transforms
    model: ModelConfig        # Model checkpoint settings
```

### Task-Specific Configurations

- **`DockingTaskConfig`**: Molecular and peptide docking
- **`SBDDTaskConfig`**: Structure-based drug design
- **`PeptideDesignTaskConfig`**: De novo peptide design
- **`FragmentGrowingTaskConfig`**: Fragment growing and linking
- **`CustomTaskConfig`**: Advanced custom constraints

## 📖 Usage Examples

### 1. Small Molecule Docking

```python
from pocketxmol_server import PocketXMolServer, PresetConfigs

# Using preset configuration
config = PresetConfigs.small_molecule_docking(
    protein_path=Path("protein.pdb"),
    ligand="CC(C)(C#N)c1cccc(C(=O)Nc2ccc(F)cc2)c1",  # SMILES
    pocket_center=[15.0, 20.0, 25.0],
    num_mols=100
)

server = PocketXMolServer(device="cuda:0")
results = server.generate(config, output_dir=Path("outputs/docking"))
```

### 2. Structure-Based Drug Design

```python
# De novo drug design with auto-regressive refinement
config = PresetConfigs.structure_based_drug_design(
    protein_path=Path("protein.pdb"),
    pocket_center=[10.5, -15.2, 22.8], 
    num_mols=100,
    use_refinement=True
)

server = PocketXMolServer(device="cuda:0")
results = server.generate(config, output_dir=Path("outputs/sbdd"))
```

### 3. Peptide Design

```python
# Design 8-mer peptides for a pocket
config = PresetConfigs.peptide_design(
    protein_path=Path("protein.pdb"),
    peptide_length=8,
    pocket_center=[5.0, 10.0, 15.0],
    num_mols=50
)

server = PocketXMolServer(device="cuda:0") 
results = server.generate(config, output_dir=Path("outputs/peptides"))
```

### 4. Advanced Custom Configuration

```python
from pocketxmol_server import *

# Manual configuration with full control
config = PocketXMolConfig(
    sample=SampleConfig(
        seed=2024,
        batch_size=50,
        num_mols=100,
        save_traj_prob=0.02
    ),
    data=DataConfig(
        protein_path=Path("protein.pdb"),
        input_ligand="ligand.sdf",
        pocket_args=PocketDefinition(
            pocket_coord=[15.0, 20.0, 25.0],
            radius=15.0
        )
    ),
    task=DockingTaskConfig().set_flexible_docking(flexible_prob=0.3),
    noise=NoiseConfig(
        name=NoiseType.DOCK,
        num_steps=150,
        level=NoiseLevel(min=0.0, max=1.0)
    )
)
```

### 5. Programmatic Modifications

```python
# Start with preset and modify
config = PresetConfigs.small_molecule_docking(...)

# Modify parameters programmatically
config.sample.batch_size = 100
config.sample.num_mols = 500
config.noise.num_steps = 200
config.data.pocket_args.radius = 12.0

# Add transforms
config.transforms = TransformConfig(
    variable_mol_size=VariableMolSizeTransform(
        num_atoms_distri={
            "mean": 30,
            "std": 5,
            "min": 20,
            "max": 40
        }
    )
)
```

### 6. Batch Processing

```python
# Process multiple targets
tasks = [
    ("target1", PresetConfigs.small_molecule_docking(...)),
    ("target2", PresetConfigs.structure_based_drug_design(...)),
    ("target3", PresetConfigs.peptide_design(...))
]

server = PocketXMolServer(device="cuda:0")
results = {}

for name, config in tasks:
    results[name] = server.generate(
        config, 
        output_dir=Path(f"outputs/batch/{name}")
    )
```

## 🎯 Supported Tasks

### Molecular Docking
- **Small molecule docking**: Standard and flexible docking
- **Peptide docking**: With sequence or PDB input
- **Constrained docking**: Fix specific atoms or residues

### Structure-Based Drug Design
- **De novo design**: Generate molecules from scratch
- **Auto-regressive refinement**: Multi-round optimization
- **Size-controlled design**: Custom molecular weight distributions

### Peptide Design
- **Full peptide design**: Complete sequence generation
- **Side-chain design**: Fixed backbone, variable side-chains
- **Constrained design**: Custom amino acid preferences

### Fragment-Based Design
- **Fragment growing**: Extend molecular fragments
- **Fragment linking**: Connect separate fragments
- **Flexible positioning**: Adjust fragment poses

### Custom Tasks
- **Advanced constraints**: Complex molecular partitioning
- **Multi-component systems**: Handle multiple ligands
- **Custom noise schedules**: Fine-tune generation process

## 📊 Input/Output Formats

### Input Formats

**Proteins:**
- PDB files: `protein.pdb`

**Small Molecules:**
- SDF files: `ligand.sdf` 
- SMILES strings: `"CC(C)(C)c1ccccc1"`
- `None` for de novo design

**Peptides:**
- PDB files: `peptide.pdb`
- Sequences: `"pepseq_DTVFALFW"` or `"DTVFALFW"`
- Length specification: `"peplen_10"`

**Pocket Definition:**
- Coordinates + radius: `pocket_coord=[x, y, z], radius=15.0`
- Reference ligand: `ref_ligand_path="reference.sdf"`
- Auto-detection: Use input ligand as reference

### Output Structure

```
outputs/
└── {experiment_name}/
    ├── {experiment_name}_SDF/      # Generated molecules
    │   ├── 0_inputs/              # Input structures
    │   ├── 0.sdf                  # Molecule 0
    │   ├── 1-incomp.sdf           # Incomplete molecule 1
    │   └── ...
    ├── SDF/                       # Generation trajectories
    ├── gen_info.csv              # Metadata and scores
    └── config.json               # Used configuration
```

### Result DataFrame

```python
results = server.generate(config)
# results is a pandas DataFrame with columns:
# - filename: output file name
# - smiles: SMILES string (molecules) 
# - aaseq: amino acid sequence (peptides)
# - tag: success/incomp/bad/nonstd status
# - cfd_pos/node/edge: confidence scores
# - data_id, pdbid: metadata
```

## ⚙️ Configuration Reference

### Core Parameters

```python
# Sample configuration
SampleConfig(
    seed=2024,                 # Random seed
    batch_size=50,             # GPU batch size
    num_mols=100,              # Total molecules to generate  
    num_repeats=1,             # Generation rounds
    save_traj_prob=0.02        # Trajectory saving probability
)

# Data configuration  
DataConfig(
    protein_path=Path("..."),  # Protein PDB file
    input_ligand="...",        # Ligand input (various formats)
    is_pep=False,              # Peptide flag (auto-detected)
    pocket_args=PocketDefinition(...),  # Pocket definition
    pocmol_args=PocMolMetadata(...)     # Experiment metadata
)

# Noise configuration
NoiseConfig(
    name=NoiseType.DOCK,       # dock/sbdd/pepdesign/maskfill/custom
    num_steps=100,             # Denoising steps
    level=NoiseLevel(          # Information preservation
        min=0.0,               # Minimum preservation level
        max=1.0                # Maximum preservation level  
    )
)
```

### Task-Specific Settings

```python
# Docking with flexibility
DockingTaskConfig().set_flexible_docking(flexible_prob=0.3)

# Fixed atom constraints  
config.task.set_fixed_atoms(
    atom_indices=[0, 1, 2, 3],
    res_bb=[0, 1],             # Fixed backbone residues
    res_sc=[0]                 # Fixed side-chain residues
)

# SBDD with refinement
SBDDTaskConfig().enable_autoregressive()

# Peptide design modes
PeptideDesignTaskConfig().set_mode("full")  # full/sc/packing
```

## 🔧 Performance Tips

### GPU Memory Optimization

```python
# Adjust batch size based on GPU memory
# Small molecules: 50-100
# Peptides: 20-50  
# Complex tasks: 10-30

config.sample.batch_size = 50  # Adjust as needed
```

### Quality vs Speed Trade-offs

```python
# Higher quality (slower)
config.noise.num_steps = 200
config.sample.save_traj_prob = 0.1

# Faster generation (lower quality) 
config.noise.num_steps = 50
config.sample.save_traj_prob = 0.01
```

### Noise Level Guidelines

```python
# De novo generation
NoiseLevel(min=0.0, max=1.0)

# Constrained generation (preserve some input)
NoiseLevel(min=0.3, max=0.9)  

# Fine-tuning existing structures
NoiseLevel(min=0.7, max=1.0)
```

## 🧪 Validation Features

The server includes automatic validation for:

- **File existence**: Protein/ligand files must exist
- **Format compatibility**: Correct file extensions and formats
- **Parameter ranges**: Sensible values for all numeric parameters
- **Configuration consistency**: Matching task and noise types
- **Pocket definitions**: Valid pocket specification methods
- **Input compatibility**: Ligand format matches task requirements

## 🔌 Extensibility

### Adding Custom Tasks

```python
class MyCustomTaskConfig(BaseModel):
    name: Literal["my_task"] = "my_task"
    transform: Dict[str, Any] = {...}
    
    def set_custom_parameter(self, value):
        self.transform["custom_param"] = value
        return self
```

### Custom Validation

```python
class MyDataConfig(DataConfig):
    @validator('input_ligand')
    def validate_my_format(cls, v):
        # Add custom validation logic
        return v
```

## 📚 Examples

See the included example files:

- **`quick_start.py`**: Immediate usage examples
- **`examples_server_usage.py`**: Comprehensive task demonstrations
- **10 detailed examples** covering all major use cases

Run examples:

```bash
# Quick start demo
python quick_start.py

# Specific example
python examples_server_usage.py --example docking

# All examples  
python examples_server_usage.py --example all
```

## 🆚 Comparison: CLI vs Server

| Feature | CLI (old) | Server (new) |
|---------|-----------|--------------|
| Configuration | YAML files | Pydantic models |
| Validation | Runtime errors | Compile-time validation |
| Type Safety | None | Full type hints |
| Modularity | Monolithic configs | Composable components |
| Batch Processing | Manual scripts | Built-in support |
| IDE Support | Limited | Full autocomplete |
| Error Messages | Generic | Specific field validation |
| Programmatic Use | Difficult | Native support |

### Migration Example

**Old CLI approach:**
```bash
python scripts/sample_use.py \
    --config_task configs/dock_smallmol.yml \
    --outdir outputs \
    --batch_size 100
```

**New Server approach:**
```python
config = PresetConfigs.small_molecule_docking(...)
config.sample.batch_size = 100
server = PocketXMolServer()
results = server.generate(config, Path("outputs"))
```

## 🐛 Troubleshooting

### Common Issues

**Out of Memory:**
```python
config.sample.batch_size = 10  # Reduce batch size
```

**Poor Results:**  
```python
config.noise.num_steps = 200   # More denoising steps
config.noise.level.min = 0.1   # Preserve more information
```

**Validation Errors:**
```python
# Check file paths
assert config.data.protein_path.exists()

# Verify input formats
print(config.data.input_ligand)  # Should be Path, SMILES, or special string
```

**Configuration Errors:**
```python
# Validate before running
config.dict()  # Will raise validation errors if any
```

## 📄 License

Same license as the original PocketXMol project.

## 🤝 Contributing

To add new task types or configuration options:

1. Create new Pydantic models in `pocketxmol_server.py`
2. Add validation methods as needed
3. Update the `PocketXMolConfig` union types
4. Add preset configurations in `PresetConfigs`
5. Create examples in `examples_server_usage.py`

---

**Ready to generate molecules?** Start with `quick_start.py` or dive into the full examples! 🧬✨