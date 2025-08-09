"""
Example usage scripts for PocketXMol Server
Demonstrates various molecular generation tasks using the Pydantic-based API
"""

from pathlib import Path
from pocketxmol_server import (
    PocketXMolServer,
    PocketXMolConfig,
    PresetConfigs,
    
    # Core configs
    SampleConfig,
    DataConfig,
    ModelConfig,
    NoiseConfig,
    TransformConfig,
    
    # Task configs
    DockingTaskConfig,
    SBDDTaskConfig,
    PeptideDesignTaskConfig,
    FragmentGrowingTaskConfig,
    CustomTaskConfig,
    
    # Supporting configs
    PocketDefinition,
    PocMolMetadata,
    VariableMolSizeTransform,
    VariableSCSizeTransform,
    NoiseLevel,
    NoiseType,
    LevelStrategy,
    
    # Quick functions
    quick_dock,
    quick_design
)


# ============================================================================
# EXAMPLE 1: SMALL MOLECULE DOCKING
# ============================================================================

def example_small_molecule_docking():
    """Example: Dock a small molecule to a protein pocket"""
    
    # Method 1: Using preset configuration
    config = PresetConfigs.small_molecule_docking(
        protein_path=Path("data/examples/1a2b/1a2b_protein.pdb"),
        ligand="CC(C)(C#N)c1cccc(C(=O)Nc2ccc(F)cc2)c1",  # SMILES string
        pocket_center=[15.0, 20.0, 25.0],  # Pocket center coordinates
        num_mols=100
    )
    
    # Method 2: Manual configuration with more control
    config = PocketXMolConfig(
        sample=SampleConfig(
            seed=2024,
            batch_size=50,
            num_mols=100,
            save_traj_prob=0.02
        ),
        data=DataConfig(
            protein_path=Path("data/examples/1a2b/1a2b_protein.pdb"),
            input_ligand="data/examples/dock/ligand.sdf",  # SDF file
            pocket_args=PocketDefinition(
                pocket_coord=[15.0, 20.0, 25.0],
                radius=15.0
            ),
            pocmol_args=PocMolMetadata(
                data_id="dock_example",
                pdbid="1a2b"
            )
        ),
        task=DockingTaskConfig(),
        noise=NoiseConfig(
            name=NoiseType.DOCK,
            num_steps=100,
            level=NoiseLevel(
                name=LevelStrategy.ADVANCE,
                min=0.0,
                max=1.0
            )
        )
    )
    
    # Run generation
    server = PocketXMolServer(device="cuda:0")
    results = server.generate(config, output_dir=Path("outputs/docking"))
    
    print(f"Generated {len(results)} molecules")
    print(f"Success rate: {(results['tag'] == '').sum() / len(results) * 100:.1f}%")
    print(f"Average confidence: {results['cfd_pos'].mean():.3f}")
    
    return results


# ============================================================================
# EXAMPLE 2: FLEXIBLE DOCKING
# ============================================================================

def example_flexible_docking():
    """Example: Flexible docking with partial constraints"""
    
    config = PocketXMolConfig(
        sample=SampleConfig(num_mols=50, batch_size=25),
        data=DataConfig(
            protein_path=Path("data/examples/1a2b/1a2b_protein.pdb"),
            input_ligand=Path("data/examples/dock/ligand.sdf"),
            pocket_args=PocketDefinition(
                ref_ligand_path=Path("data/examples/dock/ref_ligand.sdf"),
                radius=12.0
            )
        ),
        task=DockingTaskConfig().set_flexible_docking(flexible_prob=0.5),
        noise=NoiseConfig(name=NoiseType.DOCK, num_steps=100)
    )
    
    # Add fixed atoms constraint
    config.task.set_fixed_atoms(atom_indices=[0, 1, 2, 3])  # Fix first 4 atoms
    
    server = PocketXMolServer(device="cuda:0")
    results = server.generate(config, output_dir=Path("outputs/flexible_docking"))
    
    return results


# ============================================================================
# EXAMPLE 3: STRUCTURE-BASED DRUG DESIGN (SBDD)
# ============================================================================

def example_drug_design():
    """Example: De novo drug design in a protein pocket"""
    
    # Method 1: Using preset
    config = PresetConfigs.structure_based_drug_design(
        protein_path=Path("data/examples/5d3n/5d3n_protein.pdb"),
        pocket_center=[10.5, -15.2, 22.8],
        num_mols=100,
        use_refinement=True  # Enable auto-regressive refinement
    )
    
    # Method 2: Manual configuration with custom molecule size distribution
    config = PocketXMolConfig(
        sample=SampleConfig(
            num_mols=100,
            batch_size=30  # Smaller batch for SBDD
        ),
        data=DataConfig(
            protein_path=Path("data/examples/5d3n/5d3n_protein.pdb"),
            input_ligand=None,  # No input ligand for de novo design
            pocket_args=PocketDefinition(
                pocket_coord=[10.5, -15.2, 22.8],
                radius=15.0
            )
        ),
        task=SBDDTaskConfig().enable_autoregressive(),
        noise=NoiseConfig(
            name=NoiseType.SBDD,
            num_steps=150  # More steps for better quality
        ),
        transforms=TransformConfig(
            variable_mol_size=VariableMolSizeTransform(
                num_atoms_distri={
                    "strategy": "mol_atoms_based",
                    "mean": {"coef": 0, "bias": 25},  # Target ~25 atoms
                    "std": {"coef": 0, "bias": 3},
                    "min": 10,
                    "max": 35
                }
            )
        )
    )
    
    server = PocketXMolServer(device="cuda:0")
    results = server.generate(config, output_dir=Path("outputs/drug_design"))
    
    # Analyze results
    successful = results[results['tag'] == '']
    print(f"Generated {len(successful)} complete molecules")
    print(f"SMILES examples: {successful['smiles'].head(5).tolist()}")
    
    return results


# ============================================================================
# EXAMPLE 4: PEPTIDE DESIGN
# ============================================================================

def example_peptide_design():
    """Example: Design peptides for a protein pocket"""
    
    # Method 1: Using preset
    config = PresetConfigs.peptide_design(
        protein_path=Path("data/examples/hot136E/protein.pdb"),
        peptide_length=8,  # Design 8-mer peptides
        pocket_center=[5.0, 10.0, 15.0],
        num_mols=50
    )
    
    # Method 2: Manual configuration with custom side-chain variability
    config = PocketXMolConfig(
        sample=SampleConfig(
            num_mols=50,
            batch_size=20  # Smaller batch for peptides
        ),
        data=DataConfig(
            protein_path=Path("data/examples/hot136E/protein.pdb"),
            input_ligand="peplen_10",  # Design 10-mer peptides
            is_pep=True,
            pocket_args=PocketDefinition(
                pocket_coord=[5.0, 10.0, 15.0],
                radius=18.0  # Larger radius for peptides
            )
        ),
        task=PeptideDesignTaskConfig().set_mode("full"),  # Full peptide design
        noise=NoiseConfig(
            name=NoiseType.PEPDESIGN,
            num_steps=100
        ),
        transforms=TransformConfig(
            variable_sc_size=VariableSCSizeTransform(
                num_atoms_distri={
                    "mean": 10,
                    "std": {"coef": 0.4, "bias": 2}
                }
            )
        )
    )
    
    server = PocketXMolServer(device="cuda:0")
    results = server.generate(config, output_dir=Path("outputs/peptide_design"))
    
    # Analyze peptide sequences
    successful = results[results['tag'] == '']
    print(f"Generated {len(successful)} peptides")
    print(f"Sequences: {successful['aaseq'].head(5).tolist()}")
    
    return results


# ============================================================================
# EXAMPLE 5: PEPTIDE DOCKING
# ============================================================================

def example_peptide_docking():
    """Example: Dock a peptide to a protein"""
    
    # Method 1: Using preset with sequence
    config = PresetConfigs.peptide_docking(
        protein_path=Path("data/examples/peptide/protein.pdb"),
        peptide="DTVFALFW",  # Peptide sequence
        pocket_center=[12.0, 8.0, 20.0],
        num_mols=50
    )
    
    # Method 2: Using PDB file input
    config = PocketXMolConfig(
        sample=SampleConfig(num_mols=50, batch_size=20),
        data=DataConfig(
            protein_path=Path("data/examples/peptide/protein.pdb"),
            input_ligand=Path("data/examples/peptide/peptide.pdb"),  # PDB file
            is_pep=True,
            pocket_args=PocketDefinition(
                ref_ligand_path=Path("data/examples/peptide/ref_pep.pdb"),
                radius=15.0
            )
        ),
        task=DockingTaskConfig(),
        noise=NoiseConfig(name=NoiseType.DOCK, num_steps=100)
    )
    
    server = PocketXMolServer(device="cuda:0")
    results = server.generate(config, output_dir=Path("outputs/peptide_docking"))
    
    return results


# ============================================================================
# EXAMPLE 6: FRAGMENT GROWING
# ============================================================================

def example_fragment_growing():
    """Example: Grow a molecular fragment in a pocket"""
    
    config = PocketXMolConfig(
        sample=SampleConfig(num_mols=100, batch_size=50),
        data=DataConfig(
            protein_path=Path("data/examples/fragment/protein.pdb"),
            input_ligand=Path("data/examples/fragment/fragment.sdf"),
            pocket_args=PocketDefinition(
                ref_ligand_path=Path("data/examples/fragment/ref_complete.sdf"),
                radius=12.0
            )
        ),
        task=FragmentGrowingTaskConfig().set_fragment_perturbation(sigma=0.5),
        noise=NoiseConfig(
            name=NoiseType.MASKFILL,
            num_steps=100,
            level=NoiseLevel(
                min=0.3,  # Preserve some fragment information
                max=1.0
            )
        ),
        transforms=TransformConfig(
            variable_mol_size=VariableMolSizeTransform(
                num_atoms_distri={
                    "strategy": "mol_atoms_based",
                    "mean": {"coef": 0.5, "bias": 15},  # Grow by ~15 atoms
                    "std": {"coef": 0, "bias": 3},
                    "min": 5
                }
            )
        )
    )
    
    server = PocketXMolServer(device="cuda:0")
    results = server.generate(config, output_dir=Path("outputs/fragment_growing"))
    
    return results


# ============================================================================
# EXAMPLE 7: CUSTOM ADVANCED TASK
# ============================================================================

def example_custom_task():
    """Example: Custom task with complex constraints"""
    
    # Example: Fix part of molecule while redesigning another part
    config = PocketXMolConfig(
        sample=SampleConfig(num_mols=50, batch_size=25),
        data=DataConfig(
            protein_path=Path("data/examples/custom/protein.pdb"),
            input_ligand=Path("data/examples/custom/ligand.sdf"),
            pocket_args=PocketDefinition(
                pocket_coord=[10.0, 15.0, 20.0],
                radius=15.0
            )
        ),
        task=CustomTaskConfig(
            transform={
                "name": "custom",
                "is_peptide": False,
                "partition": [
                    {"name": "fixed_core", "nodes": [0, 1, 2, 3, 4]},
                    {"name": "variable_region", "nodes": "others"}
                ],
                "fixed": {
                    "node": ["fixed_core"],
                    "pos": ["fixed_core"],
                    "edge": [["fixed_core", "fixed_core"]]
                }
            }
        ),
        noise=NoiseConfig(
            name=NoiseType.CUSTOM,
            init_step=0.7,  # Start from partially noised state
            prior={
                "fixed_core": {
                    "pos_only": True,
                    "pos": {
                        "name": "allpos",
                        "pos": {
                            "name": "gaussian_simple",
                            "sigma_max": 0.5  # Small perturbation for fixed part
                        }
                    }
                },
                "variable_region": {
                    "node": {"name": "uniform"},
                    "pos": {"name": "gaussian", "sigma_max": 3},
                    "edge": {"name": "uniform"}
                }
            },
            level={
                "fixed_core": 0.2,  # Keep most information
                "variable_region": 0.9  # Redesign freely
            }
        )
    )
    
    server = PocketXMolServer(device="cuda:0")
    results = server.generate(config, output_dir=Path("outputs/custom_task"))
    
    return results


# ============================================================================
# EXAMPLE 8: BATCH PROCESSING
# ============================================================================

def example_batch_processing():
    """Example: Process multiple proteins/pockets in batch"""
    
    # Define multiple tasks
    tasks = [
        {
            "name": "target1_docking",
            "config": PresetConfigs.small_molecule_docking(
                Path("data/targets/target1.pdb"),
                "CC(=O)Nc1ccccc1",
                [10, 15, 20],
                num_mols=50
            )
        },
        {
            "name": "target2_design",
            "config": PresetConfigs.structure_based_drug_design(
                Path("data/targets/target2.pdb"),
                [5, 10, 15],
                num_mols=50
            )
        },
        {
            "name": "target3_peptide",
            "config": PresetConfigs.peptide_design(
                Path("data/targets/target3.pdb"),
                peptide_length=7,
                pocket_center=[8, 12, 18],
                num_mols=30
            )
        }
    ]
    
    # Process all tasks
    server = PocketXMolServer(device="cuda:0")
    all_results = {}
    
    for task in tasks:
        print(f"Processing {task['name']}...")
        results = server.generate(
            task['config'],
            output_dir=Path(f"outputs/batch/{task['name']}")
        )
        all_results[task['name']] = results
        print(f"Completed {task['name']}: {len(results)} molecules generated\n")
    
    return all_results


# ============================================================================
# EXAMPLE 9: PROGRAMMATIC CONFIG MODIFICATION
# ============================================================================

def example_config_modification():
    """Example: Programmatically modify configurations"""
    
    # Start with a base configuration
    base_config = PresetConfigs.small_molecule_docking(
        Path("data/examples/1a2b/1a2b_protein.pdb"),
        "CC(C)C(=O)O",
        [10, 15, 20]
    )
    
    # Modify various parameters programmatically
    base_config.sample.batch_size = 100  # Increase batch size
    base_config.sample.num_mols = 500  # Generate more molecules
    base_config.sample.save_traj_prob = 0.1  # Save more trajectories
    
    # Adjust noise parameters
    base_config.noise.num_steps = 200  # More denoising steps
    base_config.noise.level = NoiseLevel(
        name=LevelStrategy.ADVANCE,
        min=0.1,  # Start with some information preserved
        max=0.9   # Don't fully noise
    )
    
    # Add custom transforms
    base_config.transforms = TransformConfig(
        featurizer_pocket={"center": [10, 15, 20]},
        variable_mol_size=VariableMolSizeTransform(
            num_atoms_distri={
                "strategy": "gaussian",
                "mean": 30,
                "std": 5,
                "min": 20,
                "max": 40
            }
        )
    )
    
    # Run with modified config
    server = PocketXMolServer(device="cuda:0")
    results = server.generate(base_config, output_dir=Path("outputs/modified"))
    
    return results


# ============================================================================
# EXAMPLE 10: USING DICTIONARY CONFIGURATION
# ============================================================================

def example_dict_configuration():
    """Example: Create configuration from dictionary"""
    
    config_dict = {
        "sample": {
            "seed": 42,
            "batch_size": 50,
            "num_mols": 100,
            "save_traj_prob": 0.05
        },
        "data": {
            "protein_path": "data/examples/1a2b/1a2b_protein.pdb",
            "input_ligand": "CC(C)(C)c1ccccc1",  # SMILES
            "pocket_args": {
                "pocket_coord": [10.0, 15.0, 20.0],
                "radius": 15.0
            },
            "pocmol_args": {
                "data_id": "dict_example",
                "pdbid": "1a2b"
            }
        },
        "task": {
            "name": "dock",
            "transform": {
                "name": "dock",
                "settings": {
                    "free": 0.7,
                    "flexible": 0.3
                }
            }
        },
        "noise": {
            "name": "dock",
            "num_steps": 100,
            "level": {
                "name": "advance",
                "min": 0.0,
                "max": 1.0
            }
        },
        "model": {
            "checkpoint": "checkpoints/pocketxmol.ckpt"
        }
    }
    
    server = PocketXMolServer(device="cuda:0")
    results = server.generate_from_dict(config_dict, Path("outputs/dict_config"))
    
    return results


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PocketXMol Server Examples")
    parser.add_argument("--example", type=str, default="docking",
                       choices=["docking", "flexible", "design", "peptide_design",
                               "peptide_dock", "fragment", "custom", "batch",
                               "modify", "dict", "all"],
                       help="Which example to run")
    parser.add_argument("--device", type=str, default="cuda:0",
                       help="PyTorch device to use")
    
    args = parser.parse_args()
    
    # Map example names to functions
    examples = {
        "docking": example_small_molecule_docking,
        "flexible": example_flexible_docking,
        "design": example_drug_design,
        "peptide_design": example_peptide_design,
        "peptide_dock": example_peptide_docking,
        "fragment": example_fragment_growing,
        "custom": example_custom_task,
        "batch": example_batch_processing,
        "modify": example_config_modification,
        "dict": example_dict_configuration
    }
    
    if args.example == "all":
        # Run all examples
        for name, func in examples.items():
            print(f"\n{'='*60}")
            print(f"Running example: {name}")
            print('='*60)
            try:
                results = func()
                print(f"✓ {name} completed successfully")
            except Exception as e:
                print(f"✗ {name} failed: {e}")
    else:
        # Run specific example
        example_func = examples[args.example]
        print(f"Running {args.example} example...")
        results = example_func()
        print(f"\n{args.example} example completed!")
        print(f"Results saved with {len(results)} molecules generated")