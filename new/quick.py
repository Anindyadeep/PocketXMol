#!/usr/bin/env python3
"""
PocketXMol Server - Quick Start Script
Simple examples for immediate usage
"""

from pathlib import Path
from pocketxmol_server import (
    PocketXMolServer,
    PresetConfigs,
    quick_dock,
    quick_design
)


def main():
    """Quick start examples"""
    
    print("🧬 PocketXMol Server - Quick Start Examples")
    print("=" * 50)
    
    # Example 1: Quick small molecule docking
    print("\n1. Small Molecule Docking")
    print("-" * 30)
    
    try:
        results = quick_dock(
            protein_path="data/examples/1a2b/1a2b_protein.pdb",
            ligand="CC(C)(C#N)c1cccc(C(=O)Nc2ccc(F)cc2)c1",  # SMILES
            pocket_center=[15.0, 20.0, 25.0],
            num_mols=10,  # Small number for quick demo
            device="cuda:0"
        )
        print(f"✓ Generated {len(results)} docked molecules")
        successful = results[results['tag'] == '']
        print(f"✓ Success rate: {len(successful)/len(results)*100:.1f}%")
    except Exception as e:
        print(f"✗ Docking failed: {e}")
    
    # Example 2: Quick drug design
    print("\n2. Structure-Based Drug Design")
    print("-" * 30)
    
    try:
        results = quick_design(
            protein_path="data/examples/5d3n/5d3n_protein.pdb",
            pocket_center=[10.5, -15.2, 22.8],
            num_mols=10,  # Small number for quick demo
            device="cuda:0"
        )
        print(f"✓ Generated {len(results)} designed molecules")
        successful = results[results['tag'] == '']
        print(f"✓ Success rate: {len(successful)/len(results)*100:.1f}%")
        if len(successful) > 0:
            print(f"✓ Example SMILES: {successful['smiles'].iloc[0]}")
    except Exception as e:
        print(f"✗ Design failed: {e}")
    
    # Example 3: Using preset configurations
    print("\n3. Using Preset Configurations")
    print("-" * 30)
    
    try:
        # Create server instance
        server = PocketXMolServer(device="cuda:0")
        
        # Peptide design using preset
        config = PresetConfigs.peptide_design(
            protein_path=Path("data/examples/peptide/protein.pdb"),
            peptide_length=8,
            pocket_center=[5.0, 10.0, 15.0],
            num_mols=5  # Very small for demo
        )
        
        results = server.generate(config, output_dir=Path("outputs/quick_peptide"))
        print(f"✓ Generated {len(results)} peptides")
        
        successful = results[results['tag'] == '']
        if len(successful) > 0:
            print(f"✓ Example sequence: {successful['aaseq'].iloc[0]}")
    
    except Exception as e:
        print(f"✗ Peptide design failed: {e}")
    
    # Example 4: Programmatic configuration
    print("\n4. Programmatic Configuration")
    print("-" * 30)
    
    try:
        # Start with preset and modify
        config = PresetConfigs.small_molecule_docking(
            protein_path=Path("data/examples/1a2b/1a2b_protein.pdb"),
            ligand="data/examples/dock/ligand.sdf",
            num_mols=5
        )
        
        # Modify parameters
        config.sample.batch_size = 10
        config.noise.num_steps = 50  # Faster for demo
        config.data.pocket_args.radius = 12.0  # Smaller pocket
        
        server = PocketXMolServer(device="cuda:0")
        results = server.generate(config, output_dir=Path("outputs/quick_modified"))
        
        print(f"✓ Generated {len(results)} molecules with modified config")
        print(f"✓ Used pocket radius: {config.data.pocket_args.radius} Å")
        print(f"✓ Used {config.noise.num_steps} denoising steps")
    
    except Exception as e:
        print(f"✗ Modified config failed: {e}")
    
    print("\n" + "=" * 50)
    print("🎉 Quick start examples completed!")
    print("📁 Check the outputs/ directory for results")
    print("📖 See examples_server_usage.py for more detailed examples")


if __name__ == "__main__":
    main()