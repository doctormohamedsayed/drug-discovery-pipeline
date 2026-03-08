"""
Module 5: Fragment Linking & Merging (3D)
Combines top-ranked fragments using BRICS linking, scaffold merging,
and 3D conformer generation with energy minimization.
"""

import itertools
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, BRICS, Descriptors, rdFMCS, rdMolAlign
from rdkit.Chem import rdDistGeom


def _passes_lipinski(mol):
    """Check Lipinski's Rule of Five."""
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    violations = 0
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    return violations <= 1, violations


def _is_valid_druglike(mol):
    """Check if molecule is valid and drug-like."""
    if mol is None:
        return False
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        return False
    
    mw = Descriptors.MolWt(mol)
    if mw < 150 or mw > 800:
        return False
    
    num_atoms = mol.GetNumHeavyAtoms()
    if num_atoms < 10 or num_atoms > 70:
        return False
    
    passes, _ = _passes_lipinski(mol)
    return passes


def brics_link_fragments(fragment_smiles, top_n=20, max_molecules=5000):
    """
    Link top-ranked fragments using BRICS.BRICSBuild.
    
    Args:
        fragment_smiles: list of SMILES strings (ranked by score)
        top_n: use top N fragments
        max_molecules: cap on total generated molecules
        
    Returns:
        list of unique SMILES of generated molecules
    """
    # Get top fragments
    top_frags = fragment_smiles[:top_n]
    
    # Convert to RDKit Mol objects
    frag_mols = []
    for smi in top_frags:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            frag_mols.append(mol)
    
    if len(frag_mols) < 2:
        return []
    
    # BRICS decompose each fragment to get BRICS-compatible fragments
    all_brics_frags = set()
    for mol in frag_mols:
        try:
            brics_frags = BRICS.BRICSDecompose(mol, minFragmentSize=3)
            all_brics_frags.update(brics_frags)
        except Exception:
            continue
    
    if len(all_brics_frags) < 2:
        # If BRICS decomposition yields too few fragments, use them directly
        all_brics_frags = set()
        for mol in frag_mols:
            smi = Chem.MolToSmiles(mol)
            all_brics_frags.add(smi)
    
    # Convert back to mols for BRICSBuild
    brics_mol_list = []
    for smi in all_brics_frags:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            brics_mol_list.append(mol)
    
    generated = set()
    
    # Try BRICS building
    try:
        builder = BRICS.BRICSBuild(brics_mol_list)
        for i, mol in enumerate(builder):
            if i >= max_molecules:
                break
            try:
                smi = Chem.MolToSmiles(mol)
                if _is_valid_druglike(mol):
                    generated.add(smi)
            except Exception:
                continue
    except Exception:
        pass
    
    # Also do pairwise combinations if BRICS didn't give enough
    if len(generated) < 100:
        for i, mol1 in enumerate(frag_mols[:min(10, len(frag_mols))]):
            for j, mol2 in enumerate(frag_mols[:min(10, len(frag_mols))]):
                if i >= j:
                    continue
                try:
                    # Simple concatenation with common linkers
                    smi1 = Chem.MolToSmiles(mol1)
                    smi2 = Chem.MolToSmiles(mol2)
                    
                    linkers = ['CC', 'CCC', 'CCCC', 'CNC', 'COC', 'CC(=O)N', 'C(=O)N',
                               'CCNC', 'CCOC', 'c1ccc(cc1)']
                    
                    for linker in linkers:
                        combined_smi = f"{smi1}{linker}{smi2}"
                        combined_mol = Chem.MolFromSmiles(combined_smi)
                        if combined_mol is not None and _is_valid_druglike(combined_mol):
                            generated.add(Chem.MolToSmiles(combined_mol))
                        
                        if len(generated) >= max_molecules:
                            break
                except Exception:
                    continue
                
                if len(generated) >= max_molecules:
                    break
            if len(generated) >= max_molecules:
                break
    
    return list(generated)


def scaffold_merge(fragment_smiles, top_n=15, max_molecules=2000):
    """
    Merge fragments by finding common substructures (MCS) and combining.
    
    Args:
        fragment_smiles: list of SMILES strings
        top_n: use top N fragments for merging
        max_molecules: cap on output
    
    Returns:
        list of unique SMILES
    """
    top_frags = fragment_smiles[:top_n]
    frag_mols = []
    for smi in top_frags:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            frag_mols.append((smi, mol))
    
    generated = set()
    
    # Pairwise MCS-based merging
    for i in range(len(frag_mols)):
        for j in range(i + 1, len(frag_mols)):
            if len(generated) >= max_molecules:
                break
            
            smi1, mol1 = frag_mols[i]
            smi2, mol2 = frag_mols[j]
            
            try:
                # Find Maximum Common Substructure
                mcs_result = rdFMCS.FindMCS(
                    [mol1, mol2],
                    timeout=5,
                    matchValences=False,
                    ringMatchesRingOnly=True,
                    completeRingsOnly=True,
                )
                
                if mcs_result.numAtoms < 3:
                    continue
                
                mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
                if mcs_mol is None:
                    continue
                
                # Try replacing the MCS in mol1 with mol2's version
                try:
                    # Use ReplaceSubstructs to merge
                    merged_mols = AllChem.ReplaceSubstructs(mol1, mcs_mol, mol2)
                    for merged in merged_mols:
                        try:
                            Chem.SanitizeMol(merged)
                            if _is_valid_druglike(merged):
                                generated.add(Chem.MolToSmiles(merged))
                        except Exception:
                            continue
                except Exception:
                    continue
                    
            except Exception:
                continue
        
        if len(generated) >= max_molecules:
            break
    
    return list(generated)


def generate_3d_conformer(mol, num_conformers=1, optimize=True):
    """
    Generate 3D conformer(s) for a molecule with energy minimization.
    
    Args:
        mol: RDKit Mol object
        num_conformers: number of conformers to generate
        optimize: whether to energy-minimize with MMFF94
        
    Returns:
        mol with 3D conformer(s), or None if failed
    """
    try:
        mol_3d = Chem.AddHs(mol)
        
        # Use ETKDG method for conformer generation
        params = rdDistGeom.ETKDGv3()
        params.randomSeed = 42
        params.numThreads = 0  # Use all available threads
        
        conf_ids = AllChem.EmbedMultipleConfs(mol_3d, numConfs=num_conformers, params=params)
        
        if len(conf_ids) == 0:
            # Fallback: try without ETKDG constraints
            conf_ids = AllChem.EmbedMultipleConfs(
                mol_3d, numConfs=num_conformers,
                randomSeed=42, useRandomCoords=True
            )
        
        if len(conf_ids) == 0:
            return None
        
        if optimize:
            # Energy minimize with MMFF94
            results = []
            for conf_id in conf_ids:
                try:
                    ff_result = AllChem.MMFFOptimizeMolecule(mol_3d, confId=conf_id, maxIters=500)
                    # Get MMFF energy
                    ff_props = AllChem.MMFFGetMoleculeProperties(mol_3d)
                    if ff_props is not None:
                        ff = AllChem.MMFFGetMoleculeForceField(mol_3d, ff_props, confId=conf_id)
                        if ff is not None:
                            energy = ff.CalcEnergy()
                            results.append((conf_id, energy))
                        else:
                            results.append((conf_id, float('inf')))
                    else:
                        # Fallback to UFF
                        AllChem.UFFOptimizeMolecule(mol_3d, confId=conf_id, maxIters=500)
                        ff = AllChem.UFFGetMoleculeForceField(mol_3d, confId=conf_id)
                        if ff is not None:
                            energy = ff.CalcEnergy()
                            results.append((conf_id, energy))
                        else:
                            results.append((conf_id, float('inf')))
                except Exception:
                    results.append((conf_id, float('inf')))
            
            # Keep lowest energy conformer
            if results:
                results.sort(key=lambda x: x[1])
                best_conf_id = results[0][0]
                mol_3d.SetProp("mmff_energy", str(round(results[0][1], 2)))
                
                # Remove other conformers, keep only the best
                conf_ids_to_remove = [r[0] for r in results if r[0] != best_conf_id]
                for cid in sorted(conf_ids_to_remove, reverse=True):
                    mol_3d.RemoveConformer(cid)
        
        mol_3d = Chem.RemoveHs(mol_3d)
        return mol_3d
    
    except Exception:
        return None


def _compute_strain_energy(mol):
    """Compute strain energy of a 3D conformer. Lower = better."""
    try:
        mol_h = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol_h, randomSeed=42)
        
        ff_props = AllChem.MMFFGetMoleculeProperties(mol_h)
        if ff_props:
            ff = AllChem.MMFFGetMoleculeForceField(mol_h, ff_props)
            if ff:
                return ff.CalcEnergy()
        
        # Fallback to UFF
        ff = AllChem.UFFGetMoleculeForceField(mol_h)
        if ff:
            return ff.CalcEnergy()
        
        return float('inf')
    except Exception:
        return float('inf')


def link_and_merge_fragments(ranked_fragments, max_output=10000, progress_callback=None):
    """
    Main function: perform fragment linking and merging with 3D optimization.
    
    Args:
        ranked_fragments: list of dicts with 'smiles' and 'score' keys, sorted by score
        max_output: maximum molecules to generate
        progress_callback: optional callback(step, message)
        
    Returns:
        dict with generated molecules and statistics
    """
    frag_smiles = [f['smiles'] for f in ranked_fragments]
    
    if progress_callback:
        progress_callback("linking", "BRICS fragment linking...")
    
    # Step 1: BRICS linking
    brics_molecules = brics_link_fragments(frag_smiles, top_n=20, max_molecules=max_output // 2)
    
    if progress_callback:
        progress_callback("merging", f"Scaffold merging... ({len(brics_molecules)} from BRICS)")
    
    # Step 2: Scaffold merging
    merged_molecules = scaffold_merge(frag_smiles, top_n=15, max_molecules=max_output // 2)
    
    # Combine and deduplicate
    all_smiles = set(brics_molecules) | set(merged_molecules)
    
    if progress_callback:
        progress_callback("3d_gen", f"Generating 3D conformers for {len(all_smiles)} molecules...")
    
    # Step 3: Generate 3D conformers for all valid molecules
    results = []
    for idx, smi in enumerate(all_smiles):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        # Generate 3D
        mol_3d = generate_3d_conformer(mol, num_conformers=3, optimize=True)
        has_3d = mol_3d is not None
        
        energy = float('inf')
        if has_3d and mol_3d.HasProp("mmff_energy"):
            energy = float(mol_3d.GetProp("mmff_energy"))
        
        results.append({
            'smiles': smi,
            'mol': mol,
            'mol_3d': mol_3d,
            'has_3d': has_3d,
            'energy': energy,
            'mw': round(Descriptors.MolWt(mol), 2),
            'logp': round(Descriptors.MolLogP(mol), 2),
            'source': 'brics' if smi in brics_molecules else 'scaffold_merge',
        })
    
    # Filter out high-strain molecules (top 80% by energy)
    valid_results = [r for r in results if r['energy'] < float('inf')]
    if len(valid_results) > 10:
        energy_cutoff = sorted([r['energy'] for r in valid_results])[int(len(valid_results) * 0.8)]
        valid_results = [r for r in valid_results if r['energy'] <= energy_cutoff]
    
    # Add back molecules without 3D (they still have value)
    no_3d_results = [r for r in results if r['energy'] == float('inf')]
    final_results = valid_results + no_3d_results
    
    if progress_callback:
        progress_callback("done", f"Generated {len(final_results)} molecules")
    
    return {
        'molecules': final_results,
        'total_generated': len(all_smiles),
        'from_brics': len(brics_molecules),
        'from_merging': len(merged_molecules),
        'with_3d': sum(1 for r in final_results if r['has_3d']),
        'final_count': len(final_results),
    }
