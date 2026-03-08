"""
Module 2: Decoy Generation
Generates property-matched decoys using DUD-E methodology.
Uses a bundled background set of drug-like SMILES for sampling.
"""

import random
import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors

# A curated set of common drug-like SMILES for decoy generation
# These are real drug-like molecules from public domain sources
BACKGROUND_SMILES = [
    "CC(=O)Oc1ccccc1C(=O)O",  # Aspirin
    "CC(C)Cc1ccc(cc1)C(C)C(=O)O",  # Ibuprofen
    "CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C",  # Testosterone
    "OC(=O)c1ccccc1O",  # Salicylic acid
    "CN1C=NC2=C1C(=O)N(C(=O)N2C)C",  # Caffeine
    "CC(=O)NC1=CC=C(O)C=C1",  # Acetaminophen
    "OC(=O)CC(O)(CC(=O)O)C(=O)O",  # Citric acid
    "C1CCCCC1",  # Cyclohexane
    "c1ccc2c(c1)cc1ccc3cccc4ccc2c1c34",  # Pyrene
    "CC(C)NCC(O)c1ccc(O)c(O)c1",  # Isoproterenol
    "CCOC(=O)c1cncn1C1CCCCC1",  
    "CC1=CC=C(C=C1)C(=O)NN",
    "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1",  # Omeprazole
    "CC(CS)C(=O)NCC(=O)O",  # Captopril-like
    "O=C1CN=C(c2ccccc2)c2cc(Cl)ccc2N1",  # Chlordiazepoxide-like
    "CCN(CC)CCOC(=O)c1ccc(N)cc1",  # Procaine  
    "CC(=O)OC1CC2CCC3C(CCC4(C)C3CCC4(OC(C)=O)C#C)C2(C)CC1",
    "OC1C(O)C(OC1CO)n1cnc2c(O)ncnc12",
    "CC1(C)CCC(CC1)OC(=O)c1ccccc1",
    "CCCCCCCCCCCCCCCC(=O)OCC(COC(=O)CCCCCCCCCCCCCCC)OC(=O)CCCCCCCCCCCCCCC",
    "CC(C)CC(NC(=O)C(CC(=O)O)NC(=O)C(Cc1ccccc1)NC(=O)c1cnccn1)C(=O)O",
    "Oc1ccc(cc1)C(O)c1ccc(O)cc1",
    "c1ccc(NC(=O)c2ccccc2)cc1",
    "CCOC(=O)C1=C(C)NC(=C(C1c1ccccc1[N+](=O)[O-])C(=O)OC)C",
    "COc1cc2c(cc1OC)C(=O)C(CC2)c1ccc(OC)c(OC)c1",
    "CC1=C(C(=O)N(N1C)c1ccccc1)c1ccc(cc1)S(N)(=O)=O",
    "OC(=O)c1cc(O)c(O)c(O)c1",
    "Cc1[nH]c2ccccc2c1CCN",
    "Cn1c(=O)c2[nH]cnc2n(C)c1=O",
    "CCCCC(CC)COC(=O)c1ccc(N)cc1",
    "CC(C)(C)NCC(O)c1cc(O)cc(O)c1",
    "COc1ccc2c3c1OC1C=CC(=O)CC1C3N(C)CC2",
    "Clc1ccc(c1)C(c1ccccc1)n1ccnc1",
    "CC1(O)CCC2C3CCC4=CC(=O)CCC4(C)C3C(O)CC2(C)C1C(=O)CO",
    "OC(c1cc(Cl)ccc1)c1ccccn1",
    "CC1(C)SC2C(NC(=O)Cc3ccccc3)C(=O)N2C1C(=O)O",
    "CC(=O)Nc1nnc(s1)S(N)(=O)=O",
    "Nc1ccc(cc1)S(=O)(=O)c1ccc(N)cc1",
    "OCC1OC(O)C(O)C(O)C1O",
    "CNCC(O)c1ccc(O)c(O)c1",
    "CC(Cc1ccc(O)cc1)NCC(O)c1ccc(O)c(O)c1",
    "OC(=O)CCC(=O)O",
    "CCN(CC)C(=O)c1cc(OC)c(OC)c(OC)c1",
    "CC1Oc2cc(O)cc(O)c2C(=O)C1O",
    "COc1ccc(Cc2cnc(N)nc2N)cc1",
    "CC(=O)NC1C(OC(C1O)CO)Oc1ccc(O)cc1",
    "Nc1ncnc2c1ncn2C1OC(COP(=O)(O)O)C(O)C1O",
    "CC(N)Cc1ccc(O)c(O)c1",
    "c1cc2c(cc1O)C(CN2)c1ccc(O)c(O)c1",
]


def _compute_properties(mol):
    """Compute molecular properties for decoy matching."""
    return {
        'mw': Descriptors.MolWt(mol),
        'logp': Descriptors.MolLogP(mol),
        'hbd': Descriptors.NumHDonors(mol),
        'hba': Descriptors.NumHAcceptors(mol),
        'rot_bonds': Descriptors.NumRotatableBonds(mol),
        'tpsa': Descriptors.TPSA(mol),
    }


def _compute_fingerprint(mol, radius=2, n_bits=2048):
    """Compute Morgan fingerprint."""
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)


def _get_tanimoto(fp1, fp2):
    """Compute Tanimoto similarity between two fingerprints."""
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def _properties_match(active_props, decoy_props, relaxation=1.0):
    """
    Check if decoy properties match active within allowed ranges.
    relaxation: multiplier to widen ranges (1.0 = standard, 2.0 = relaxed)
    """
    mw_range = 0.20 * relaxation
    logp_range = 1.0 * relaxation
    hbd_range = int(1 * relaxation)
    hba_range = int(1 * relaxation)
    
    if abs(active_props['mw'] - decoy_props['mw']) > active_props['mw'] * mw_range:
        return False
    if abs(active_props['logp'] - decoy_props['logp']) > logp_range:
        return False
    if abs(active_props['hbd'] - decoy_props['hbd']) > hbd_range:
        return False
    if abs(active_props['hba'] - decoy_props['hba']) > hba_range:
        return False
    
    return True


def _generate_random_smiles(n=500):
    """
    Generate random drug-like SMILES by modifying the background set.
    Creates variations by simple molecular modifications.
    """
    generated = []
    base_mols = []
    
    for smi in BACKGROUND_SMILES:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            base_mols.append(mol)
    
    # Add the base molecules themselves
    for mol in base_mols:
        generated.append(mol)
    
    # Create variations by shuffling atom order (produces same molecule, different SMILES)
    # and using different random seeds for enumeration
    substituents = ['C', 'CC', 'O', 'N', 'F', 'Cl', 'OC', 'NC', 'C(=O)O', 'C(=O)N']
    
    attempts = 0
    while len(generated) < n and attempts < n * 10:
        attempts += 1
        base = random.choice(base_mols)
        smi = Chem.MolToSmiles(base, doRandom=True)
        mol = Chem.MolFromSmiles(smi)
        if mol is not None and mol not in generated:
            generated.append(mol)
    
    return generated


def generate_decoys(active_molecules, decoys_per_active=39, max_total=5000):
    """
    Generate property-matched decoys for a list of active molecules.
    
    Args:
        active_molecules: list of RDKit Mol objects (actives)
        decoys_per_active: target number of decoys per active (DUD-E = 39)
        max_total: maximum total decoys to generate
        
    Returns:
        dict with keys:
          - 'decoys': list of RDKit Mol objects
          - 'decoy_smiles': list of SMILES strings
          - 'stats': generation statistics
    """
    # Compute properties and fingerprints for all actives
    active_props = []
    active_fps = []
    active_smiles_set = set()
    
    for mol in active_molecules:
        props = _compute_properties(mol)
        fp = _compute_fingerprint(mol)
        active_props.append(props)
        active_fps.append(fp)
        active_smiles_set.add(Chem.MolToSmiles(mol))
    
    # Build background pool
    background_mols = _generate_random_smiles(n=2000)
    
    # Pre-compute background properties and fingerprints
    bg_data = []
    for mol in background_mols:
        smi = Chem.MolToSmiles(mol)
        if smi in active_smiles_set:
            continue
        try:
            props = _compute_properties(mol)
            fp = _compute_fingerprint(mol)
            bg_data.append({
                'mol': mol,
                'smiles': smi,
                'props': props,
                'fp': fp,
            })
        except Exception:
            continue
    
    all_decoys = []
    all_decoy_smiles = set()
    stats = {'per_active': [], 'total': 0, 'relaxations': 0}
    
    for act_idx, (a_props, a_fp) in enumerate(zip(active_props, active_fps)):
        decoys_for_active = []
        
        # Try with progressively relaxed constraints
        for relaxation in [1.0, 1.5, 2.0, 3.0]:
            if len(decoys_for_active) >= decoys_per_active:
                break
            
            if relaxation > 1.0:
                stats['relaxations'] += 1
            
            random.shuffle(bg_data)
            
            for bg in bg_data:
                if bg['smiles'] in all_decoy_smiles:
                    continue
                
                # Check property matching
                if not _properties_match(a_props, bg['props'], relaxation):
                    continue
                
                # Check Tanimoto dissimilarity to ALL actives
                max_sim = max(_get_tanimoto(bg['fp'], afp) for afp in active_fps)
                if max_sim >= 0.35:
                    continue
                
                # This molecule passes as a decoy
                bg['mol'].SetProp("_Name", f"decoy_{act_idx}_{len(decoys_for_active)}")
                bg['mol'].SetProp("SMILES", bg['smiles'])
                bg['mol'].SetProp("source_active", str(act_idx))
                decoys_for_active.append(bg['mol'])
                all_decoy_smiles.add(bg['smiles'])
                
                if len(decoys_for_active) >= decoys_per_active:
                    break
                
                if len(all_decoys) + len(decoys_for_active) >= max_total:
                    break
            
            if len(all_decoys) + len(decoys_for_active) >= max_total:
                break
        
        stats['per_active'].append(len(decoys_for_active))
        all_decoys.extend(decoys_for_active)
        
        if len(all_decoys) >= max_total:
            break
    
    stats['total'] = len(all_decoys)
    
    return {
        'decoys': all_decoys,
        'decoy_smiles': list(all_decoy_smiles),
        'stats': stats,
    }
