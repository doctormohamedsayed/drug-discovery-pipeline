"""
Module 6: Ranking
Multi-criteria ranking of generated molecules using classifier scores,
drug-likeness, synthetic accessibility, and structural diversity.
"""

import numpy as np
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem, rdMolDescriptors
from rdkit.Chem import RDConfig
import os


def _compute_sa_score(mol):
    """
    Compute Synthetic Accessibility Score (1=easy to synthesize, 10=hard).
    Simplified version based on fragment complexity.
    """
    try:
        # Count features that affect synthetic accessibility
        ring_count = rdMolDescriptors.CalcNumRings(mol)
        num_stereo = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
        num_spiro = rdMolDescriptors.CalcNumSpiroAtoms(mol)
        num_bridgehead = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
        heavy_atoms = mol.GetNumHeavyAtoms()
        
        # Complexity score (higher = harder to synthesize)
        complexity = 0
        complexity += ring_count * 0.5
        complexity += num_stereo * 1.0
        complexity += num_spiro * 2.0
        complexity += num_bridgehead * 2.0
        
        # Penalize very large molecules
        if heavy_atoms > 35:
            complexity += (heavy_atoms - 35) * 0.1
        
        # Normalize to 1-10 scale
        sa_score = 1.0 + min(9.0, complexity)
        
        return round(sa_score, 2)
    except Exception:
        return 5.0  # Default middle score


def _lipinski_score(mol):
    """
    Compute drug-likeness score based on Lipinski's Rule of Five.
    Returns 0-1 (1 = fully compliant, 0 = many violations).
    """
    violations = 0
    
    mw = Descriptors.MolWt(mol)
    logp = Descriptors.MolLogP(mol)
    hbd = Descriptors.NumHDonors(mol)
    hba = Descriptors.NumHAcceptors(mol)
    
    if mw > 500: violations += 1
    if logp > 5: violations += 1
    if hbd > 5: violations += 1
    if hba > 10: violations += 1
    
    return max(0, 1.0 - violations * 0.25)


def _diversity_score(mol, active_fps, radius=2, n_bits=2048):
    """
    Compute structural diversity from actives (higher = more different = more novel).
    Returns 0-1 (1 = completely different, 0 = identical to an active).
    """
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    
    if not active_fps:
        return 0.5
    
    max_sim = max(DataStructs.TanimotoSimilarity(fp, afp) for afp in active_fps)
    return 1.0 - max_sim


def rank_molecules(molecules_data, classifier_model, predict_fn, active_mols,
                   top_n=10, weights=None):
    """
    Rank generated molecules using multi-criteria scoring.
    
    Args:
        molecules_data: list of dicts with 'smiles' key (from fragment_linker)
        classifier_model: trained model
        predict_fn: function(model, [smiles]) -> list of prediction dicts
        active_mols: list of active RDKit Mol objects
        top_n: number of top molecules to return
        weights: dict of scoring weights (default: balanced)
        
    Returns:
        list of top-N ranked molecules with scores
    """
    if weights is None:
        weights = {
            'activity': 0.45,
            'druglikeness': 0.20,
            'sa_score': 0.15,
            'diversity': 0.20,
        }
    
    # Compute active fingerprints
    active_fps = []
    for mol in active_mols:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            active_fps.append(fp)
        except Exception:
            continue
    
    # Get all SMILES
    smiles_list = [m['smiles'] for m in molecules_data]
    
    # Batch predict activity scores
    predictions = predict_fn(classifier_model, smiles_list)
    pred_map = {p['smiles']: p for p in predictions}
    
    ranked = []
    
    for mol_data in molecules_data:
        smi = mol_data['smiles']
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        
        pred = pred_map.get(smi, {})
        if not pred.get('valid', False):
            continue
        
        # Score components
        activity_score = pred.get('score', 0.0)
        druglikeness = _lipinski_score(mol)
        
        sa_raw = _compute_sa_score(mol)
        sa_score = max(0, 1.0 - (sa_raw - 1.0) / 9.0)  # Normalize: 1→1.0, 10→0.0
        
        diversity = _diversity_score(mol, active_fps)
        
        # Weighted composite score
        composite = (
            weights['activity'] * activity_score +
            weights['druglikeness'] * druglikeness +
            weights['sa_score'] * sa_score +
            weights['diversity'] * diversity
        )
        
        ranked.append({
            'smiles': smi,
            'composite_score': round(composite, 4),
            'activity_score': round(activity_score, 4),
            'druglikeness': round(druglikeness, 4),
            'sa_score': round(sa_score, 4),
            'sa_raw': sa_raw,
            'diversity': round(diversity, 4),
            'mw': round(Descriptors.MolWt(mol), 2),
            'logp': round(Descriptors.MolLogP(mol), 2),
            'hbd': Descriptors.NumHDonors(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'tpsa': round(Descriptors.TPSA(mol), 2),
            'has_3d': mol_data.get('has_3d', False),
            'energy': mol_data.get('energy', None),
            'source': mol_data.get('source', 'unknown'),
        })
    
    # Sort by composite score (descending)
    ranked.sort(key=lambda x: x['composite_score'], reverse=True)
    
    return ranked[:top_n]
