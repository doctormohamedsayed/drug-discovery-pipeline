"""
Module 4: Fragment Library & Evaluation
Provides a built-in fragment library and evaluates fragments using the trained classifier.
"""

from rdkit import Chem
from rdkit.Chem import BRICS, AllChem, Descriptors

# Built-in fragment library: common drug-like fragments
# These are real pharmacophore fragments found in approved drugs
BUILT_IN_FRAGMENTS = [
    # Aromatic rings
    "c1ccccc1",           # Benzene
    "c1ccncc1",           # Pyridine
    "c1ccoc1",            # Furan
    "c1ccsc1",            # Thiophene
    "c1cc[nH]c1",         # Pyrrole
    "c1cnc2ccccc2n1",     # Quinazoline
    "c1ccc2[nH]ccc2c1",   # Indole
    "c1ccc2ncccc2c1",     # Quinoline
    "c1cnc[nH]1",         # Imidazole
    "c1ccnnc1",           # Pyridazine
    "c1cnccn1",           # Pyrimidine
    "c1ccncn1",           # Pyrazine
    "c1ccc2ccccc2c1",     # Naphthalene
    "c1cn[nH]c1",         # Pyrazole
    "c1cocn1",            # Oxazole
    "c1cscn1",            # Thiazole
    "c1nnc[nH]1",         # Triazole (1,2,4)
    "c1nnn[nH]1",         # Tetrazole
    "c1cc2ccccc2o1",      # Benzofuran
    "c1cc2ccccc2s1",      # Benzothiophene
    
    # Aliphatic rings
    "C1CCCCC1",           # Cyclohexane
    "C1CCCC1",            # Cyclopentane
    "C1CCNCC1",           # Piperidine
    "C1CNCCN1",           # Piperazine
    "C1CCOCC1",           # Tetrahydropyran
    "C1CCOC1",            # Tetrahydrofuran
    "C1CCNC1",            # Pyrrolidine
    "C1COCCO1",           # 1,3-Dioxane
    "C1CCC(CC1)O",        # Cyclohexanol
    "C1CCC(CC1)N",        # Cyclohexylamine
    
    # Functional groups as fragments
    "CC(=O)N",            # Acetamide
    "CC(=O)O",            # Acetyl
    "C(=O)N",             # Amide
    "C(=O)O",             # Carboxylic acid
    "S(=O)(=O)N",         # Sulfonamide
    "CS(=O)(=O)",         # Methylsulfonyl
    "CN",                 # Methylamine
    "CCO",                # Ethanol
    "CC(C)N",             # Isopropylamine
    "CCNCC",              # Diethylamine
    "OCC(O)CO",           # Glycerol
    "CC(=O)",             # Acetyl group
    "C(F)(F)F",           # Trifluoromethyl
    "OC(F)(F)F",          # Trifluoromethoxy
    
    # Heterocyclic drug fragments
    "O=c1[nH]c(=O)c2[nH]cnc2[nH]1",  # Xanthine core
    "c1nc2ccccc2[nH]1",              # Benzimidazole
    "c1nc2ccccc2o1",                  # Benzoxazole
    "c1nc2ccccc2s1",                  # Benzothiazole
    "c1cc2c(cc1)OCO2",                # Methylenedioxy
    "c1ccc(cc1)O",                    # Phenol
    "c1ccc(cc1)N",                    # Aniline
    "c1ccc(cc1)F",                    # Fluorobenzene
    "c1ccc(cc1)Cl",                   # Chlorobenzene
    "c1ccc(cc1)C(F)(F)F",             # Trifluoromethylbenzene
    "c1ccc(cc1)OC",                   # Anisole
    "c1ccc(cc1)C(=O)O",              # Benzoic acid
    "c1ccc(cc1)C(=O)N",              # Benzamide
    "c1ccc(cc1)S(=O)(=O)N",          # Benzenesulfonamide
    "CC(C)(C)c1ccccc1",               # tert-Butylbenzene
    "c1ccc(-c2ccccc2)cc1",            # Biphenyl
    
    # Linker fragments
    "CC",                  # Ethane
    "CCC",                 # Propane
    "CCCC",                # Butane
    "CC(C)C",              # Isobutane
    "C=C",                 # Ethylene
    "C#C",                 # Acetylene
    "CCO",                 # Ethanol
    "CCOC",                # Ethyl methyl ether
    "CCNC",                # Ethyl methyl amine
    "CCS",                 # Ethyl thiol
    "CC(=O)NCC",           # N-methylacetamide chain
    "CCOC(=O)",            # Ethyl ester
    "CCNC(=O)",            # Ethyl amide
]


def get_builtin_fragments():
    """Return built-in fragment library as RDKit Mol objects."""
    fragments = []
    for smi in BUILT_IN_FRAGMENTS:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            mol.SetProp("SMILES", Chem.MolToSmiles(mol))
            mol.SetProp("source", "built-in")
            fragments.append(mol)
    return fragments


def decompose_molecule(mol):
    """
    Decompose a molecule into BRICS fragments.
    Returns list of fragment SMILES.
    """
    try:
        frags = BRICS.BRICSDecompose(mol, minFragmentSize=3)
        return list(frags)
    except Exception:
        return []


def extract_fragments_from_actives(active_mols):
    """
    Extract BRICS fragments from all active molecules.
    Returns unique fragment SMILES and their source molecules.
    """
    frag_to_sources = {}
    
    for idx, mol in enumerate(active_mols):
        frags = decompose_molecule(mol)
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"active_{idx}"
        
        for frag_smi in frags:
            # Clean BRICS dummy atoms for scoring
            clean_smi = _clean_brics_smiles(frag_smi)
            if clean_smi:
                if clean_smi not in frag_to_sources:
                    frag_to_sources[clean_smi] = []
                frag_to_sources[clean_smi].append(name)
    
    return frag_to_sources


def _clean_brics_smiles(brics_smi):
    """
    Clean BRICS SMILES by removing dummy atom labels.
    [1*], [2*], etc. → [*]
    """
    import re
    cleaned = re.sub(r'\[\d+\*\]', '[*]', brics_smi)
    mol = Chem.MolFromSmiles(cleaned)
    if mol is not None:
        return Chem.MolToSmiles(mol)
    
    # Try removing dummy atoms entirely
    cleaned2 = re.sub(r'\[\d*\*\]', '', brics_smi)
    mol2 = Chem.MolFromSmiles(cleaned2)
    if mol2 is not None and mol2.GetNumAtoms() >= 3:
        return Chem.MolToSmiles(mol2)
    
    return None


def evaluate_fragments(fragments_smiles, classifier_model, predict_fn):
    """
    Evaluate and rank fragments using the trained classifier.
    
    Args:
        fragments_smiles: list of SMILES strings
        classifier_model: trained model
        predict_fn: function that takes (model, [smiles]) and returns predictions
        
    Returns:
        list of dicts sorted by score (descending)
    """
    predictions = predict_fn(classifier_model, fragments_smiles)
    
    # Add molecular properties
    results = []
    for pred in predictions:
        if not pred.get('valid', False):
            continue
        
        mol = Chem.MolFromSmiles(pred['smiles'])
        if mol is None:
            continue
        
        results.append({
            'smiles': pred['smiles'],
            'score': pred['score'],
            'mw': round(Descriptors.MolWt(mol), 2),
            'logp': round(Descriptors.MolLogP(mol), 2),
            'num_atoms': mol.GetNumHeavyAtoms(),
        })
    
    # Sort by score descending
    results.sort(key=lambda x: x['score'], reverse=True)
    return results


def build_fragment_library(active_mols, classifier_model, predict_fn, include_builtin=True):
    """
    Build and evaluate a complete fragment library.
    
    Args:
        active_mols: list of active RDKit Mol objects
        classifier_model: trained classifier model
        predict_fn: prediction function
        include_builtin: whether to include built-in fragments
        
    Returns:
        dict with fragment evaluation results
    """
    all_fragments = {}
    
    # Extract fragments from actives
    active_frags = extract_fragments_from_actives(active_mols)
    for smi, sources in active_frags.items():
        all_fragments[smi] = {'source': 'active_decomposition', 'parents': sources}
    
    # Add built-in fragments
    if include_builtin:
        for smi in BUILT_IN_FRAGMENTS:
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                canon_smi = Chem.MolToSmiles(mol)
                if canon_smi not in all_fragments:
                    all_fragments[canon_smi] = {'source': 'built-in', 'parents': []}
    
    # Evaluate all fragments
    frag_smiles_list = list(all_fragments.keys())
    evaluated = evaluate_fragments(frag_smiles_list, classifier_model, predict_fn)
    
    # Enrich with source information
    for item in evaluated:
        smi = item['smiles']
        if smi in all_fragments:
            item['source'] = all_fragments[smi]['source']
            item['parents'] = all_fragments[smi]['parents']
        else:
            item['source'] = 'unknown'
            item['parents'] = []
    
    return {
        'fragments': evaluated,
        'total_count': len(evaluated),
        'from_actives': sum(1 for f in evaluated if f.get('source') == 'active_decomposition'),
        'from_builtin': sum(1 for f in evaluated if f.get('source') == 'built-in'),
    }
