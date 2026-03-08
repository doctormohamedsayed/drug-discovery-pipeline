"""
Module 1: Structure Input & Parsing
Handles parsing of molecular structure files (SDF, SMILES CSV/TXT).
"""

import os
import csv
from rdkit import Chem
from rdkit.Chem import Descriptors, AllChem


def parse_smiles_file(filepath):
    """Parse a file containing SMILES strings (one per line, or CSV with SMILES column)."""
    molecules = []
    
    ext = os.path.splitext(filepath)[1].lower()
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read().strip()
    except UnicodeDecodeError:
        with open(filepath, 'r', encoding='latin-1') as f:
            content = f.read().strip()
    
    lines = content.split('\n')
    
    # Detect if it's CSV (has commas or tabs)
    if ',' in lines[0] or '\t' in lines[0]:
        delimiter = ',' if ',' in lines[0] else '\t'
        reader = csv.reader(lines, delimiter=delimiter)
        header = next(reader, None)
        
        # Find SMILES column
        smiles_col = 0
        name_col = None
        if header:
            for i, col_name in enumerate(header):
                col_lower = col_name.strip().lower()
                if col_lower in ['smiles', 'smi', 'canonical_smiles', 'molecule']:
                    smiles_col = i
                elif col_lower in ['name', 'id', 'mol_name', 'compound_id', 'title']:
                    name_col = i
        
        for row_idx, row in enumerate(reader):
            if len(row) > smiles_col:
                smi = row[smiles_col].strip()
                name = row[name_col].strip() if name_col is not None and len(row) > name_col else f"mol_{row_idx}"
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    mol.SetProp("_Name", name)
                    mol.SetProp("SMILES", Chem.MolToSmiles(mol))
                    molecules.append(mol)
    else:
        # Plain text, one SMILES per line (optionally with name after space/tab)
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            smi = parts[0]
            name = parts[1] if len(parts) > 1 else f"mol_{idx}"
            mol = Chem.MolFromSmiles(smi)
            if mol is not None:
                mol.SetProp("_Name", name)
                mol.SetProp("SMILES", Chem.MolToSmiles(mol))
                molecules.append(mol)
    
    return molecules


def parse_sdf_file(filepath):
    """Parse an SDF file and return list of RDKit Mol objects."""
    molecules = []
    supplier = Chem.SDMolSupplier(filepath, removeHs=True, sanitize=True)
    
    for idx, mol in enumerate(supplier):
        if mol is not None:
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"mol_{idx}"
            mol.SetProp("_Name", name)
            mol.SetProp("SMILES", Chem.MolToSmiles(mol))
            molecules.append(mol)
    
    return molecules


def parse_structure_file(filepath):
    """
    Auto-detect file type and parse molecular structures.
    Supports: .sdf, .smi, .smiles, .csv, .txt
    Returns: list of RDKit Mol objects with _Name and SMILES properties
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")
    
    ext = os.path.splitext(filepath)[1].lower()
    
    if ext == '.sdf':
        molecules = parse_sdf_file(filepath)
    elif ext in ['.smi', '.smiles', '.csv', '.txt']:
        molecules = parse_smiles_file(filepath)
    else:
        # Try SMILES first, then SDF
        try:
            molecules = parse_smiles_file(filepath)
            if not molecules:
                molecules = parse_sdf_file(filepath)
        except Exception:
            molecules = parse_sdf_file(filepath)
    
    if not molecules:
        raise ValueError("No valid molecules found in the input file.")
    
    return molecules


def validate_molecules(molecules):
    """Validate molecules and compute basic properties."""
    valid = []
    for mol in molecules:
        try:
            Chem.SanitizeMol(mol)
            # Compute basic properties
            mol.SetProp("MW", str(round(Descriptors.MolWt(mol), 2)))
            mol.SetProp("LogP", str(round(Descriptors.MolLogP(mol), 2)))
            mol.SetProp("HBD", str(Descriptors.NumHDonors(mol)))
            mol.SetProp("HBA", str(Descriptors.NumHAcceptors(mol)))
            mol.SetProp("TPSA", str(round(Descriptors.TPSA(mol), 2)))
            mol.SetProp("RotBonds", str(Descriptors.NumRotatableBonds(mol)))
            valid.append(mol)
        except Exception:
            continue
    return valid


def get_molecule_info(mol):
    """Get a dictionary of molecule properties."""
    return {
        'name': mol.GetProp("_Name") if mol.HasProp("_Name") else "Unknown",
        'smiles': mol.GetProp("SMILES") if mol.HasProp("SMILES") else Chem.MolToSmiles(mol),
        'mw': float(mol.GetProp("MW")) if mol.HasProp("MW") else Descriptors.MolWt(mol),
        'logp': float(mol.GetProp("LogP")) if mol.HasProp("LogP") else Descriptors.MolLogP(mol),
        'hbd': int(mol.GetProp("HBD")) if mol.HasProp("HBD") else Descriptors.NumHDonors(mol),
        'hba': int(mol.GetProp("HBA")) if mol.HasProp("HBA") else Descriptors.NumHAcceptors(mol),
        'tpsa': float(mol.GetProp("TPSA")) if mol.HasProp("TPSA") else Descriptors.TPSA(mol),
        'rot_bonds': int(mol.GetProp("RotBonds")) if mol.HasProp("RotBonds") else Descriptors.NumRotatableBonds(mol),
    }
