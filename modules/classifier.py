"""
Module 3: Classifier Training
Trains a Random Forest classifier on actives vs decoys using
Morgan fingerprints and molecular descriptors.
"""

import os
import numpy as np
import joblib
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, AllChem
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    classification_report
)


def _mol_to_features(mol, fp_radius=2, fp_bits=2048):
    """
    Convert a molecule to a feature vector combining:
    - Morgan fingerprint (2048 bits)
    - Molecular descriptors (MW, LogP, TPSA, HBD, HBA, RotBonds, etc.)
    """
    # Morgan fingerprint
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, fp_radius, nBits=fp_bits)
    fp_array = np.zeros(fp_bits, dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, fp_array)
    
    # Molecular descriptors
    desc = np.array([
        Descriptors.MolWt(mol),
        Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.RingCount(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.FractionCSP3(mol),
        Descriptors.HeavyAtomCount(mol),
    ], dtype=np.float64)
    
    # Combine
    features = np.concatenate([fp_array.astype(np.float64), desc])
    return features


def smiles_to_features(smiles, fp_radius=2, fp_bits=2048):
    """Convert a SMILES string to feature vector."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return _mol_to_features(mol, fp_radius, fp_bits)


def prepare_dataset(active_mols, decoy_mols):
    """
    Prepare training dataset from active and decoy molecules.
    
    Returns:
        X: feature matrix (n_samples, n_features)
        y: labels (1=active, 0=decoy)
        valid_smiles: list of SMILES for valid molecules
    """
    X = []
    y = []
    valid_smiles = []
    
    for mol in active_mols:
        try:
            features = _mol_to_features(mol)
            X.append(features)
            y.append(1)  # Active
            valid_smiles.append(Chem.MolToSmiles(mol))
        except Exception:
            continue
    
    for mol in decoy_mols:
        try:
            features = _mol_to_features(mol)
            X.append(features)
            y.append(0)  # Decoy
            valid_smiles.append(Chem.MolToSmiles(mol))
        except Exception:
            continue
    
    return np.array(X), np.array(y), valid_smiles


def train_classifier(active_mols, decoy_mols, n_estimators=100, random_state=42):
    """
    Train a Random Forest classifier on actives vs decoys.
    
    Args:
        active_mols: list of RDKit Mol objects (actives)
        decoy_mols: list of RDKit Mol objects (decoys)
        n_estimators: number of trees in the forest
        random_state: random seed for reproducibility
        
    Returns:
        dict with keys:
          - 'model': trained RandomForestClassifier
          - 'metrics': dict of performance metrics
          - 'feature_importances': array of feature importances
    """
    X, y, smiles = prepare_dataset(active_mols, decoy_mols)
    
    if len(X) == 0:
        raise ValueError("No valid molecules for training!")
    
    if len(set(y)) < 2:
        raise ValueError("Need both actives and decoys for training!")
    
    # Train classifier
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1,
    )
    
    # Cross-validation
    n_splits = min(5, min(np.sum(y == 0), np.sum(y == 1)))
    n_splits = max(2, n_splits)  # At least 2-fold
    
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    cv_accuracy = cross_val_score(clf, X, y, cv=cv, scoring='accuracy')
    cv_auc = cross_val_score(clf, X, y, cv=cv, scoring='roc_auc')
    cv_precision = cross_val_score(clf, X, y, cv=cv, scoring='precision')
    cv_recall = cross_val_score(clf, X, y, cv=cv, scoring='recall')
    
    # Train on full data
    clf.fit(X, y)
    
    # Final predictions
    y_pred = clf.predict(X)
    y_proba = clf.predict_proba(X)[:, 1]
    
    metrics = {
        'cv_accuracy': float(np.mean(cv_accuracy)),
        'cv_accuracy_std': float(np.std(cv_accuracy)),
        'cv_auc_roc': float(np.mean(cv_auc)),
        'cv_auc_std': float(np.std(cv_auc)),
        'cv_precision': float(np.mean(cv_precision)),
        'cv_recall': float(np.mean(cv_recall)),
        'train_accuracy': float(accuracy_score(y, y_pred)),
        'train_auc_roc': float(roc_auc_score(y, y_proba)),
        'n_actives': int(np.sum(y == 1)),
        'n_decoys': int(np.sum(y == 0)),
        'n_total': len(y),
        'n_folds': n_splits,
    }
    
    return {
        'model': clf,
        'metrics': metrics,
        'feature_importances': clf.feature_importances_,
    }


def predict_activity(model, smiles_list):
    """
    Predict activity probability for a list of SMILES strings.
    
    Args:
        model: trained RandomForestClassifier
        smiles_list: list of SMILES strings
        
    Returns:
        list of dicts with 'smiles', 'score' (probability of being active),
        'prediction' (1=active, 0=inactive)
    """
    results = []
    
    for smi in smiles_list:
        features = smiles_to_features(smi)
        if features is None:
            results.append({
                'smiles': smi,
                'score': 0.0,
                'prediction': 0,
                'valid': False,
            })
            continue
        
        features = features.reshape(1, -1)
        proba = model.predict_proba(features)[0][1]
        pred = model.predict(features)[0]
        
        results.append({
            'smiles': smi,
            'score': float(proba),
            'prediction': int(pred),
            'valid': True,
        })
    
    return results


def save_model(model, filepath):
    """Save trained model to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(model, filepath)


def load_model(filepath):
    """Load trained model from disk."""
    return joblib.load(filepath)
