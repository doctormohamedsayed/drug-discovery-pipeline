"""
Module 7: Novelty Checking
Checks generated molecules against PubChem, ChEMBL, Reaxys, and internal actives.
"""

import os
import time
import requests
import urllib.parse
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem


def _tanimoto_vs_actives(smiles, active_fps):
    """Compute max Tanimoto similarity against active molecules."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return 0.0
    
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
    if not active_fps:
        return 0.0
    
    return max(DataStructs.TanimotoSimilarity(fp, afp) for afp in active_fps)


def check_pubchem(smiles, threshold=0.85):
    """
    Check novelty against PubChem using REST API.
    
    Args:
        smiles: SMILES string
        threshold: Tanimoto threshold (>= this = "known")
    
    Returns:
        dict with 'found', 'similarity', 'cid', 'source'
    """
    try:
        encoded = urllib.parse.quote(smiles, safe='')
        
        # First try exact structure search
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/smiles/{encoded}/cids/JSON"
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            cids = data.get('IdentifierList', {}).get('CID', [])
            if cids:
                return {
                    'found': True,
                    'similarity': 1.0,
                    'hit_id': f"CID:{cids[0]}",
                    'source': 'PubChem',
                    'match_type': 'exact',
                }
        
        time.sleep(0.5)  # Rate limiting
        
        # Try similarity search
        url = (f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/fastsimilarity_2d/"
               f"smiles/{encoded}/cids/JSON?Threshold={int(threshold * 100)}")
        response = requests.get(url, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            cids = data.get('IdentifierList', {}).get('CID', [])
            if cids:
                return {
                    'found': True,
                    'similarity': threshold,
                    'hit_id': f"CID:{cids[0]}",
                    'source': 'PubChem',
                    'match_type': 'similar',
                }
        
        return {
            'found': False,
            'similarity': 0.0,
            'hit_id': None,
            'source': 'PubChem',
            'match_type': 'none',
        }
    
    except requests.exceptions.RequestException:
        return {
            'found': False,
            'similarity': 0.0,
            'hit_id': None,
            'source': 'PubChem',
            'match_type': 'error',
            'error': 'API request failed',
        }


def check_chembl(smiles, threshold=85):
    """
    Check novelty against ChEMBL using REST API.
    
    Args:
        smiles: SMILES string
        threshold: similarity percentage (85 = 85%)
    
    Returns:
        dict with search results
    """
    try:
        encoded = urllib.parse.quote(smiles, safe='')
        url = f"https://www.ebi.ac.uk/chembl/api/data/similarity/{encoded}/{threshold}.json"
        
        headers = {'Accept': 'application/json'}
        response = requests.get(url, headers=headers, timeout=15)
        
        if response.status_code == 200:
            data = response.json()
            molecules = data.get('molecules', [])
            if molecules:
                best = molecules[0]
                return {
                    'found': True,
                    'similarity': float(best.get('similarity', threshold)) / 100.0,
                    'hit_id': best.get('molecule_chembl_id', 'Unknown'),
                    'source': 'ChEMBL',
                    'match_type': 'similar',
                    'pref_name': best.get('pref_name', ''),
                }
        
        return {
            'found': False,
            'similarity': 0.0,
            'hit_id': None,
            'source': 'ChEMBL',
            'match_type': 'none',
        }
    
    except requests.exceptions.RequestException:
        return {
            'found': False,
            'similarity': 0.0,
            'hit_id': None,
            'source': 'ChEMBL',
            'match_type': 'error',
            'error': 'API request failed',
        }


def check_reaxys(smiles, api_key=None, threshold=0.85):
    """
    Check novelty against Reaxys using REST API.
    Requires API key from Elsevier.
    
    Args:
        smiles: SMILES string
        api_key: Reaxys API key (or set REAXYS_API_KEY env variable)
        threshold: similarity threshold
    
    Returns:
        dict with search results
    """
    if api_key is None:
        api_key = os.environ.get('REAXYS_API_KEY', '')
    
    if not api_key:
        return {
            'found': False,
            'similarity': 0.0,
            'hit_id': None,
            'source': 'Reaxys',
            'match_type': 'skipped',
            'error': 'No API key provided. Set REAXYS_API_KEY environment variable.',
        }
    
    try:
        # Reaxys API: create session and search
        base_url = "https://api.elsevier.com/content/reaxys"
        
        headers = {
            'Accept': 'application/xml',
            'apikey': api_key,
        }
        
        # Start session
        session_url = f"{base_url}/session.do"
        session_data = f'<xf name="session.create"><xf name="caller">DrugDiscoveryPipeline</xf></xf>'
        
        session_resp = requests.post(
            session_url, data=session_data, headers=headers, timeout=15
        )
        
        if session_resp.status_code != 200:
            return {
                'found': False,
                'similarity': 0.0,
                'hit_id': None,
                'source': 'Reaxys',
                'match_type': 'error',
                'error': f'Session creation failed: {session_resp.status_code}',
            }
        
        # Extract session ID from response
        import xml.etree.ElementTree as ET
        root = ET.fromstring(session_resp.text)
        session_id = root.text if root.text else ''
        
        if not session_id:
            return {
                'found': False,
                'similarity': 0.0,
                'hit_id': None,
                'source': 'Reaxys',
                'match_type': 'error',
                'error': 'Could not obtain session ID',
            }
        
        # Structure similarity search
        search_url = f"{base_url}/search.do"
        search_data = (
            f'<xf name="search.start">'
            f'<xf name="session">{session_id}</xf>'
            f'<xf name="dbname">RX</xf>'
            f'<xf name="context">S</xf>'
            f'<xf name="where">'
            f'<xf name="criterion">'
            f'<xf name="field">structure</xf>'
            f'<xf name="value">{smiles}</xf>'
            f'<xf name="similarityThreshold">{int(threshold * 100)}</xf>'
            f'</xf>'
            f'</xf>'
            f'</xf>'
        )
        
        search_resp = requests.post(
            search_url, data=search_data, headers=headers, timeout=30
        )
        
        if search_resp.status_code == 200:
            search_root = ET.fromstring(search_resp.text)
            # Check if any hits found
            hits = search_root.findall('.//hit') or search_root.findall('.//result')
            
            if hits:
                return {
                    'found': True,
                    'similarity': threshold,
                    'hit_id': f"Reaxys hit",
                    'source': 'Reaxys',
                    'match_type': 'similar',
                }
        
        return {
            'found': False,
            'similarity': 0.0,
            'hit_id': None,
            'source': 'Reaxys',
            'match_type': 'none',
        }
    
    except Exception as e:
        return {
            'found': False,
            'similarity': 0.0,
            'hit_id': None,
            'source': 'Reaxys',
            'match_type': 'error',
            'error': str(e),
        }


def check_novelty(smiles_list, active_mols, reaxys_api_key=None,
                  threshold=0.85, progress_callback=None):
    """
    Check novelty of molecules against all databases.
    
    Args:
        smiles_list: list of SMILES strings to check
        active_mols: list of active RDKit Mol objects (for internal check)
        reaxys_api_key: optional Reaxys API key
        threshold: similarity threshold for "known" classification
        progress_callback: optional callback(step, message)
        
    Returns:
        list of dicts with novelty results for each molecule
    """
    # Pre-compute active fingerprints
    active_fps = []
    for mol in active_mols:
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            active_fps.append(fp)
        except Exception:
            continue
    
    results = []
    
    for idx, smi in enumerate(smiles_list):
        if progress_callback:
            progress_callback("novelty", f"Checking molecule {idx + 1}/{len(smiles_list)}...")
        
        mol_result = {
            'smiles': smi,
            'databases': {},
            'overall_status': 'Novel',
            'max_similarity': 0.0,
            'closest_hit': None,
        }
        
        # Internal check against actives
        internal_sim = _tanimoto_vs_actives(smi, active_fps)
        mol_result['databases']['internal'] = {
            'found': internal_sim >= threshold,
            'similarity': round(internal_sim, 4),
            'source': 'Input Actives',
            'match_type': 'similar' if internal_sim >= threshold else 'none',
        }
        
        if internal_sim > mol_result['max_similarity']:
            mol_result['max_similarity'] = internal_sim
            mol_result['closest_hit'] = 'Input Active'
        
        # PubChem check
        pubchem_result = check_pubchem(smi, threshold)
        mol_result['databases']['pubchem'] = pubchem_result
        time.sleep(0.3)  # Rate limiting
        
        if pubchem_result.get('found'):
            if pubchem_result.get('similarity', 0) > mol_result['max_similarity']:
                mol_result['max_similarity'] = pubchem_result['similarity']
                mol_result['closest_hit'] = pubchem_result.get('hit_id', 'PubChem')
        
        # ChEMBL check
        chembl_result = check_chembl(smi, int(threshold * 100))
        mol_result['databases']['chembl'] = chembl_result
        time.sleep(0.3)  # Rate limiting
        
        if chembl_result.get('found'):
            if chembl_result.get('similarity', 0) > mol_result['max_similarity']:
                mol_result['max_similarity'] = chembl_result['similarity']
                mol_result['closest_hit'] = chembl_result.get('hit_id', 'ChEMBL')
        
        # Reaxys check
        reaxys_result = check_reaxys(smi, api_key=reaxys_api_key, threshold=threshold)
        mol_result['databases']['reaxys'] = reaxys_result
        
        if reaxys_result.get('found'):
            if reaxys_result.get('similarity', 0) > mol_result['max_similarity']:
                mol_result['max_similarity'] = reaxys_result['similarity']
                mol_result['closest_hit'] = reaxys_result.get('hit_id', 'Reaxys')
        
        # Determine overall status
        any_found = any(
            db.get('found', False)
            for db in mol_result['databases'].values()
        )
        
        if any_found:
            if mol_result['max_similarity'] >= 0.95:
                mol_result['overall_status'] = 'Known'
            else:
                mol_result['overall_status'] = 'Similar-to-known'
        else:
            mol_result['overall_status'] = 'Novel'
        
        mol_result['max_similarity'] = round(mol_result['max_similarity'], 4)
        results.append(mol_result)
    
    return results
