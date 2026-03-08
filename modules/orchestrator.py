"""
Module 8: Pipeline Orchestrator
Coordinates the full drug discovery pipeline from structure input through novelty checking.
"""

import os
import json
import time
import uuid
import traceback
from datetime import datetime
from rdkit import Chem

from . import structure_input
from . import decoy_generator
from . import classifier
from . import fragment_library
from . import fragment_linker
from . import ranker
from . import novelty_checker


class PipelineState:
    """Track pipeline execution state for progress reporting."""
    
    STEPS = [
        'parsing',
        'decoy_generation',
        'classifier_training',
        'fragment_evaluation',
        'fragment_linking',
        'ranking',
        'novelty_checking',
        'complete',
    ]
    
    STEP_LABELS = {
        'parsing': 'Parsing Input Structures',
        'decoy_generation': 'Generating Decoys',
        'classifier_training': 'Training Classifier',
        'fragment_evaluation': 'Evaluating Fragment Library',
        'fragment_linking': 'Linking & Merging Fragments (3D)',
        'ranking': 'Ranking Generated Molecules',
        'novelty_checking': 'Checking Novelty Against Databases',
        'complete': 'Pipeline Complete',
    }
    
    def __init__(self, job_id=None):
        self.job_id = job_id or str(uuid.uuid4())[:8]
        self.current_step = None
        self.current_message = ''
        self.progress = 0
        self.error = None
        self.results = {}
        self.start_time = time.time()
        self.step_times = {}
    
    def set_step(self, step, message=''):
        self.current_step = step
        self.current_message = message
        step_idx = self.STEPS.index(step) if step in self.STEPS else 0
        self.progress = int((step_idx / len(self.STEPS)) * 100)
        self.step_times[step] = time.time()
    
    def to_dict(self):
        elapsed = time.time() - self.start_time
        return {
            'job_id': self.job_id,
            'current_step': self.current_step,
            'step_label': self.STEP_LABELS.get(self.current_step, ''),
            'message': self.current_message,
            'progress': self.progress,
            'error': self.error,
            'elapsed_seconds': round(elapsed, 1),
            'has_results': bool(self.results),
        }


# Global state storage (simple in-memory for the web app)
_pipeline_states = {}


def get_pipeline_state(job_id):
    """Get the current state of a pipeline job."""
    return _pipeline_states.get(job_id)


def run_pipeline(filepath, job_id=None, reaxys_api_key=None, output_dir=None):
    """
    Run the full drug discovery pipeline.
    
    Args:
        filepath: path to structure file (SDF, SMILES, etc.)
        job_id: optional job ID for tracking
        reaxys_api_key: optional Reaxys API key
        output_dir: directory to save results
        
    Returns:
        dict with all pipeline results
    """
    state = PipelineState(job_id)
    _pipeline_states[state.job_id] = state
    
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(filepath), 'results', state.job_id)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # ===== STEP 1: Parse Input Structures =====
        state.set_step('parsing', 'Reading and validating molecular structures...')
        
        molecules = structure_input.parse_structure_file(filepath)
        molecules = structure_input.validate_molecules(molecules)
        
        if len(molecules) < 2:
            raise ValueError(f"Need at least 2 valid structures, got {len(molecules)}")
        
        active_info = [structure_input.get_molecule_info(m) for m in molecules]
        state.results['actives'] = {
            'count': len(molecules),
            'molecules': active_info,
        }
        state.set_step('parsing', f'Found {len(molecules)} valid active molecules')
        
        # ===== STEP 2: Generate Decoys =====
        state.set_step('decoy_generation', 'Generating property-matched decoys...')
        
        decoy_result = decoy_generator.generate_decoys(
            molecules,
            decoys_per_active=39,
            max_total=5000,
        )
        
        state.results['decoys'] = {
            'count': decoy_result['stats']['total'],
            'per_active': decoy_result['stats']['per_active'],
            'relaxations': decoy_result['stats']['relaxations'],
        }
        state.set_step('decoy_generation',
                       f'Generated {decoy_result["stats"]["total"]} decoys')
        
        # ===== STEP 3: Train Classifier =====
        state.set_step('classifier_training', 'Training Random Forest classifier...')
        
        clf_result = classifier.train_classifier(
            molecules,
            decoy_result['decoys'],
            n_estimators=100,
        )
        
        trained_model = clf_result['model']
        state.results['classifier'] = {
            'metrics': clf_result['metrics'],
        }
        
        # Save model
        model_path = os.path.join(output_dir, 'classifier_model.pkl')
        classifier.save_model(trained_model, model_path)
        
        state.set_step('classifier_training',
                       f'AUC-ROC: {clf_result["metrics"]["cv_auc_roc"]:.3f}')
        
        # ===== STEP 4: Evaluate Fragment Library =====
        state.set_step('fragment_evaluation', 'Building and evaluating fragment library...')
        
        frag_result = fragment_library.build_fragment_library(
            molecules,
            trained_model,
            classifier.predict_activity,
            include_builtin=True,
        )
        
        state.results['fragments'] = {
            'total': frag_result['total_count'],
            'from_actives': frag_result['from_actives'],
            'from_builtin': frag_result['from_builtin'],
            'top_fragments': frag_result['fragments'][:20],  # Top 20
        }
        state.set_step('fragment_evaluation',
                       f'Evaluated {frag_result["total_count"]} fragments')
        
        # ===== STEP 5: Fragment Linking & Merging (3D) =====
        state.set_step('fragment_linking',
                       'Linking and merging fragments with 3D optimization...')
        
        def link_progress(step, msg):
            state.set_step('fragment_linking', msg)
        
        link_result = fragment_linker.link_and_merge_fragments(
            frag_result['fragments'],
            max_output=5000,
            progress_callback=link_progress,
        )
        
        state.results['linking'] = {
            'total_generated': link_result['total_generated'],
            'from_brics': link_result['from_brics'],
            'from_merging': link_result['from_merging'],
            'with_3d': link_result['with_3d'],
            'final_count': link_result['final_count'],
        }
        state.set_step('fragment_linking',
                       f'Generated {link_result["final_count"]} molecules')
        
        # ===== STEP 6: Ranking =====
        state.set_step('ranking', 'Ranking molecules by multi-criteria score...')
        
        top_molecules = ranker.rank_molecules(
            link_result['molecules'],
            trained_model,
            classifier.predict_activity,
            molecules,
            top_n=10,
        )
        
        state.results['top_molecules'] = top_molecules
        state.set_step('ranking', f'Selected top {len(top_molecules)} molecules')
        
        # ===== STEP 7: Novelty Checking =====
        state.set_step('novelty_checking', 'Checking novelty against databases...')
        
        top_smiles = [m['smiles'] for m in top_molecules]
        
        def novelty_progress(step, msg):
            state.set_step('novelty_checking', msg)
        
        novelty_results = novelty_checker.check_novelty(
            top_smiles,
            molecules,
            reaxys_api_key=reaxys_api_key,
            threshold=0.85,
            progress_callback=novelty_progress,
        )
        
        # Merge novelty results into top molecules
        for mol_data, nov_data in zip(top_molecules, novelty_results):
            mol_data['novelty'] = nov_data['overall_status']
            mol_data['max_db_similarity'] = nov_data['max_similarity']
            mol_data['closest_hit'] = nov_data['closest_hit']
            mol_data['db_details'] = nov_data['databases']
        
        state.results['novelty'] = {
            'novel': sum(1 for n in novelty_results if n['overall_status'] == 'Novel'),
            'similar': sum(1 for n in novelty_results if n['overall_status'] == 'Similar-to-known'),
            'known': sum(1 for n in novelty_results if n['overall_status'] == 'Known'),
        }
        
        # ===== COMPLETE =====
        state.set_step('complete', 'Pipeline complete!')
        
        # Save results to file
        results_path = os.path.join(output_dir, 'results.json')
        save_results = {
            'job_id': state.job_id,
            'timestamp': datetime.now().isoformat(),
            'input_file': os.path.basename(filepath),
            'actives': state.results['actives'],
            'decoys': state.results['decoys'],
            'classifier': state.results['classifier'],
            'fragments': {
                'total': state.results['fragments']['total'],
                'top_fragments': state.results['fragments']['top_fragments'][:10],
            },
            'linking': state.results['linking'],
            'top_molecules': state.results['top_molecules'],
            'novelty': state.results['novelty'],
        }
        
        with open(results_path, 'w') as f:
            json.dump(save_results, f, indent=2, default=str)
        
        # Save results as SMILES file
        smi_path = os.path.join(output_dir, 'results.smi')
        with open(smi_path, 'w') as f:
            for i, mol_data in enumerate(top_molecules):
                # Also include some basic properties in the SMILES line
                score = round(mol_data.get('composite_score', 0), 3)
                f.write(f"{mol_data['smiles']}\\tMolecule_{i+1}\\tScore:{score}\\n")
                
        # Save results as SDF file (with 3D coords if available)
        sdf_path = os.path.join(output_dir, 'results.sdf')
        from rdkit.Chem import AllChem
        writer = Chem.SDWriter(sdf_path)
        
        # Create a mapping of SMILES to 3D mols from the linking step
        smiles_to_mol3d = {}
        for r in link_result['molecules']:
            if r['has_3d'] and r['mol_3d'] is not None:
                smiles_to_mol3d[r['smiles']] = r['mol_3d']
                
        for i, mol_data in enumerate(top_molecules):
            smi = mol_data['smiles']
            # Try to get the 3D conformer, otherwise create a 2D one
            mol = smiles_to_mol3d.get(smi)
            if mol is None:
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    AllChem.Compute2DCoords(mol)
            
            if mol is not None:
                # Add properties to the SDF record
                mol.SetProp("_Name", f"Molecule_{i+1}")
                mol.SetProp("CompositeScore", str(round(mol_data.get('composite_score', 0), 4)))
                mol.SetProp("ActivityScore", str(round(mol_data.get('activity_score', 0), 4)))
                mol.SetProp("DrugLikeness", str(round(mol_data.get('druglikeness', 0), 4)))
                mol.SetProp("SAScore", str(round(mol_data.get('sa_score', 0), 4)))
                mol.SetProp("Diversity", str(round(mol_data.get('diversity', 0), 4)))
                mol.SetProp("Novelty", str(mol_data.get('novelty', 'Unknown')))
                mol.SetProp("ExactSmiles", smi)
                writer.write(mol)
                
        writer.close()
        
        return state.results
    
    except Exception as e:
        state.error = str(e)
        state.current_message = f'Error: {str(e)}'
        traceback.print_exc()
        raise
