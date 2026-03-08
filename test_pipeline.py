"""Quick test to verify the pipeline modules work."""
import sys
import traceback

def test_step(name, fn):
    try:
        result = fn()
        print(f"  [OK] {name}: PASSED")
        return result
    except Exception as e:
        print(f"  [FAIL] {name}: FAILED - {e}")
        traceback.print_exc()
        return None

print("=" * 50)
print("Drug Discovery Pipeline - Module Tests")
print("=" * 50)

# Test 1: Structure Input
print("\n[1/5] Testing Structure Input...")
from modules.structure_input import parse_structure_file, validate_molecules

def test_parse():
    mols = parse_structure_file("test_actives.smi")
    mols = validate_molecules(mols)
    print(f"       Parsed {len(mols)} molecules")
    for m in mols[:3]:
        print(f"       - {m.GetProp('_Name')}: {m.GetProp('SMILES')}")
    assert len(mols) >= 5, "Need at least 5 molecules"
    return mols

active_mols = test_step("Parse SMILES file", test_parse)
if not active_mols:
    print("CRITICAL: Cannot continue without molecules!")
    sys.exit(1)

# Test 2: Decoy Generation
print("\n[2/5] Testing Decoy Generation...")
from modules.decoy_generator import generate_decoys

def test_decoys():
    result = generate_decoys(active_mols, decoys_per_active=5, max_total=100)
    print(f"       Generated {result['stats']['total']} decoys")
    assert len(result['decoys']) > 0, "No decoys generated"
    return result

decoy_result = test_step("Generate decoys", test_decoys)

# Test 3: Classifier
print("\n[3/5] Testing Classifier Training...")
from modules.classifier import train_classifier, predict_activity

def test_classifier():
    if not decoy_result:
        raise Exception("No decoys to train with")
    result = train_classifier(active_mols, decoy_result['decoys'], n_estimators=20)
    m = result['metrics']
    print(f"       AUC-ROC: {m['cv_auc_roc']:.3f}, Accuracy: {m['cv_accuracy']:.3f}")
    return result

clf_result = test_step("Train classifier", test_classifier)

# Test 4: Fragment Library
print("\n[4/5] Testing Fragment Library...")
from modules.fragment_library import build_fragment_library

def test_fragments():
    if not clf_result:
        raise Exception("No classifier")
    result = build_fragment_library(active_mols, clf_result['model'], predict_activity)
    print(f"       Total fragments: {result['total_count']}")
    print(f"       From actives: {result['from_actives']}, Built-in: {result['from_builtin']}")
    if result['fragments']:
        print(f"       Top fragment: {result['fragments'][0]['smiles']} (score: {result['fragments'][0]['score']:.3f})")
    return result

frag_result = test_step("Build fragment library", test_fragments)

# Test 5: Fragment Linking (quick test)
print("\n[5/5] Testing Fragment Linking (limited)...")
from modules.fragment_linker import brics_link_fragments

def test_linking():
    if not frag_result:
        raise Exception("No fragments")
    frag_smiles = [f['smiles'] for f in frag_result['fragments'][:10]]
    generated = brics_link_fragments(frag_smiles, top_n=5, max_molecules=50)
    print(f"       Generated {len(generated)} molecules from BRICS linking")
    return generated

test_step("BRICS linking", test_linking)

print("\n" + "=" * 50)
print("All basic tests completed!")
print("=" * 50)
print("\nTo launch the web app, run:")
print("  py -3 app.py")
print("Then open http://localhost:5000 in your browser")
