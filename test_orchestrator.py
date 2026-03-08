import sys
import os

from modules.orchestrator import run_pipeline

print("Running pipeline via orchestrator...")
output_dir = 'test_orchestrator'
job_id = 'test_001'

try:
    results = run_pipeline(
        'test_actives.smi', 
        job_id=job_id, 
        output_dir=output_dir
    )
    print("Pipeline finished.")
    files_created = os.listdir(output_dir)
    print(f"Files created in {output_dir}:", files_created)
    if 'results.sdf' in files_created and 'results.smi' in files_created:
        print("SDF and SMI files generated successfully!")
        
        # Verify SDF contents
        sdf_path = os.path.join(output_dir, 'results.sdf')
        with open(sdf_path, 'r') as f:
            lines = f.readlines()
            print(f"SDF file length: {len(lines)} lines")
            
        # Verify SMI contents
        smi_path = os.path.join(output_dir, 'results.smi')
        with open(smi_path, 'r') as f:
            lines = f.readlines()
            print(f"SMI file length: {len(lines)} lines")
    else:
        print("FAILED: SDF or SMI files missing!")
        sys.exit(1)
        
except Exception as e:
    print(f"Error: {e}")
    sys.exit(1)
