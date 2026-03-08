"""
Drug Discovery Pipeline - Flask Web Application
A fragment-based drug discovery tool with modern web UI.
"""

import os
import json
import uuid
import threading
from flask import Flask, render_template, request, jsonify, send_file
from werkzeug.utils import secure_filename

from modules import orchestrator

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB max upload
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
app.config['RESULTS_FOLDER'] = os.path.join(os.path.dirname(__file__), 'results')

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

# Allowed file extensions
ALLOWED_EXTENSIONS = {'sdf', 'smi', 'smiles', 'csv', 'txt'}


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Render the main page."""
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and start pipeline."""
    if 'file' not in request.files:
        # Check if SMILES were pasted directly
        smiles_text = request.form.get('smiles_text', '').strip()
        if not smiles_text:
            return jsonify({'error': 'No file uploaded and no SMILES provided'}), 400
        
        # Save pasted SMILES to a temp file
        job_id = str(uuid.uuid4())[:8]
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}_input.smi')
        with open(filepath, 'w') as f:
            f.write(smiles_text)
    else:
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': f'Invalid file type. Allowed: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400
        
        job_id = str(uuid.uuid4())[:8]
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f'{job_id}_{filename}')
        file.save(filepath)
    
    # Get optional Reaxys API key
    reaxys_key = request.form.get('reaxys_api_key', '').strip() or None
    
    # Start pipeline in background thread
    output_dir = os.path.join(app.config['RESULTS_FOLDER'], job_id)
    
    def run_bg():
        try:
            orchestrator.run_pipeline(
                filepath,
                job_id=job_id,
                reaxys_api_key=reaxys_key,
                output_dir=output_dir,
            )
        except Exception as e:
            state = orchestrator.get_pipeline_state(job_id)
            if state:
                state.error = str(e)
    
    thread = threading.Thread(target=run_bg, daemon=True)
    thread.start()
    
    return jsonify({'job_id': job_id, 'status': 'started'})


@app.route('/api/status/<job_id>')
def get_status(job_id):
    """Get pipeline status."""
    state = orchestrator.get_pipeline_state(job_id)
    if state is None:
        return jsonify({'error': 'Job not found'}), 404
    return jsonify(state.to_dict())


@app.route('/api/results/<job_id>')
def get_results(job_id):
    """Get pipeline results."""
    state = orchestrator.get_pipeline_state(job_id)
    if state is None:
        return jsonify({'error': 'Job not found'}), 404
    
    if state.error:
        return jsonify({'error': state.error}), 500
    
    if state.current_step != 'complete':
        return jsonify({'error': 'Pipeline still running', 'status': state.to_dict()}), 202
    
    return jsonify(state.results)


@app.route('/api/molecule_image/<smiles_encoded>')
def molecule_image(smiles_encoded):
    """Generate 2D molecule image as SVG."""
    import urllib.parse
    from rdkit import Chem
    from rdkit.Chem import Draw
    from io import BytesIO
    
    smiles = urllib.parse.unquote(smiles_encoded)
    mol = Chem.MolFromSmiles(smiles)
    
    if mol is None:
        return 'Invalid SMILES', 400
    
    # Generate SVG image
    from rdkit.Chem.Draw import rdMolDraw2D
    drawer = rdMolDraw2D.MolDraw2DSVG(350, 250)
    drawer.drawOptions().addStereoAnnotation = True
    drawer.drawOptions().addAtomIndices = False
    drawer.drawOptions().bondLineWidth = 2.0
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText()
    
    return svg, 200, {'Content-Type': 'image/svg+xml'}


@app.route('/api/download/<job_id>')
def download_results(job_id):
    """Download results in requested format."""
    fmt = request.args.get('format', 'json').lower()
    
    if fmt == 'sdf':
        file_path = os.path.join(app.config['RESULTS_FOLDER'], job_id, 'results.sdf')
        filename = f'results_{job_id}.sdf'
    elif fmt == 'smi':
        file_path = os.path.join(app.config['RESULTS_FOLDER'], job_id, 'results.smi')
        filename = f'results_{job_id}.smi'
    else:
        file_path = os.path.join(app.config['RESULTS_FOLDER'], job_id, 'results.json')
        filename = f'results_{job_id}.json'
        
    if not os.path.exists(file_path):
        return jsonify({'error': 'Results file not found. Please wait for the job to finish.'}), 404
        
    return send_file(file_path, as_attachment=True, download_name=filename)


    import socket
    
    def find_free_port(start_port=5000):
        """Find the first available port starting from start_port."""
        port = start_port
        while port < 65535:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            try:
                sock.bind(('0.0.0.0', port))
                sock.close()
                return port
            except OSError:
                port += 1
        return start_port
        
    port = find_free_port(5000)
    
    print("\n" + "=" * 60)
    print("  Drug Discovery Pipeline")
    print("  Fragment-Based Drug Design Tool")
    print("=" * 60)
    print(f"\n  ✅ Successfully Bound to Port: {port}")
    print(f"  🌐 Open your browser to: http://127.0.0.1:{port}")
    print(f"\n  Supported file formats: {', '.join(ALLOWED_EXTENSIONS)}")
    print("=" * 60 + "\n")
    app.run(host='0.0.0.0', debug=True, port=port, use_reloader=False)
