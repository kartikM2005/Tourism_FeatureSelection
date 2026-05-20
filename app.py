import os
import json
import traceback
import pandas as pd
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Import pipeline helper modules
from src.preprocess_data import preprocess_pipeline
from src.feature_selection import run_feature_selection_pipeline
from src.model_training import run_model_training_pipeline
from src.visualization import generate_plots

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
UPLOADS_DIR = os.path.join(DATA_DIR, 'uploads')
PLOTS_DIR = os.path.join(DATA_DIR, 'plots')

os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

@app.route('/')
def serve_index():
    return app.send_static_file('index.html')

@app.route('/api/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part provided'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ['.csv', '.xlsx', '.xls']:
        return jsonify({'error': 'Only CSV and Excel (.xlsx, .xls) files are supported'}), 400
        
    try:
        # Secure filename fallback
        filename = file.filename.replace('/', '').replace('\\', '')
        filepath = os.path.join(UPLOADS_DIR, filename)
        file.save(filepath)
        
        # Read header to get columns depending on extension
        if ext == '.csv':
            df = pd.read_csv(filepath, nrows=5)
        else:
            df = pd.read_excel(filepath, nrows=5)
        columns = list(df.columns)
        
        return jsonify({
            'message': 'File uploaded successfully',
            'filename': filename,
            'filepath': filepath,
            'columns': columns
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Failed to process file: {str(e)}'}), 500

@app.route('/api/run-pipeline', methods=['POST'])
def run_pipeline():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'Invalid JSON payload'}), 400
        
    filepath = data.get('filepath')
    target_col = data.get('target_col')
    k = int(data.get('k', 20))
    
    if not filepath or not os.path.exists(filepath):
        return jsonify({'error': 'Uploaded file path not found'}), 400
        
    # Auto-detect target column if missing or set to 'auto'
    if not target_col or target_col == 'auto':
        ext = os.path.splitext(filepath)[1].lower()
        if ext in ['.xlsx', '.xls']:
            df_temp = pd.read_excel(filepath, nrows=5)
        else:
            df_temp = pd.read_csv(filepath, nrows=5)
        cols = list(df_temp.columns)
        
        candidates = ['is_canceled', 'churn', 'target', 'class', 'label', 'outcome', 'status', 'prodtaken', 'booking_complete', 'liked', 'recommended']
        found = False
        for c in cols:
            if c.lower() in candidates:
                target_col = c
                found = True
                break
        if not found and cols:
            target_col = cols[-1]
            
    if not target_col:
        return jsonify({'error': 'Could not detect target column automatically'}), 400
        
    try:
        print(f"Starting pipeline execution for file: {filepath} | target: {target_col} | k: {k}")
        
        # 1. Preprocessing
        print("Running Preprocessing...")
        X, y = preprocess_pipeline(filepath, target_col)
        
        # 2. Feature Selection
        print("Running Feature Selection...")
        selected_features = run_feature_selection_pipeline(X, y, k=k)
        
        # Save feature selection results locally
        with open(os.path.join(DATA_DIR, 'api_selected_features.json'), 'w') as f:
            json.dump(selected_features, f, indent=4)
            
        # 3. Model Training
        print("Running Model Training & Evaluation...")
        results = run_model_training_pipeline(X, y, selected_features)
        
        with open(os.path.join(DATA_DIR, 'api_model_results.json'), 'w') as f:
            json.dump(results, f, indent=4)
            
        # 4. Visualizations
        print("Generating Visualizations...")
        plots = generate_plots(results, PLOTS_DIR)
        
        print("Pipeline Complete successfully!")
        return jsonify({
            'success': True,
            'target_col': target_col,
            'selected_features': selected_features,
            'results': results,
            'plots': {
                'accuracy': f'/api/plots/{plots["accuracy_plot"]}',
                'f1': f'/api/plots/{plots["f1_plot"]}',
                'time': f'/api/plots/{plots["time_plot"]}'
            }
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'error': f'Pipeline failure: {str(e)}'}), 500

@app.route('/api/plots/<path:filename>')
def serve_plot(filename):
    return send_from_directory(PLOTS_DIR, filename)

if __name__ == '__main__':
    print("Starting Flask Backend API Server on port 5000...")
    app.run(host='0.0.0.0', port=5000, debug=True, threaded=True)
