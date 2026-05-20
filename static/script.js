document.addEventListener('DOMContentLoaded', () => {
    // --- State variables ---
    let currentFilePath = null;
    let availableColumns = [];

    // --- DOM Elements ---
    const dropzone = document.getElementById('dropzone');
    const fileInput = document.getElementById('file-input');
    const fileInfo = document.getElementById('file-info');
    const fileNameDisplay = document.getElementById('file-name');
    const removeFileBtn = document.getElementById('remove-file-btn');
    
    const configCard = document.getElementById('config-card');
    const detectedTargetText = document.getElementById('detected-target-text');

    const placeholderView = document.getElementById('placeholder-view');
    const loaderView = document.getElementById('loader-view');
    const resultsView = document.getElementById('results-view');

    // Tabs
    const tabButtons = document.querySelectorAll('.tab-btn');
    const tabPanes = document.querySelectorAll('.tab-pane');

    // Modal
    const imgModal = document.getElementById('img-modal');
    const modalImg = document.getElementById('modal-img');
    const modalClose = document.querySelector('.modal-close');
    const modalBackdrop = document.querySelector('.modal-backdrop');

    // --- Event Listeners ---

    // Dropzone logic
    dropzone.addEventListener('click', () => fileInput.click());

    ['dragenter', 'dragover'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.add('dragover');
        }, false);
    });

    ['dragleave', 'drop'].forEach(eventName => {
        dropzone.addEventListener(eventName, (e) => {
            e.preventDefault();
            dropzone.classList.remove('dragover');
        }, false);
    });

    dropzone.addEventListener('drop', (e) => {
        const files = e.dataTransfer.files;
        if (files.length) handleFileUpload(files[0]);
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length) handleFileUpload(e.target.files[0]);
    });

    removeFileBtn.addEventListener('click', () => {
        resetWorkspace();
    });

    // Tabs switching
    tabButtons.forEach(btn => {
        btn.addEventListener('click', () => {
            tabButtons.forEach(b => b.classList.remove('active'));
            tabPanes.forEach(p => p.classList.add('hidden'));

            btn.classList.add('active');
            const targetId = btn.getAttribute('data-target');
            document.getElementById(targetId).classList.remove('hidden');
        });
    });

    // Zoom Image click handlers
    document.querySelectorAll('.zoom-img').forEach(img => {
        img.addEventListener('click', () => {
            modalImg.src = img.src;
            imgModal.classList.remove('hidden');
        });
    });

    [modalClose, modalBackdrop].forEach(elem => {
        elem.addEventListener('click', () => imgModal.classList.add('hidden'));
    });

    // --- Core Functions ---

    async function handleFileUpload(file) {
        const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
        if (!['.csv', '.xlsx', '.xls'].includes(ext)) {
            alert('Please select a valid CSV or Excel dataset file.');
            return;
        }

        // Show uploading state
        fileNameDisplay.textContent = `Uploading: ${file.name}...`;
        dropzone.classList.add('hidden');
        fileInfo.classList.remove('hidden');
        document.getElementById('file-status').textContent = 'Uploading...';
        document.getElementById('file-status').className = 'status-badge warning';

        const formData = new FormData();
        formData.append('file', file);

        try {
            const response = await fetch('/api/upload', {
                method: 'POST',
                body: formData
            });

            const data = await response.json();

            if (!response.ok) throw new Error(data.error || 'Upload failure');

            currentFilePath = data.filepath;
            availableColumns = data.columns;

            // Populate file info display
            fileNameDisplay.textContent = file.name;
            document.getElementById('file-status').textContent = 'Optimizing...';
            document.getElementById('file-status').className = 'status-badge success';

            // Unlock AI status card
            configCard.classList.remove('disabled');
            detectedTargetText.textContent = 'Analyzing columns...';

            // Pre-populate raw dataset columns
            const allFeaturesCount = document.getElementById('all-features-count');
            const allFeaturesList = document.getElementById('all-features-list');
            if (allFeaturesCount && allFeaturesList && data.columns) {
                allFeaturesCount.textContent = data.columns.length;
                allFeaturesList.innerHTML = data.columns.map(f => `<span class="feature-pill" style="border-color: rgba(168, 85, 247, 0.4); background: rgba(168, 85, 247, 0.05);">${f}</span>`).join('');
            }

            // Instantly trigger optimization suite automatically without asking anything
            executeOptimizationSuite('auto');

        } catch (error) {
            alert(`Error uploading file: ${error.message}`);
            resetWorkspace();
        }
    }

    function resetWorkspace() {
        currentFilePath = null;
        availableColumns = [];
        fileInput.value = '';

        dropzone.classList.remove('hidden');
        fileInfo.classList.add('hidden');

        configCard.classList.add('disabled');
        detectedTargetText.textContent = 'Auto-detecting...';

        placeholderView.classList.remove('hidden');
        loaderView.classList.add('hidden');
        resultsView.classList.add('hidden');

        // Reset features list
        const allFeaturesCount = document.getElementById('all-features-count');
        const allFeaturesList = document.getElementById('all-features-list');
        if (allFeaturesCount) allFeaturesCount.textContent = '0';
        if (allFeaturesList) allFeaturesList.innerHTML = `<span style="color: var(--text-muted); font-style: italic; font-size: 0.9rem;">No dataset uploaded yet</span>`;
    }

    async function executeOptimizationSuite(targetColParam = 'auto') {
        // Lock file removal button during active pipeline convergence
        removeFileBtn.disabled = true;

        placeholderView.classList.add('hidden');
        resultsView.classList.add('hidden');
        loaderView.classList.remove('hidden');

        // Setup sequencing timeline animation to keep users wowed
        startSequencedProgressIndicators();

        try {
            const response = await fetch('/api/run-pipeline', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    filepath: currentFilePath,
                    target_col: targetColParam,
                    k: 20
                })
            });

            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Pipeline execution failed');

            // Force finish all sequential indicators instantly
            completeAllProgressSteps();

            // Proudly display inferred target prediction column
            if (data.target_col) {
                detectedTargetText.textContent = data.target_col;
                document.getElementById('file-status').textContent = 'Finished';
            }

            // Populate dashboard metrics
            renderResultsDashboard(data);

            // Transition main views smoothly
            setTimeout(() => {
                loaderView.classList.add('hidden');
                resultsView.classList.remove('hidden');
            }, 600);

        } catch (error) {
            alert(`Execution suite encountered an error: ${error.message}`);
            loaderView.classList.add('hidden');
            placeholderView.classList.remove('hidden');
            document.getElementById('file-status').textContent = 'Error';
            document.getElementById('file-status').className = 'status-badge danger';
        } finally {
            removeFileBtn.disabled = false;
        }
    }

    let progressTimers = [];
    function startSequencedProgressIndicators() {
        // Reset steps
        document.querySelectorAll('.progress-item').forEach(el => {
            el.className = 'progress-item pending';
        });

        // Sequence simulated real-time state steps
        const steps = [
            { id: 'step-preprocess', delay: 200, duration: 3000 },
            { id: 'step-filter', delay: 3200, duration: 2500 },
            { id: 'step-wrapper', delay: 5700, duration: 8000 },
            { id: 'step-embedded', delay: 13700, duration: 5000 },
            { id: 'step-models', delay: 18700, duration: 10000 },
            { id: 'step-plots', delay: 28700, duration: 4000 }
        ];

        progressTimers.forEach(t => clearTimeout(t));
        progressTimers = [];

        steps.forEach((step, idx) => {
            const timerStart = setTimeout(() => {
                // Complete previous step
                if (idx > 0) {
                    const prevStep = document.getElementById(steps[idx-1].id);
                    if (prevStep && !prevStep.classList.contains('completed')) {
                        prevStep.className = 'progress-item completed';
                    }
                }
                // Set current active
                const currStep = document.getElementById(step.id);
                if (currStep) currStep.className = 'progress-item active';
            }, step.delay);

            progressTimers.push(timerStart);
        });
    }

    function completeAllProgressSteps() {
        progressTimers.forEach(t => clearTimeout(t));
        document.querySelectorAll('.progress-item').forEach(el => {
            el.className = 'progress-item completed';
        });
    }

    function renderResultsDashboard(data) {
        const results = data.results;
        const features = data.selected_features;

        // 1. Compute top strategy based on highest Accuracy retention
        let bestMethod = 'Baseline';
        let maxAcc = 0;
        let baselineAcc = results['Baseline']?.['XGBoost']?.['Accuracy'] || 0;
        let baselineTime = results['Baseline']?.['XGBoost']?.['Time (s)'] || 1;

        Object.keys(results).forEach(method => {
            if (method !== 'Baseline') {
                const xgbAcc = results[method]?.['XGBoost']?.['Accuracy'] || 0;
                if (xgbAcc > maxAcc) {
                    maxAcc = xgbAcc;
                    bestMethod = method;
                }
            }
        });

        // Fallback / Format Metrics
        const bestAccFormatted = (maxAcc * 100).toFixed(2) + '%';
        const baseAccFormatted = (baselineAcc * 100).toFixed(2) + '%';
        const bestTime = results[bestMethod]?.['XGBoost']?.['Time (s)'] || 0.1;
        const speedup = (baselineTime / bestTime).toFixed(1) + 'x';

        document.getElementById('best-method-name').textContent = bestMethod.replace('_', ' ');
        document.getElementById('best-method-acc').textContent = bestAccFormatted;
        document.getElementById('baseline-acc').textContent = baseAccFormatted;
        document.getElementById('best-method-speedup').textContent = `~${speedup}`;

        // 2. Set dynamic chart vector references with timestamp cache buster
        const ts = Date.now();
        document.getElementById('plot-accuracy').src = `${data.plots.accuracy}?t=${ts}`;
        document.getElementById('plot-f1').src = `${data.plots.f1}?t=${ts}`;
        document.getElementById('plot-time').src = `${data.plots.time}?t=${ts}`;

        // Render All Features list (preprocessed)
        const allFeaturesCount = document.getElementById('all-features-count');
        const allFeaturesList = document.getElementById('all-features-list');
        if (allFeaturesCount && allFeaturesList && data.all_features) {
            allFeaturesCount.textContent = data.all_features.length;
            allFeaturesList.innerHTML = data.all_features.map(f => `<span class="feature-pill" style="border-color: rgba(59, 130, 246, 0.4); background: rgba(59, 130, 246, 0.05);">${f}</span>`).join('');
        }

        // 3. Render Feature Subsets List Table
        const featuresTbody = document.getElementById('features-tbody');
        featuresTbody.innerHTML = '';

        Object.entries(features).forEach(([methodName, featureList]) => {
            const tr = document.createElement('tr');
            
            // Generate distinct stylized attribute pills
            const pillsHtml = featureList.map(f => `<span class="feature-pill">${f}</span>`).join('');
            
            tr.innerHTML = `
                <td><strong>${methodName.replace('_', ' ')}</strong></td>
                <td><span class="step-number">${featureList.length}</span></td>
                <td><div style="max-height: 180px; overflow-y: auto;">${pillsHtml}</div></td>
            `;
            featuresTbody.appendChild(tr);
        });

        // 4. Render Raw Evaluation Matrix Table
        const matrixTbody = document.getElementById('matrix-tbody');
        matrixTbody.innerHTML = '';

        Object.entries(results).forEach(([methodName, models]) => {
            Object.entries(models).forEach(([modelName, metrics], mIdx) => {
                const tr = document.createElement('tr');
                
                // Show method row-span effect using simple text grouping
                const methodDisplay = mIdx === 0 ? `<strong>${methodName.replace('_', ' ')}</strong>` : `<span style="color: var(--text-muted)">↳</span>`;
                
                tr.innerHTML = `
                    <td>${methodDisplay}</td>
                    <td><span class="status-badge ${modelName === 'XGBoost' ? 'success' : ''}" style="font-size:0.8rem">${modelName}</span></td>
                    <td><strong>${(metrics['Accuracy'] * 100).toFixed(2)}%</strong></td>
                    <td>${metrics['F1 Score'].toFixed(4)}</td>
                    <td>${metrics['Time (s)'].toFixed(3)}s</td>
                `;
                matrixTbody.appendChild(tr);
            });
        });
    }
});
