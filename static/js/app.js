document.addEventListener('DOMContentLoaded', () => {
    
    // --- 1. Init Leaflet Map ---
    let map;
    let marker;

    function initMap(lat, lon, zoom = 3) {
        if (!map) {
            document.getElementById('map-placeholder').classList.add('hidden-el');
            map = L.map('leaflet-map').setView([lat, lon], zoom);
            L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
                attribution: '&copy; <a href="https://carto.com/">CARTO</a>',
                subdomains: 'abcd',
                maxZoom: 20
            }).addTo(map);
            marker = L.marker([lat, lon]).addTo(map);
        } else {
            map.flyTo([lat, lon], zoom);
            marker.setLatLng([lat, lon]);
        }
    }

    // --- 2. Init Chart.js ---
    const ctx = document.getElementById('confidenceChart').getContext('2d');
    let confChart = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: ['Metadata (Forensics)', 'Visual (ELA/AI)', 'Geospace (Shadows)', 'Weather Match', 'Entities'],
            datasets: [{
                label: 'Confidence Level',
                data: [0, 0, 0, 0, 0],
                backgroundColor: 'rgba(59, 130, 246, 0.2)', // terminal-accent
                borderColor: 'rgba(59, 130, 246, 1)',
                pointBackgroundColor: 'rgba(0, 255, 65, 1)', // terminal-green
                pointBorderColor: '#fff',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: 'rgba(0, 255, 65, 1)',
                borderWidth: 2,
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: { color: '#888', font: { family: 'Courier New', size: 10 } },
                    ticks: { display: false, max: 1, min: 0 }
                }
            },
            plugins: {
                legend: { display: false }
            }
        }
    });

    // --- 3. DOM Elements ---
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');
    const errText = document.getElementById('upload-error');
    const errMsg = document.getElementById('upload-error-text');
    const progContainer = document.getElementById('progress-container');
    const progBar = document.getElementById('progress-bar');
    const curStatus = document.getElementById('current-task-status');
    const curPercent = document.getElementById('current-task-percent');
    
    // UI Panels
    const forensList = document.getElementById('forens-list');
    const entityList = document.getElementById('entity-list');
    const geoTele = document.getElementById('geo-telemetry');
    const synthesisV = document.getElementById('synthesis-verdict');
    const vText = document.getElementById('verdict-text');
    const vNotes = document.getElementById('verdict-notes');
    const cPlaceholder = document.getElementById('chart-placeholder');
    const cCanvas = document.getElementById('confidenceChart');
    const actionContainer = document.getElementById('action-container');
    const downloadBtn = document.getElementById('download-report-btn');

    // Módní funkce k "typování" textu pro terminal effect
    function typeText(element, htmlContent) {
        element.innerHTML = htmlContent;
        element.classList.add('glow-text');
        setTimeout(() => element.classList.remove('glow-text'), 500);
    }

    // --- 4. File Upload & Drag-Drop ---
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, preventDefaults, false);
    });
    function preventDefaults(e) { e.preventDefault(); e.stopPropagation(); }

    ['dragenter', 'dragover'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.add('border-terminal-green', 'bg-terminal-border/20'), false);
    });
    ['dragleave', 'drop'].forEach(eventName => {
        dropZone.addEventListener(eventName, () => dropZone.classList.remove('border-terminal-green', 'bg-terminal-border/20'), false);
    });

    dropZone.addEventListener('drop', (e) => {
        let dt = e.dataTransfer;
        let files = dt.files;
        if(files.length) handleFile(files[0]);
    });

    fileInput.addEventListener('change', function() {
        if(this.files.length) handleFile(this.files[0]);
    });

    function handleFile(file) {
        errText.classList.add('hidden-el');
        
        // Validate type
        const validTypes = ['image/jpeg', 'image/jpg', 'image/png', 'image/tiff'];
        if (!validTypes.includes(file.type)) {
            errMsg.textContent = 'Invalid file format. Use JPG, PNG, or TIFF.';
            errText.classList.remove('hidden-el');
            return;
        }

        uploadFile(file);
    }

    function uploadFile(file) {
        progContainer.classList.remove('hidden-el');
        actionContainer.classList.add('hidden-el');
        curStatus.textContent = 'UPLOADING DATA STREAM...';
        progBar.style.width = '5%';
        curPercent.textContent = '5%';
        
        // Reset panels
        resetUI();

        let formData = new FormData();
        formData.append('file', file);

        fetch('/analyze', {
            method: 'POST',
            body: formData
        })
        .then(response => response.json())
        .then(data => {
            if (data.error) {
                errMsg.textContent = data.error;
                errText.classList.remove('hidden-el');
                progContainer.classList.add('hidden-el');
            } else if (data.task_id) {
                connectWebSocket(data.task_id);
            }
        })
        .catch(error => {
            errMsg.textContent = 'Upload failed: ' + error.message;
            errText.classList.remove('hidden-el');
            progContainer.classList.add('hidden-el');
        });
    }

    function resetUI() {
        forensList.innerHTML = '<li>> Awaiting stream...</li>';
        entityList.innerHTML = '<div class="text-terminal-dim font-mono">> Awaiting object detection...</div>';
        geoTele.innerHTML = '<li>LAT: --</li><li>LON: --</li><li>GEO_MATCH: <span class="text-terminal-dim">Pending</span></li>';
        synthesisV.classList.add('hidden-el');
        cPlaceholder.classList.remove('hidden-el');
        cCanvas.classList.add('hidden-el');
        confChart.data.datasets[0].data = [0,0,0,0,0];
        confChart.update();
        if(map) map.flyTo([0,0], 1);
        document.getElementById('ela-stats').classList.add('hidden-el');
        document.getElementById('elaCanvas').classList.add('hidden-el');
        document.getElementById('ela-placeholder').classList.remove('hidden-el');
    }

    // --- 5. WebSocket Connection ---
    let ws = null;
    const connStatus = document.getElementById('connection-status');

    function connectWebSocket(taskId) {
        if (ws) { ws.close(); }
        
        const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        ws = new WebSocket(`${protocol}//${window.location.host}/ws/progress/${taskId}`);

        ws.onopen = () => {
            connStatus.textContent = 'WS: CONNECTED';
            connStatus.classList.replace('text-terminal-dim', 'text-terminal-green');
            connStatus.classList.add('border-terminal-green');
        };

        ws.onmessage = (event) => {
            const msg = JSON.parse(event.data);
            
            // Update Progress Bar
            curStatus.textContent = msg.status;
            progBar.style.width = `${msg.progress}%`;
            curPercent.textContent = `${msg.progress}%`;

            // Update Modules Based on Data Available
            if (msg.data) {
                updateDashboard(msg.data);
            }

            if (msg.progress >= 100 || msg.progress < 0) {
                ws.close();
            }
        };

        ws.onclose = () => {
            connStatus.textContent = 'WS: DISCONNECTED';
            connStatus.classList.replace('text-terminal-green', 'text-terminal-dim');
            connStatus.classList.remove('border-terminal-green');
        };
    }

    // --- 6. Update Dashboard Logic ---
    function updateDashboard(data) {
        // Forensics
        if (data.forensic) {
            let metaHtml = '';
            if(data.forensic.metadata) {
                metaHtml += `<li><span class="text-terminal-text">TIME:</span> ${data.forensic.metadata.datetime_original || 'UNKNOWN'}</li>`;
                metaHtml += `<li><span class="text-terminal-text">CAM:</span> ${data.forensic.metadata.make || '-'} / ${data.forensic.metadata.model || '-'}</li>`;
            }
            if(data.forensic.ai_generation_check && data.forensic.ai_generation_check.verdict) {
                const aiCol = ['authentic', 'likely_authentic'].includes(data.forensic.ai_generation_check.verdict) ? 'text-terminal-green' : 'text-terminal-danger';
                metaHtml += `<li><span class="text-terminal-text">AI CHECK:</span> <span class="${aiCol}">${data.forensic.ai_generation_check.verdict.toUpperCase()}</span> (${(data.forensic.ai_generation_check.confidence*100).toFixed(1)}%)</li>`;
            }
            if(metaHtml) typeText(forensList, metaHtml);
            
            if(data.forensic.ela && data.forensic.ela.ela_image_path) {
                let webElaPath = data.forensic.ela.ela_image_path.replace(/\\/g, '/');
                let dataIdx = webElaPath.indexOf('data/ela_output');
                if(dataIdx !== -1) {
                    webElaPath = '/' + webElaPath.substring(dataIdx);
                } else if(!webElaPath.startsWith('/')) {
                    webElaPath = '/' + webElaPath;
                }
                
                const cEla = document.getElementById('elaCanvas');
                const pEla = document.getElementById('ela-placeholder');
                const sEla = document.getElementById('ela-stats');
                
                cEla.classList.remove('hidden-el');
                pEla.classList.add('hidden-el');
                sEla.classList.remove('hidden-el');
                
                document.getElementById('ela-max').textContent = data.forensic.ela.max_error || 0;
                document.getElementById('ela-sus').textContent = data.forensic.ela.suspicious_regions ? data.forensic.ela.suspicious_regions.length : 0;
                
                const ctxEla = cEla.getContext('2d');
                const img = new Image();
                img.onload = function() {
                    cEla.width = img.width;
                    cEla.height = img.height;
                    ctxEla.drawImage(img, 0, 0);
                };
                img.src = webElaPath;
            }
            
            // Updating Chart Data
            confChart.data.datasets[0].data[0] = 0.9; // Mock confidence
            confChart.data.datasets[0].data[1] = data.forensic.ai_generation_check ? data.forensic.ai_generation_check.confidence : 0.5;
            updateChartVis();
        }

        // Environment
        if (data.environment) {
            if (data.environment.shadow_analysis && data.environment.shadow_analysis.candidate_locations.length > 0) {
                const loc = data.environment.shadow_analysis.candidate_locations[0];
                initMap(loc.latitude, loc.longitude, 8);
                
                let teleHtml = `<li>LAT: <span class="text-terminal-text">${loc.latitude.toFixed(4)}</span></li>`;
                teleHtml += `<li>LON: <span class="text-terminal-text">${loc.longitude.toFixed(4)}</span></li>`;
                
                if(data.environment.sky_weather_match) {
                    const wMatch = data.environment.sky_weather_match.match;
                    const wCol = wMatch ? 'text-terminal-green' : 'text-terminal-danger';
                    const wTxt = wMatch ? 'CONFIRMED' : 'REJECTED';
                    teleHtml += `<li>GEO_W_MATCH: <span class="${wCol}">${wTxt}</span></li>`;
                    confChart.data.datasets[0].data[3] = data.environment.sky_weather_match.confidence;
                }
                typeText(geoTele, teleHtml);
                confChart.data.datasets[0].data[2] = 0.85; // shadow conf
                updateChartVis();
            }
        }

        // Entities
        if (data.entities) {
            let entHtml = '';
            if(data.entities.persons && data.entities.persons.length > 0) {
                data.entities.persons.forEach(p => {
                    entHtml += `<div class="flex justify-between items-center border border-terminal-border p-2 rounded">
                        <span>> ${p.description}</span>
                        <span class="text-xs text-terminal-green">${(p.confidence*100).toFixed(0)}%</span>
                    </div>`;
                });
            }
            if(data.entities.objects && data.entities.objects.length > 0) {
                entHtml += `<div class="text-xs text-terminal-dim mt-2">OBJECTS: ${data.entities.objects.join(', ')}</div>`;
            }
            if(entHtml) typeText(entityList, entHtml);
            confChart.data.datasets[0].data[4] = 0.92;
            updateChartVis();
        }

        // Synthesis (Final)
        if (data.synthesis) {
            synthesisV.classList.remove('hidden-el');
            vText.textContent = data.synthesis.is_authentic ? 'AUTHENTIC' : 'MANIPULATED';
            vText.className = data.synthesis.is_authentic ? 'text-xl font-bold text-terminal-green' : 'text-xl font-bold text-terminal-danger glow-text';
            
            if (data.synthesis.reliability_notes) {
                vNotes.innerHTML = data.synthesis.reliability_notes.map(n => `<li>${n}</li>`).join('');
            }
        }

        // Report Action
        if (data.report_url) {
            actionContainer.classList.remove('hidden-el');
            downloadBtn.href = data.report_url;
        }
    }

    function updateChartVis() {
        if(cCanvas.classList.contains('hidden-el')) {
            cCanvas.classList.remove('hidden-el');
            cPlaceholder.classList.add('hidden-el');
            confChart.resize();
        }
        confChart.update();
    }
});
