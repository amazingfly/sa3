// Shared Logic
console.log("SA3 Script Loaded: v7");

async function api(path, options) {
    console.log(`API Call: ${path}`, options?.method || 'GET');
    try {
        const res = await fetch(path, options);
        const data = await res.json();
        if (!res.ok) throw new Error(data.error || res.statusText);
        return data;
    } catch (err) {
        console.error(`API Error (${path}):`, err);
        throw err;
    }
}

// Common Tab Logic
function initTabs() {
    console.log("Initializing Tabs...");
    document.querySelectorAll('.tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const targetId = tab.dataset.target;
            console.log(`Tab Clicked: ${targetId}`);
            if (!targetId) return;

            const parent = tab.parentElement;
            parent.querySelectorAll('.tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const view = document.getElementById(targetId);
            if (view && view.classList.contains('view')) {
                const container = view.parentElement;
                container.querySelectorAll(':scope > .view').forEach(v => v.classList.remove('active'));
                view.classList.add('active');
                
                if (targetId === 'imageLibrary') loadGallery();
                if (targetId === 'audioLibrary') loadLibrary();
            } else if (view && view.classList.contains('sd-view')) {
                const container = view.parentElement;
                container.querySelectorAll('.sd-view').forEach(v => v.style.display = 'none');
                view.style.display = 'block';
            } else {
                const standard = document.getElementById('standardInputs');
                const config = document.getElementById('configInput');
                if (standard && config) {
                    const isConfig = targetId === 'configInput';
                    standard.style.display = isConfig ? 'none' : 'block';
                    config.style.display = isConfig ? 'block' : 'none';
                }
            }
        });
    });
}

// --- Generator Logic (Index) ---
function initGenerator() {
    const form = document.getElementById('generateForm');
    const sdManualBtn = document.getElementById('sdManualBtn');
    const sdFullBtn = document.getElementById('sdFullBtn');
    
    if (!form && !sdManualBtn && !sdFullBtn) return;
    console.log("Initializing Generator...");

    const jobsEl = document.getElementById('jobs');
    const sdJobsEl = document.getElementById('sdJobs');
    const statusEl = document.getElementById('serverStatus');
    const generateBtn = document.getElementById('generateBtn');
    const generateQueueBtn = document.getElementById('generateQueueBtn');
    const clearQueueBtn = document.getElementById('clearQueueBtn');
    const mediumQueueEl = document.getElementById('mediumQueue');
    const modelNameEl = document.getElementById('modelName');
    const durationEl = document.getElementById('duration');
    const continuationModeEl = document.getElementById('continuationMode');
    const modelHelpEl = document.getElementById('modelHelp');
    const rawConfigEl = document.getElementById('rawConfig');
    const loadDefaultBtn = document.getElementById('loadDefaultBtn');

    let knownJobs = [];
    let knownSdJobs = [];
    let smallContinuation = continuationModeEl ? continuationModeEl.checked : true;
    const queueStorageKey = 'sa3-medium-colab-queue-v1';
    let mediumQueue = [];

    try {
        const storedQueue = JSON.parse(localStorage.getItem(queueStorageKey) || '[]');
        if (Array.isArray(storedQueue)) mediumQueue = storedQueue.slice(0, 20);
    } catch (err) {
        localStorage.removeItem(queueStorageKey);
    }

    function payloadModel(payload) {
        return payload.raw_config?.model?.name || payload.model || '';
    }

    function queueSummary(payload) {
        const config = payload.raw_config || payload;
        const generation = config.generation || {};
        const track = Array.isArray(config.tracks) ? config.tracks[0] || {} : {};
        return {
            name: config.track_id || track.id || generation.output_name || 'medium-track',
            duration: generation.duration_seconds || config.duration_seconds || 380,
            prompt: generation.prompt || config.prompt || track.prompt || ''
        };
    }

    function saveQueue() {
        localStorage.setItem(queueStorageKey, JSON.stringify(mediumQueue));
    }

    function renderQueue() {
        if (!mediumQueueEl) return;
        mediumQueueEl.innerHTML = '';
        if (!mediumQueue.length) {
            const empty = document.createElement('div');
            empty.className = 'muted';
            empty.textContent = 'Queue is empty.';
            mediumQueueEl.append(empty);
        } else {
            mediumQueue.forEach((payload, index) => {
                const summary = queueSummary(payload);
                const item = document.createElement('div');
                item.className = 'queue-item';
                const text = document.createElement('div');
                const title = document.createElement('div');
                title.className = 'queue-item-title';
                title.textContent = `${index + 1}. ${summary.name}`;
                const meta = document.createElement('div');
                meta.className = 'queue-item-meta';
                meta.textContent = `${summary.duration}s · ${summary.prompt.slice(0, 140)}`;
                text.append(title, meta);
                const remove = document.createElement('button');
                remove.type = 'button';
                remove.className = 'secondary';
                remove.textContent = 'Remove';
                remove.addEventListener('click', () => {
                    mediumQueue.splice(index, 1);
                    saveQueue();
                    renderQueue();
                });
                item.append(text, remove);
                mediumQueueEl.append(item);
            });
        }
        if (generateQueueBtn) generateQueueBtn.disabled = mediumQueue.length === 0;
        if (clearQueueBtn) clearQueueBtn.disabled = mediumQueue.length === 0;
    }

    function buildAudioPayload() {
        const activeTab = document.querySelector('#audioView .tab.active');
        const currentMode = activeTab ? activeTab.dataset.target : 'standardInputs';
        if (currentMode === 'configInput') {
            return { raw_config: JSON.parse(rawConfigEl.value) };
        }
        return {
            prompt: document.getElementById('prompt').value,
            model: modelNameEl.value,
            continuation: continuationModeEl.checked,
            track_id: document.getElementById('trackId').value,
            duration_seconds: Number(durationEl.value),
            steps: Number(document.getElementById('steps').value),
            cfg_scale: Number(document.getElementById('cfgScale').value),
            init_noise_level: Number(document.getElementById('initNoise').value),
            seed_start: document.getElementById('seed').value ? Number(document.getElementById('seed').value) : null
        };
    }

    function applyModelDefaults() {
        if (!modelNameEl || !durationEl || !continuationModeEl) return;
        const medium = modelNameEl.value === 'medium';
        if (medium) {
            smallContinuation = continuationModeEl.checked;
            continuationModeEl.checked = false;
            continuationModeEl.disabled = true;
            durationEl.max = '380';
            durationEl.value = '380';
            const stepsEl = document.getElementById('steps');
            if (stepsEl && Number(stepsEl.value) === 30) stepsEl.value = '8';
            if (modelHelpEl) {
                modelHelpEl.textContent = 'Medium tracks are queued and generated in one Colab session with one model load (10-380 seconds; default 380).';
            }
            if (generateBtn) generateBtn.textContent = 'Queue';
        } else {
            continuationModeEl.disabled = false;
            continuationModeEl.checked = smallContinuation;
            durationEl.max = '120';
            durationEl.value = '120';
            const stepsEl = document.getElementById('steps');
            if (stepsEl && Number(stepsEl.value) === 8) stepsEl.value = '30';
            if (modelHelpEl) {
                modelHelpEl.textContent = 'Small Music uses the existing local continuation workflow.';
            }
            if (generateBtn) generateBtn.textContent = 'Generate';
        }
    }

    if (modelNameEl) {
        modelNameEl.addEventListener('change', applyModelDefaults);
        applyModelDefaults();
    }

    if (loadDefaultBtn) {
        loadDefaultBtn.addEventListener('click', async () => {
            try {
                const data = await api('/api/defaults');
                rawConfigEl.value = JSON.stringify(data, null, 2);
            } catch (err) {
                alert("Failed to load defaults: " + err.message);
            }
        });
    }

    // ... (rest of helper functions same as before) ...
    function ensureJobElement(container, job) {
        if (!container) return null;
        let el = container.querySelector(`[data-job-id="${job.id}"]`);
        if (el) return el;
        el = document.createElement('article');
        el.className = 'job';
        el.dataset.jobId = job.id;
        el.innerHTML = `<div class="job-head"><div><strong class="job-title"></strong><div class="muted job-created"></div></div><span class="badge job-status"></span></div><div class="job-audio"></div><div class="muted job-error"></div><pre class="job-log"></pre>`;
        container.prepend(el);
        return el;
    }

    function updateJobElement(el, job) {
        if (!el) return;
        const badgeClass = job.status === 'done' ? 'done' : job.status === 'failed' ? 'failed' : 'running';
        el.querySelector('.job-title').textContent = job.track_id || (job.mode ? `SD ${job.mode.toUpperCase()}` : job.id);
        el.querySelector('.job-created').textContent = job.created_at || '';
        const badge = el.querySelector('.job-status');
        badge.className = `badge job-status ${badgeClass}`;
        badge.textContent = job.status;
        el.querySelector('.job-error').textContent = job.error || '';
        
        const log = el.querySelector('.job-log');
        if (log && job.log_tail) {
            const wasAtBottom = Math.abs(log.scrollHeight - log.clientHeight - log.scrollTop) < 20;
            log.textContent = job.log_tail;
            if (wasAtBottom) log.scrollTop = log.scrollHeight;
        }
        
        const outputs = job.outputs?.length ? job.outputs : (job.audio_url ? [job] : []);
        if (outputs.length) {
            const audioBox = el.querySelector('.job-audio');
            const renderKey = outputs.map(output => output.audio_url).join('|');
            if (audioBox.dataset.renderKey !== renderKey) {
                audioBox.innerHTML = '';
                audioBox.dataset.renderKey = renderKey;
                outputs.forEach(output => {
                    const outputBox = document.createElement('div');
                    outputBox.className = 'queue-output';
                    if (output.track_id) {
                        const title = document.createElement('strong');
                        title.textContent = output.track_id;
                        outputBox.append(title);
                    }
                const audio = document.createElement('audio');
                audio.controls = true;
                    audio.src = output.audio_url;
                const link = document.createElement('a');
                    link.href = output.audio_url;
                    link.download = output.download_name || '';
                    link.textContent = `Download ${(output.audio_format || 'audio').toUpperCase()}`;
                    outputBox.append(audio, link);
                    audioBox.append(outputBox);
                });
            }
        }
    }

    async function refresh() {
        console.log("Refreshing jobs...");
        // Audio Jobs
        if (jobsEl) {
            try {
                const data = await api('/api/jobs');
                knownJobs = data.jobs;
                knownJobs.slice().reverse().forEach(j => updateJobElement(ensureJobElement(jobsEl, j), j));
            } catch (err) { console.error("Refresh Audio Jobs failed:", err); }
        }
        
        // SD Jobs
        if (sdJobsEl) {
            try {
                const sdData = await api('/api/sd15/jobs');
                knownSdJobs = sdData.jobs;
                knownSdJobs.slice().reverse().forEach(j => updateJobElement(ensureJobElement(sdJobsEl, j), j));
                
                if (statusEl) {
                    statusEl.textContent = (knownJobs.some(j => j.status === 'running') || sdData.running) ? 'Generating' : 'Ready';
                }
                
                if (knownSdJobs.length > 0 && knownSdJobs[0].status === 'done') {
                    showLatestSdResult();
                }
            } catch (err) { console.error("Refresh SD Jobs failed:", err); }
        }
    }

    async function showLatestSdResult() {
        const viewer = document.getElementById('sdOutputViewer');
        const latest = document.getElementById('sdLatestResult');
        const dlBtn = document.getElementById('sdDownloadBtn');
        if (!viewer || !latest) return;
        
        try {
            const data = await api('/api/images');
            if (data.images.length > 0) {
                const img = data.images[0]; // Most recent
                latest.innerHTML = `<img src="${img.url}" style="max-width:100%; border-radius:8px;">`;
                dlBtn.href = img.url;
                dlBtn.download = img.filename;
                viewer.style.display = 'block';
                
                document.getElementById('sdLikeBtn').onclick = () => handleImageActionExplicit(img, 'like');
                document.getElementById('sdTrashBtn').onclick = () => {
                    handleImageActionExplicit(img, 'delete');
                    viewer.style.display = 'none';
                };
            }
        } catch (err) {}
    }

    async function handleImageActionExplicit(img, action) {
        try {
            await api('/api/images/action', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({action, filename: img.filename})
            });
            if (action === 'like') alert("Liked!");
        } catch (err) {
            alert("Action failed: " + err.message);
        }
    }

    if (form) {
        form.addEventListener('submit', async (event) => {
            event.preventDefault();
            console.log("Audio Gen Form Submitted");
            try {
                const payload = buildAudioPayload();
                if (payloadModel(payload) === 'medium') {
                    if (mediumQueue.length >= 20) {
                        throw new Error('Medium queue is limited to 20 tracks.');
                    }
                    mediumQueue.push(payload);
                    saveQueue();
                    renderQueue();
                } else {
                    await api('/api/generate', {
                        method: 'POST',
                        headers: {'Content-Type': 'application/json'},
                        body: JSON.stringify(payload)
                    });
                    refresh();
                }
            } catch (err) { alert("Audio Gen Failed: " + err.message); }
        });
    }

    if (generateQueueBtn) {
        generateQueueBtn.addEventListener('click', async () => {
            if (!mediumQueue.length) return;
            try {
                await api('/api/generate-queue', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({queue_version: 1, items: mediumQueue})
                });
                mediumQueue = [];
                saveQueue();
                renderQueue();
                refresh();
            } catch (err) {
                alert("Queue generation failed: " + err.message);
            }
        });
    }

    if (clearQueueBtn) {
        clearQueueBtn.addEventListener('click', () => {
            mediumQueue = [];
            saveQueue();
            renderQueue();
        });
    }

    renderQueue();

    if (document.getElementById('refreshBtn')) {
        document.getElementById('refreshBtn').addEventListener('click', refresh);
    }
    
    // --- SD Image Gen Buttons ---
    if (sdManualBtn) {
        console.log("Attaching SD Manual Listener");
        sdManualBtn.addEventListener('click', async () => {
            console.log("SD Manual Clicked");
            const prompt = document.getElementById('sdPrompt').value;
            try {
                await api('/api/sd15/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({prompt, mode: 'manual'})
                });
                refresh();
            } catch (err) {
                alert("Failed: " + err.message);
            }
        });
    }

    if (sdFullBtn) {
        console.log("Attaching SD Full Listener");
        sdFullBtn.addEventListener('click', async () => {
            console.log("SD Full Clicked");
            const payload = {
                mode: 'full',
                prompt_count: Number(document.getElementById('sdCount').value),
                steps: Number(document.getElementById('sdSteps').value),
                resize: document.getElementById('sdResize').value,
                gemma_prompt: document.getElementById('gemmaPrompt').value
            };
            try {
                await api('/api/sd15/generate', {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify(payload)
                });
                refresh();
            } catch (err) {
                alert("Failed: " + err.message);
            }
        });
    }

    setInterval(refresh, 5000);
    refresh();
}

// --- Library Logic ---
let images = [];
let currentIndex = 0;
let slideshowInterval = null;
let touchStartX = 0;

async function loadGallery() {
    const galleryContainer = document.getElementById('galleryContainer');
    if (!galleryContainer) return;
    console.log("Loading Gallery...");
    try {
        const data = await api('/api/images');
        images = data.images;
        renderGallery();
    } catch (err) {
        console.error("Failed to load gallery:", err);
    }
}

function renderGallery() {
    const galleryContainer = document.getElementById('galleryContainer');
    if (images.length === 0) {
        galleryContainer.innerHTML = '<div class="muted">No images found.</div>';
        return;
    }
    let strip = document.getElementById('galleryStrip');
    if (!strip) {
        strip = document.createElement('div');
        strip.id = 'galleryStrip';
        strip.className = 'gallery-strip';
        galleryContainer.appendChild(strip);
    }
    strip.innerHTML = '';
    images.forEach((img, i) => {
        const slide = document.createElement('div');
        slide.className = 'img-slide';
        slide.innerHTML = `<img src="${img.url}" loading="${Math.abs(i-currentIndex) < 3 ? 'eager' : 'lazy'}">`;
        strip.appendChild(slide);
    });
    updateSlide();
}

function updateSlide() {
    const strip = document.getElementById('galleryStrip');
    if (strip) {
        strip.style.transform = `translateX(-${currentIndex * 100}%)`;
    }
    const dlBtn = document.getElementById('galleryDownloadBtn');
    if (dlBtn && images[currentIndex]) {
        dlBtn.href = images[currentIndex].url;
        dlBtn.download = images[currentIndex].filename;
    }
}

function nextSlide() {
    if (images.length === 0) return;
    currentIndex = (currentIndex + 1) % images.length;
    updateSlide();
}

function prevSlide() {
    if (images.length === 0) return;
    currentIndex = (currentIndex - 1 + images.length) % images.length;
    updateSlide();
}

async function loadLibrary() {
    const tracksEl = document.getElementById('tracks');
    if (!tracksEl) return;
    console.log("Loading Library...");
    const pageStatus = document.getElementById('pageStatus');
    pageStatus.textContent = 'Loading library';
    try {
        const data = await api('/api/library');
        const tracks = data.tracks;
        tracksEl.innerHTML = '';
        for (const track of tracks) {
            const el = document.createElement('article');
            el.className = 'track';
            el.innerHTML = `
                <div class="track-head">
                    <input type="checkbox" class="track-check" checked value="${track.filename}">
                    <div>
                        <div class="track-title">${track.title}</div>
                        <div class="track-meta">${(track.format || 'audio').toUpperCase()} · ${Math.round(track.size/1024)} KB · ${track.modified}</div>
                    </div>
                    <a class="download" download href="${track.url}">Download</a>
                </div>
                <audio controls preload="metadata" src="${track.url}"></audio>
            `;
            tracksEl.append(el);
        }
        pageStatus.textContent = `${tracks.length} tracks`;
    } catch (err) {
        pageStatus.textContent = "Error loading library";
    }
}

function initLibrary() {
    const galleryContainer = document.getElementById('galleryContainer');
    if (galleryContainer) {
        console.log("Initializing Library UI...");
        document.getElementById('tapPrev').addEventListener('click', (e) => { e.stopPropagation(); prevSlide(); });
        document.getElementById('tapNext').addEventListener('click', (e) => { e.stopPropagation(); nextSlide(); });
        document.getElementById('tapFullscreen').addEventListener('click', (e) => {
            e.stopPropagation();
            document.body.classList.toggle('gallery-fullscreen');
        });

        galleryContainer.addEventListener('touchstart', e => touchStartX = e.changedTouches[0].screenX);
        galleryContainer.addEventListener('touchend', e => {
            const diff = e.changedTouches[0].screenX - touchStartX;
            if (diff > 50) prevSlide();
            else if (diff < -50) nextSlide();
        });

        document.getElementById('likeBtn').addEventListener('click', () => handleImageActionExplicit(images[currentIndex], 'like'));
        document.getElementById('trashBtn').addEventListener('click', () => {
            if (confirm("Delete this image?")) {
                handleImageActionExplicit(images[currentIndex], 'delete');
                images.splice(currentIndex, 1);
                renderGallery();
            }
        });
        document.getElementById('slideshowBtn').addEventListener('click', () => {
            if (slideshowInterval) {
                clearInterval(slideshowInterval);
                slideshowInterval = null;
                document.getElementById('slideshowBtn').textContent = '▶️ Slideshow';
            } else {
                slideshowInterval = setInterval(nextSlide, 1000);
                document.getElementById('slideshowBtn').textContent = '⏹️ Stop Slideshow';
            }
        });
    }

    const deviceSelect = document.getElementById('deviceSelect');
    if (deviceSelect) {
        document.getElementById('refreshLibraryBtn').addEventListener('click', loadLibrary);
        document.getElementById('refreshDevicesBtn').addEventListener('click', async () => {
            try {
                const data = await api('/api/cast/devices');
                deviceSelect.innerHTML = '';
                ['speaker group alpha', ...data.devices.map(d => d.name)].forEach(name => {
                    const opt = document.createElement('option');
                    opt.value = name;
                    opt.textContent = name;
                    deviceSelect.append(opt);
                });
            } catch (err) {}
        });
    }
}

// --- Initialization ---
document.addEventListener('DOMContentLoaded', () => {
    console.log("DOM Content Loaded");
    initTabs();
    initGenerator();
    initLibrary();
    
    document.querySelectorAll('.tab.active').forEach(tab => {
        const targetId = tab.dataset.target;
        if (targetId === 'imageLibrary') loadGallery();
        if (targetId === 'audioLibrary') loadLibrary();
    });
});
