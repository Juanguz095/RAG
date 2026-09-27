/* ===== GLOBALS ===== */
let documents = [];
let currentView = 'dashboard';
let pdfUrls = [];
let progressPollTimer = null;
let currentDocId = null;

/* ===== ZOOM ===== */
let pdfZoom = 1.0;
const ZOOM_MIN = 0.25;
const ZOOM_MAX = 3.0;
const ZOOM_STEP = 0.25;

/* ===== VIEWER SEARCH ===== */
let searchMatches = [];
let currentMatchIndex = -1;

/* ===== UTILITIES ===== */
const $ = (id) => document.getElementById(id);
const esc = (v) => String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;' }[c]));
const cleanText = (v) => String(v ?? '').replace(/\s+/g, ' ').trim();

async function apiFetch(url, opts = {}) {
  const resp = await Auth.apiFetch(url, opts);
  const text = await resp.text();
  try { return text ? JSON.parse(text) : {}; }
  catch { throw Error(resp.ok ? 'Respuesta inválida del servidor' : `Servidor (${resp.status}): ${text.slice(0, 200)}`); }
}

function showToast(message, type = 'info') {
  const container = $('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast toast-${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => { toast.style.opacity = '0'; setTimeout(() => toast.remove(), 300); }, 3500);
}

/* ===== USER INFO ===== */
function loadUserInfo() {
  const user = Auth.getUser();
  if (!user) return;
  $('userName').textContent = user.full_name || user.username;
  $('userRole').textContent = user.role;
  $('userAvatar').textContent = Auth.getInitials(user.full_name || user.username);
}

/* ===== NAVIGATION ===== */
function switchView(view) {
  currentView = view;
  document.querySelectorAll('.nav-item').forEach(el => {
    el.classList.toggle('active', el.dataset.view === view);
  });
  document.querySelectorAll('[id^="view-"]').forEach(el => {
    el.hidden = !el.id.endsWith(`-${view}`);
  });
  const titles = { dashboard: 'Dashboard', documents: 'Documentos', query: 'Consultar' };
  $('viewTitle').textContent = titles[view] || 'Dashboard';
  if (view === 'dashboard') { loadDocuments(); }
  if (view === 'query') { checkLlmStatus(); }
}

async function checkLlmStatus() {
  try {
    const d = await apiFetch('/api/v1/health');
    const warning = $('llmWarning');
    if (warning) warning.style.display = d.llm_enabled === false ? 'block' : 'none';
  } catch { /* ignore */ }
}

/* ===== PROGRESS TRACKING ===== */
function showProgress(doc) {
  const container = $('progressContainer');
  if (!doc || doc.status === 'completed' || doc.status === 'failed') {
    container.style.display = 'none';
    return;
  }
  container.style.display = 'block';
  updateProgressUI(doc);
}

function updateProgressUI(doc) {
  const pct = doc.progress_pct || 0;
  const phase = doc.progress_phase || '';
  const message = doc.progress_message || '';

  $('progressFill').style.width = `${pct}%`;
  $('progressPct').textContent = `${Math.round(pct)}%`;

  const phaseLabels = {
    processing: 'Procesando',
    ocr: 'OCR',
    chunking: 'Fragmentando',
    embedding: 'Generando embeddings',
    completed: 'Completado',
    failed: 'Error',
  };
  $('progressPhase').textContent = phaseLabels[phase] || phase;
  $('progressMessage').textContent = message;
}

function startProgressPolling(docId) {
  stopProgressPolling();
  progressPollTimer = setInterval(async () => {
    try {
      const d = await apiFetch(`/api/v1/documents/${encodeURIComponent(docId)}`);
      updateProgressUI(d);

      if (d.status === 'completed' || d.status === 'failed') {
        stopProgressPolling();
        await loadDocuments();
        const doc = documents.find(x => x.id === docId);
        if (doc && d.status === 'completed') {
          loadPdf(doc);
          renderSummary(doc);
          showToast('Documento procesado correctamente', 'success');
        } else if (d.status === 'failed') {
          showToast('Error al procesar el documento', 'error');
        }
      }
    } catch (e) {
      stopProgressPolling();
    }
  }, 1500);
}

function stopProgressPolling() {
  if (progressPollTimer) {
    clearInterval(progressPollTimer);
    progressPollTimer = null;
  }
}

/* ===== DOCUMENTS ===== */
function updateUploadState() {
  const occupied = documents.length > 0;
  $('fileInput').disabled = occupied;
  $('uploadBtn').disabled = occupied;
}

function renderDocList() {
  updateUploadState();
  const total = documents.length;
  const processing = documents.filter(d => d.status === 'processing' || d.status === 'ocr' || d.status === 'chunking' || d.status === 'embedding');
  $('docCount').textContent = `${total} documento${total !== 1 ? 's' : ''}`;
  $('statDocs').textContent = total;
  $('statProcessing').textContent = processing.length;

  if (!total) {
    $('docList').innerHTML = '<div class="empty-state"><p>No hay documentos cargados</p></div>';
    clearViewer();
    renderSummary(null);
    $('progressContainer').style.display = 'none';
    return;
  }

  const activeDoc = processing[0];
  if (activeDoc) {
    showProgress(activeDoc);
    startProgressPolling(activeDoc.id);
  } else {
    $('progressContainer').style.display = 'none';
  }

  $('docList').innerHTML = documents.map(d => `
    <div class="doc-item">
      <div class="doc-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/><polyline points="14 2 14 8 20 8"/></svg>
      </div>
      <div class="doc-details">
        <div class="doc-name" onclick="selectDocument('${esc(d.id)}')">${esc(d.original_name)}</div>
        <div class="doc-meta">${d.page_count || '?'} páginas · ${esc(d.created_at?.split('T')[0] || d.created_at)}</div>
      </div>
      <span class="badge ${d.status === 'completed' ? 'badge-success' : (d.status === 'processing' || d.status === 'ocr' || d.status === 'chunking' || d.status === 'embedding') ? 'badge-warning' : d.status === 'failed' ? 'badge-danger' : 'badge-neutral'}">
        <span class="badge-dot"></span>${esc(d.status)}
      </span>
      <div class="doc-actions">
        <button class="btn btn-ghost btn-sm" onclick="reprocessDoc('${esc(d.id)}')" title="Reprocesar">↻</button>
        <button class="btn btn-danger btn-sm" onclick="deleteDoc('${esc(d.id)}','${esc(d.original_name)}')" title="Eliminar">✕</button>
      </div>
    </div>
  `).join('');

  const firstCompleted = documents.find(d => d.status === 'completed');
  if (firstCompleted) {
    loadPdf(firstCompleted);
    renderSummary(firstCompleted);
  }
}

async function loadDocuments() {
  try {
    const d = await apiFetch('/api/v1/documents?size=100');
    documents = d.items || [];
    renderDocList();
  } catch (e) {
    documents = [];
    $('docList').innerHTML = `<p style="color:var(--danger)">${esc(e.message)}</p>`;
    updateUploadState();
  }
}

function selectDocument(id) {
  const doc = documents.find(d => d.id === id);
  if (!doc) return;
  if (currentView !== 'documents') switchView('documents');
  loadPdf(doc);
  renderSummary(doc);
}

/* ===== PDF VIEWER ===== */
function clearViewer() {
  pdfUrls.forEach(url => URL.revokeObjectURL(url));
  pdfUrls = [];
  $('pdfPages').replaceChildren();
  $('pdfPages').hidden = true;
  $('viewerEmpty').hidden = false;
  $('viewerStatus').textContent = 'Sin documento';
  pdfZoom = 1.0;
  currentDocId = null;
  updateZoomDisplay();
  clearSearch();
}

async function loadPdf(doc) {
  if (!doc) return clearViewer();
  clearViewer();
  currentDocId = doc.id;
  $('viewerStatus').textContent = 'Cargando páginas...';
  try {
    const count = Math.max(1, Number(doc.page_count) || 1);
    $('pdfPages').innerHTML = Array.from({ length: count }, (_, i) =>
      `<img class="pdf-page" src="/api/v1/documents/${encodeURIComponent(doc.id)}/pages/${i + 1}" alt="Página ${i + 1}" loading="lazy">`
    ).join('');
    $('pdfPages').hidden = false;
    $('viewerEmpty').hidden = true;
    $('viewerStatus').textContent = `${doc.original_name} · ${count} páginas`;
  } catch (e) {
    $('viewerStatus').textContent = `Error: ${e.message}`;
  }
}

/* ===== VIEWER SEARCH ===== */
async function searchInViewer() {
  const query = $('viewerSearchInput').value.trim();
  if (!query || !currentDocId) return;

  try {
    const d = await apiFetch(`/api/v1/documents/${encodeURIComponent(currentDocId)}/search?q=${encodeURIComponent(query)}`);
    searchMatches = d.matches || [];
    currentMatchIndex = -1;

    const nav = $('matchNavigator');
    if (searchMatches.length === 0) {
      nav.hidden = false;
      $('matchCounter').textContent = '0 / 0';
      $('matchScoreBadge').textContent = 'Sin resultados';
      $('matchScoreBadge').className = 'match-score-badge match-score-low';
      return;
    }

    nav.hidden = false;
    $('matchCounter').textContent = `1 / ${searchMatches.length}`;
    $('matchScoreBadge').textContent = `${searchMatches[0].score}%`;
    $('matchScoreBadge').className = `match-score-badge ${getScoreClass(searchMatches[0].score)}`;
    currentMatchIndex = 0;

    scrollToMatch(0);
    showToast(`${searchMatches.length} coincidencias encontradas`, 'info');
  } catch (e) {
    showToast(`Error en la búsqueda: ${e.message}`, 'error');
  }
}

function getScoreClass(score) {
  if (score >= 80) return 'match-score-high';
  if (score >= 50) return 'match-score-medium';
  return 'match-score-low';
}

function nextMatch() {
  if (searchMatches.length === 0) return;
  currentMatchIndex = (currentMatchIndex + 1) % searchMatches.length;
  updateMatchUI();
  scrollToMatch(currentMatchIndex);
}

function prevMatch() {
  if (searchMatches.length === 0) return;
  currentMatchIndex = (currentMatchIndex - 1 + searchMatches.length) % searchMatches.length;
  updateMatchUI();
  scrollToMatch(currentMatchIndex);
}

function updateMatchUI() {
  $('matchCounter').textContent = `${currentMatchIndex + 1} / ${searchMatches.length}`;
  const match = searchMatches[currentMatchIndex];
  $('matchScoreBadge').textContent = `${match.score}%`;
  $('matchScoreBadge').className = `match-score-badge ${getScoreClass(match.score)}`;
}

function scrollToMatch(index) {
  const match = searchMatches[index];
  if (!match || !match.page) return;

  const pageImages = $('pdfPages').querySelectorAll('.pdf-page');
  if (match.page <= pageImages.length) {
    pageImages[match.page - 1].scrollIntoView({ behavior: 'smooth', block: 'center' });
  }
}

function clearSearch() {
  searchMatches = [];
  currentMatchIndex = -1;
  const input = $('viewerSearchInput');
  if (input) input.value = '';
  const nav = $('matchNavigator');
  if (nav) nav.hidden = true;
}

/* ===== ZOOM CONTROLS ===== */
function zoomIn() {
  pdfZoom = Math.min(ZOOM_MAX, pdfZoom + ZOOM_STEP);
  applyZoom();
}

function zoomOut() {
  pdfZoom = Math.max(ZOOM_MIN, pdfZoom - ZOOM_STEP);
  applyZoom();
}

function zoomFit() {
  const viewer = $('viewerBody');
  const firstPage = $('pdfPages').querySelector('.pdf-page');
  if (firstPage && viewer) {
    const viewerWidth = viewer.clientWidth - 24;
    const pageWidth = firstPage.naturalWidth || firstPage.width;
    pdfZoom = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, viewerWidth / pageWidth));
  } else {
    pdfZoom = 1.0;
  }
  applyZoom();
}

function zoomReset() {
  pdfZoom = 1.0;
  applyZoom();
}

function applyZoom() {
  document.querySelectorAll('.pdf-page').forEach(img => {
    img.style.transform = `scale(${pdfZoom})`;
    img.style.transformOrigin = 'top center';
    img.style.width = '100%';
    img.style.height = 'auto';
  });
  updateZoomDisplay();
}

function updateZoomDisplay() {
  const el = $('zoomLevel');
  if (el) el.textContent = `${Math.round(pdfZoom * 100)}%`;
}

/* ===== UPLOAD ===== */
async function uploadDocument() {
  if (documents.length) {
    $('uploadStatus').textContent = 'Ya hay un PDF. Elimínalo para subir otro.';
    $('uploadStatus').className = 'status-msg error';
    return;
  }
  const file = $('fileInput').files[0];
  if (!file) {
    $('uploadStatus').textContent = 'Selecciona un PDF.';
    $('uploadStatus').className = 'status-msg error';
    return;
  }
  const form = new FormData();
  form.append('file', file);
  $('uploadBtn').disabled = true;
  $('uploadStatus').textContent = 'Subiendo y validando...';
  $('uploadStatus').className = 'status-msg info';
  try {
    const d = await apiFetch('/api/v1/documents', { method: 'POST', body: form });
    if (d.detail) throw Error(d.detail);
    $('uploadStatus').textContent = `Aceptado: ${d.filename || file.name}. Procesando...`;
    $('uploadStatus').className = 'status-msg ok';
    showToast('Documento subido. Procesando...', 'success');
    await loadDocuments();
  } catch (e) {
    $('uploadStatus').textContent = e.message;
    $('uploadStatus').className = 'status-msg error';
    updateUploadState();
  }
}

async function reprocessDoc(id) {
  if (!confirm('¿Volver a procesar el PDF con OCR mejorado?')) return;
  try {
    await apiFetch(`/api/v1/documents/${encodeURIComponent(id)}/reprocess`, { method: 'POST' });
    showToast('Reprocesando documento...', 'info');
    await loadDocuments();
  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
  }
}

async function deleteDoc(id, name) {
  if (!confirm(`¿Eliminar "${name}"? Esta acción no se puede deshacer.`)) return;
  try {
    await apiFetch(`/api/v1/documents/${encodeURIComponent(id)}`, { method: 'DELETE' });
    showToast('Documento eliminado', 'success');
    $('progressContainer').style.display = 'none';
    stopProgressPolling();
    await loadDocuments();
  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
  }
}

/* ===== SUMMARY ===== */
function renderSummary(doc) {
  if (!doc || doc.status !== 'completed') {
    $('summaryArea').innerHTML = '<div class="empty-state"><svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M9 12h6m-6 4h6m2 5H7a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5.586a1 1 0 0 1 .707.293l5.414 5.414a1 1 0 0 1 .293.707V19a2 2 0 0 1-2 2z"/></svg><p>' + (doc ? `Estado: ${doc.status}. Espera a que termine.` : 'El resumen aparecerá cuando termine el procesamiento.') + '</p></div>';
    return;
  }
  $('summaryArea').innerHTML = '<p style="color:var(--text-muted)"><span class="spinner"></span> Cargando resumen...</p>';
  loadDocDetails(doc);
}

async function loadDocDetails(doc) {
  try {
    const d = await apiFetch(`/api/v1/documents/${encodeURIComponent(doc.id)}`);
    const chunks = d.chunks || [];
    const rawText = chunks.map(c => c.content || '').join('\n');
    const upper = rawText.toUpperCase();

    const medicines = ['AMOXICILINA','METFORMINA','NAPROXENO','OMEPRAZOL','PARACETAMOL','CIPROFLOXACINO','ENALAPRIL','IBUPROFENO','LEVOTIROXINA']
      .filter(m => upper.includes(m)).join(', ') || 'No detectados';
    const hospital = ['HOSPITAL REGIONAL','HOSPITAL DOCENTE','HOSPITAL GENERAL']
      .find(h => upper.includes(h)) || 'No detectado';

    const fields = [
      ['Centro de salud', hospital],
      ['Páginas', doc.page_count || '-'],
      ['Fragmentos', chunks.length],
      ['Medicamentos', medicines],
    ];

    $('summaryArea').innerHTML = `
      <div class="field-grid">
        ${fields.map(([k, v]) => `<div class="field-row"><span class="field-label">${esc(k)}</span><span class="field-value">${esc(v)}</span></div>`).join('')}
      </div>
    `;
  } catch (e) {
    $('summaryArea').innerHTML = `<p style="color:var(--danger)">Error: ${esc(e.message)}</p>`;
  }
}

/* ===== QUERY ===== */
async function askQuestion() {
  const q = $('questionInput').value.trim();
  if (!q) {
    $('queryStatus').textContent = 'Escribe una pregunta.';
    $('queryStatus').className = 'status-msg error';
    return;
  }
  if (!documents.some(d => d.status === 'completed')) {
    $('queryStatus').textContent = 'Espera a que un documento termine de procesarse.';
    $('queryStatus').className = 'status-msg error';
    return;
  }
  $('askBtn').disabled = true;
  $('queryStatus').textContent = 'Buscando evidencia y generando respuesta...';
  $('queryStatus').className = 'status-msg info';
  $('answerArea').hidden = true;
  try {
    const d = await apiFetch('/api/v1/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ query: q, top_k: 5, stream: false }),
    });
    if (d.detail) throw Error(d.detail);
    $('answerText').textContent = d.answer || 'No se encontró evidencia suficiente.';
    $('citationsBox').innerHTML = (d.citations || []).length
      ? '<h4 style="font-size:12px;color:var(--text-muted);margin-bottom:8px;">Citas</h4>' +
        d.citations.map(c => `<div class="citation-item">${esc(typeof c === 'string' ? c : JSON.stringify(c))}</div>`).join('')
      : '<p style="font-size:12px;color:var(--text-muted);margin-top:8px;">Sin citas disponibles.</p>';
    $('answerArea').hidden = false;
    $('queryStatus').textContent = `Completado en ${d.latency_ms ?? '-'} ms`;
    $('queryStatus').className = 'status-msg ok';
  } catch (e) {
    $('queryStatus').textContent = e.message;
    $('queryStatus').className = 'status-msg error';
  } finally {
    $('askBtn').disabled = false;
  }
}

/* ===== KEYBOARD SHORTCUTS ===== */
document.addEventListener('keydown', (e) => {
  if (e.ctrlKey && e.key === '=') { e.preventDefault(); zoomIn(); }
  if (e.ctrlKey && e.key === '-') { e.preventDefault(); zoomOut(); }
  if (e.ctrlKey && e.key === '0') { e.preventDefault(); zoomReset(); }
});

/* ===== INIT ===== */
document.addEventListener('DOMContentLoaded', () => {
  loadUserInfo();
  loadDocuments();
});
