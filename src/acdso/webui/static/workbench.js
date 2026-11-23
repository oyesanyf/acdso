/* ACDSO Workbench — Modular Frontend Controller (SSE + Queue + Charts + Compare) */
(function (globalScope) {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const MAX_CLIENT_LOG_LINES = 2000;
  const FLAG_META = Array.isArray(globalScope.__ACDSO_FLAG_META__) ? globalScope.__ACDSO_FLAG_META__ : [];
  const Charts = globalScope.ACDSOCharts || {};

  const state = {
    uploadId: null,
    uploadFileName: null,
    columns: [],
    dtypes: {},
    columnStats: [],
    predictableSet: new Set(),
    goalMode: 'predict', // 'predict' | 'train'
    jobId: null,
    cursor: 0,
    poll: null,
    pollInFlight: false,
    pollFailures: 0,
    sse: null,
    columnsBlocked: false,
    logLines: [],
    inspectorData: null,
    inspectorTab: 'overview',
    paretoAxis: 'train_time_s',
    autoInspectedJob: null,
    selectedCompare: new Set(),
  };

  function esc(s) {
    return Charts.esc
      ? Charts.esc(s)
      : String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
          { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
        ));
  }

  function flash(text, kind, targetId = 'uploadMsg') {
    const el = $(targetId);
    if (!el) return;
    el.textContent = text || '';
    el.className = text ? 'msg ' + kind : 'msg';
  }

  async function safeFetchJson(url, opts) {
    try {
      const res = await fetch(url, opts);
      const raw = await res.text();
      let data = null;
      if (raw) {
        try { data = JSON.parse(raw); } catch (_) { data = null; }
      }
      if (!res.ok) {
        const msg = (data && (data.error || data.message)) || ('HTTP ' + res.status);
        return { ok: false, status: res.status, data, error: msg };
      }
      return { ok: true, status: res.status, data, error: null };
    } catch (err) {
      return { ok: false, status: 0, data: null, error: String(err) };
    }
  }

  function fmtSize(n) {
    if (n == null) return '';
    if (n < 1024) return n + ' B';
    if (n < 1024 * 1024) return (n / 1024).toFixed(1) + ' KB';
    return (n / 1048576).toFixed(1) + ' MB';
  }

  function fmtNum(v, digits) {
    if (Charts.fmtNum) return Charts.fmtNum(v, digits);
    if (v == null || v === '') return 'n/a';
    const n = Number(v);
    return Number.isFinite(n) ? n.toFixed(digits) : 'n/a';
  }

  function fmtValue(n, col) {
    if (Charts.fmtValue) return Charts.fmtValue(n, col);
    if (n == null || Number.isNaN(Number(n))) return '-';
    return Number(n).toFixed(2);
  }

  /* ── uploads & recent uploads ──────────────────────────────────────── */
  const drop = $('drop');
  const fileInput = $('file');

  if (drop && fileInput) {
    drop.addEventListener('click', () => fileInput.click());
    drop.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); fileInput.click(); }
    });
    drop.addEventListener('dragover', (e) => { e.preventDefault(); drop.classList.add('over'); });
    drop.addEventListener('dragleave', () => drop.classList.remove('over'));
    drop.addEventListener('drop', (e) => {
      e.preventDefault();
      drop.classList.remove('over');
      if (e.dataTransfer.files.length) sendFile(e.dataTransfer.files[0]);
    });
    fileInput.addEventListener('change', () => {
      if (fileInput.files.length) sendFile(fileInput.files[0]);
    });
  }

  function applyLoadedUpload(data) {
    state.uploadId = data.upload_id;
    state.uploadFileName = data.stored_as || data.filename || null;
    state.columns = data.columns || [];
    state.dtypes = data.dtypes || {};
    state.columnStats = data.column_stats || [];
    state.predictableSet = new Set(data.predictable_columns || state.columns);
    state.columnsBlocked = false;

    clearTimeout(columnCheckTimer);
    clearTimeout(horizonCheckTimer);
    if ($('targetSearch')) $('targetSearch').value = '';
    if ($('excludeSearch')) $('excludeSearch').value = '';
    $('colCheck').innerHTML = '';
    $('horizonSuggestion').innerHTML = '';
    $('horizonSuggestion').style.display = 'none';
    $('nlPlanOut').innerHTML = '';
    $('nlPlanStatus').textContent = '';
    $('fcOut').innerHTML = '';
    flash('', 'err', 'runMsg');
    flash('', 'err', 'fcMsg');

    $('fname').textContent = data.filename || data.stored_as;
    $('fmeta').textContent =
      state.columns.length + ' columns' +
      (data.rows != null ? ' · ' + Number(data.rows).toLocaleString() + ' rows' : '') +
      (data.size_bytes ? ' · ' + fmtSize(data.size_bytes) : '');

    renderDatasetSummaryPills(state.columnStats, data.rows);
    renderPreview(state.columns, data.preview || [], state.dtypes);
    $('fileinfo').style.display = 'block';

    fillColumnSelects(state.columns);
    restrictPredictTargetOptions(data.predictable_columns || state.columns, state.columns);
    renderTargetPicker();
    preselectForecastColumns(state.columns, state.dtypes);
    refreshCommand();
    checkColumns();
    loadTargetSuggestion();
  }

  function renderDatasetSummaryPills(stats, rows) {
    const box = $('datasetSummaryPills');
    if (!box) return;
    if (!stats || !stats.length) { box.innerHTML = ''; return; }
    const numCols = stats.filter((s) => s.role_hint === 'numeric').length;
    const catCols = stats.filter((s) => s.role_hint === 'categorical').length;
    const dtCols = stats.filter((s) => s.role_hint === 'datetime').length;
    const withNulls = stats.filter((s) => s.null_count > 0).length;
    box.innerHTML = [
      rows != null ? '<span class="pill">' + Number(rows).toLocaleString() + ' rows</span>' : '',
      '<span class="pill">' + stats.length + ' cols</span>',
      '<span class="pill">' + numCols + ' numeric</span>',
      '<span class="pill">' + catCols + ' categorical</span>',
      dtCols ? '<span class="pill">' + dtCols + ' datetime</span>' : '',
      '<span class="pill">' + withNulls + ' cols with nulls</span>',
    ].filter(Boolean).join('');
  }

  async function sendFile(file) {
    flash('Uploading and profiling ' + file.name + '…', 'ok', 'uploadMsg');
    drop.classList.add('busy');
    $('recentUploads').disabled = true;
    try {
      const body = new FormData();
      body.append('file', file);
      const r = await safeFetchJson('/api/upload', { method: 'POST', body });
      if (!r.ok) {
        flash(r.error || 'Upload failed', 'err', 'uploadMsg');
        return;
      }
      applyLoadedUpload(r.data);
      flash('Loaded ' + r.data.filename + '. Pick a target below.', 'ok', 'uploadMsg');
      loadRecentUploads(r.data.upload_id);
    } finally {
      drop.classList.remove('busy');
      $('recentUploads').disabled = false;
      fileInput.value = '';
    }
  }

  async function loadRecentUploads(selectedId) {
    const sel = $('recentUploads');
    if (!sel) return;
    const r = await safeFetchJson('/api/uploads');
    if (!r.ok) {
      sel.innerHTML = '<option value="">Could not list uploads</option>';
      return;
    }
    const list = (r.data && r.data.uploads) || [];
    if (!list.length) {
      sel.innerHTML = '<option value="">No uploaded datasets yet</option>';
      return;
    }
    const nameCounts = new Map();
    for (const u of list) {
      const key = u.filename || '';
      nameCounts.set(key, (nameCounts.get(key) || 0) + 1);
    }
    sel.innerHTML = list.map((u) => {
      const info = u.rows != null
        ? Number(u.rows).toLocaleString() + ' rows, ' + (u.columns ? u.columns.length : '?') + ' cols'
        : fmtSize(u.size_bytes);
      const dupTag = (nameCounts.get(u.filename || '') || 0) > 1 && u.upload_id
        ? ' · #' + String(u.upload_id).slice(0, 6)
        : '';
      return '<option value="' + esc(u.upload_id) + '">' + esc(u.filename) + ' (' + esc(info) + esc(dupTag) + ')</option>';
    }).join('');
    if (selectedId) sel.value = selectedId;
    else if (state.uploadId) sel.value = state.uploadId;
  }

  async function loadUploadById(uid) {
    if (!uid) return;
    flash('Loading saved upload…', 'ok', 'uploadMsg');
    $('loadRecentBtn').disabled = true;
    try {
      const r = await safeFetchJson('/api/uploads/' + encodeURIComponent(uid));
      if (!r.ok) {
        flash(r.error || 'Could not load upload', 'err', 'uploadMsg');
        return;
      }
      applyLoadedUpload(r.data);
      flash('Loaded ' + r.data.filename + '.', 'ok', 'uploadMsg');
    } finally {
      $('loadRecentBtn').disabled = false;
    }
  }

  if ($('loadRecentBtn')) $('loadRecentBtn').addEventListener('click', () => loadUploadById($('recentUploads').value));
  if ($('refreshUploadsBtn')) $('refreshUploadsBtn').addEventListener('click', () => loadRecentUploads());

  /* ── target goal toggle & searchable picker ────────────────────────── */
  function setGoalMode(mode) {
    state.goalMode = mode;
    const isPredict = mode === 'predict';
    $('goalPredictBtn').classList.toggle('active', isPredict);
    $('goalTrainBtn').classList.toggle('active', !isPredict);
    $('predictSelectWrap').style.display = isPredict ? 'block' : 'none';
    $('trainSelectWrap').style.display = isPredict ? 'none' : 'block';
    if (isPredict) {
      Array.from($('f_target').options).forEach((o) => { o.selected = false; });
    } else {
      Array.from($('f_predict').options).forEach((o) => { o.selected = false; });
    }
    renderTargetPicker();
    refreshCommand();
    checkColumns();
  }

  if ($('goalPredictBtn')) $('goalPredictBtn').addEventListener('click', () => setGoalMode('predict'));
  if ($('goalTrainBtn')) $('goalTrainBtn').addEventListener('click', () => setGoalMode('train'));

  function activeTargetSelectId() {
    return state.goalMode === 'predict' ? 'f_predict' : 'f_target';
  }

  function renderTargetPicker() {
    const listEl = $('targetPickerList');
    if (!listEl) return;
    if (!state.columns.length) {
      listEl.innerHTML = '<div class="hint" style="padding:10px; margin:0">Upload or load a dataset to browse columns.</div>';
      return;
    }
    const term = ($('targetSearch').value || '').trim().toLowerCase();
    const sel = $(activeTargetSelectId());
    const selectedVals = new Set(Array.from(sel.selectedOptions).map((o) => o.value));
    const statsByName = new Map(state.columnStats.map((s) => [s.name, s]));

    const rows = state.columns.filter((col) => {
      if (!term) return true;
      const st = statsByName.get(col);
      const dt = (st ? st.dtype : (state.dtypes[col] || '')).toLowerCase();
      return col.toLowerCase().includes(term) || dt.includes(term);
    });

    if (!rows.length) {
      listEl.innerHTML = '<div class="hint" style="padding:10px; margin:0">No columns match "' + esc(term) + '".</div>';
      return;
    }

    listEl.innerHTML = rows.map((col) => {
      const st = statsByName.get(col) || {
        name: col, dtype: state.dtypes[col] || '', nunique: null, null_pct: null,
        role_hint: 'categorical', predictable: state.predictableSet.has(col), exclude_reason: null,
      };
      const isPred = st.predictable !== false && state.predictableSet.has(col);
      const isSel = selectedVals.has(col);
      const roleClass = st.role_hint === 'numeric' ? 'badge-num'
        : st.role_hint === 'datetime' ? 'badge-date'
        : (st.role_hint === 'id' || st.role_hint === 'constant') ? 'badge-id' : 'badge-cat';
      const metaBadges = [
        st.dtype ? '<span class="badge ' + roleClass + '">' + esc(st.dtype) + '</span>' : '',
        st.nunique != null ? '<span class="badge">' + st.nunique + ' uniq</span>' : '',
        st.null_pct != null && st.null_pct > 0 ? '<span class="badge">' + st.null_pct + '% null</span>' : '',
        !isPred && st.role_hint ? '<span class="badge badge-id">' + esc(st.role_hint) + '</span>' : '',
      ].filter(Boolean).join(' ');

      return '<div class="picker-item' + (isSel ? ' selected' : '') + (!isPred ? ' disabled' : '') + '" ' +
        'data-col="' + esc(col) + '" data-pred="' + (isPred ? '1' : '0') + '" ' +
        (st.exclude_reason ? 'title="' + esc(st.exclude_reason) + '"' : '') + '>' +
        '<span><strong>' + esc(col) + '</strong>' +
        (!isPred && st.exclude_reason ? ' <span class="hint" style="margin:0">(' + esc(st.exclude_reason) + ')</span>' : '') +
        '</span><span class="row" style="gap:4px">' + metaBadges + '</span></div>';
    }).join('');

    listEl.querySelectorAll('.picker-item').forEach((row) => {
      row.addEventListener('click', () => {
        if (row.dataset.pred !== '1') return;
        const col = row.dataset.col;
        const targetSel = $(activeTargetSelectId());
        Array.from(targetSel.options).forEach((o) => {
          if (o.value === col) o.selected = !o.selected;
        });
        targetSel.dispatchEvent(new Event('change', { bubbles: true }));
        renderTargetPicker();
      });
    });
  }

  if ($('targetSearch')) $('targetSearch').addEventListener('input', renderTargetPicker);
  if ($('f_predict')) $('f_predict').addEventListener('change', renderTargetPicker);
  if ($('f_target')) $('f_target').addEventListener('change', renderTargetPicker);

  if ($('excludeSearch')) {
    $('excludeSearch').addEventListener('input', () => {
      const term = ($('excludeSearch').value || '').trim().toLowerCase();
      Array.from($('f_exclude').options).forEach((opt) => {
        opt.hidden = Boolean(term) && !opt.value.toLowerCase().includes(term);
      });
    });
  }

  /* ── target suggestion ─────────────────────────────────────────────── */
  async function loadTargetSuggestion() {
    const predictBox = $('targetSuggestion');
    const trainBox = $('trainTargetSuggestion');
    const btn = $('suggestTargetBtn');
    if (!state.uploadId || !btn) return;
    btn.disabled = true;
    btn.textContent = 'Analyzing targets…';
    try {
      const r = await safeFetchJson('/api/suggest-target', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ upload_id: state.uploadId }),
      });
      if (!r.ok) {
        const box = state.goalMode === 'predict' ? predictBox : trainBox;
        box.style.display = 'block';
        box.innerHTML = '<div class="msg warn" style="display:block">Could not auto-suggest targets: ' + esc(r.error) + '</div>';
        return;
      }
      const suggestions = (r.data && r.data.suggestions) || [];
      if (!suggestions.length) {
        const box = state.goalMode === 'predict' ? predictBox : trainBox;
        box.style.display = 'block';
        box.innerHTML = '<div class="msg warn" style="display:block">No predictable target columns were detected automatically. Check the excluded reasons above.</div>';
        return;
      }
      renderSuggestionBox(predictBox, r.data, 'f_predict');
      renderSuggestionBox(trainBox, r.data, 'f_target');
    } finally {
      btn.disabled = false;
      btn.textContent = 'Suggest target';
    }
  }

  if ($('suggestTargetBtn')) $('suggestTargetBtn').addEventListener('click', loadTargetSuggestion);

  function fmtVotes(votes) {
    if (!votes) return '';
    const parts = [];
    if (votes.rule_based != null) parts.push('naming/dtype: ' + votes.rule_based.toFixed(2));
    if (votes.learnability) parts.push(votes.learnability.metric + '=' + votes.learnability.score.toFixed(2));
    else parts.push('learnability: n/a');
    return parts.length ? ' · votes: ' + parts.join(', ') : '';
  }

  function renderSuggestionBox(box, data, targetFieldId) {
    const sourceNote = data.source === 'llm'
      ? 'LLM verdict, weighing votes against column semantics'
      : 'rules + fast CV learnability';
    box.style.display = 'block';
    box.innerHTML = data.suggestions.map((s, i) =>
      '<div class="msg ok" style="display:block; margin-bottom:6px">' +
      '<strong>Suggested target: ' + esc(s.column) + '</strong> ' +
      '<button type="button" class="sm use-suggestion" data-col="' + esc(s.column) + '" style="margin-left:8px">Use this</button>' +
      '<div class="hint" style="margin:4px 0 0">' + esc(s.reason) + esc(fmtVotes(s.votes)) +
      (i === 0 ? ' (' + sourceNote + ')' : '') + '</div></div>'
    ).join('');
    box.querySelectorAll('.use-suggestion').forEach((btn) => {
      btn.addEventListener('click', () => {
        const sel = $(targetFieldId);
        const col = btn.dataset.col;
        Array.from(sel.options).forEach((o) => { if (o.value === col) o.selected = true; });
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        renderTargetPicker();
      });
    });
  }

  function renderPreview(columns, rows, dtypes) {
    const head = '<thead><tr>' + columns.map(
      (c) => '<th scope="col">' + esc(c) + '<br><span class="hint" style="margin:0">' + esc(dtypes[c] || '') + '</span></th>'
    ).join('') + '</tr></thead>';
    const body = '<tbody>' + rows.map(
      (r) => '<tr>' + columns.map((c) => '<td>' + esc(r[c] === null || r[c] === undefined ? '' : String(r[c])) + '</td>').join('') + '</tr>'
    ).join('') + '</tbody>';
    $('preview').innerHTML = head + body;
  }

  function fillColumnSelects(columns) {
    document.querySelectorAll('select[data-columns]').forEach((sel) => {
      const blank = sel.multiple ? '' : '<option value="">(none)</option>';
      sel.innerHTML = blank +
        columns.map((c) => '<option value="' + esc(c) + '">' + esc(c) + '</option>').join('');
    });
  }

  function restrictPredictTargetOptions(predictable, allColumns) {
    const blockedCount = allColumns.length - predictable.length;
    ['f_predict', 'f_target'].forEach((id) => {
      const sel = $(id);
      if (!sel) return;
      sel.innerHTML = predictable.map((c) => '<option value="' + esc(c) + '">' + esc(c) + '</option>').join('');
    });
    const note = $('predictHiddenNote');
    if (note) {
      note.textContent = blockedCount > 0
        ? blockedCount + ' column(s) marked non-predictable (date/period, constant, ID, or high-cardinality categorical).'
        : '';
    }
  }

  /* ── run mode syncing ──────────────────────────────────────────────── */
  const MODE_HINTS = {
    standard: 'Runs full EDA, Risk Analysis, AutoML, and Time Series on your dataset.',
    quick: 'Fast EDA + AutoML (--simple). Skips Time Series and benchmark comparisons.',
    benchmark: 'Compares ACDSO against external AutoML benchmarks (--benchmark).',
    decision: 'Runs causal uplift & action optimization (--decision) with optional treatment & budget.',
    modes: 'Select a specialized built-in mode below.',
  };

  function syncDecisionOptsVisibility() {
    if ($('decisionOptsWrap') && $('f_decision')) {
      $('decisionOptsWrap').style.display = $('f_decision').checked ? 'block' : 'none';
    }
  }

  if ($('runMode')) {
    $('runMode').addEventListener('change', () => {
      const m = $('runMode').value;
      $('runModeHint').textContent = MODE_HINTS[m] || '';
      $('modeSwitchesWrap').style.display = m === 'modes' ? 'block' : 'none';

      $('f_simple').checked = (m === 'quick');
      $('f_benchmark').checked = (m === 'benchmark');
      $('f_skip_benchmark').checked = (m !== 'benchmark');
      $('f_decision').checked = (m === 'decision');
      if (m !== 'modes') {
        $('f_timeseries').checked = false;
        $('f_policy').checked = false;
        $('f_v3').checked = false;
        $('f_all').checked = false;
      }
      syncDecisionOptsVisibility();
      refreshCommand();
    });
  }
  if ($('f_decision')) $('f_decision').addEventListener('change', syncDecisionOptsVisibility);

  /* ── forecast + SVG chart ──────────────────────────────────────────── */
  function preselectForecastColumns(columns, dtypes) {
    const isNumeric = (c) => /int|float/i.test(dtypes[c] || '');
    const dateCandidates = columns.filter((c) => !isNumeric(c));
    const dateGuess = dateCandidates.find((c) => /^(date|month|period|day|time|timestamp)$/i.test(c))
      || dateCandidates.find((c) => /date|month|period|time/i.test(c));
    const numeric = columns.filter(isNumeric);
    const valueGuess = numeric.find((c) => /total|amount|bytes|transfer|value|cost|qty|count|price/i.test(c))
      || numeric[0];
    if (dateGuess && $('fc_date')) $('fc_date').value = dateGuess;
    if (valueGuess && $('fc_value')) $('fc_value').value = valueGuess;
  }

  function renderForecast(d) {
    const col = d.value_col;
    const agg = d.aggregation;
    const rows = (label, items) => items.map((p) =>
      '<tr><td>' + esc(p.period) + '</td><td>' + fmtValue(p.value, col) + '</td>' +
      '<td class="hint" style="margin:0">' + label + '</td></tr>').join('');

    const hist = rows('actual', d.history);
    const fc = d.forecast.map((p) =>
      '<tr style="background:rgba(91,157,255,.10)"><td><strong>' + esc(p.period) + '</strong></td>' +
      '<td><strong>' + fmtValue(p.value, col) + '</strong></td>' +
      '<td class="hint" style="margin:0">forecast</td></tr>').join('');

    const methods = d.methods.map((m) =>
      '<tr' + (m.method === d.chosen_method ? ' style="background:rgba(79,209,139,.12)"' : '') +
      '><td>' + esc(m.method) + '</td><td>' +
      (m.mae == null ? '<span class="hint" style="margin:0">not testable</span>' : fmtValue(m.mae, col)) +
      '</td><td>' + (m.method === d.chosen_method ? '&#10003; selected' : '') + '</td></tr>').join('');

    const chartSvg = Charts.renderForecastChart ? Charts.renderForecastChart(d) : '';
    $('fcOut').innerHTML =
      '<div class="row" style="gap:18px; margin-bottom:10px">' +
        '<div><div class="hint" style="margin:0">next ' + d.forecast.length + ' periods</div>' +
        '<strong style="font-size:20px">' + fmtValue(d.forecast_total, col) + '</strong></div>' +
        '<div><div class="hint" style="margin:0">method (lowest backtest MAE)</div>' +
        '<strong>' + esc(d.chosen_method) + '</strong></div>' +
        '<div><div class="hint" style="margin:0">grain / periods</div>' +
        '<strong>' + esc(agg.grain) + ' &middot; ' + agg.periods + '</strong></div>' +
      '</div>' +
      '<div class="hint">' + agg.rows_used.toLocaleString() + ' of ' +
        agg.rows_in.toLocaleString() + ' rows aggregated by ' + esc(agg.agg) +
        (agg.dropped_periods.length ? ' &middot; dropped ' + agg.dropped_periods.map((p) => esc(p)).join(', ') : '') +
      '</div>' +
      chartSvg +
      '<div class="row" style="margin:10px 0">' +
        '<a class="btn primary" href="/api/artifact?path=' + encodeURIComponent(d.csv_path) + '">Download CSV</a>' +
        '<button type="button" class="sm inspect-btn" data-path="' + esc(d.csv_path) + '">Inspect CSV</button>' +
        '<span class="hint" style="margin:0">' + esc(d.csv_path) + '</span>' +
      '</div>' +
      '<div class="scroll-x" tabindex="0" role="region" aria-label="Forecast table" style="max-height:240px; overflow-y:auto">' +
        '<table><thead><tr><th>period</th><th>' + esc(col) + '</th><th></th></tr></thead><tbody>' + hist + fc + '</tbody></table></div>' +
      '<details style="margin-top:12px"><summary class="hint" style="cursor:pointer">' +
        'method comparison (one-step-ahead backtest)</summary>' +
        '<table><thead><tr><th>method</th><th>MAE</th><th></th></tr></thead><tbody>' + methods + '</tbody></table></details>' +
      (d.warnings && d.warnings.length
        ? '<div class="msg warn" style="display:block; margin-top:12px"><strong>Notes:</strong><ul style="margin:6px 0 0 18px">' +
          d.warnings.map((w) => '<li>' + esc(w) + '</li>').join('') + '</ul></div>'
        : '');
    $('fcOut').querySelectorAll('.inspect-btn').forEach((b) => {
      b.addEventListener('click', () => inspectReport(b.dataset.path));
    });
  }

  if ($('fcBtn')) {
    $('fcBtn').addEventListener('click', async () => {
      flash('', 'err', 'fcMsg');
      if (!state.uploadId) { flash('Upload or load a dataset first.', 'err', 'fcMsg'); return; }
      const rawH = $('fc_horizon').value;
      const hNum = Number(rawH);
      if (!rawH || !Number.isInteger(hNum) || hNum < 1 || hNum > 36) {
        flash('Forecast periods must be a whole number between 1 and 36.', 'err', 'fcMsg');
        return;
      }
      const body = {
        upload_id: state.uploadId,
        date_col: $('fc_date').value,
        value_col: $('fc_value').value,
        agg: $('fc_agg').value,
        horizon: hNum,
        drop_first: $('fc_drop_first').checked,
        drop_last: $('fc_drop_last').checked,
      };
      if (!body.date_col || !body.value_col) {
        flash('Pick both a period column and a value column.', 'err', 'fcMsg');
        return;
      }
      $('fcBtn').disabled = true;
      $('fcStatus').textContent = 'aggregating and forecasting…';
      try {
        const r = await safeFetchJson('/api/forecast', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(body),
        });
        if (!r.ok) {
          flash(r.error || 'Forecast failed', 'err', 'fcMsg');
          $('fcOut').innerHTML = '';
          return;
        }
        renderForecast(r.data);
      } finally {
        $('fcBtn').disabled = false;
        $('fcStatus').textContent = '';
      }
    });
  }

  /* ── payload + command preview ──────────────────────────────────────── */
  function collectPayload() {
    const payload = { upload_id: state.uploadId };
    const decisionActive = $('f_decision') ? $('f_decision').checked : false;
    const formEl = $('form');
    if (formEl) {
      formEl.querySelectorAll('[data-kind]').forEach((el) => {
        if ((el.name === 'treatment' || el.name === 'budget') && !decisionActive) return;
        const kind = el.dataset.kind;
        if (kind === 'bool') {
          payload[el.name] = el.checked;
        } else if (el.multiple) {
          payload[el.name] = Array.from(el.selectedOptions).map((o) => o.value).filter(Boolean);
        } else if (kind === 'int' || kind === 'float') {
          payload[el.name] = el.value === '' ? null : Number(el.value);
        } else {
          payload[el.name] = el.value;
        }
      });
    }
    if ($('queueIfBusy') && $('queueIfBusy').checked) {
      payload.queue = true;
    }
    return payload;
  }

  function refreshCommand() {
    const cmdEl = $('cmd');
    if (!cmdEl) return;
    const p = collectPayload();
    const parts = ['python', 'run_acdso.py'];
    if (state.uploadId) parts.push('--file', 'uploads/' + (state.uploadFileName || '<your file>'));
    for (const meta of FLAG_META) {
      const v = p[meta.name];
      if (meta.kind === 'bool') { if (v) parts.push(meta.cli); continue; }
      if (meta.kind === 'columns') {
        const joined = (Array.isArray(v) ? v : String(v || '').split(','))
          .map((s) => s.trim()).filter(Boolean).join(',');
        if (joined) parts.push(meta.cli, joined);
        continue;
      }
      if (v === null || v === '' || v === undefined) continue;
      if (meta.kind === 'choice' && String(v) === String(meta.default)) continue;
      parts.push(meta.cli, String(v));
    }
    cmdEl.textContent = parts.join(' ');
  }

  if ($('form')) {
    $('form').addEventListener('input', refreshCommand);
    $('form').addEventListener('change', refreshCommand);
  }

  /* ── column-role validation & horizon suggestion ────────────────────── */
  const ROLE_FIELDS = ['f_predict', 'f_target', 'f_exclude', 'f_datetime_col', 'f_treatment'];
  let columnCheckTimer = null;
  let horizonCheckTimer = null;

  function renderColumnCheck(result) {
    state.columnsBlocked = !result.ok;
    const box = $('colCheck');
    if (!box) return;
    if (!result.issues || !result.issues.length) { box.innerHTML = ''; return; }
    const rows = result.issues.map((i) =>
      '<div class="msg ' + (i.severity === 'block' ? 'err' : 'warn') + '" style="display:block; margin-bottom:6px">' +
      (i.severity === 'block' ? '&#10060; ' : '&#9888; ') + esc(i.message) + '</div>'
    ).join('');
    const sourceNote = result.source === 'llm' ? 'rules + LLM check' : 'rule-based check';
    box.innerHTML = rows + '<div class="hint" style="margin:0">' + sourceNote + '</div>';
  }

  function collectColumnValues(id) {
    const el = $(id);
    if (!el) return [];
    return el.multiple ? Array.from(el.selectedOptions).map((o) => o.value).filter(Boolean)
                        : (el.value ? [el.value] : []);
  }

  async function checkColumns() {
    if (!state.uploadId) { renderColumnCheck({ ok: true, issues: [] }); return; }
    const roles = {
      upload_id: state.uploadId,
      predict: collectColumnValues('f_predict'),
      target: collectColumnValues('f_target'),
      exclude: collectColumnValues('f_exclude'),
      datetime_col: collectColumnValues('f_datetime_col'),
      treatment: collectColumnValues('f_treatment'),
    };
    if (!roles.predict.length && !roles.target.length && !roles.exclude.length && !roles.datetime_col.length && !roles.treatment.length) {
      renderColumnCheck({ ok: true, issues: [] });
      return;
    }
    const r = await safeFetchJson('/api/validate-columns', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(roles),
    });
    if (r.ok && r.data) renderColumnCheck(r.data);
    else if ($('colCheck')) $('colCheck').innerHTML = '<div class="msg warn" style="display:block">Column validation check unavailable: ' + esc(r.error) + '</div>';
  }

  ROLE_FIELDS.forEach((id) => {
    const el = $(id);
    if (!el) return;
    el.addEventListener('change', () => {
      clearTimeout(columnCheckTimer);
      columnCheckTimer = setTimeout(checkColumns, 300);
      clearTimeout(horizonCheckTimer);
      horizonCheckTimer = setTimeout(loadHorizonSuggestion, 350);
    });
  });

  async function loadHorizonSuggestion() {
    const box = $('horizonSuggestion');
    if (!box) return;
    box.style.display = 'none';
    const datetimeCol = collectColumnValues('f_datetime_col')[0];
    const targetCol = collectColumnValues('f_predict')[0] || collectColumnValues('f_target')[0];
    if (!state.uploadId || !datetimeCol || !targetCol) return;

    const r = await safeFetchJson('/api/suggest-horizon', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ upload_id: state.uploadId, datetime_col: datetimeCol, target_col: targetCol }),
    });
    if (!r.ok) {
      box.style.display = 'block';
      box.innerHTML = '<div class="msg warn" style="display:block">Could not suggest horizon: ' + esc(r.error) + '</div>';
      return;
    }
    const data = r.data;
    if (!data || data.horizon == null) return;
    const sourceNote = data.source === 'llm' ? 'LLM + seasonality' : 'seasonality data vote';
    let voteText = '';
    if (data.votes && data.votes.data) {
      const dv = data.votes.data;
      voteText = dv.seasonality_period
        ? ' · seasonality period ' + dv.seasonality_period + ' (strength ' + dv.seasonality_strength.toFixed(2) + ')'
        : ' · no strong seasonality detected';
    }
    box.style.display = 'block';
    box.innerHTML =
      '<div class="msg ok" style="display:block">' +
      '<strong>Suggested horizon: ' + esc(String(data.horizon)) + '</strong> ' +
      '<button type="button" class="sm" id="useSuggestedHorizon" style="margin-left:8px">Use this</button>' +
      '<div class="hint" style="margin:4px 0 0">' + esc(data.reason) + esc(voteText) +
      ' (' + sourceNote + ')</div></div>';
    $('useSuggestedHorizon').addEventListener('click', () => {
      $('f_horizon').value = data.horizon;
      $('f_horizon').dispatchEvent(new Event('input', { bubbles: true }));
    });
  }

  /* ── natural-language plan ─────────────────────────────────────────── */
  if ($('nlPlanBtn')) {
    $('nlPlanBtn').addEventListener('click', async () => {
      const prompt = $('nlPrompt').value.trim();
      const status = $('nlPlanStatus');
      $('nlPlanOut').innerHTML = '';
      if (!state.uploadId) { status.textContent = 'Upload or load a dataset first.'; return; }
      if (!prompt) { status.textContent = 'Type a request first.'; return; }
      $('nlPlanBtn').disabled = true;
      status.textContent = 'Thinking…';
      try {
        const r = await safeFetchJson('/api/nl-plan', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ upload_id: state.uploadId, prompt }),
        });
        if (!r.ok) { status.textContent = r.error || 'Could not build a plan.'; return; }
        status.textContent = '';
        renderNlPlan(r.data);
      } finally {
        $('nlPlanBtn').disabled = false;
      }
    });
  }

  function renderNlPlan(plan) {
    const sourceNote = plan.method === 'llm' ? 'LLM-parsed' : 'keyword match';
    const warnings = (plan.warnings || []).map((w) =>
      '<div class="msg warn" style="display:block; margin-bottom:6px">&#9888; ' + esc(w) + '</div>'
    ).join('');
    $('nlPlanOut').innerHTML =
      warnings +
      '<div class="msg ok" style="display:block">' +
      '<strong>Predict:</strong> ' + esc((plan.predict || []).join(', ') || '(none found)') + '<br>' +
      '<strong>Task:</strong> ' + esc(plan.task) + '<br>' +
      (plan.exclude && plan.exclude.length ? '<strong>Exclude:</strong> ' + esc(plan.exclude.join(', ')) + '<br>' : '') +
      (plan.datetime_col && plan.datetime_col.length ? '<strong>Datetime column:</strong> ' + esc(plan.datetime_col.join(', ')) + '<br>' : '') +
      '<div class="hint" style="margin:6px 0 0">' + esc(plan.explanation) + ' (' + sourceNote + ')</div>' +
      '<div class="row" style="margin-top:10px">' +
      '<button type="button" class="sm" id="nlApply">Use this plan</button>' +
      '<span class="hint" style="margin:0">Applies to Task &amp; Target above &mdash; review, then press Run.</span>' +
      '</div></div>';
    $('nlApply').addEventListener('click', () => applyNlPlan(plan));
  }

  function setColumnSelect(id, values) {
    const sel = $(id);
    if (!sel) return;
    const wanted = new Set(values || []);
    Array.from(sel.options).forEach((o) => { o.selected = wanted.has(o.value); });
    sel.dispatchEvent(new Event('change', { bubbles: true }));
  }

  function applyNlPlan(plan) {
    setGoalMode('predict');
    const rawPred = plan.predict || [];
    const allowedPred = state.predictableSet && state.predictableSet.size
      ? rawPred.filter((c) => state.predictableSet.has(c))
      : rawPred;
    const skipped = rawPred.filter((c) => !allowedPred.includes(c));
    setColumnSelect('f_predict', allowedPred);
    setColumnSelect('f_exclude', plan.exclude);
    setColumnSelect('f_datetime_col', plan.datetime_col);
    if ($('f_task')) {
      $('f_task').value = plan.task || 'auto';
      $('f_task').dispatchEvent(new Event('change', { bubbles: true }));
    }
    if (skipped.length && $('nlPlanStatus')) {
      $('nlPlanStatus').textContent = 'Skipped non-predictable target(s): ' + skipped.join(', ');
    }
    renderTargetPicker();
    refreshCommand();
  }

  /* ── preflight, run, stop, SSE streaming + polling fallback & queue ── */
  function validateRunPreflight(payload) {
    if (state.columnsBlocked) return 'Resolve the column selection conflict above before running.';
    const noFileMode = Boolean(
      payload.all || payload.v3 || payload.policy || payload.timeseries || payload.decision ||
      (payload.benchmark && !state.uploadId)
    );
    if (!state.uploadId && !noFileMode) return 'Upload or select a dataset first (or pick a built-in demo mode).';
    const predict = payload.predict || [];
    const target = payload.target || [];
    if (state.uploadId && !noFileMode && !predict.length && !target.length) {
      return 'Pick at least one target column to predict or train on.';
    }
    const exSet = new Set(payload.exclude || []);
    const overlap = [...predict, ...target].filter((c) => exSet.has(c));
    if (overlap.length) return 'Column conflict: ' + overlap.join(', ') + ' cannot be both target/predict and excluded.';
    if (predict.length && target.length) {
      const pSet = new Set(predict), tSet = new Set(target);
      const same = pSet.size === tSet.size && [...pSet].every((x) => tSet.has(x));
      if (!same) return 'Predict target(s) and Train-only target(s) differ; use the Goal toggle or clear one.';
    }
    if (payload.horizon != null && (!Number.isInteger(payload.horizon) || payload.horizon < 1)) {
      return 'Forecast horizon (--horizon) must be a positive integer (>= 1).';
    }
    if (payload.budget != null && (!Number.isFinite(payload.budget) || payload.budget < 0)) {
      return 'Budget constraint (--budget) must be a finite number >= 0.';
    }
    if (payload.time_budget != null && (!Number.isInteger(payload.time_budget) || payload.time_budget < 1)) {
      return 'Time budget (--time-budget) must be a positive integer (>= 1).';
    }
    return null;
  }

  function resetLogView(initialText) {
    state.logLines = [];
    if ($('log')) $('log').textContent = initialText || '';
    if ($('logCapNotice')) {
      $('logCapNotice').style.display = 'none';
      $('logCapNotice').textContent = '';
    }
  }

  function appendLogChunk(lines) {
    if (!lines || !lines.length) return;
    const logEl = $('log');
    if (!logEl) return;
    if (!state.logLines.length && logEl.textContent === 'Nothing running yet.') logEl.textContent = '';
    for (const l of lines) state.logLines.push(l);
    if (state.logLines.length > MAX_CLIENT_LOG_LINES) {
      state.logLines.splice(0, state.logLines.length - MAX_CLIENT_LOG_LINES);
      logEl.textContent = state.logLines.join('\n') + '\n';
      $('logCapNotice').style.display = 'block';
      $('logCapNotice').textContent = 'Showing last ' + MAX_CLIENT_LOG_LINES.toLocaleString() + ' lines (full log saved on disk).';
    } else {
      logEl.appendChild(document.createTextNode(lines.join('\n') + '\n'));
    }
    if ($('autoscroll') && $('autoscroll').checked) logEl.scrollTop = logEl.scrollHeight;
  }

  function renderProgress(prog, jobStatus) {
    const wrap = $('progressWrap');
    if (!wrap) return;
    const steps = Array.from($('pipelineStepper').querySelectorAll('.step'));
    if (!prog) {
      if (jobStatus === 'running' || jobStatus === 'queued') {
        wrap.style.display = 'block';
        $('progressLabel').textContent = jobStatus === 'queued' ? 'Queued — waiting for active run…' : 'Running pipeline…';
        $('progressPct').textContent = jobStatus === 'queued' ? 'queued' : '…';
        $('progressFill').style.width = jobStatus === 'queued' ? '4%' : '10%';
      } else {
        wrap.style.display = 'none';
      }
      steps.forEach((el) => { el.className = 'step'; });
      return;
    }
    wrap.style.display = 'block';
    const pct = prog.percent != null ? prog.percent : 0;
    $('progressFill').style.width = pct + '%';
    $('progressPct').textContent = pct + '%';
    $('progressLabel').textContent = (prog.name || 'Pipeline') + (prog.detail ? ' — ' + prog.detail : '');

    const nameLower = String(prog.name || '').toLowerCase();
    let activeIdx = -1;
    if (nameLower) {
      activeIdx = steps.findIndex((el) => {
        const tokens = String(el.dataset.match || '').split(',').map((s) => s.trim()).filter(Boolean);
        return tokens.some((tok) => nameLower.includes(tok));
      });
    }
    if (activeIdx < 0 && steps.length > 0) {
      const total = Number(prog.total || 0), phase = Number(prog.phase || 0);
      const ratio = total > 0 ? Math.min(1, Math.max(0, phase / total)) : Math.min(1, Math.max(0, pct / 100));
      activeIdx = Math.min(steps.length - 1, Math.max(0, Math.round(ratio * (steps.length - 1))));
    }

    steps.forEach((el, idx) => {
      el.className = 'step';
      if (jobStatus === 'finished') el.classList.add('done');
      else if (prog.status === 'error' && idx === activeIdx) el.classList.add('error');
      else if (idx < activeIdx || (idx === activeIdx && prog.status === 'done')) el.classList.add('done');
      else if (idx === activeIdx) el.classList.add('running');
    });
  }

  function setStreamBadge(mode) {
    const el = $('streamBadge');
    if (!el) return;
    if (mode === 'sse') {
      el.textContent = 'SSE live';
      el.className = 'stream-badge sse';
      el.style.display = 'inline';
    } else if (mode === 'poll') {
      el.textContent = 'polling';
      el.className = 'stream-badge';
      el.style.display = 'inline';
    } else {
      el.style.display = 'none';
    }
  }

  function teardownTracking() {
    if (state.sse) {
      try { state.sse.close(); } catch (_) {}
      state.sse = null;
    }
    if (state.poll) {
      clearInterval(state.poll);
      state.poll = null;
    }
    setStreamBadge(null);
  }

  async function loadQueue() {
    const box = $('queuePanel');
    if (!box) return;
    const r = await safeFetchJson('/api/queue');
    if (!r.ok || !r.data) return;
    const q = r.data.queue || [];
    if (!q.length) {
      box.style.display = 'none';
      box.innerHTML = '';
      return;
    }
    box.style.display = 'block';
    box.innerHTML =
      '<div class="hint" style="margin:0 0 6px"><strong>Queued Runs (' + q.length + ')</strong> — will start automatically in FIFO order</div>' +
      q.map((item) =>
        '<div class="queue-item"><span><strong>#' + item.queue_position + '</strong> <code class="inline">' + esc(item.job_id) + '</code> ' +
        '<span class="hint" style="margin:0">' + esc(String(item.command || '').slice(0, 80)) + '</span></span>' +
        '<button type="button" class="sm danger cancel-q-btn" data-jid="' + esc(item.job_id) + '">Cancel</button></div>'
      ).join('');
    box.querySelectorAll('.cancel-q-btn').forEach((btn) => {
      btn.addEventListener('click', async () => {
        await safeFetchJson('/api/queue/' + encodeURIComponent(btn.dataset.jid), { method: 'DELETE' });
        loadQueue();
      });
    });
  }

  function handleJobTerminal(data) {
    teardownTracking();
    $('runBtn').disabled = false;
    $('stopBtn').disabled = true;
    $('stopBtn').textContent = 'Stop';
    loadReportHistory();
    loadQueue();
    if (data.status === 'finished' && state.autoInspectedJob !== state.jobId && data.artifacts && data.artifacts.length) {
      const preferred =
        data.artifacts.find((a) => /_combined\.json$/i.test(a.path)) ||
        data.artifacts.find((a) => /^audit_.*\.json$/i.test(a.name || a.path)) ||
        data.artifacts.find((a) => /^predictions_.*\.csv$/i.test(a.name || a.path));
      if (preferred) {
        state.autoInspectedJob = state.jobId;
        inspectReport(preferred.path);
      }
    }
    // Check if a queued job automatically started
    safeFetchJson('/api/health').then((hr) => {
      if (hr.ok && hr.data && hr.data.active_job && hr.data.active_job !== state.jobId) {
        startJobTracking(hr.data.active_job);
      }
    });
  }

  function startPollingFallback() {
    if (state.poll) return;
    setStreamBadge('poll');
    poll();
    state.poll = setInterval(poll, 1200);
  }

  function startJobTracking(jobId, command) {
    teardownTracking();
    state.jobId = jobId;
    state.cursor = 0;
    state.pollInFlight = false;
    state.pollFailures = 0;
    state.autoInspectedJob = null;
    resetLogView('');
    if (command && $('cmd')) $('cmd').textContent = command;
    $('runBtn').disabled = !($('queueIfBusy') && $('queueIfBusy').checked);
    $('stopBtn').disabled = false;
    $('stopBtn').textContent = 'Stop';
    setStatus('running');
    renderProgress(null, 'running');

    if (typeof globalScope.EventSource === 'function') {
      try {
        const es = new globalScope.EventSource('/api/job/' + encodeURIComponent(jobId) + '/stream?cursor=' + state.cursor);
        state.sse = es;
        setStreamBadge('sse');

        es.addEventListener('log', (ev) => {
          if (jobId !== state.jobId) return;
          try {
            const payload = JSON.parse(ev.data);
            if (payload.lines && payload.lines.length) {
              const nextC = typeof payload.cursor === 'number' ? payload.cursor : (state.cursor + payload.lines.length);
              if (nextC > state.cursor) {
                appendLogChunk(payload.lines);
                state.cursor = nextC;
              }
            }
          } catch (_) {}
        });

        es.addEventListener('progress', (ev) => {
          if (jobId !== state.jobId) return;
          try {
            const payload = JSON.parse(ev.data);
            renderProgress(payload.progress, payload.status || 'running');
          } catch (_) {}
        });

        es.addEventListener('status', (ev) => {
          if (jobId !== state.jobId) return;
          try {
            const payload = JSON.parse(ev.data);
            if (payload.command && $('cmd')) $('cmd').textContent = payload.command;
            if (payload.elapsed != null) $('elapsed').textContent = payload.elapsed + 's elapsed';
            if (payload.log_file) $('logfile').textContent = 'log: ' + payload.log_file;
            setStatus(payload.status, payload.returncode);
          } catch (_) {}
        });

        es.addEventListener('done', (ev) => {
          if (jobId !== state.jobId) return;
          try {
            const payload = JSON.parse(ev.data);
            if (payload.elapsed != null) $('elapsed').textContent = payload.elapsed + 's elapsed';
            if (payload.log_file) $('logfile').textContent = 'log: ' + payload.log_file;
            setStatus(payload.status, payload.returncode);
            renderProgress(payload.progress, payload.status);
            renderArtifacts(payload.artifacts);
            handleJobTerminal(payload);
          } catch (_) {
            startPollingFallback();
          }
        });

        es.onerror = () => {
          if (state.sse === es) {
            try { es.close(); } catch (_) {}
            state.sse = null;
            if (state.jobId === jobId) startPollingFallback();
          }
        };
        return;
      } catch (_) {
        // Fall through to polling
      }
    }
    startPollingFallback();
  }

  async function poll() {
    if (!state.jobId || state.pollInFlight) return;
    const reqJobId = state.jobId;
    const reqCursor = state.cursor;
    state.pollInFlight = true;
    let r;
    try {
      r = await safeFetchJson('/api/job/' + encodeURIComponent(reqJobId) + '?cursor=' + reqCursor);
    } finally {
      state.pollInFlight = false;
    }
    if (reqJobId !== state.jobId) return;
    if (!r.ok) {
      state.pollFailures += 1;
      if (r.status === 404 || state.pollFailures >= 5) {
        teardownTracking();
        $('runBtn').disabled = false;
        $('stopBtn').disabled = true;
        $('stopBtn').textContent = 'Stop';
        setStatus('failed');
        flash('Job polling stopped: ' + r.error, 'err', 'runMsg');
      }
      return;
    }
    state.pollFailures = 0;
    const data = r.data;
    if (data.command && $('cmd')) $('cmd').textContent = data.command;
    if (data.lines && data.lines.length && reqCursor === state.cursor) {
      appendLogChunk(data.lines);
      if (typeof data.cursor === 'number' && data.cursor > state.cursor) state.cursor = data.cursor;
      else state.cursor += data.lines.length;
    }
    $('elapsed').textContent = data.elapsed + 's elapsed';
    $('logfile').textContent = 'log: ' + data.log_file;
    setStatus(data.status, data.returncode);
    renderProgress(data.progress, data.status);
    renderArtifacts(data.artifacts);

    if (data.status !== 'running' && data.status !== 'queued') {
      handleJobTerminal(data);
    }
  }

  if ($('jumpLatestBtn')) $('jumpLatestBtn').addEventListener('click', () => { $('log').scrollTop = $('log').scrollHeight; });
  if ($('clearLogBtn')) $('clearLogBtn').addEventListener('click', () => { resetLogView(''); });
  if ($('queueIfBusy')) {
    $('queueIfBusy').addEventListener('change', () => {
      if (state.sse || state.poll) {
        $('runBtn').disabled = !$('queueIfBusy').checked;
      }
    });
  }

  if ($('runBtn')) {
    $('runBtn').addEventListener('click', async () => {
      flash('', 'err', 'runMsg');
      const payload = collectPayload();
      const errMsg = validateRunPreflight(payload);
      if (errMsg) { flash(errMsg, 'err', 'runMsg'); return; }
      $('runBtn').disabled = true;
      $('runBtn').textContent = 'Starting…';
      const r = await safeFetchJson('/api/run', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      $('runBtn').textContent = 'Run pipeline';
      if (!r.ok) {
        $('runBtn').disabled = false;
        flash(r.error || 'Could not start the run', 'err', 'runMsg');
        if (r.data && r.data.column_issues) renderColumnCheck(r.data.column_issues);
        return;
      }
      if (r.status === 202 || (r.data && r.data.status === 'queued')) {
        $('runBtn').disabled = !($('queueIfBusy') && $('queueIfBusy').checked);
        flash('Enqueued job #' + r.data.queue_position + ' (' + r.data.job_id + '). Will run automatically.', 'ok', 'runMsg');
        loadQueue();
        return;
      }
      startJobTracking(r.data.job_id, r.data.command);
      loadQueue();
    });
  }

  if ($('stopBtn')) {
    $('stopBtn').addEventListener('click', async () => {
      if (!state.jobId) return;
      $('stopBtn').disabled = true;
      $('stopBtn').textContent = 'Stopping…';
      const r = await safeFetchJson('/api/job/' + encodeURIComponent(state.jobId) + '/stop', { method: 'POST' });
      if (!r.ok) {
        flash('Stop failed: ' + r.error, 'err', 'runMsg');
        $('stopBtn').disabled = false;
        $('stopBtn').textContent = 'Stop';
        return;
      }
      poll();
    });
  }

  function setStatus(status, code) {
    const el = $('status');
    if (!el) return;
    el.style.display = 'inline';
    el.className = 'status ' + status;
    el.textContent = status + (code != null && status !== 'running' && status !== 'queued' ? ' (exit ' + code + ')' : '');
  }

  /* ── artifacts & 5-tab results inspector with visual analytics ─────── */
  const INSPECTABLE_KINDS = new Set(['json', 'csv']);
  const VIEWABLE_KINDS = new Set(['html', 'json', 'csv', 'txt', 'md', 'png', 'svg']);

  function renderArtifacts(items) {
    const box = $('artifacts');
    if (!box) return;
    if (!items || !items.length) { box.textContent = 'None yet.'; return; }
    box.innerHTML = '<div class="scroll-x"><table><thead><tr><th>file</th><th>size</th><th></th></tr></thead><tbody>' +
      items.map((a) => {
        const q = encodeURIComponent(a.path);
        const inspect = INSPECTABLE_KINDS.has(a.kind)
          ? '<button type="button" class="sm inspect-btn" data-path="' + esc(a.path) + '">Inspect</button> '
          : '';
        const view = VIEWABLE_KINDS.has(a.kind)
          ? '<a href="/api/artifact?path=' + q + '&inline=1" target="_blank">view</a> ' : '';
        return '<tr><td>' + esc(a.path) + '</td><td>' + fmtSize(a.size) + '</td>' +
               '<td>' + inspect + view + '<a href="/api/artifact?path=' + q + '">download</a></td></tr>';
      }).join('') + '</tbody></table></div>';
    box.querySelectorAll('.inspect-btn').forEach((b) => {
      b.addEventListener('click', () => inspectReport(b.dataset.path));
    });
  }

  async function inspectReport(path) {
    if (!path) return;
    flash('', 'err', 'inspectorMsg');
    $('inspectorFile').textContent = path;
    $('inspectorBody').innerHTML = 'Loading report summary…';
    const r = await safeFetchJson('/api/report-summary?path=' + encodeURIComponent(path) + '&limit=50');
    if (!r.ok) {
      flash('Could not inspect report: ' + r.error, 'err', 'inspectorMsg');
      $('inspectorBody').textContent = 'Unable to inspect "' + path + '".';
      return;
    }
    state.inspectorData = r.data;
    if (r.data.kind === 'csv') state.inspectorTab = 'predictions';
    else if (r.data.kind === 'audit') state.inspectorTab = 'models';
    else state.inspectorTab = 'overview';
    renderInspector();
  }

  function setInspectorTab(tab) {
    state.inspectorTab = tab;
    document.querySelectorAll('#reportInspector .tab-btn').forEach((btn) => {
      btn.setAttribute('aria-selected', btn.dataset.tab === tab ? 'true' : 'false');
    });
    renderInspector();
  }

  document.querySelectorAll('#reportInspector .tab-btn').forEach((btn) => {
    btn.addEventListener('click', () => setInspectorTab(btn.dataset.tab));
  });

  function wireParetoInteractions(body, cands, paretoNames, selName) {
    body.querySelectorAll('.axis-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        state.paretoAxis = btn.dataset.axis;
        renderInspector();
      });
    });
    const readout = $('paretoReadout');
    if (readout && Charts.formatParetoReadout) {
      body.querySelectorAll('.pareto-point').forEach((pt) => {
        const update = () => {
          const idx = Number(pt.dataset.pidx);
          const c = cands[idx];
          if (!c) return;
          const name = c.model_name || c.name || '';
          readout.innerHTML = Charts.formatParetoReadout(c, name, paretoNames.has(name), name === selName);
        };
        pt.addEventListener('mouseenter', update);
        pt.addEventListener('focus', update);
        pt.addEventListener('click', update);
      });
    }
  }

  function renderInspector() {
    document.querySelectorAll('#reportInspector .tab-btn').forEach((btn) => {
      btn.setAttribute('aria-selected', btn.dataset.tab === state.inspectorTab ? 'true' : 'false');
    });
    const d = state.inspectorData;
    const body = $('inspectorBody');
    if (!body) return;
    if (!d) {
      body.innerHTML = 'Select <strong>Inspect</strong> on any JSON or CSV report below.';
      return;
    }
    const tab = state.inspectorTab;
    const statusEl = $('tableSortStatus');

    if (d.kind === 'csv') {
      if (tab !== 'predictions') {
        body.innerHTML = '<div class="hint">CSV reports show row previews under the <strong>Predictions Preview</strong> tab.</div>';
        return;
      }
      const cols = d.columns || [];
      const rows = d.rows || [];
      const head = '<thead><tr>' + cols.map((c) => '<th scope="col">' + esc(c) + '</th>').join('') + '</tr></thead>';
      const trs = '<tbody>' + rows.map((r) =>
        '<tr>' + cols.map((c) => {
          const hl = /_predicted$|_probability$|^type$/i.test(c) ? ' style="background:rgba(91,157,255,.12); font-weight:600"' : '';
          return '<td' + hl + '>' + esc(r[c] == null ? '' : r[c]) + '</td>';
        }).join('') + '</tr>'
      ).join('') + '</tbody>';
      body.innerHTML =
        '<div class="row" style="justify-content:space-between; margin-bottom:8px">' +
          '<span class="hint" style="margin:0">Showing ' + rows.length + ' of ' + Number(d.total_rows || 0).toLocaleString() + ' rows</span>' +
          '<a class="btn sm" href="/api/artifact?path=' + encodeURIComponent(d.path) + '">Download full CSV</a>' +
        '</div>' +
        '<div class="scroll-x" tabindex="0" role="region" aria-label="CSV predictions preview" style="max-height:320px; overflow-y:auto"><table>' + head + trs + '</table></div>';
      return;
    }

    if (tab === 'overview') {
      const sel = d.selected_model || {};
      const metrics = sel.metrics || {};
      const tp = d.test_performance || {};
      const sum = d.summary || {};
      const ds = d.dataset_info || {};
      const phases = d.phases || [];
      const targetReports = d.target_reports && typeof d.target_reports === 'object' ? d.target_reports : null;
      const mtRows = targetReports
        ? Object.entries(targetReports).map(([tname, tr]) => {
            const ts = (tr && tr.selected_model) || {};
            const tm = ts.metrics || {};
            const sc = tm.metric != null ? tm.metric : ts.metric;
            const tt = tm.train_time_s != null ? tm.train_time_s : ts.train_time_s;
            return '<tr><td><strong>' + esc(tname) + '</strong></td><td>' +
              esc(ts.name || ts.model_name || 'None') + '</td><td>' +
              fmtNum(sc, 4) + '</td><td>' +
              (tt != null && Number.isFinite(Number(tt)) ? fmtNum(tt, 2) + 's' : 'n/a') +
              '</td></tr>';
          }).join('')
        : '';
      const dsRowsNum = ds.rows != null ? Number(ds.rows) : NaN;
      body.innerHTML =
        '<div class="kpi-grid">' +
          '<div class="kpi-card"><div class="k-label">Selected Model</div><div class="k-val">' + esc(sel.name || sel.model_name || 'None') + '</div></div>' +
          '<div class="kpi-card"><div class="k-label">CV Score</div><div class="k-val">' + (metrics.metric != null ? fmtNum(metrics.metric, 4) : fmtNum(sel.metric, 4)) + '</div></div>' +
          '<div class="kpi-card"><div class="k-label">Train Time</div><div class="k-val">' + (metrics.train_time_s != null && Number.isFinite(Number(metrics.train_time_s)) ? fmtNum(metrics.train_time_s, 2) + 's' : (sum.total_time != null && Number.isFinite(Number(sum.total_time)) ? fmtNum(sum.total_time, 1) + 's' : 'n/a')) + '</div></div>' +
          '<div class="kpi-card"><div class="k-label">Peak Memory</div><div class="k-val">' + (metrics.peak_mem_mb != null && Number.isFinite(Number(metrics.peak_mem_mb)) ? fmtNum(metrics.peak_mem_mb, 1) + ' MB' : 'n/a') + '</div></div>' +
          '<div class="kpi-card"><div class="k-label">Dataset</div><div class="k-val">' + (Number.isFinite(dsRowsNum) ? dsRowsNum.toLocaleString() + ' x ' + esc(ds.columns || '?') : 'n/a') + '</div></div>' +
          '<div class="kpi-card"><div class="k-label">Phases OK</div><div class="k-val">' + (sum.successful != null ? esc(sum.successful + '/' + sum.total_phases) : (phases.length || 'n/a')) + '</div></div>' +
        '</div>' +
        (mtRows
          ? '<div class="scroll-x" style="max-height:220px; overflow-y:auto; margin-bottom:8px"><table><thead><tr><th>Target</th><th>Selected Model</th><th>CV Score</th><th>Train (s)</th></tr></thead><tbody>' +
            mtRows + '</tbody></table></div>'
          : '') +
        (Object.keys(tp).length
          ? '<div class="row" style="gap:8px; margin:8px 0">' +
            '<span class="hint" style="margin:0"><strong>Test performance:</strong></span>' +
            Object.entries(tp).map(([k, v]) => {
              const valStr = typeof v === 'number' && Number.isFinite(v) ? v.toFixed(4) : String(v);
              return '<span class="pill">' + esc(k.toUpperCase()) + ': <strong>' + esc(valStr) + '</strong></span>';
            }).join('') + '</div>'
          : '') +
        (phases.length
          ? '<div class="scroll-x" tabindex="0" role="region" aria-label="Pipeline phases" style="max-height:220px; overflow-y:auto"><table><thead><tr><th>Phase</th><th>Status</th><th>Time (s)</th></tr></thead><tbody>' +
            phases.map((p) => '<tr><td>' + esc(p.name || '') + '</td><td>' + esc(p.status || '') + '</td><td>' + (p.time != null && Number.isFinite(Number(p.time)) ? fmtNum(p.time, 2) : '') + '</td></tr>').join('') +
            '</tbody></table></div>'
          : '');
    } else if (tab === 'models') {
      const cands = d.candidates_evaluated || [];
      const paretoNames = new Set((d.pareto_front || []).map((p) => (typeof p === 'string' ? p : (p.model_name || p.name || ''))));
      const selName = (d.selected_model && (d.selected_model.name || d.selected_model.model_name)) || '';
      if (!cands.length) {
        body.innerHTML = '<div class="hint">No candidate model evaluation table in this report.</div>';
        return;
      }
      const paretoPlot = Charts.renderParetoPlotHtml
        ? Charts.renderParetoPlotHtml(cands, d.pareto_front || [], selName, state.paretoAxis)
        : '';
      body.innerHTML = paretoPlot +
        '<div class="scroll-x" tabindex="0" role="region" aria-label="Candidate models leaderboard" style="max-height:300px; overflow-y:auto">' +
        '<table id="modelsLeaderboard">' +
        '<thead><tr>' +
          '<th scope="col" data-type="text"><button type="button" class="sort-btn">Model<span class="sort-icon" aria-hidden="true"></span></button></th>' +
          '<th scope="col" data-type="number"><button type="button" class="sort-btn">Score<span class="sort-icon" aria-hidden="true"></span></button></th>' +
          '<th scope="col" data-type="number"><button type="button" class="sort-btn">Train (s)<span class="sort-icon" aria-hidden="true"></span></button></th>' +
          '<th scope="col" data-type="number"><button type="button" class="sort-btn">Mem (MB)<span class="sort-icon" aria-hidden="true"></span></button></th>' +
          '<th scope="col" data-type="number"><button type="button" class="sort-btn">Risk<span class="sort-icon" aria-hidden="true"></span></button></th>' +
          '<th scope="col">Flags</th>' +
        '</tr></thead><tbody>' +
        cands.map((c) => {
          const name = c.model_name || c.name || '';
          const badges = [
            name === selName ? '<span class="badge badge-cat">selected</span>' : '',
            paretoNames.has(name) ? '<span class="badge badge-num">pareto</span>' : '',
          ].filter(Boolean).join(' ');
          return '<tr><td data-value="' + esc(name) + '"><strong>' + esc(name) + '</strong></td>' +
            '<td data-value="' + esc(c.metric == null ? '' : c.metric) + '">' + fmtNum(c.metric, 4) + '</td>' +
            '<td data-value="' + esc(c.train_time_s == null ? '' : c.train_time_s) + '">' + fmtNum(c.train_time_s, 2) + '</td>' +
            '<td data-value="' + esc(c.peak_mem_mb == null ? '' : c.peak_mem_mb) + '">' + fmtNum(c.peak_mem_mb, 1) + '</td>' +
            '<td data-value="' + esc(c.risk_penalty == null ? '' : c.risk_penalty) + '">' + fmtNum(c.risk_penalty, 3) + '</td>' +
            '<td>' + badges + '</td></tr>';
        }).join('') + '</tbody></table></div>';
      wireParetoInteractions(body, cands, paretoNames, selName);
      if (Charts.enhanceSortableTable) Charts.enhanceSortableTable($('modelsLeaderboard'), statusEl);
    } else if (tab === 'risk') {
      const rr = d.risk_report || {};
      if (!Object.keys(rr).length) {
        body.innerHTML = '<div class="hint">No risk_report section in this report.</div>';
        return;
      }
      const notes = rr.notes || [];
      const missPct = rr.missing_ratio != null && Number.isFinite(Number(rr.missing_ratio)) ? (Number(rr.missing_ratio) * 100).toFixed(1) + '%' : 'n/a';
      body.innerHTML =
        '<div class="kpi-grid">' +
          '<div class="kpi-card"><div class="k-label">Task</div><div class="k-val">' + esc(rr.task || 'n/a') + '</div></div>' +
          '<div class="kpi-card"><div class="k-label">Missing Ratio</div><div class="k-val">' + missPct + '</div></div>' +
          '<div class="kpi-card"><div class="k-label">Max VIF</div><div class="k-val">' + fmtNum(rr.vif_max, 2) + '</div></div>' +
          '<div class="kpi-card"><div class="k-label">Leakage Score</div><div class="k-val">' + fmtNum(rr.leakage_proxy_score_max, 3) + '</div></div>' +
        '</div>' +
        (notes.length
          ? '<ul style="margin:6px 0 0 18px">' + notes.map((n) => '<li>' + esc(n) + '</li>').join('') + '</ul>'
          : '<div class="hint">No risk warnings recorded.</div>');
    } else if (tab === 'eda') {
      const eda = d.eda_report || {};
      const htmlPath = d.meta && d.meta.html_report_path ? d.meta.html_report_path.split(/[\\/]/).pop() : '';
      const edaRowsNum = eda.n_rows != null ? Number(eda.n_rows) : NaN;
      const fiChart = Charts.renderFeatureImportanceHtml ? Charts.renderFeatureImportanceHtml(d.feature_importance) : '';
      body.innerHTML =
        '<div class="kpi-grid">' +
          '<div class="kpi-card"><div class="k-label">Quality Score</div><div class="k-val">' + (eda.quality_score != null && Number.isFinite(Number(eda.quality_score)) ? fmtNum(eda.quality_score, 1) + '/100' : 'n/a') + '</div></div>' +
          '<div class="kpi-card"><div class="k-label">Rows</div><div class="k-val">' + (Number.isFinite(edaRowsNum) ? edaRowsNum.toLocaleString() : 'n/a') + '</div></div>' +
          '<div class="kpi-card"><div class="k-label">Columns</div><div class="k-val">' + (eda.n_cols != null ? esc(eda.n_cols) : 'n/a') + '</div></div>' +
        '</div>' +
        fiChart +
        (htmlPath ? '<a class="btn sm" href="/api/artifact?path=' + encodeURIComponent(htmlPath) + '&inline=1" target="_blank">Open full HTML report</a>' : '');
    } else if (tab === 'predictions') {
      const pred = d.predictions;
      const task = (d.meta && d.meta.task) || (d.risk_report && d.risk_report.task) || '';
      if (!pred) {
        body.innerHTML = '<div class="hint">Inspect a <code class="inline">predictions_*.csv</code> or <code class="inline">forecast_*.csv</code> file in Report History to preview predicted rows.</div>';
      } else if (typeof pred === 'object' && !Array.isArray(pred)) {
        const stats = pred.statistics && typeof pred.statistics === 'object' ? pred.statistics : {};
        const samples = Array.isArray(pred.sample_values) ? pred.sample_values : [];
        const kpis = [
          pred.total_predictions != null
            ? '<div class="kpi-card"><div class="k-label">Predictions</div><div class="k-val">' + Number(pred.total_predictions).toLocaleString() + '</div></div>'
            : '',
          stats.mean != null ? '<div class="kpi-card"><div class="k-label">Mean</div><div class="k-val">' + fmtNum(stats.mean, 4) + '</div></div>' : '',
          stats.std != null ? '<div class="kpi-card"><div class="k-label">Std Dev</div><div class="k-val">' + fmtNum(stats.std, 4) + '</div></div>' : '',
          stats.min != null && stats.max != null
            ? '<div class="kpi-card"><div class="k-label">Range</div><div class="k-val">' + fmtNum(stats.min, 3) + ' – ' + fmtNum(stats.max, 3) + '</div></div>'
            : '',
        ].filter(Boolean).join('');
        const diagChart = Charts.renderPredictionDiagnosticsHtml ? Charts.renderPredictionDiagnosticsHtml(pred, task) : '';
        const sampleCols = samples.length && typeof samples[0] === 'object' ? Object.keys(samples[0]) : [];
        const sampleTable = sampleCols.length
          ? '<div class="scroll-x" tabindex="0" role="region" aria-label="Sample predictions" style="max-height:240px; overflow-y:auto"><table><thead><tr>' +
            sampleCols.map((c) => '<th scope="col">' + esc(c) + '</th>').join('') + '</tr></thead><tbody>' +
            samples.map((row) => '<tr>' + sampleCols.map((c) => {
              const v = row[c];
              const fmt = typeof v === 'number' && Number.isFinite(v) && c !== 'row' ? v.toFixed(4) : (v == null ? '' : String(v));
              return '<td>' + esc(fmt) + '</td>';
            }).join('') + '</tr>').join('') + '</tbody></table></div>'
          : '<div class="hint">' + esc(JSON.stringify(pred, null, 2)) + '</div>';
        body.innerHTML = (kpis ? '<div class="kpi-grid">' + kpis + '</div>' : '') + diagChart + sampleTable;
      } else {
        body.innerHTML = '<div class="hint">' + esc(JSON.stringify(pred, null, 2)) + '</div>';
      }
    }
  }

  /* ── report history + side-by-side comparison ──────────────────────── */
  let reportCache = [];

  function fmtDuration(seconds) {
    if (!seconds || seconds < 1) return '<1s';
    const m = Math.floor(seconds / 60), s = Math.round(seconds % 60);
    return m ? m + 'm ' + s + 's' : s + 's';
  }

  function updateCompareButton() {
    const btn = $('compareRunsBtn');
    if (!btn) return;
    const count = state.selectedCompare.size;
    btn.textContent = 'Compare selected (' + count + ')';
    btn.disabled = count < 2 || count > 4;
  }

  function reportRow(a) {
    const q = encodeURIComponent(a.path);
    const isJson = a.kind === 'json';
    const cmpBox = isJson
      ? '<input type="checkbox" class="cmp-check" data-path="' + esc(a.path) + '"' +
        (state.selectedCompare.has(a.path) ? ' checked' : '') + ' aria-label="Select ' + esc(a.path) + ' for comparison">'
      : '';
    const inspect = INSPECTABLE_KINDS.has(a.kind)
      ? '<button type="button" class="sm inspect-btn" data-path="' + esc(a.path) + '">Inspect</button> '
      : '';
    const view = VIEWABLE_KINDS.has(a.kind)
      ? '<a href="/api/artifact?path=' + q + '&inline=1" target="_blank">view</a> ' : '';
    const when = new Date(a.modified * 1000).toLocaleString();
    const start = new Date(a.run_start * 1000).toLocaleTimeString();
    const end = new Date(a.run_end * 1000).toLocaleTimeString();
    const runtime = '<span title="' + esc(start + ' -> ' + end) + '">' + fmtDuration(a.run_seconds) + '</span>';
    return '<tr><td>' + cmpBox + '</td>' +
           '<td data-value="' + esc(a.path) + '">' + esc(a.path) + '</td>' +
           '<td data-value="' + esc(a.modified) + '" class="hint" style="margin:0">' + when + '</td>' +
           '<td data-value="' + esc(a.run_seconds || 0) + '">' + runtime + '</td>' +
           '<td data-value="' + esc(a.size || 0) + '">' + fmtSize(a.size) + '</td>' +
           '<td>' + inspect + view + '<a href="/api/artifact?path=' + q + '">download</a> ' +
           '<a href="#" class="report-delete link-danger" data-path="' + esc(a.path) + '">delete</a></td></tr>';
  }

  function renderReportHistory() {
    const box = $('reportHistory');
    if (!box) return;
    const term = ($('reportFilter').value || '').trim().toLowerCase();
    const items = term ? reportCache.filter((a) => a.path.toLowerCase().includes(term)) : reportCache;
    if (!items.length) { box.textContent = term ? 'No files match "' + term + '".' : 'No reports yet.'; return; }
    box.innerHTML = '<div class="scroll-x" tabindex="0" role="region" aria-label="Report history" style="max-height:360px; overflow-y:auto">' +
      '<table id="reportHistoryTable"><thead><tr>' +
        '<th scope="col" title="Compare 2-4 JSON runs">Cmp</th>' +
        '<th scope="col" data-type="text"><button type="button" class="sort-btn">file<span class="sort-icon" aria-hidden="true"></span></button></th>' +
        '<th scope="col" data-type="number"><button type="button" class="sort-btn">modified<span class="sort-icon" aria-hidden="true"></span></button></th>' +
        '<th scope="col" data-type="number"><button type="button" class="sort-btn">runtime<span class="sort-icon" aria-hidden="true"></span></button></th>' +
        '<th scope="col" data-type="number"><button type="button" class="sort-btn">size<span class="sort-icon" aria-hidden="true"></span></button></th>' +
        '<th scope="col"></th>' +
      '</tr></thead><tbody>' +
      items.map(reportRow).join('') + '</tbody></table></div>';
    box.querySelectorAll('.inspect-btn').forEach((b) => {
      b.addEventListener('click', () => inspectReport(b.dataset.path));
    });
    box.querySelectorAll('.report-delete').forEach((el) => {
      el.addEventListener('click', (e) => { e.preventDefault(); deleteReport(el.dataset.path); });
    });
    box.querySelectorAll('.cmp-check').forEach((cb) => {
      cb.addEventListener('change', () => {
        if (cb.checked) {
          if (state.selectedCompare.size >= 4) {
            cb.checked = false;
            flash('Compare supports up to 4 JSON reports at once.', 'warn', 'reportMsg');
            return;
          }
          state.selectedCompare.add(cb.dataset.path);
        } else {
          state.selectedCompare.delete(cb.dataset.path);
        }
        updateCompareButton();
      });
    });
    if (Charts.enhanceSortableTable) Charts.enhanceSortableTable($('reportHistoryTable'), $('tableSortStatus'));
    updateCompareButton();
  }

  async function runCompareReports() {
    const paths = Array.from(state.selectedCompare);
    if (paths.length < 2) return;
    const card = $('compareCard');
    const out = $('compareBody');
    if (card) card.style.display = 'block';
    if (out) out.innerHTML = 'Comparing selected runs…';
    const r = await safeFetchJson('/api/compare-reports', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ paths }),
    });
    if (!r.ok) {
      flash(r.error || 'Could not compare reports', 'err', 'reportMsg');
      if (out) out.textContent = r.error || 'Comparison failed.';
      return;
    }
    if (out && Charts.renderRunComparisonHtml) {
      out.innerHTML = Charts.renderRunComparisonHtml(r.data);
    }
  }

  if ($('compareRunsBtn')) $('compareRunsBtn').addEventListener('click', runCompareReports);
  if ($('closeCompareBtn')) {
    $('closeCompareBtn').addEventListener('click', () => {
      if ($('compareCard')) $('compareCard').style.display = 'none';
    });
  }

  async function loadReportHistory() {
    flash('', 'err', 'reportMsg');
    const r = await safeFetchJson('/api/reports?limit=500');
    if (!r.ok) {
      flash('Could not load report history: ' + r.error, 'err', 'reportMsg');
      if ($('reportHistory')) $('reportHistory').textContent = 'Could not load report history.';
      return;
    }
    reportCache = (r.data && r.data.reports) || [];
    const existingPaths = new Set(reportCache.map((x) => x.path));
    for (const p of Array.from(state.selectedCompare)) {
      if (!existingPaths.has(p)) state.selectedCompare.delete(p);
    }
    renderReportHistory();
    if (r.data && r.data.total > r.data.shown) {
      $('reportHistory').insertAdjacentHTML(
        'beforeend',
        '<div class="hint" style="margin-top:8px">Showing ' + r.data.shown + ' of ' + r.data.total +
        ' files &mdash; narrow with the filter box to find older ones.</div>'
      );
    }
  }

  async function deleteReport(path) {
    if (!confirm('Delete "' + path + '"? This cannot be undone.')) return;
    const r = await safeFetchJson('/api/reports?path=' + encodeURIComponent(path), { method: 'DELETE' });
    if (!r.ok) { flash(r.error || 'Delete failed', 'err', 'reportMsg'); return; }
    state.selectedCompare.delete(path);
    await loadReportHistory();
  }

  async function deleteAllReports() {
    if (!confirm('Delete ALL ' + reportCache.length + ' report files? This cannot be undone.')) return;
    const r = await safeFetchJson('/api/reports?all=1', { method: 'DELETE' });
    if (!r.ok) { flash(r.error || 'Delete failed', 'err', 'reportMsg'); return; }
    state.selectedCompare.clear();
    await loadReportHistory();
  }

  if ($('reportFilter')) $('reportFilter').addEventListener('input', renderReportHistory);
  if ($('reportRefresh')) $('reportRefresh').addEventListener('click', loadReportHistory);
  if ($('reportDeleteAll')) $('reportDeleteAll').addEventListener('click', deleteAllReports);

  /* ── workbench initialization & active-job reconnect ───────────────── */
  async function initWorkbench() {
    refreshCommand();
    const [healthRes] = await Promise.all([
      safeFetchJson('/api/health'),
      loadRecentUploads(),
      loadReportHistory(),
      loadQueue(),
    ]);
    if (healthRes.ok && healthRes.data && healthRes.data.active_job) {
      startJobTracking(healthRes.data.active_job);
    }
  }

  // Expose testable functions for Node VM tests & inspection
  globalScope.state = state;
  globalScope.renderInspector = renderInspector;
  globalScope.renderForecast = renderForecast;

  initWorkbench();
})(typeof globalThis !== 'undefined' ? globalThis : window);
