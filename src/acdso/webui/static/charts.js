/* ACDSO Workbench — Interactive SVG Visual Analytics & Accessible Table Sorting */
(function (globalScope) {
  'use strict';

  function esc(s) {
    return String(s == null ? '' : s).replace(/[&<>"']/g, (c) => (
      { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]
    ));
  }

  function fmtNum(v, digits) {
    if (v == null || v === '') return 'n/a';
    const n = Number(v);
    return Number.isFinite(n) ? n.toFixed(digits) : 'n/a';
  }

  function fmtValue(n, col) {
    if (n == null || Number.isNaN(Number(n))) return '-';
    const num = Number(n);
    if (/\(B\)|bytes?\b/i.test(col || '')) {
      return (num / 1e9).toLocaleString(undefined, { maximumFractionDigits: 1 }) + ' GB';
    }
    if (Math.abs(num) >= 1e9) return (num / 1e9).toFixed(2) + 'B';
    if (Math.abs(num) >= 1e6) return (num / 1e6).toFixed(2) + 'M';
    if (Math.abs(num) >= 1e3) return (num / 1e3).toFixed(1) + 'K';
    return num.toFixed(2);
  }

  /* ── 1. Period Forecast SVG Chart (collision-safe labels) ───────────── */
  function renderForecastChart(d) {
    const hist = d.history || [];
    const fc = d.forecast || [];
    const all = hist.concat(fc);
    if (all.length < 2) return '';

    const W = 640, H = 190, padL = 62, padR = 16, padT = 16, padB = 28;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;
    const vals = all.map((p) => Number(p.value));
    let minV = Math.min(...vals);
    let maxV = Math.max(...vals);
    if (maxV === minV) { maxV = minV + 1; minV = Math.max(0, minV - 1); }

    const xAt = (i) => padL + (i / Math.max(all.length - 1, 1)) * plotW;
    const yAt = (v) => padT + (1 - (v - minV) / (maxV - minV)) * plotH;

    const histPts = hist.map((p, i) => xAt(i).toFixed(1) + ',' + yAt(p.value).toFixed(1)).join(' ');
    const fcStart = Math.max(hist.length - 1, 0);
    const fcSeries = (hist.length ? [hist[hist.length - 1]] : []).concat(fc);
    const fcPts = fcSeries.map((p, i) => xAt(fcStart + i).toFixed(1) + ',' + yAt(p.value).toFixed(1)).join(' ');
    const splitX = xAt(fcStart).toFixed(1);
    const splitNum = Number(splitX);

    const dotStride = all.length > 60 ? Math.ceil(hist.length / 24) : 1;
    const circles = all.map((p, i) => {
      const isFc = i >= hist.length;
      if (!isFc && dotStride > 1 && i !== 0 && i !== hist.length - 1 && i % dotStride !== 0) {
        return '';
      }
      const color = isFc ? '#4fd18b' : '#5b9dff';
      const r = isFc ? '3.5' : (all.length > 60 ? '2.2' : '3.5');
      return '<circle cx="' + xAt(i).toFixed(1) + '" cy="' + yAt(p.value).toFixed(1) + '" r="' + r + '" fill="' + color + '">' +
        '<title>' + esc(p.period) + ': ' + esc(fmtValue(p.value, d.value_col)) + (isFc ? ' (forecast)' : ' (actual)') + '</title></circle>';
    }).join('');

    const midV = (minV + maxV) / 2;
    const splitClear = hist.length && (splitNum - padL >= 75) && ((W - padR) - splitNum >= 75);
    const midIdx = Math.floor((all.length - 1) / 2);
    const midLabelEl = splitClear
      ? '<text x="' + splitX + '" y="' + (H - 8) + '" fill="#98a1b3" font-size="10" text-anchor="middle">' + esc(hist[hist.length - 1].period) + '</text>'
      : (all.length > 2
        ? '<text x="' + ((padL + (W - padR)) / 2).toFixed(1) + '" y="' + (H - 8) + '" fill="#98a1b3" font-size="10" text-anchor="middle">' + esc(all[midIdx].period) + '</text>'
        : '');

    return '<div class="chart-wrap" style="margin:10px 0">' +
      '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="History and forecast chart">' +
      '<rect x="' + splitX + '" y="' + padT + '" width="' + Math.max(0, W - padR - splitNum).toFixed(1) + '" height="' + plotH + '" fill="rgba(79,209,139,0.08)" />' +
      '<line x1="' + padL + '" y1="' + padT + '" x2="' + (W - padR) + '" y2="' + padT + '" stroke="#2a2f3a" stroke-dasharray="3,3" />' +
      '<line x1="' + padL + '" y1="' + (padT + plotH / 2).toFixed(1) + '" x2="' + (W - padR) + '" y2="' + (padT + plotH / 2).toFixed(1) + '" stroke="#2a2f3a" stroke-dasharray="3,3" />' +
      '<line x1="' + padL + '" y1="' + (padT + plotH) + '" x2="' + (W - padR) + '" y2="' + (padT + plotH) + '" stroke="#2a2f3a" />' +
      '<text x="' + (padL - 6) + '" y="' + (padT + 4) + '" fill="#98a1b3" font-size="10" text-anchor="end">' + esc(fmtValue(maxV, d.value_col)) + '</text>' +
      '<text x="' + (padL - 6) + '" y="' + (padT + plotH / 2 + 4).toFixed(1) + '" fill="#98a1b3" font-size="10" text-anchor="end">' + esc(fmtValue(midV, d.value_col)) + '</text>' +
      '<text x="' + (padL - 6) + '" y="' + (padT + plotH + 4) + '" fill="#98a1b3" font-size="10" text-anchor="end">' + esc(fmtValue(minV, d.value_col)) + '</text>' +
      '<text x="' + padL + '" y="' + (H - 8) + '" fill="#98a1b3" font-size="10">' + esc(all[0].period) + '</text>' +
      midLabelEl +
      '<text x="' + (W - padR) + '" y="' + (H - 8) + '" fill="#4fd18b" font-size="10" text-anchor="end">' + esc(all[all.length - 1].period) + '</text>' +
      (hist.length > 1 ? '<polyline fill="none" stroke="#5b9dff" stroke-width="2" points="' + histPts + '" />' : '') +
      (fcSeries.length > 1 ? '<polyline fill="none" stroke="#4fd18b" stroke-width="2" stroke-dasharray="5,4" points="' + fcPts + '" />' : '') +
      circles + '</svg></div>';
  }

  /* ── 2. Interactive Pareto Frontier Scatter Plot ────────────────────── */
  const PARETO_AXES = [
    { key: 'train_time_s', label: 'Train Time (s)', digits: 2 },
    { key: 'risk_penalty', label: 'Risk Penalty', digits: 3 },
    { key: 'peak_mem_mb', label: 'Peak Memory (MB)', digits: 1 },
    { key: 'latency_ms_per_row', label: 'Latency (ms/row)', digits: 3 },
  ];

  function renderParetoPlotHtml(candidates, paretoFront, selectedName, xKey) {
    const axis = PARETO_AXES.find((a) => a.key === xKey) || PARETO_AXES[0];
    const paretoNames = new Set((paretoFront || []).map((p) => (typeof p === 'string' ? p : (p.model_name || p.name || ''))));
    const pts = (candidates || []).map((c, idx) => {
      const name = c.model_name || c.name || ('Model_' + idx);
      const yVal = Number(c.metric);
      const rawX = c[axis.key] != null ? c[axis.key] : (axis.key === 'latency_ms_per_row' ? c.latency_ms : 0);
      const xVal = Number(rawX);
      return {
        idx,
        name,
        x: Number.isFinite(xVal) ? xVal : 0,
        y: Number.isFinite(yVal) ? yVal : NaN,
        isPareto: paretoNames.has(name),
        isSelected: name === selectedName,
        raw: c,
      };
    }).filter((p) => Number.isFinite(p.y));

    if (!pts.length) return '';

    const W = 620, H = 220, padL = 64, padR = 24, padT = 20, padB = 34;
    const plotW = W - padL - padR;
    const plotH = H - padT - padB;

    let minX = Math.min(...pts.map((p) => p.x));
    let maxX = Math.max(...pts.map((p) => p.x));
    if (maxX === minX) { maxX = minX + 1; minX = Math.max(0, minX - 0.1); }
    else { const dx = (maxX - minX) * 0.08; minX = Math.max(0, minX - dx); maxX += dx; }

    let minY = Math.min(...pts.map((p) => p.y));
    let maxY = Math.max(...pts.map((p) => p.y));
    if (maxY === minY) { maxY = minY + 0.1; minY -= 0.1; }
    else { const dy = (maxY - minY) * 0.12; minY -= dy; maxY += dy; }

    const xAt = (v) => padL + ((v - minX) / (maxX - minX)) * plotW;
    const yAt = (v) => padT + (1 - (v - minY) / (maxY - minY)) * plotH;

    const frontPts = pts.filter((p) => p.isPareto).sort((a, b) => a.x - b.x);
    const frontPoly = frontPts.length > 1
      ? '<polyline fill="none" stroke="#5b9dff" stroke-width="1.8" stroke-dasharray="4,3" opacity="0.8" points="' +
        frontPts.map((p) => xAt(p.x).toFixed(1) + ',' + yAt(p.y).toFixed(1)).join(' ') + '" />'
      : '';

    const dots = pts.map((p) => {
      const cx = xAt(p.x).toFixed(1);
      const cy = yAt(p.y).toFixed(1);
      const fill = p.isSelected ? '#4fd18b' : (p.isPareto ? '#5b9dff' : '#6b768c');
      const r = p.isSelected ? 6.5 : (p.isPareto ? 5 : 4);
      const ring = p.isSelected
        ? '<circle cx="' + cx + '" cy="' + cy + '" r="10.5" fill="none" stroke="#4fd18b" stroke-width="1.5" opacity="0.7" />'
        : '';
      const tip = p.name + ' · Score ' + fmtNum(p.y, 4) + ' · ' + axis.label + ' ' + fmtNum(p.x, axis.digits);
      return ring +
        '<circle class="pareto-point" tabindex="0" role="button" aria-label="' + esc(tip) + '" ' +
        'data-pidx="' + p.idx + '" cx="' + cx + '" cy="' + cy + '" r="' + r + '" fill="' + fill + '" stroke="#0b0d12" stroke-width="1.2">' +
        '<title>' + esc(tip) + '</title></circle>';
    }).join('');

    const axisBtns = PARETO_AXES.map((a) =>
      '<button type="button" class="axis-btn' + (a.key === axis.key ? ' active' : '') + '" data-axis="' + esc(a.key) + '">' + esc(a.label) + '</button>'
    ).join('');

    const selPt = pts.find((p) => p.isSelected) || pts[0];
    const defaultReadout = formatParetoReadout(selPt.raw, selPt.name, selPt.isPareto, selPt.isSelected);

    return '<div class="pareto-widget" style="margin-bottom:12px">' +
      '<div class="chart-toolbar">' +
        '<span class="hint" style="margin:0"><strong>Pareto Frontier</strong> (Score vs. ' + esc(axis.label) + ' &mdash; <span style="color:#4fd18b">&#9679;</span> selected, <span style="color:#5b9dff">&#9679;</span> pareto, <span style="color:#6b768c">&#9679;</span> dominated)</span>' +
        '<div class="row" style="gap:4px">' + axisBtns + '</div>' +
      '</div>' +
      '<div class="chart-wrap">' +
        '<svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Pareto frontier scatter plot">' +
          '<line x1="' + padL + '" y1="' + padT + '" x2="' + (W - padR) + '" y2="' + padT + '" stroke="#2a2f3a" stroke-dasharray="3,3" />' +
          '<line x1="' + padL + '" y1="' + (padT + plotH / 2).toFixed(1) + '" x2="' + (W - padR) + '" y2="' + (padT + plotH / 2).toFixed(1) + '" stroke="#2a2f3a" stroke-dasharray="3,3" />' +
          '<line x1="' + padL + '" y1="' + (padT + plotH) + '" x2="' + (W - padR) + '" y2="' + (padT + plotH) + '" stroke="#2a2f3a" />' +
          '<line x1="' + padL + '" y1="' + padT + '" x2="' + padL + '" y2="' + (padT + plotH) + '" stroke="#2a2f3a" />' +
          '<text x="' + (padL - 6) + '" y="' + (padT + 4) + '" fill="#98a1b3" font-size="10" text-anchor="end">' + esc(maxY.toFixed(3)) + '</text>' +
          '<text x="' + (padL - 6) + '" y="' + (padT + plotH / 2 + 4).toFixed(1) + '" fill="#98a1b3" font-size="10" text-anchor="end">' + esc(((minY + maxY) / 2).toFixed(3)) + '</text>' +
          '<text x="' + (padL - 6) + '" y="' + (padT + plotH + 4) + '" fill="#98a1b3" font-size="10" text-anchor="end">' + esc(minY.toFixed(3)) + '</text>' +
          '<text x="' + padL + '" y="' + (H - 12) + '" fill="#98a1b3" font-size="10">' + esc(minX.toFixed(axis.digits)) + '</text>' +
          '<text x="' + ((padL + W - padR) / 2).toFixed(1) + '" y="' + (H - 12) + '" fill="#98a1b3" font-size="10" text-anchor="middle">' + esc(axis.label) + ' (lower is better &larr;)</text>' +
          '<text x="' + (W - padR) + '" y="' + (H - 12) + '" fill="#98a1b3" font-size="10" text-anchor="end">' + esc(maxX.toFixed(axis.digits)) + '</text>' +
          frontPoly + dots +
        '</svg>' +
      '</div>' +
      '<div class="chart-readout" id="paretoReadout">' + defaultReadout + '</div>' +
    '</div>';
  }

  function formatParetoReadout(c, name, isPareto, isSelected) {
    const tags = [isSelected ? 'selected' : '', isPareto ? 'pareto' : ''].filter(Boolean).join(', ');
    return '<strong>' + esc(name) + '</strong>' + (tags ? ' (' + esc(tags) + ')' : '') +
      ' &middot; Score: <strong>' + fmtNum(c.metric, 4) + '</strong>' +
      ' &middot; Train: ' + fmtNum(c.train_time_s, 2) + 's' +
      ' &middot; Mem: ' + fmtNum(c.peak_mem_mb, 1) + ' MB' +
      ' &middot; Risk: ' + fmtNum(c.risk_penalty, 3);
  }

  /* ── 3. Feature Importance Horizontal Bar Chart ─────────────────────── */
  function normalizeImportanceItems(raw) {
    if (!raw) return [];
    let list = [];
    if (Array.isArray(raw)) {
      list = raw.map((item) => {
        if (!item || typeof item !== 'object') return null;
        const name = item.feature_name || item.feature || item.name || '';
        const val = Number(item.importance_mean != null ? item.importance_mean : (item.importance != null ? item.importance : item.score));
        return name && Number.isFinite(val) ? { name: String(name), value: val } : null;
      }).filter(Boolean);
    } else if (typeof raw === 'object') {
      const inner = raw.feature_importances || raw.importances || raw.top_features;
      if (Array.isArray(inner)) return normalizeImportanceItems(inner);
      list = Object.entries(raw)
        .map(([k, v]) => ({ name: String(k), value: Number(v) }))
        .filter((x) => Number.isFinite(x.value));
    }
    return list.sort((a, b) => Math.abs(b.value) - Math.abs(a.value)).slice(0, 12);
  }

  function renderFeatureImportanceHtml(raw) {
    const items = normalizeImportanceItems(raw);
    if (!items.length) return '';
    const W = 600, rowH = 22, padL = 148, padR = 56, padT = 10, padB = 10;
    const H = padT + padB + items.length * rowH;
    const barW = W - padL - padR;
    const maxAbs = Math.max(...items.map((x) => Math.abs(x.value)), 1e-9);

    const bars = items.map((item, i) => {
      const y = padT + i * rowH;
      const w = Math.max(2, (Math.abs(item.value) / maxAbs) * barW);
      const color = item.value >= 0 ? '#5b9dff' : '#ffc860';
      const label = item.name.length > 20 ? item.name.slice(0, 19) + '…' : item.name;
      return '<text x="' + (padL - 8) + '" y="' + (y + 14) + '" fill="#e6e9ef" font-size="11" text-anchor="end">' + esc(label) + '</text>' +
        '<rect x="' + padL + '" y="' + (y + 3) + '" width="' + w.toFixed(1) + '" height="14" rx="3" fill="' + color + '"><title>' + esc(item.name + ': ' + item.value.toFixed(4)) + '</title></rect>' +
        '<text x="' + (padL + w + 6).toFixed(1) + '" y="' + (y + 14) + '" fill="#98a1b3" font-size="10">' + esc(item.value.toFixed(4)) + '</text>';
    }).join('');

    return '<div style="margin:12px 0"><div class="hint" style="margin:0 0 6px"><strong>Top Feature Importances</strong></div>' +
      '<div class="chart-wrap"><svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Feature importance bar chart">' + bars + '</svg></div></div>';
  }

  /* ── 4. Prediction Diagnostics Chart (Actual vs Predicted / Classes) ── */
  function renderPredictionDiagnosticsHtml(pred, task) {
    if (!pred || typeof pred !== 'object' || !Array.isArray(pred.sample_values) || !pred.sample_values.length) {
      return '';
    }
    const samples = pred.sample_values.filter((r) => r && r.predicted != null && r.actual != null);
    if (samples.length < 2) return '';

    const numericPairs = samples.map((r) => ({ a: Number(r.actual), p: Number(r.predicted), row: r.row }))
      .filter((x) => Number.isFinite(x.a) && Number.isFinite(x.p));

    const isReg = (task === 'regression') || (numericPairs.length === samples.length && new Set(numericPairs.map((x) => x.a)).size > 5);
    if (isReg && numericPairs.length >= 2) {
      const W = 600, H = 195, padL = 56, padR = 20, padT = 16, padB = 30;
      const plotW = W - padL - padR, plotH = H - padT - padB;
      const allV = numericPairs.flatMap((x) => [x.a, x.p]);
      let lo = Math.min(...allV), hi = Math.max(...allV);
      if (hi === lo) { hi += 1; lo -= 1; }
      const pad = (hi - lo) * 0.08;
      lo -= pad; hi += pad;
      const xAt = (v) => padL + ((v - lo) / (hi - lo)) * plotW;
      const yAt = (v) => padT + (1 - (v - lo) / (hi - lo)) * plotH;

      const errs = numericPairs.map((x) => x.p - x.a);
      const mae = errs.reduce((s, e) => s + Math.abs(e), 0) / errs.length;
      const bias = errs.reduce((s, e) => s + e, 0) / errs.length;
      const maxErr = Math.max(...errs.map(Math.abs));

      const dots = numericPairs.map((x) =>
        '<circle cx="' + xAt(x.a).toFixed(1) + '" cy="' + yAt(x.p).toFixed(1) + '" r="3.8" fill="#5b9dff" opacity="0.88">' +
        '<title>Row ' + esc(x.row) + ': actual=' + x.a.toFixed(4) + ', pred=' + x.p.toFixed(4) + ', err=' + (x.p - x.a).toFixed(4) + '</title></circle>'
      ).join('');

      return '<div style="margin:10px 0">' +
        '<div class="row" style="justify-content:space-between; margin-bottom:6px">' +
          '<span class="hint" style="margin:0"><strong>Actual vs. Predicted</strong> (sample n=' + numericPairs.length + ', dashed line = ideal y=x)</span>' +
          '<span class="hint" style="margin:0">Sample MAE: <strong>' + mae.toFixed(4) + '</strong> &middot; Bias: ' + bias.toFixed(4) + ' &middot; Max |err|: ' + maxErr.toFixed(4) + '</span>' +
        '</div>' +
        '<div class="chart-wrap"><svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Actual versus predicted scatter plot">' +
          '<line x1="' + padL + '" y1="' + (padT + plotH) + '" x2="' + (W - padR) + '" y2="' + (padT + plotH) + '" stroke="#2a2f3a" />' +
          '<line x1="' + padL + '" y1="' + padT + '" x2="' + padL + '" y2="' + (padT + plotH) + '" stroke="#2a2f3a" />' +
          '<line x1="' + xAt(lo).toFixed(1) + '" y1="' + yAt(lo).toFixed(1) + '" x2="' + xAt(hi).toFixed(1) + '" y2="' + yAt(hi).toFixed(1) + '" stroke="#4fd18b" stroke-width="1.5" stroke-dasharray="5,4" opacity="0.75" />' +
          '<text x="' + (padL - 6) + '" y="' + (padT + 4) + '" fill="#98a1b3" font-size="10" text-anchor="end">' + esc(hi.toFixed(2)) + '</text>' +
          '<text x="' + (padL - 6) + '" y="' + (padT + plotH + 4) + '" fill="#98a1b3" font-size="10" text-anchor="end">' + esc(lo.toFixed(2)) + '</text>' +
          '<text x="' + ((padL + W - padR) / 2).toFixed(1) + '" y="' + (H - 8) + '" fill="#98a1b3" font-size="10" text-anchor="middle">Actual Value</text>' +
          dots +
        '</svg></div></div>';
    }

    // Classification sample distribution bar chart
    const countsA = new Map(), countsP = new Map();
    let matches = 0;
    for (const r of samples) {
      const a = String(r.actual), p = String(r.predicted);
      countsA.set(a, (countsA.get(a) || 0) + 1);
      countsP.set(p, (countsP.get(p) || 0) + 1);
      if (a === p) matches += 1;
    }
    const labels = Array.from(new Set([...countsA.keys(), ...countsP.keys()])).slice(0, 10);
    const maxC = Math.max(...labels.flatMap((l) => [countsA.get(l) || 0, countsP.get(l) || 0]), 1);
    const W = 600, H = 155, padL = 44, padR = 16, padT = 16, padB = 30;
    const plotW = W - padL - padR, plotH = H - padT - padB;
    const grpW = plotW / labels.length;
    const barW = Math.max(8, Math.min(26, grpW * 0.34));

    const bars = labels.map((lbl, i) => {
      const gx = padL + i * grpW + grpW / 2;
      const hA = ((countsA.get(lbl) || 0) / maxC) * plotH;
      const hP = ((countsP.get(lbl) || 0) / maxC) * plotH;
      return '<rect x="' + (gx - barW - 2).toFixed(1) + '" y="' + (padT + plotH - hA).toFixed(1) + '" width="' + barW.toFixed(1) + '" height="' + hA.toFixed(1) + '" rx="2" fill="#5b9dff"><title>' + esc(lbl + ' actual: ' + (countsA.get(lbl) || 0)) + '</title></rect>' +
        '<rect x="' + (gx + 2).toFixed(1) + '" y="' + (padT + plotH - hP).toFixed(1) + '" width="' + barW.toFixed(1) + '" height="' + hP.toFixed(1) + '" rx="2" fill="#4fd18b"><title>' + esc(lbl + ' predicted: ' + (countsP.get(lbl) || 0)) + '</title></rect>' +
        '<text x="' + gx.toFixed(1) + '" y="' + (H - 10) + '" fill="#98a1b3" font-size="10" text-anchor="middle">' + esc(lbl.slice(0, 12)) + '</text>';
    }).join('');

    return '<div style="margin:10px 0">' +
      '<div class="hint" style="margin:0 0 6px"><strong>Sample Class Balance</strong> (<span style="color:#5b9dff">&#9632;</span> Actual, <span style="color:#4fd18b">&#9632;</span> Predicted &middot; Sample match: ' + matches + '/' + samples.length + ')</div>' +
      '<div class="chart-wrap"><svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Class distribution comparison">' +
        '<line x1="' + padL + '" y1="' + (padT + plotH) + '" x2="' + (W - padR) + '" y2="' + (padT + plotH) + '" stroke="#2a2f3a" />' +
        bars + '</svg></div></div>';
  }

  /* ── 5. Side-by-Side Experiment Run Comparison ──────────────────────── */
  function fmtDelta(val, digits, higherIsBetter) {
    if (val == null || !Number.isFinite(Number(val))) return '<span class="delta-flat">&mdash;</span>';
    const n = Number(val);
    if (Math.abs(n) < 1e-9) return '<span class="delta-flat">baseline</span>';
    const sign = n > 0 ? '+' : '';
    const good = higherIsBetter ? n > 0 : n < 0;
    return '<span class="' + (good ? 'delta-pos' : 'delta-neg') + '">' + sign + n.toFixed(digits) + '</span>';
  }

  function renderRunComparisonHtml(comp) {
    const items = (comp && comp.reports) || [];
    if (items.length < 2) return '<div class="hint">Select 2 to 4 JSON reports to compare.</div>';

    // Score bar SVG
    const scores = items.map((r) => Number(r.metric));
    const finiteScores = scores.filter(Number.isFinite);
    let chartHtml = '';
    if (finiteScores.length) {
      const W = 580, rowH = 26, padL = 170, padR = 64, padT = 8, padB = 8;
      const H = padT + padB + items.length * rowH;
      const barW = W - padL - padR;
      const maxS = Math.max(...finiteScores.map((s) => Math.max(0, s)), 1e-6);
      const bars = items.map((r, i) => {
        const y = padT + i * rowH;
        const sc = Number(r.metric);
        const w = Number.isFinite(sc) ? Math.max(2, (Math.max(0, sc) / maxS) * barW) : 0;
        const shortName = String(r.path || '').replace(/^autonomous_/, '').replace(/_combined\.json$/, '');
        const lbl = shortName.length > 24 ? shortName.slice(0, 23) + '…' : shortName;
        const color = i === 0 ? '#5b9dff' : '#4fd18b';
        return '<text x="' + (padL - 8) + '" y="' + (y + 16) + '" fill="#e6e9ef" font-size="11" text-anchor="end">' + esc(lbl) + '</text>' +
          '<rect x="' + padL + '" y="' + (y + 4) + '" width="' + w.toFixed(1) + '" height="15" rx="3" fill="' + color + '" />' +
          '<text x="' + (padL + w + 6).toFixed(1) + '" y="' + (y + 16) + '" fill="#98a1b3" font-size="11">' + esc(fmtNum(r.metric, 4)) + '</text>';
      }).join('');
      chartHtml = '<div class="chart-wrap" style="margin-bottom:10px"><svg viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="Experiment score comparison">' + bars + '</svg></div>';
    }

    const rows = items.map((r, i) => {
      const d = r.deltas || {};
      return '<tr>' +
        '<td><strong>' + esc(r.path) + '</strong>' + (i === 0 ? ' <span class="badge badge-num">baseline</span>' : '') + '</td>' +
        '<td>' + esc(Array.isArray(r.target) ? r.target.join(',') : (r.target || 'n/a')) + '</td>' +
        '<td>' + esc(r.selected_model || 'None') + '</td>' +
        '<td>' + fmtNum(r.metric, 4) + ' (' + fmtDelta(d.metric, 4, true) + ')</td>' +
        '<td>' + fmtNum(r.train_time_s, 2) + 's (' + fmtDelta(d.train_time_s, 2, false) + ')</td>' +
        '<td>' + fmtNum(r.peak_mem_mb, 1) + ' MB (' + fmtDelta(d.peak_mem_mb, 1, false) + ')</td>' +
        '<td>' + fmtNum(r.risk_penalty, 3) + ' (' + fmtDelta(d.risk_penalty, 3, false) + ')</td>' +
        '</tr>';
    }).join('');

    return chartHtml +
      '<div class="scroll-x" tabindex="0" role="region" aria-label="Experiment comparison table">' +
      '<table><thead><tr><th>Report</th><th>Target</th><th>Model</th><th>CV Score (Δ)</th><th>Train Time (Δ)</th><th>Peak Mem (Δ)</th><th>Risk (Δ)</th></tr></thead>' +
      '<tbody>' + rows + '</tbody></table></div>';
  }

  /* ── 6. Accessible Sortable Tables (Intl.Collator + moveBefore) ─────── */
  const _collator = typeof Intl !== 'undefined' && Intl.Collator
    ? new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' })
    : { compare: (a, b) => (a < b ? -1 : a > b ? 1 : 0) };

  function sortTableByHeader(table, th, statusEl) {
    if (!table || !th || !table.tBodies || !table.tBodies.length) return;
    const tbody = table.tBodies[0];
    const headerCells = table.tHead && table.tHead.rows.length ? Array.from(table.tHead.rows[0].cells) : [];
    const colIdx = th.cellIndex;
    const type = th.dataset.type || 'text';
    const nextDir = th.getAttribute('aria-sort') === 'ascending' ? 'descending' : 'ascending';

    for (const h of headerCells) {
      if (h !== th) h.removeAttribute('aria-sort');
    }
    th.setAttribute('aria-sort', nextDir);

    const factor = nextDir === 'ascending' ? 1 : -1;
    const rows = Array.from(tbody.rows);
    const keyed = rows.map((row) => {
      const cell = row.cells[colIdx];
      const raw = cell ? (cell.dataset.value != null ? cell.dataset.value : cell.textContent.trim()) : '';
      return { row, raw, num: type === 'number' ? Number(raw) : NaN };
    });

    keyed.sort((a, b) => {
      if (type === 'number') {
        const aOk = Number.isFinite(a.num), bOk = Number.isFinite(b.num);
        if (!aOk && !bOk) return 0;
        if (!aOk) return 1;
        if (!bOk) return -1;
        return factor * (a.num - b.num);
      }
      return factor * _collator.compare(a.raw, b.raw);
    });

    for (const item of keyed) {
      if (typeof tbody.moveBefore === 'function') {
        tbody.moveBefore(item.row, null);
      } else {
        tbody.appendChild(item.row);
      }
    }

    if (statusEl) {
      const btn = th.querySelector('button');
      const colLabel = btn ? btn.textContent.trim() : th.textContent.trim();
      statusEl.textContent = 'Sorted by ' + colLabel + ', ' + nextDir + ', ' + keyed.length + ' rows';
    }
  }

  function enhanceSortableTable(table, statusEl) {
    if (!table) return;
    const btns = table.querySelectorAll('thead th button.sort-btn');
    btns.forEach((btn) => {
      btn.addEventListener('click', () => {
        const th = btn.closest('th');
        sortTableByHeader(table, th, statusEl);
      });
    });
  }

  globalScope.ACDSOCharts = {
    PARETO_AXES,
    esc,
    fmtNum,
    fmtValue,
    renderForecastChart,
    renderParetoPlotHtml,
    formatParetoReadout,
    renderFeatureImportanceHtml,
    renderPredictionDiagnosticsHtml,
    renderRunComparisonHtml,
    enhanceSortableTable,
    sortTableByHeader,
  };
})(typeof globalThis !== 'undefined' ? globalThis : window);
