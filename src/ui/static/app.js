/* Grade change intelligence - dashboard behaviour.
 *
 * Plain fetch + vanilla DOM. Every number rendered here comes from an API response; this
 * file formats and positions, it does not compute process quantities.
 *
 * Colour discipline: the semantic hues (green in spec, amber at risk, red off spec, blue
 * for prediction) are declared once in PALETTE and used only for those meanings. Source
 * chips and impact bars use identity palettes chosen clear of the semantic hues.
 */
'use strict';

const PALETTE = {
  measured: '#111318',
  setpoint: '#6b7280',
  predicted: '#1d4ed8',
  uncertainty: 'rgba(29, 78, 216, 0.16)',
  spec: 'rgba(21, 128, 61, 0.13)',
  ok: '#15803d',
  risk: '#b45309',
  bad: '#b91c1c',
  grid: '#e8eaee',
  axis: '#6b7280',
};

/* One hue per variable for the impact ranking. Deliberately not green/amber/red/blue. */
const VAR_COLOURS = {
  speed: '#6d28d9',
  filler_flow: '#0e7490',
  steam_p: '#a21caf',
  stock_flow: '#4338ca',
  stock_cons: '#57534e',
};

/* One hue per evidence source type. Matches the chip borders in style.css. */
const SOURCE_COLOURS = {
  physics: '#4338ca',
  causal: '#0e7490',
  historical: '#6d28d9',
  recipe: '#92400e',
  model: '#a21caf',
};

const VAR_LABELS = {
  speed: 'Machine speed',
  filler_flow: 'Filler flow',
  steam_p: 'Steam pressure',
  stock_flow: 'Thick stock flow',
  stock_cons: 'Stock consistency',
};

const REASON_LABELS = {
  unsafe: 'Unsafe',
  already_handling: 'Already handling it',
  wrong_variable: 'Wrong variable',
  too_aggressive: 'Too aggressive',
  too_late: 'Too late',
  disagree_with_cause: 'Disagree with the cause',
  other: 'Other',
};

const PLAY_TICK_MS = 350;
const PLAY_STEP_SEC = 10;

const state = {
  episodeId: null,
  episode: null,
  t: 0,
  stepSec: 5,
  prediction: null,
  suggestion: null,
  timer: null,
  inFlight: false,
  dirty: false,
  yRange: null,
  autoPausedAt: null,
  impactLagged: true,
  impact: null,
  reasonCodes: [],
  history: [],
  sessionAvoided: 0,
  charts: {},
};

const $ = (id) => document.getElementById(id);
const fmt = (v, d = 1) => (v === null || v === undefined || Number.isNaN(v) ? '\u2014' : Number(v).toFixed(d));
const el = (tag, cls, text) => {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
};

async function api(path, options) {
  const res = await fetch(path, options);
  const body = await res.json().catch(() => ({ error: `${res.status} ${res.statusText}` }));
  if (!res.ok) throw new Error(body.error || `request failed: ${path}`);
  return body;
}

/* ------------------------------------------------------------------ chart plugins */

/* Vertical "now" rule plus the breach callout, drawn straight onto the chart so the
 * operator never has to cross-reference a legend to find the breach point. */
const timelineMarkers = {
  id: 'timelineMarkers',
  afterDatasetsDraw(chart, _args, opts) {
    const { ctx, chartArea, scales } = chart;
    if (!scales.x || !scales.y) return;
    ctx.save();
    ctx.font = '600 11px "Segoe UI", sans-serif';

    const xNow = scales.x.getPixelForValue(opts.tNow);
    if (xNow >= chartArea.left && xNow <= chartArea.right) {
      ctx.strokeStyle = '#111318';
      ctx.lineWidth = 1;
      ctx.setLineDash([]);
      ctx.beginPath();
      ctx.moveTo(xNow, chartArea.top);
      ctx.lineTo(xNow, chartArea.bottom);
      ctx.stroke();
      const label = `now  t = ${opts.tNow.toFixed(0)} s`;
      const w = ctx.measureText(label).width + 10;
      const left = Math.min(xNow + 4, chartArea.right - w - 2);
      ctx.fillStyle = '#111318';
      ctx.fillRect(left, chartArea.top + 2, w, 17);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, left + 5, chartArea.top + 14);
    }

    /* Where the open-loop mass balance stops being a statement about sheet that already
     * exists. No breach is claimed to the right of this line. */
    if (opts.validityEnd !== undefined && opts.validityEnd !== null) {
      const xv = scales.x.getPixelForValue(opts.validityEnd);
      if (xv >= chartArea.left && xv <= chartArea.right) {
        ctx.strokeStyle = '#c9ccd2';
        ctx.lineWidth = 1;
        ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.moveTo(xv, chartArea.top);
        ctx.lineTo(xv, chartArea.bottom);
        ctx.stroke();
        ctx.setLineDash([]);
        ctx.fillStyle = '#8b9099';
        ctx.font = '400 10px "Segoe UI", sans-serif';
        ctx.fillText('open-loop limit', xv + 4, chartArea.bottom - 5);
        ctx.font = '600 11px "Segoe UI", sans-serif';
      }
    }

    if (opts.breach) {
      const bx = scales.x.getPixelForValue(opts.breach.x);
      const by = scales.y.getPixelForValue(opts.breach.y);
      ctx.strokeStyle = PALETTE.risk;
      ctx.fillStyle = PALETTE.risk;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(bx, by, 5, 0, Math.PI * 2);
      ctx.stroke();
      const text = opts.breach.label;
      const w = ctx.measureText(text).width + 12;
      const flip = bx + w + 14 > chartArea.right;
      const tx = flip ? bx - w - 10 : bx + 10;
      const ty = Math.max(by - 24, chartArea.top + 2);
      ctx.fillRect(tx, ty, w, 17);
      ctx.beginPath();
      ctx.moveTo(flip ? tx + w : tx, ty + 17);
      ctx.lineTo(bx, by - 6);
      ctx.stroke();
      ctx.fillStyle = '#ffffff';
      ctx.fillText(text, tx + 6, ty + 13);
    }
    ctx.restore();
  },
};

/* Value label at the end of each horizontal bar - the discovered lag, per the impact
 * ranking panel spec. */
const barEndLabels = {
  id: 'barEndLabels',
  afterDatasetsDraw(chart, _args, opts) {
    const { ctx, scales } = chart;
    const meta = chart.getDatasetMeta(0);
    if (!meta || !opts.labels) return;
    ctx.save();
    ctx.font = '600 11px "Segoe UI", sans-serif';
    ctx.fillStyle = '#454b54';
    ctx.textBaseline = 'middle';
    meta.data.forEach((bar, i) => {
      const text = opts.labels[i];
      if (!text) return;
      const x = Math.min(bar.x + 8, scales.x.right - ctx.measureText(text).width - 2);
      ctx.fillText(text, x, bar.y);
    });
    ctx.restore();
  },
};

/* Sample count above each bar in the by-source acceptance chart. */
const barTopLabels = {
  id: 'barTopLabels',
  afterDatasetsDraw(chart, _args, opts) {
    const { ctx } = chart;
    const meta = chart.getDatasetMeta(0);
    if (!meta || !opts.labels) return;
    ctx.save();
    ctx.font = '400 11px "Segoe UI", sans-serif';
    ctx.fillStyle = '#6b7280';
    ctx.textAlign = 'center';
    meta.data.forEach((bar, i) => {
      const text = opts.labels[i];
      if (text) ctx.fillText(text, bar.x, bar.y - 6);
    });
    ctx.restore();
  },
};

if (window.Chart) {
  Chart.register(timelineMarkers, barEndLabels, barTopLabels);
  Chart.defaults.font.family = '"Segoe UI", -apple-system, Roboto, Helvetica, Arial, sans-serif';
  Chart.defaults.font.size = 11;
  Chart.defaults.color = PALETTE.axis;
  Chart.defaults.animation = false;
  Chart.defaults.maintainAspectRatio = false;
}

/* ------------------------------------------------------------------ timeline chart */

function buildBwChart() {
  const ctx = $('bw-chart').getContext('2d');
  const line = (label, colour, extra) => Object.assign({
    label,
    data: [],
    parsing: false,
    borderColor: colour,
    backgroundColor: colour,
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.15,
    spanGaps: false,
  }, extra || {});

  state.charts.bw = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        line('spec low', 'rgba(0,0,0,0)', { borderWidth: 0 }),
        line('in spec band', 'rgba(0,0,0,0)', { borderWidth: 0, fill: '-1', backgroundColor: PALETTE.spec }),
        line('setpoint', PALETTE.setpoint, { borderWidth: 1.5, borderDash: [2, 3] }),
        line('uncertainty low', 'rgba(0,0,0,0)', { borderWidth: 0 }),
        line('uncertainty', 'rgba(0,0,0,0)', { borderWidth: 0, fill: '-1', backgroundColor: PALETTE.uncertainty }),
        line('forecast', PALETTE.predicted, { borderDash: [6, 4] }),
        line('measured', PALETTE.measured, { borderWidth: 2.2 }),
        /* Amber for a forecast excursion, red only for sheet that measured off spec.
         * Predicted and realised must not share a hue. */
        line('forecast outside band', PALETTE.risk, {
          showLine: false,
          pointRadius: 2.8,
          pointBackgroundColor: PALETTE.risk,
          pointBorderColor: PALETTE.risk,
        }),
        line('measured off spec', PALETTE.bad, {
          showLine: false,
          pointRadius: 2.4,
          pointBackgroundColor: PALETTE.bad,
          pointBorderColor: PALETTE.bad,
        }),
      ],
    },
    options: {
      layout: { padding: { top: 6, right: 10 } },
      scales: {
        x: {
          type: 'linear',
          title: { display: true, text: 'seconds from transition start', color: PALETTE.axis },
          grid: { color: PALETTE.grid },
          ticks: { callback: (v) => `${v}` },
        },
        y: {
          title: { display: true, text: 'basis weight  g/m2', color: PALETTE.axis },
          grid: { color: PALETTE.grid },
        },
      },
      plugins: {
        legend: { display: false },
        tooltip: {
          mode: 'nearest',
          intersect: false,
          callbacks: {
            // The filter below can empty the item list, so this must not assume item 0.
            title: (items) => (items.length ? `t = ${Number(items[0].parsed.x).toFixed(0)} s` : ''),
            label: (item) => `${item.dataset.label}: ${item.parsed.y.toFixed(2)} g/m2`,
          },
          filter: (item) => item.dataset.borderWidth > 0 || item.dataset.showLine === false,
        },
        timelineMarkers: { tNow: 0, breach: null },
        barEndLabels: false,
        barTopLabels: false,
      },
    },
  });
}

function updateBwChart() {
  const chart = state.charts.bw;
  const ep = state.episode;
  const pred = state.prediction;
  if (!chart || !ep) return;

  const s = ep.series;
  const band = ep.spec_band_pct / 100;
  const tNow = state.t;

  const specLo = [];
  const specHi = [];
  const sp = [];
  const measured = [];
  const measuredOff = [];
  for (let i = 0; i < s.t_sec.length; i += 1) {
    const t = s.t_sec[i];
    if (s.bw_sp[i] !== null) {
      specLo.push({ x: t, y: s.bw_sp[i] * (1 - band) });
      specHi.push({ x: t, y: s.bw_sp[i] * (1 + band) });
      sp.push({ x: t, y: s.bw_sp[i] });
    }
    if (t <= tNow && s.bw[i] !== null) {
      measured.push({ x: t, y: s.bw[i] });
      if (t >= 0 && s.bw_sp[i] && Math.abs((s.bw[i] - s.bw_sp[i]) / s.bw_sp[i]) > band) {
        measuredOff.push({ x: t, y: s.bw[i] });
      }
    }
  }

  const uncLo = [];
  const uncHi = [];
  const forecast = [];
  const outside = [];
  let breach = null;
  if (pred) {
    const traj = pred.predicted_trajectory || [];
    const spTraj = pred.setpoint_trajectory || [];
    traj.forEach((point, i) => {
      const [t, bw] = point;
      forecast.push({ x: t, y: bw });
      if (pred.uncertainty) {
        uncLo.push({ x: t, y: pred.uncertainty.lo[i] });
        uncHi.push({ x: t, y: pred.uncertainty.hi[i] });
      }
      const spAt = spTraj[i] ? spTraj[i][1] : null;
      if (spAt) {
        const dev = Math.abs((bw - spAt) / spAt) * 100;
        if (dev > pred.spec_band_pct) {
          outside.push({ x: t, y: bw });
          if (!breach) breach = { x: t, y: bw, label: '' };
        }
      }
    });
    /* The callout says which side of the open-loop validity horizon the excursion falls
     * on. Inside it the sheet is already made and the breach is a claim; outside it the
     * loop has had a chance to act, so the crossing is flagged but not claimed. */
    if (breach) {
      const ahead = breach.x - tNow;
      const ttb = pred.time_to_breach_sec;
      breach.label = ahead <= pred.validity_sec && ttb !== null
        ? `breach forecast  +${Number(ttb).toFixed(0)} s`
        : `outside band  +${ahead.toFixed(0)} s, past the open-loop limit`;
    }
  }

  const sets = chart.data.datasets;
  sets[0].data = specLo;
  sets[1].data = specHi;
  sets[2].data = sp;
  sets[3].data = uncLo;
  sets[4].data = uncHi;
  sets[5].data = forecast;
  sets[6].data = measured;
  sets[7].data = outside;
  sets[8].data = measuredOff;

  const xMax = Math.min(Math.max(tNow + 240, 360), ep.replay.t_end + 20);
  chart.options.scales.x.min = Math.max(-90, ep.series.t_sec[0]);
  chart.options.scales.x.max = xMax;

  /* Y axis is never truncated: it always contains the whole spec band plus every value
   * on screen, and it only ever grows during a replay so the trace does not appear to
   * move because the axis moved. */
  const visible = [];
  const push = (arr) => arr.forEach((p) => { if (p.x >= chart.options.scales.x.min && p.x <= xMax) visible.push(p.y); });
  push(specLo); push(specHi); push(measured); push(forecast); push(uncLo); push(uncHi);
  if (visible.length) {
    const lo = Math.min.apply(null, visible);
    const hi = Math.max.apply(null, visible);
    const pad = Math.max((hi - lo) * 0.08, 1.5);
    const next = { min: lo - pad, max: hi + pad };
    if (!state.yRange || state.episodeChanged) state.yRange = next;
    else state.yRange = { min: Math.min(state.yRange.min, next.min), max: Math.max(state.yRange.max, next.max) };
    chart.options.scales.y.min = state.yRange.min;
    chart.options.scales.y.max = state.yRange.max;
  }
  state.episodeChanged = false;

  chart.options.plugins.timelineMarkers = {
    tNow,
    breach,
    validityEnd: pred ? tNow + pred.validity_sec : null,
  };
  chart.update();
}

/* ------------------------------------------------------------------ impact ranking */

function buildImpactChart() {
  state.charts.impact = new Chart($('impact-chart').getContext('2d'), {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderWidth: 0, barThickness: 22 }] },
    options: {
      indexAxis: 'y',
      layout: { padding: { right: 74 } },
      scales: {
        x: {
          beginAtZero: true,
          max: 1,
          title: { display: true, text: 'strength  |corr| with basis-weight deviation', color: PALETTE.axis },
          grid: { color: PALETTE.grid },
        },
        y: { grid: { display: false } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (item) => `strength ${item.parsed.x.toFixed(3)}` } },
        timelineMarkers: false,
        barTopLabels: false,
        barEndLabels: { labels: [] },
      },
    },
  });
}

function renderImpact() {
  const data = state.impact;
  const chart = state.charts.impact;
  if (!data || !chart) return;

  const rows = (data.rankings || [])
    .filter((r) => r.kind === 'deviation')
    .map((r) => ({
      variable: r.variable,
      lagged: r.strength,
      zero: r.zero_lag_strength,
      raw: r.raw_level_corr,
      lag: r.best_lag_sec,
      n: r.n_episodes,
    }));

  const lagged = state.impactLagged;
  rows.sort((a, b) => (lagged ? b.lagged - a.lagged : b.zero - a.zero));

  chart.data.labels = rows.map((r) => VAR_LABELS[r.variable] || r.variable);
  chart.data.datasets[0].data = rows.map((r) => (lagged ? r.lagged : r.zero));
  chart.data.datasets[0].backgroundColor = rows.map((r) => VAR_COLOURS[r.variable] || '#57534e');
  chart.options.plugins.barEndLabels.labels = rows.map((r) =>
    lagged ? `lag ${fmt(r.lag, 0)} s` : 'lag 0 s'
  );
  chart.update();

  $('btn-lag-toggle').textContent = lagged ? 'Showing: lagged' : 'Showing: raw, zero lag';
  $('btn-lag-toggle').setAttribute('aria-pressed', String(lagged));
  $('impact-note').textContent = lagged
    ? `Lagged discovery over ${rows[0] ? rows[0].n : 0} episodes. Each bar is the strongest correlation found across the swept lags, with the lag that produced it.`
    : 'The same variables at zero lag. Ranking and separation both collapse, which is why a contemporaneous correlation matrix cannot tell you which loop to touch.';

  const amb = data.zero_lag_ambiguity || {};
  $('impact-foot').textContent = amb.raw_level_corr_min
    ? `Raw level correlations sit between ${fmt(amb.raw_level_corr_min, 2)} and ${fmt(amb.raw_level_corr_max, 2)} - everything looks connected to everything. Lagging first is what separates them.`
    : '';
}

/* ------------------------------------------------------------------ trust charts */

function buildTrustCharts() {
  state.charts.trust = new Chart($('trust-chart').getContext('2d'), {
    type: 'line',
    data: {
      labels: [],
      datasets: [{
        label: 'cumulative acceptance rate',
        data: [],
        borderColor: PALETTE.predicted,
        backgroundColor: 'rgba(29, 78, 216, 0.10)',
        borderWidth: 2,
        fill: true,
        tension: 0.2,
        pointRadius: 4,
        pointBackgroundColor: [],
        pointBorderColor: [],
      }],
    },
    options: {
      scales: {
        x: { title: { display: true, text: 'decisions, in order', color: PALETTE.axis }, grid: { color: PALETTE.grid } },
        y: { min: 0, max: 100, title: { display: true, text: 'acceptance rate  %', color: PALETTE.axis }, grid: { color: PALETTE.grid } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (i) => `${i.parsed.y.toFixed(0)}% after ${i.label}` } },
        timelineMarkers: false,
        barEndLabels: false,
        barTopLabels: false,
      },
    },
  });

  state.charts.source = new Chart($('source-chart').getContext('2d'), {
    type: 'bar',
    data: { labels: [], datasets: [{ data: [], backgroundColor: [], borderWidth: 0, barThickness: 34 }] },
    options: {
      layout: { padding: { top: 16 } },
      scales: {
        x: { title: { display: true, text: 'dominant evidence source', color: PALETTE.axis }, grid: { display: false } },
        y: { min: 0, max: 100, title: { display: true, text: 'acceptance rate  %', color: PALETTE.axis }, grid: { color: PALETTE.grid } },
      },
      plugins: {
        legend: { display: false },
        tooltip: { callbacks: { label: (i) => `${i.parsed.y.toFixed(0)}% accepted` } },
        timelineMarkers: false,
        barEndLabels: false,
        barTopLabels: { labels: [] },
      },
    },
  });
}

function renderTrust(stats) {
  const log = stats.decisions_log || [];
  let accepted = 0;
  const labels = [];
  const values = [];
  const colours = [];
  log.forEach((row, i) => {
    if (row.decision === 'accepted') accepted += 1;
    labels.push(String(i + 1));
    values.push((accepted / (i + 1)) * 100);
    colours.push(row.decision === 'accepted' ? PALETTE.ok : PALETTE.bad);
  });
  const trust = state.charts.trust;
  trust.data.labels = labels;
  trust.data.datasets[0].data = values;
  trust.data.datasets[0].pointBackgroundColor = colours;
  trust.data.datasets[0].pointBorderColor = colours;
  trust.update();

  const bySource = stats.by_dominant_source || {};
  const keys = Object.keys(bySource);
  const source = state.charts.source;
  source.data.labels = keys;
  source.data.datasets[0].data = keys.map((k) => bySource[k].acceptance_rate * 100);
  source.data.datasets[0].backgroundColor = keys.map((k) => SOURCE_COLOURS[k] || '#57534e');
  source.options.plugins.barTopLabels.labels = keys.map((k) => `n = ${bySource[k].n}`);
  source.update();

  const rate = stats.acceptance_rate === null ? null : stats.acceptance_rate * 100;
  $('trust-note').textContent = stats.n
    ? `${stats.n} decision${stats.n === 1 ? '' : 's'} logged, ${fmt(rate, 0)}% accepted. Accept and reject are weighted equally in the interface, so this is a usable signal.`
    : 'No decisions logged yet. Accept or reject a suggestion to start the calibration record.';
}

/* ------------------------------------------------------------------ header + lag */

function renderStatus() {
  const pred = state.prediction;
  const card = state.suggestion && state.suggestion.card;
  const meta = state.episode ? state.episode.meta : null;
  const dot = $('risk-dot');

  if (!pred || !meta) {
    $('status-line').textContent = 'Loading episode\u2026';
    dot.dataset.risk = 'idle';
    return;
  }

  const ttb = pred.time_to_breach_sec;
  if (pred.will_breach) {
    dot.dataset.risk = ttb !== null && ttb <= 0 ? 'bad' : 'risk';
    $('status-line').textContent = card
      ? card.claim.statement
      : `Basis weight is on track to run ${fmt(Math.abs(pred.predicted_max_dev_pct), 1)}% off target.`;
  } else {
    dot.dataset.risk = 'ok';
    $('status-line').textContent =
      `Tracking in spec. Forecast peak deviation ${fmt(Math.abs(pred.predicted_max_dev_pct), 2)}% against the ${fmt(pred.spec_band_pct, 1)}% band.`;
  }

  const share = `physics ${fmt(pred.physics_contribution * 100, 0)}% / residual ${fmt(pred.model_correction * 100, 0)}%`;
  const basis = pred.residual_applied ? share : 'physics only before the 90 s feature horizon';
  $('status-meta').textContent =
    `t = ${fmt(state.t, 0)} s  \u00b7  ${meta.grade_from} \u2192 ${meta.grade_to}  \u00b7  ${basis}  \u00b7  confidence ${fmt(pred.confidence, 2)}`;
}

function renderLag() {
  const pred = state.prediction;
  const lag = pred ? pred.lag_components : (state.episode && state.episode.measurement_lag);
  if (!lag) return;
  const transport = lag.transport !== undefined ? lag.transport : lag.transport_sec;
  const scanner = lag.scanner !== undefined ? lag.scanner : lag.scanner_sec;
  const composed = lag.composed !== undefined ? lag.composed : lag.composed_sec;

  $('lag-lead').textContent =
    `You are seeing paper made ${fmt(composed, 0)} seconds ago (transport ${fmt(transport, 0)} s + scanner hold ${fmt(scanner, 0)} s).`;

  const parts = $('lag-parts');
  parts.textContent = '';
  const rows = [
    ['Transport delay', `${fmt(transport, 1)} s`],
    ['Scanner traverse and hold', `${fmt(scanner, 1)} s`],
    ['Effective measurement lag', `${fmt(composed, 1)} s`],
  ];
  if (pred) {
    rows.push(['Headbox lead over the reading', `${fmt(pred.headbox_lead_g_m2, 2)} g/m2`]);
  }
  rows.forEach(([term, value]) => {
    const wrap = el('div');
    wrap.appendChild(el('dt', null, term));
    wrap.appendChild(el('dd', null, value));
    parts.appendChild(wrap);
  });

  $('lag-foot').textContent = pred
    ? `Headbox basis weight now is ${fmt(pred.headbox_bw_now, 2)} g/m2 against a scanner reading of ${fmt(pred.measured_bw_now, 2)} g/m2. The mass balance is valid open loop for ${fmt(pred.validity_sec, 0)} s, because until that sheet reaches the scanner the loop has no information to react to. That window is what makes an early call possible.`
    : '';
}

/* ------------------------------------------------------------------ suggestion */

function kvRow(list, term, value) {
  const wrap = el('div');
  wrap.appendChild(el('dt', null, term));
  wrap.appendChild(el('dd', null, value));
  list.appendChild(wrap);
}

function renderSuggestion() {
  const payload = state.suggestion;
  const card = payload && payload.card;
  const body = $('suggestion-body');
  const empty = $('suggestion-empty');

  if (!card) {
    body.hidden = true;
    empty.hidden = false;
    empty.textContent = (payload && payload.reason) || 'No suggestion at this replay position.';
    $('card-id-tag').textContent = '';
    $('econ-episode').textContent = '\u2014';
    $('econ-basis').textContent = '';
    return;
  }

  empty.hidden = true;
  body.hidden = false;
  $('card-id-tag').textContent = `${card.card_id}  \u00b7  issued t = ${fmt(card.issued_t_sec, 0)} s`;

  const a = card.action;
  const label = VAR_LABELS[a.variable] || a.variable;
  $('claim-statement').textContent = card.claim.statement;
  $('action-line').textContent =
    `${label}: ${fmt(a.from, a.decimals)} \u2192 ${fmt(a.to, a.decimals)} ${a.unit_display} over the next ${fmt(a.window_sec, 0)} s`;

  const detail = $('action-detail');
  detail.textContent = '';
  kvRow(detail, 'Ramp rate', `${fmt(a.ramp_rate_per_min, a.decimals)} ${a.unit_display} / min`);
  kvRow(detail, 'Estimated effect on bw', `${fmt(a.estimated_bw_effect_pct, 2)} %`);
  kvRow(detail, 'Peak deviation with the move', `${fmt(a.expected_max_dev_pct, 2)} %`);
  kvRow(detail, 'Peak deviation if left alone', `${fmt(card.counterfactual.no_action_max_dev_pct, 2)} %`);
  kvRow(detail, 'Stabilisation gain', a.expected_stabilisation_gain_sec === null ? '\u2014' : `${fmt(a.expected_stabilisation_gain_sec, 0)} s`);
  kvRow(detail, 'Predicted basis weight', `${fmt(card.claim.predicted_value, 2)} ${card.claim.unit}  (${fmt(card.claim.interval[0], 2)} \u2013 ${fmt(card.claim.interval[1], 2)})`);
  kvRow(detail, 'Confidence', fmt(card.claim.confidence, 2));
  kvRow(detail, 'Signed deviation', `${fmt(card.claim.dev_pct, 2)} % against a \u00b1${fmt(card.claim.spec_band_pct, 1)} % band`);

  $('narration-text').textContent = card.narration || 'Narration withheld: it failed the numeral validator, so it is not shown.';

  const chips = $('source-chips');
  chips.textContent = '';
  const sources = (card.sources || []).slice().sort((x, y) => y.weight - x.weight);
  sources.forEach((src) => {
    const chip = el('div', 'chip');
    chip.dataset.source = src.type;
    const head = el('div', 'chip-head');
    head.appendChild(el('span', null, src.type));
    head.appendChild(el('span', 'chip-weight', `weight ${fmt(src.weight, 3)}`));
    chip.appendChild(head);
    chip.appendChild(el('p', 'chip-detail', src.detail));
    const bar = el('div', 'chip-bar');
    const fill = el('span');
    fill.style.width = `${Math.max(src.weight * 100, 2)}%`;
    bar.appendChild(fill);
    chip.appendChild(bar);
    chips.appendChild(chip);
  });
  const sum = sources.reduce((acc, s) => acc + s.weight, 0);
  $('weight-sum').textContent = `\u00b7 ${sources.length} sources, weights sum to ${fmt(sum, 2)}`;

  const hist = sources.find((s) => s.type === 'historical') || {};
  const inSpec = hist.episode_ids || [];
  const offSpec = hist.off_spec_episode_ids || [];
  $('retrieved-count').textContent = `\u00b7 ${inSpec.length} held spec, ${offSpec.length} went off, k = ${hist.k || 0}`;
  const rb = $('retrieved-body');
  rb.textContent = '';
  [['Held spec', inSpec], ['Went off spec', offSpec]].forEach(([title, ids]) => {
    const group = el('div', 'id-group');
    group.appendChild(el('h4', null, `${title} (${ids.length})`));
    const list = el('div', 'id-list');
    ids.forEach((id) => list.appendChild(el('code', null, id)));
    group.appendChild(list);
    rb.appendChild(group);
  });

  const cc = card.constraints_checked;
  const cb = $('constraints-body');
  cb.textContent = '';
  kvRow(cb, 'Recipe limits', cc.recipe_limits);
  kvRow(cb, 'Actuator rate', cc.actuator_rate);
  kvRow(cb, 'Filtered before scoring', cc.filtered_before_scoring ? 'yes' : 'no');
  kvRow(cb, 'Candidates generated', String(cc.candidates_generated));
  kvRow(cb, 'Scored', String(cc.candidates_scored));
  kvRow(cb, 'Discarded as inadmissible', String(cc.candidates_discarded));
  kvRow(cb, 'Discarded as immaterial', String(cc.candidates_immaterial));
  const recipe = sources.find((s) => s.type === 'recipe');
  if (recipe && recipe.recipe_detail) {
    const rd = recipe.recipe_detail;
    kvRow(cb, 'Binding limit', `${rd.binding_bound || '\u2014'} ${rd.binding_value === null ? '' : fmt(rd.binding_value, 1)}`);
    kvRow(cb, 'Headroom used', `${fmt(rd.headroom_used_pct, 1)} %`);
    kvRow(cb, 'Rate used', `${fmt(rd.rate_used_per_min, 1)} of ${fmt(rd.rate_limit_per_min, 1)} per min`);
  }

  resetDecisionControls();
  trackCard(payload);
  renderEconomics();
}

function resetDecisionControls() {
  const known = state.history.find((r) => r.card_id === (state.suggestion.card || {}).card_id);
  const decided = known && known.decision !== 'pending';
  $('btn-accept').disabled = Boolean(decided);
  $('btn-reject').disabled = Boolean(decided);
  $('reject-reason').hidden = true;
  $('decision-confirm').textContent = decided
    ? `Already logged this session: ${known.decision}${known.reason_code ? ` (${REASON_LABELS[known.reason_code] || known.reason_code})` : ''}.`
    : '';
}

/* ------------------------------------------------------------------ history + tile */

function trackCard(payload) {
  const card = payload.card;
  if (state.history.some((r) => r.card_id === card.card_id)) return;
  const sources = (card.sources || []).slice().sort((x, y) => y.weight - x.weight);
  state.history.unshift({
    card_id: card.card_id,
    episode_id: card.episode_id,
    summary: `${VAR_LABELS[card.action.variable] || card.action.variable} ${fmt(card.action.from, card.action.decimals)} \u2192 ${fmt(card.action.to, card.action.decimals)} ${card.action.unit_display}`,
    dominant: payload.dominant_source || (sources[0] && sources[0].type),
    mix: sources.slice(0, 3).map((s) => `${s.type} ${fmt(s.weight * 100, 0)}%`).join(', '),
    decision: 'pending',
    reason_code: null,
    avoided: payload.economics ? payload.economics.avoided_broke_tonnes : 0,
  });
  renderHistory();
}

function renderHistory() {
  const body = $('history-body');
  body.textContent = '';
  if (!state.history.length) {
    const row = el('tr', 'empty-row');
    const cell = el('td', null, 'No cards issued yet this session.');
    cell.colSpan = 5;
    row.appendChild(cell);
    body.appendChild(row);
    return;
  }
  state.history.forEach((r) => {
    const row = el('tr');
    const idCell = el('td');
    idCell.appendChild(el('code', null, r.card_id));
    row.appendChild(idCell);
    row.appendChild(el('td', null, r.summary));

    const mixCell = el('td');
    const mix = el('div', 'mix');
    const dot = el('span', 'src-dot');
    dot.style.background = SOURCE_COLOURS[r.dominant] || '#57534e';
    dot.title = `dominant source: ${r.dominant}`;
    mix.appendChild(dot);
    mix.appendChild(el('span', null, r.mix));
    mixCell.appendChild(mix);
    row.appendChild(mixCell);

    const decisionCell = el('td');
    const badge = el('span', 'badge', r.decision === 'pending' ? 'Pending' : r.decision[0].toUpperCase() + r.decision.slice(1));
    badge.dataset.decision = r.decision;
    decisionCell.appendChild(badge);
    row.appendChild(decisionCell);

    row.appendChild(el('td', null, r.reason_code ? REASON_LABELS[r.reason_code] || r.reason_code : '\u2014'));
    body.appendChild(row);
  });
}

function renderEconomics() {
  const econ = state.suggestion && state.suggestion.economics;
  $('econ-episode').textContent = econ ? fmt(econ.avoided_broke_tonnes, 2) : '\u2014';
  $('econ-session').textContent = fmt(state.sessionAvoided, 2);
  $('econ-basis').textContent = econ ? econ.basis : '';
}

/* ------------------------------------------------------------------ replay engine */

const TIMELINE_NOTE = $('timeline-note').textContent;

async function refresh() {
  if (!state.episodeId) return;
  if (state.inFlight) { state.dirty = true; return; }
  state.inFlight = true;
  const t = state.t;
  try {
    const [pred, sug] = await Promise.all([
      api(`/api/predict/${state.episodeId}?t=${t}`),
      api(`/api/suggest/${state.episodeId}?t=${t}`),
    ]);
    if (t === state.t) {
      state.prediction = pred;
      state.suggestion = sug;
      renderStatus();
      renderLag();
      updateBwChart();
      renderSuggestion();
      maybeAutoPause();
    }
  } catch (err) {
    $('status-line').textContent = `Request failed: ${err.message}`;
    $('risk-dot').dataset.risk = 'idle';
  } finally {
    state.inFlight = false;
    if (state.dirty) { state.dirty = false; refresh(); }
  }
}

/* Stop on the first card of an episode. The operator is meant to read the evidence, not
 * watch it scroll past; the replay is a review tool, not a video. */
function maybeAutoPause() {
  if (!state.timer || state.autoPausedAt !== null) return;
  if (!(state.suggestion && state.suggestion.card)) return;
  state.autoPausedAt = state.t;
  pause();
  $('timeline-note').textContent = `${TIMELINE_NOTE} Paused automatically at the first suggestion, t = ${fmt(state.t, 0)} s.`;
}

function setT(value) {
  const ep = state.episode;
  if (!ep) return;
  const clamped = Math.min(Math.max(value, ep.replay.t_start), ep.replay.t_end);
  state.t = Math.round(clamped / state.stepSec) * state.stepSec;
  $('t-slider').value = String(state.t);
  $('t-readout').textContent = `t = ${fmt(state.t, 0)} s`;
  refresh();
}

function play() {
  if (state.timer || !state.episode) return;
  state.timer = window.setInterval(() => {
    if (state.t >= state.episode.replay.t_end) { pause(); return; }
    setT(state.t + PLAY_STEP_SEC);
  }, PLAY_TICK_MS);
  $('btn-play').textContent = 'Pause';
  $('btn-play').setAttribute('aria-pressed', 'true');
}

function pause() {
  if (state.timer) window.clearInterval(state.timer);
  state.timer = null;
  $('btn-play').textContent = 'Play';
  $('btn-play').setAttribute('aria-pressed', 'false');
}

async function loadEpisode(episodeId) {
  pause();
  state.episodeId = episodeId;
  state.autoPausedAt = null;
  state.yRange = null;
  state.episodeChanged = true;
  state.prediction = null;
  state.suggestion = null;
  $('timeline-note').textContent = TIMELINE_NOTE;

  const ep = await api(`/api/episode/${episodeId}`);
  state.episode = ep;
  state.stepSec = ep.replay.step_sec;

  const slider = $('t-slider');
  slider.min = String(ep.replay.t_start);
  slider.max = String(ep.replay.t_end);
  slider.step = String(ep.replay.step_sec);

  renderLag();
  setT(ep.replay.t_start);
}

/* ------------------------------------------------------------------ feedback */

async function submitDecision(decision, reasonCode) {
  const payload = state.suggestion;
  if (!payload || !payload.card) return;
  const card = payload.card;
  try {
    const res = await api('/api/feedback', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        card_id: card.card_id,
        episode_id: card.episode_id,
        decision,
        reason_code: reasonCode || null,
        dominant_source: payload.dominant_source,
      }),
    });
    const row = state.history.find((r) => r.card_id === card.card_id);
    if (row) {
      row.decision = decision;
      row.reason_code = reasonCode || null;
      if (decision === 'accepted') state.sessionAvoided += row.avoided || 0;
    }
    renderHistory();
    renderTrust(res.stats);
    renderEconomics();
    $('btn-accept').disabled = true;
    $('btn-reject').disabled = true;
    $('reject-reason').hidden = true;
    $('decision-confirm').textContent =
      `Logged ${decision}${reasonCode ? ` \u00b7 ${REASON_LABELS[reasonCode] || reasonCode}` : ''} against ${card.card_id} at ${res.recorded.timestamp}. Nothing was applied to the machine.`;
  } catch (err) {
    $('decision-confirm').textContent = `Not logged: ${err.message}`;
  }
}

/* ------------------------------------------------------------------ wiring */

function wire() {
  $('episode-select').addEventListener('change', (e) => { loadEpisode(e.target.value); });
  $('btn-play').addEventListener('click', () => { if (state.timer) pause(); else play(); });
  $('btn-step').addEventListener('click', () => { pause(); setT(state.t + state.stepSec); });
  $('btn-step-back').addEventListener('click', () => { pause(); setT(state.t - state.stepSec); });
  $('btn-reset').addEventListener('click', () => {
    pause();
    state.autoPausedAt = null;
    $('timeline-note').textContent = TIMELINE_NOTE;
    setT(state.episode ? state.episode.replay.t_start : 0);
  });
  $('t-slider').addEventListener('input', (e) => { pause(); setT(Number(e.target.value)); });

  $('btn-lag-toggle').addEventListener('click', () => {
    state.impactLagged = !state.impactLagged;
    renderImpact();
  });

  $('btn-accept').addEventListener('click', () => submitDecision('accepted', null));
  $('btn-reject').addEventListener('click', () => {
    $('reject-reason').hidden = false;
    $('reason-select').focus();
  });
  $('btn-reject-cancel').addEventListener('click', () => { $('reject-reason').hidden = true; });
  $('btn-reject-confirm').addEventListener('click', () => {
    const code = $('reason-select').value;
    if (!code) { $('decision-confirm').textContent = 'A rejection needs a reason code.'; return; }
    submitDecision('rejected', code);
  });
}

async function init() {
  buildBwChart();
  buildImpactChart();
  buildTrustCharts();
  wire();

  const [episodes, impact, stats] = await Promise.all([
    api('/api/episodes'),
    api('/api/impact-ranking'),
    api('/api/feedback/stats'),
  ]);

  const select = $('episode-select');
  episodes.episodes.forEach((row) => {
    const opt = el('option', null,
      `${row.episode_id}  \u00b7  ${row.grade_from} \u2192 ${row.grade_to}  \u00b7  ${row.off_spec ? 'off spec' : 'in spec'} ${fmt(row.max_dev_pct, 1)}%`);
    opt.value = row.episode_id;
    select.appendChild(opt);
  });

  state.impact = impact;
  renderImpact();

  state.reasonCodes = stats.reason_codes || [];
  const reason = $('reason-select');
  reason.appendChild(el('option', null, 'Select a reason\u2026')).value = '';
  state.reasonCodes.forEach((code) => {
    const opt = el('option', null, REASON_LABELS[code] || code);
    opt.value = code;
    reason.appendChild(opt);
  });
  renderTrust(stats);

  const first = episodes.default_episode_id || (episodes.episodes[0] && episodes.episodes[0].episode_id);
  select.value = first;
  await loadEpisode(first);
}

init().catch((err) => {
  $('status-line').textContent = `Startup failed: ${err.message}`;
});
