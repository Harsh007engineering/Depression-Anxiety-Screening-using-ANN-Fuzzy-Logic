/**
 * NeuroFuzzy Clinical Screening Platform - UI Controller
 * Dual Light/Dark Theme, Interactive SVG Gauge, Progress Tracker, Persona Engine
 */

import { fetchQuestions, fetchPersonas, submitScreening, fetchFuzzyVisuals, fetchBenchmark } from './api.js';
import { initRadarChart, updateRadarChart, initFuzzyCurvesChart, refreshChartsTheme } from './charts.js';

let allQuestions = [];
let responses = new Array(21).fill(0);
let fuzzyVisualsData = null;
let currentTab = 'all';
let activePersonaId = null;

document.addEventListener('DOMContentLoaded', async () => {
  initTheme();
  initRadarChart('radarCanvas');

  try {
    allQuestions = await fetchQuestions();
    renderQuestions();
    updateProgress();

    const personas = await fetchPersonas();
    renderPersonas(personas);

    fuzzyVisualsData = await fetchFuzzyVisuals();
    initFuzzyCurvesChart('fuzzyCanvas', fuzzyVisualsData, 'depression');

    // Run baseline initial screening
    handleScreening();
  } catch (err) {
    console.error('Initialization error:', err);
  }

  setupEventListeners();
});

/* --------------------------------------------------------------------------
   Theme Engine (Light / Dark)
   -------------------------------------------------------------------------- */
function initTheme() {
  const savedTheme = localStorage.getItem('neurofuzzy-theme') || 
    (window.matchMedia('(prefers-color-scheme: light)').matches ? 'light' : 'dark');
  applyTheme(savedTheme);

  const toggleBtn = document.getElementById('themeToggleBtn');
  toggleBtn.addEventListener('click', () => {
    const currentTheme = document.documentElement.getAttribute('data-theme') || 'dark';
    const nextTheme = currentTheme === 'dark' ? 'light' : 'dark';
    applyTheme(nextTheme);
  });
}

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('neurofuzzy-theme', theme);

  const sunIcon = document.getElementById('themeIconSun');
  const moonIcon = document.getElementById('themeIconMoon');

  if (theme === 'light') {
    sunIcon.style.display = 'none';
    moonIcon.style.display = 'block';
  } else {
    sunIcon.style.display = 'block';
    moonIcon.style.display = 'none';
  }

  // Refresh charts for new contrast
  refreshChartsTheme();
  if (fuzzyVisualsData) {
    const subscale = document.getElementById('fuzzySubscaleSelect').value;
    initFuzzyCurvesChart('fuzzyCanvas', fuzzyVisualsData, subscale);
  }
}

/* --------------------------------------------------------------------------
   Event Listeners Setup
   -------------------------------------------------------------------------- */
function setupEventListeners() {
  // Environmental stress slider
  const slider = document.getElementById('stressSlider');
  const sliderVal = document.getElementById('stressSliderVal');
  slider.addEventListener('input', (e) => {
    sliderVal.textContent = parseFloat(e.target.value).toFixed(1);
  });

  // Run assessment button
  document.getElementById('analyzeBtn').addEventListener('click', () => {
    handleScreening();
  });

  // Subscale category tabs
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      const target = e.currentTarget;
      target.classList.add('active');
      currentTab = target.dataset.tab;
      filterQuestions();
    });
  });

  // Fuzzy curve subscale select dropdown
  document.getElementById('fuzzySubscaleSelect').addEventListener('change', (e) => {
    if (fuzzyVisualsData) {
      initFuzzyCurvesChart('fuzzyCanvas', fuzzyVisualsData, e.target.value);
    }
  });

  // Benchmark modal open / close
  const benchmarkModal = document.getElementById('benchmarkModal');
  document.getElementById('openBenchmarkBtn').addEventListener('click', async () => {
    benchmarkModal.style.display = 'flex';
    try {
      const bdata = await fetchBenchmark();
      renderBenchmarkTable(bdata);
    } catch (e) {
      console.error(e);
    }
  });

  document.getElementById('closeBenchmarkBtn').addEventListener('click', () => {
    benchmarkModal.style.display = 'none';
  });

  benchmarkModal.addEventListener('click', (e) => {
    if (e.target === benchmarkModal) {
      benchmarkModal.style.display = 'none';
    }
  });

  // Print clinical report sheet
  document.getElementById('printReportBtn').addEventListener('click', () => {
    window.print();
  });
}

/* --------------------------------------------------------------------------
   Questionnaire Rendering & Management
   -------------------------------------------------------------------------- */
function renderQuestions() {
  const container = document.getElementById('questionsContainer');
  container.innerHTML = '';

  const options = [
    { val: 0, text: 'Never', label: '0' },
    { val: 1, text: 'Sometimes', label: '1' },
    { val: 2, text: 'Often', label: '2' },
    { val: 3, text: 'Almost Always', label: '3' }
  ];

  allQuestions.forEach((q, idx) => {
    const itemEl = document.createElement('div');
    itemEl.className = 'question-item';
    itemEl.dataset.subscale = q.subscale.toLowerCase();

    const subClass = `subscale-${q.subscale.toLowerCase()}`;

    itemEl.innerHTML = `
      <div class="question-header">
        <div class="question-text">
          <span class="q-id">Q${q.id}.</span> ${q.text}
        </div>
        <span class="subscale-badge ${subClass}">${q.subscale}</span>
      </div>
      <div class="likert-options">
        ${options.map(opt => `
          <button type="button" 
                  class="likert-btn ${responses[idx] === opt.val ? 'selected' : ''}" 
                  data-idx="${idx}" 
                  data-val="${opt.val}">
            <span>${opt.text}</span>
            <span class="val">(${opt.label})</span>
          </button>
        `).join('')}
      </div>
    `;

    container.appendChild(itemEl);
  });

  // Attach button click listeners
  container.querySelectorAll('.likert-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const target = e.currentTarget;
      const qIdx = parseInt(target.dataset.idx);
      const val = parseInt(target.dataset.val);
      responses[qIdx] = val;

      const parentGrid = target.parentElement;
      parentGrid.querySelectorAll('.likert-btn').forEach(b => b.classList.remove('selected'));
      target.classList.add('selected');

      // Clear persona highlight on custom edit
      clearActivePersona();
      updateProgress();
    });
  });
}

function filterQuestions() {
  document.querySelectorAll('.question-item').forEach(item => {
    if (currentTab === 'all' || item.dataset.subscale === currentTab) {
      item.style.display = 'block';
    } else {
      item.style.display = 'none';
    }
  });
}

function updateProgress() {
  // Count how many questions have answers (in 0-3 range)
  const answeredCount = responses.filter(v => v !== undefined && v !== null).length;
  const pct = Math.round((answeredCount / 21) * 100);
  document.getElementById('progressCount').textContent = `${answeredCount} / 21 Answered (${pct}%)`;
  document.getElementById('progressFill').style.width = `${pct}%`;
}

/* --------------------------------------------------------------------------
   Clinical Personas
   -------------------------------------------------------------------------- */
function renderPersonas(personas) {
  const grid = document.getElementById('personaGrid');
  grid.innerHTML = '';

  const avatarColors = [
    'linear-gradient(135deg, #10b981 0%, #059669 100%)',
    'linear-gradient(135deg, #06b6d4 0%, #0284c7 100%)',
    'linear-gradient(135deg, #f59e0b 0%, #d97706 100%)',
    'linear-gradient(135deg, #ef4444 0%, #b91c1c 100%)'
  ];

  personas.forEach((p, idx) => {
    const initials = p.name.split(' ')[0].substring(0, 2).toUpperCase();
    const btn = document.createElement('button');
    btn.className = 'persona-btn';
    btn.dataset.id = p.id;

    btn.innerHTML = `
      <div class="persona-avatar" style="background: ${avatarColors[idx % avatarColors.length]}">
        ${initials}
      </div>
      <div class="persona-info">
        <strong>${p.name}</strong>
        <span>${p.subtitle}</span>
      </div>
    `;

    btn.addEventListener('click', () => {
      responses = [...p.responses];
      document.getElementById('stressSlider').value = p.contextual_stress;
      document.getElementById('stressSliderVal').textContent = parseFloat(p.contextual_stress).toFixed(1);

      activePersonaId = p.id;
      document.querySelectorAll('.persona-btn').forEach(b => b.classList.remove('active'));
      btn.classList.add('active');

      renderQuestions();
      filterQuestions();
      updateProgress();

      // Trigger real-time screening
      handleScreening();
    });

    grid.appendChild(btn);
  });
}

function clearActivePersona() {
  activePersonaId = null;
  document.querySelectorAll('.persona-btn').forEach(b => b.classList.remove('active'));
}

/* --------------------------------------------------------------------------
   Inference & Dashboard Updates
   -------------------------------------------------------------------------- */
async function handleScreening() {
  const analyzeBtn = document.getElementById('analyzeBtn');
  analyzeBtn.innerHTML = `
    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" class="spin">
      <circle cx="12" cy="12" r="10" stroke-opacity="0.3"></circle>
      <path d="M12 2a10 10 0 0 1 10 10" stroke-linecap="round"></path>
    </svg>
    <span>Synthesizing Neuro-Fuzzy Assessment...</span>
  `;
  analyzeBtn.disabled = true;

  try {
    const stressVal = document.getElementById('stressSlider').value;
    const data = await submitScreening(responses, stressVal);

    // 1. Update SVG Circular Gauge
    const risk = data.summary.risk_index;
    const triage = data.summary.triage_category;
    const color = data.summary.triage_color;

    document.getElementById('gaugeVal').textContent = `${risk.toFixed(1)}%`;

    // SVG gauge circle calculation (Circumference ~ 440)
    const gaugeCircle = document.getElementById('gaugeCircle');
    const circumference = 440;
    const offset = circumference - (circumference * (risk / 100));
    gaugeCircle.style.strokeDashoffset = Math.max(0, offset);
    gaugeCircle.style.stroke = color;

    const triageBadge = document.getElementById('triageBadge');
    triageBadge.textContent = triage;
    triageBadge.style.background = `${color}20`;
    triageBadge.style.color = color;
    triageBadge.style.border = `1px solid ${color}60`;

    // 2. Update Subscale Triad Box Metrics
    const ann = data.ann_projections;
    updateSubscaleBox('dep', ann.depression);
    updateSubscaleBox('anx', ann.anxiety);
    updateSubscaleBox('str', ann.stress);

    // 3. Update Chart.js Radar Profile
    updateRadarChart(ann.depression.score, ann.anxiety.score, ann.stress.score);

    // 4. Update Explainable AI (XAI) Fired Rules Feed
    renderFiredRules(data.fuzzy_triage.top_fired_rules || []);

    // 5. Update Action Plan Box
    renderActionPlan(data.fuzzy_triage.action_plan, color);

  } catch (err) {
    console.error('Screening failed:', err);
    alert(`Assessment Error: ${err.message}`);
  } finally {
    analyzeBtn.innerHTML = `
      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2">
        <polygon points="5 3 19 12 5 21 5 3"></polygon>
      </svg>
      <span>Run Neuro-Fuzzy Assessment</span>
    `;
    analyzeBtn.disabled = false;
  }
}

function updateSubscaleBox(prefix, dataObj) {
  document.getElementById(`${prefix}ScoreVal`).textContent = `${dataObj.score.toFixed(1)}/42`;
  const tagEl = document.getElementById(`${prefix}LabelTag`);
  tagEl.textContent = dataObj.label;

  const colorMap = {
    'Normal': { bg: 'var(--color-normal-bg)', color: 'var(--color-normal)' },
    'Mild': { bg: 'var(--color-mild-bg)', color: 'var(--color-mild)' },
    'Moderate': { bg: 'var(--color-moderate-bg)', color: 'var(--color-moderate)' },
    'Severe': { bg: 'var(--color-severe-bg)', color: 'var(--color-severe)' },
    'Extremely Severe': { bg: 'var(--color-critical-bg)', color: 'var(--color-critical)' }
  };

  const style = colorMap[dataObj.label] || colorMap['Normal'];
  tagEl.style.background = style.bg;
  tagEl.style.color = style.color;
}

function renderFiredRules(rules) {
  const container = document.getElementById('rulesList');
  container.innerHTML = '';

  if (rules.length === 0) {
    container.innerHTML = `
      <div style="font-size:0.82rem; color:var(--text-muted); padding:0.5rem 0;">
        All indicators within normal baseline. No pathological triage rules activated.
      </div>
    `;
    return;
  }

  rules.forEach(r => {
    const card = document.createElement('div');
    card.className = 'rule-card';
    card.innerHTML = `
      <div class="rule-top-line">
        <span class="rule-title">Rule ${r.rule_id} — ${r.consequent} Tier</span>
        <span class="rule-weight-tag">Weight α = ${r.weight.toFixed(3)}</span>
      </div>
      <div class="rule-desc">${r.description}</div>
    `;
    container.appendChild(card);
  });
}

function renderActionPlan(plan, color) {
  const actionCard = document.getElementById('actionCard');
  actionCard.style.borderColor = `${color}40`;
  actionCard.style.background = `${color}10`;

  const titleEl = document.getElementById('actionTitle');
  titleEl.style.color = color;
  titleEl.innerHTML = `
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
      <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path>
      <polyline points="22 4 12 14.01 9 11.01"></polyline>
    </svg>
    ${plan.title}
  `;

  document.getElementById('actionText').textContent = plan.recommendation;

  const stepsList = document.getElementById('actionSteps');
  stepsList.innerHTML = '';
  plan.next_steps.forEach(s => {
    const li = document.createElement('li');
    li.textContent = s;
    stepsList.appendChild(li);
  });
}

/* --------------------------------------------------------------------------
   Benchmark Modal Table
   -------------------------------------------------------------------------- */
function renderBenchmarkTable(bdata) {
  const tbody = document.getElementById('benchmarkTbody');
  tbody.innerHTML = '';
  bdata.metrics.forEach(m => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${m.dimension}</strong></td>
      <td style="color:var(--text-muted);">${m.pure_ann}</td>
      <td style="color:var(--text-muted);">${m.pure_fis}</td>
      <td style="color:var(--primary-light); font-weight:700;">${m.hybrid_neuro_fuzzy}</td>
      <td>
        <span style="background:var(--color-normal-bg); color:var(--color-normal); font-size:0.75rem; padding:3px 8px; border-radius:6px; font-weight:700;">
          ${m.winner}
        </span>
      </td>
    `;
    tbody.appendChild(tr);
  });
}
