/**
 * Wizard and Dashboard Controller
 * Focus: Depression & Anxiety Screening using ANN & Fuzzy Logic
 */

import { fetchQuestions, fetchPersonas, submitScreening, fetchFuzzyVisuals, fetchBenchmark } from './api.js';
import { initRadarChart, updateRadarChart, initFuzzyCurvesChart } from './charts.js';

let allQuestions = [];
let responses = new Array(21).fill(0);
let fuzzyVisualsData = null;
let currentTab = 'all';

document.addEventListener('DOMContentLoaded', async () => {
  initRadarChart('radarCanvas');

  // Load questions and personas
  try {
    allQuestions = await fetchQuestions();
    renderQuestions();
    
    const personas = await fetchPersonas();
    renderPersonas(personas);

    fuzzyVisualsData = await fetchFuzzyVisuals();
    initFuzzyCurvesChart('fuzzyCanvas', fuzzyVisualsData, 'depression');

    // Run initial baseline screening
    handleScreening();
  } catch (err) {
    console.error('Initialization error:', err);
  }

  setupEventListeners();
});

function setupEventListeners() {
  // Contextual stress slider
  const slider = document.getElementById('stressSlider');
  const sliderVal = document.getElementById('stressSliderVal');
  slider.addEventListener('input', (e) => {
    sliderVal.textContent = e.target.value;
  });

  // Analyze button
  document.getElementById('analyzeBtn').addEventListener('click', () => {
    handleScreening();
  });

  // Tab buttons
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      e.target.classList.add('active');
      currentTab = e.target.dataset.tab;
      filterQuestions();
    });
  });

  // Modal triggers
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

  // Fuzzy curve subscale selector
  document.getElementById('fuzzySubscaleSelect').addEventListener('change', (e) => {
    if (fuzzyVisualsData) {
      initFuzzyCurvesChart('fuzzyCanvas', fuzzyVisualsData, e.target.value);
    }
  });

  // Print Report Button
  document.getElementById('printReportBtn').addEventListener('click', () => {
    window.print();
  });
}

function renderQuestions() {
  const container = document.getElementById('questionsContainer');
  container.innerHTML = '';

  const optionsLabels = [
    { val: 0, text: 'Never (0)' },
    { val: 1, text: 'Sometimes (1)' },
    { val: 2, text: 'Often (2)' },
    { val: 3, text: 'Almost Always (3)' }
  ];

  allQuestions.forEach((q, idx) => {
    const itemEl = document.createElement('div');
    itemEl.className = 'question-item';
    itemEl.dataset.subscale = q.subscale.toLowerCase();

    itemEl.innerHTML = `
      <div class="question-text">
        <span style="color:#818cf8; font-weight:700;">Q${q.id}.</span> ${q.text} 
        <span style="font-size:0.75rem; color:#94a3b8; margin-left:0.5rem;">[${q.subscale}]</span>
      </div>
      <div class="likert-options">
        ${optionsLabels.map(opt => `
          <button type="button" class="likert-btn ${responses[idx] === opt.val ? 'selected' : ''}" data-idx="${idx}" data-val="${opt.val}">
            ${opt.text}
          </button>
        `).join('')}
      </div>
    `;

    container.appendChild(itemEl);
  });

  // Add click listeners to likert buttons
  container.querySelectorAll('.likert-btn').forEach(btn => {
    btn.addEventListener('click', (e) => {
      const qIdx = parseInt(e.target.dataset.idx);
      const val = parseInt(e.target.dataset.val);
      responses[qIdx] = val;

      // Update button selection visual in this row
      const parentRow = e.target.parentElement;
      parentRow.querySelectorAll('.likert-btn').forEach(b => b.classList.remove('selected'));
      e.target.classList.add('selected');
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

function renderPersonas(personas) {
  const grid = document.getElementById('personaGrid');
  grid.innerHTML = '';

  personas.forEach(p => {
    const btn = document.createElement('button');
    btn.className = 'persona-btn';
    btn.innerHTML = `
      <strong>${p.name}</strong>
      <span>${p.subtitle}</span>
    `;
    btn.addEventListener('click', () => {
      responses = [...p.responses];
      document.getElementById('stressSlider').value = p.contextual_stress;
      document.getElementById('stressSliderVal').textContent = p.contextual_stress;

      // Re-render question selections
      renderQuestions();
      filterQuestions();

      // Trigger instant screening analysis
      handleScreening();
    });
    grid.appendChild(btn);
  });
}

async function handleScreening() {
  const analyzeBtn = document.getElementById('analyzeBtn');
  analyzeBtn.innerHTML = '<span>⚡ Evaluating ANN & Fuzzy Inference...</span>';
  analyzeBtn.disabled = true;

  try {
    const stressVal = document.getElementById('stressSlider').value;
    const data = await submitScreening(responses, stressVal);

    // Update Results UI
    const risk = data.summary.risk_index;
    const triage = data.summary.triage_category;
    const color = data.summary.triage_color;

    document.getElementById('gaugeVal').textContent = `${risk}%`;
    document.getElementById('gaugeDial').style.borderColor = color;
    
    const triageBadge = document.getElementById('triageBadge');
    triageBadge.textContent = triage;
    triageBadge.style.background = `${color}25`;
    triageBadge.style.color = color;
    triageBadge.style.border = `1px solid ${color}60`;

    // Subscale scores
    const ann = data.ann_projections;
    document.getElementById('depScoreVal').textContent = `${ann.depression.score}/42`;
    document.getElementById('depLabelTag').textContent = ann.depression.label;

    document.getElementById('anxScoreVal').textContent = `${ann.anxiety.score}/42`;
    document.getElementById('anxLabelTag').textContent = ann.anxiety.label;

    document.getElementById('strScoreVal').textContent = `${ann.stress.score}/42`;
    document.getElementById('strLabelTag').textContent = ann.stress.label;

    // Update Radar
    updateRadarChart(ann.depression.score, ann.anxiety.score, ann.stress.score);

    // Render Fired Fuzzy Rules (XAI)
    const rulesList = document.getElementById('rulesList');
    rulesList.innerHTML = '';
    const fired = data.fuzzy_triage.top_fired_rules || [];
    if (fired.length === 0) {
      rulesList.innerHTML = '<div style="color:#94a3b8; font-size:0.85rem;">No high-activation rules triggered. Baseline normal state.</div>';
    } else {
      fired.forEach(r => {
        const rEl = document.createElement('div');
        rEl.className = 'rule-chip';
        rEl.innerHTML = `
          <span class="weight-badge">α = ${r.weight}</span>
          <strong>Rule ${r.rule_id} [${r.consequent}]:</strong> ${r.description}
        `;
        rulesList.appendChild(rEl);
      });
    }

    // Action plan
    const plan = data.fuzzy_triage.action_plan;
    document.getElementById('actionTitle').textContent = plan.title;
    document.getElementById('actionText').textContent = plan.recommendation;
    const stepsList = document.getElementById('actionSteps');
    stepsList.innerHTML = '';
    plan.next_steps.forEach(s => {
      const li = document.createElement('li');
      li.textContent = s;
      stepsList.appendChild(li);
    });

  } catch (err) {
    console.error('Screening failed:', err);
    alert(`Screening analysis error: ${err.message}`);
  } finally {
    analyzeBtn.innerHTML = '<span>🚀 Run Neuro-Fuzzy Assessment</span>';
    analyzeBtn.disabled = false;
  }
}

function renderBenchmarkTable(bdata) {
  const tbody = document.getElementById('benchmarkTbody');
  tbody.innerHTML = '';
  bdata.metrics.forEach(m => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td><strong>${m.dimension}</strong></td>
      <td style="color:#94a3b8;">${m.pure_ann}</td>
      <td style="color:#94a3b8;">${m.pure_fis}</td>
      <td style="color:#818cf8; font-weight:600;">${m.hybrid_neuro_fuzzy}</td>
      <td><span style="background:rgba(16,185,129,0.2); color:#34d399; font-size:0.75rem; padding:2px 8px; border-radius:4px; font-weight:700;">${m.winner}</span></td>
    `;
    tbody.appendChild(tr);
  });
}
