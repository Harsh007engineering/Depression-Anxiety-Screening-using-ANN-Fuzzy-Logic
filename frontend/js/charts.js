/**
 * Visualization Module using Chart.js
 * Renders Radar charts and Fuzzy Membership Function curves
 */

let radarChartInstance = null;
let fuzzyChartInstance = null;

export function initRadarChart(canvasId) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  radarChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Depression Severity', 'Anxiety Arousal', 'Tension / Stress'],
      datasets: [
        {
          label: 'Patient Score (0-42)',
          data: [0, 0, 0],
          backgroundColor: 'rgba(99, 102, 241, 0.35)',
          borderColor: '#818cf8',
          borderWidth: 2,
          pointBackgroundColor: '#ffffff',
          pointBorderColor: '#6366f1',
          pointHoverRadius: 6
        },
        {
          label: 'Moderate Threshold',
          data: [14, 10, 19],
          borderColor: 'rgba(245, 158, 11, 0.6)',
          borderDash: [4, 4],
          borderWidth: 1.5,
          pointRadius: 0,
          fill: false
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0,
          max: 42,
          ticks: {
            stepSize: 7,
            color: '#94a3b8',
            backdropColor: 'transparent'
          },
          grid: { color: 'rgba(255, 255, 255, 0.08)' },
          angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
          pointLabels: {
            color: '#f8fafc',
            font: { size: 11, weight: '600' }
          }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#cbd5e1', font: { size: 11 } }
        }
      }
    }
  });
}

export function updateRadarChart(depScore, anxScore, strScore) {
  if (!radarChartInstance) return;
  radarChartInstance.data.datasets[0].data = [depScore, anxScore, strScore];
  radarChartInstance.update();
}

export function initFuzzyCurvesChart(canvasId, visualsData, currentSubscale = 'depression', currentScore = 0) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  
  const xLabels = visualsData.dass_x;
  const subscaleData = visualsData[currentSubscale];

  const colors = {
    Normal: '#10b981',
    Mild: '#06b6d4',
    Moderate: '#f59e0b',
    Severe: '#f97316',
    Extremely_Severe: '#ef4444'
  };

  const datasets = Object.keys(subscaleData).map(term => ({
    label: term.replace('_', ' '),
    data: subscaleData[term],
    borderColor: colors[term] || '#818cf8',
    backgroundColor: 'transparent',
    borderWidth: 2,
    pointRadius: 0,
    tension: 0.1
  }));

  if (fuzzyChartInstance) {
    fuzzyChartInstance.destroy();
  }

  fuzzyChartInstance = new Chart(ctx, {
    type: 'line',
    data: {
      labels: xLabels,
      datasets: datasets
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: {
          title: { display: true, text: `${currentSubscale.toUpperCase()} Score (0-42)`, color: '#94a3b8' },
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        },
        y: {
          min: 0,
          max: 1.0,
          title: { display: true, text: 'Membership Degree μ(x)', color: '#94a3b8' },
          ticks: { color: '#64748b' },
          grid: { color: 'rgba(255, 255, 255, 0.05)' }
        }
      },
      plugins: {
        legend: {
          labels: { color: '#cbd5e1', font: { size: 10 } }
        }
      }
    }
  });
}
