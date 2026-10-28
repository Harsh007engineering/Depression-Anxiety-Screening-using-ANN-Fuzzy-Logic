/**
 * High-Performance Chart.js Visualizer
 * Dual Theme Support (Light/Dark) + Dynamic Patient Score Markers
 */

let radarChartInstance = null;
let fuzzyChartInstance = null;

function getThemeColors() {
  const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
  return {
    gridColor: isDark ? 'rgba(255, 255, 255, 0.08)' : 'rgba(15, 23, 42, 0.08)',
    labelColor: isDark ? '#cbd5e1' : '#334155',
    tickColor: isDark ? '#64748b' : '#94a3b8',
    pointLabelColor: isDark ? '#f8fafc' : '#0f172a'
  };
}

export function initRadarChart(canvasId) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const theme = getThemeColors();

  if (radarChartInstance) {
    radarChartInstance.destroy();
  }

  radarChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: ['Depression Severity', 'Autonomic Anxiety', 'Nervous Tension / Stress'],
      datasets: [
        {
          label: 'Patient Severity (0-42)',
          data: [0, 0, 0],
          backgroundColor: 'rgba(99, 102, 241, 0.28)',
          borderColor: '#6366f1',
          borderWidth: 2.5,
          pointBackgroundColor: '#ffffff',
          pointBorderColor: '#4f46e5',
          pointBorderWidth: 2,
          pointRadius: 4,
          pointHoverRadius: 7
        },
        {
          label: 'Moderate Clinical Threshold',
          data: [14, 10, 19],
          borderColor: 'rgba(245, 158, 11, 0.7)',
          borderDash: [5, 5],
          borderWidth: 1.8,
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
            color: theme.tickColor,
            backdropColor: 'transparent',
            font: { size: 10, weight: '500' }
          },
          grid: { color: theme.gridColor },
          angleLines: { color: theme.gridColor },
          pointLabels: {
            color: theme.pointLabelColor,
            font: { size: 11, weight: '700' }
          }
        }
      },
      plugins: {
        legend: {
          position: 'bottom',
          labels: {
            color: theme.labelColor,
            font: { size: 11, weight: '600' },
            boxWidth: 14,
            padding: 12
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          titleFont: { weight: '700' },
          padding: 10,
          cornerRadius: 8
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

export function initFuzzyCurvesChart(canvasId, visualsData, currentSubscale = 'depression', currentPatientScore = 0) {
  const canvas = document.getElementById(canvasId);
  if (!canvas) return;
  const ctx = canvas.getContext('2d');
  const theme = getThemeColors();

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
    borderWidth: 2.2,
    pointRadius: 0,
    tension: 0.1
  }));

  // Add Patient Score Marker line/point
  if (currentPatientScore !== undefined && currentPatientScore !== null) {
    const clampedScore = Math.max(0, Math.min(42, currentPatientScore));
    // Find closest index in xLabels
    let closestIdx = 0;
    let minDiff = 999;
    xLabels.forEach((x, i) => {
      const diff = Math.abs(x - clampedScore);
      if (diff < minDiff) {
        minDiff = diff;
        closestIdx = i;
      }
    });

    const markerData = new Array(xLabels.length).fill(null);
    markerData[closestIdx] = 1.0;

    datasets.push({
      label: `Patient Score: ${clampedScore.toFixed(1)} pt`,
      data: markerData,
      borderColor: '#ec4899',
      backgroundColor: '#ec4899',
      pointBackgroundColor: '#ffffff',
      pointBorderColor: '#ec4899',
      pointBorderWidth: 3,
      pointRadius: 8,
      pointHoverRadius: 10,
      showLine: false
    });
  }

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
          title: {
            display: true,
            text: `${currentSubscale.toUpperCase()} Score Scale (0 to 42 points)`,
            color: theme.labelColor,
            font: { weight: '700', size: 11 }
          },
          ticks: { color: theme.tickColor, maxTicksLimit: 15 },
          grid: { color: theme.gridColor }
        },
        y: {
          min: 0,
          max: 1.0,
          title: {
            display: true,
            text: 'Degree of Membership μ(x) [0.0 - 1.0]',
            color: theme.labelColor,
            font: { weight: '700', size: 11 }
          },
          ticks: { color: theme.tickColor, stepSize: 0.2 },
          grid: { color: theme.gridColor }
        }
      },
      plugins: {
        legend: {
          position: 'top',
          labels: {
            color: theme.labelColor,
            font: { size: 11, weight: '600' },
            boxWidth: 16,
            padding: 10
          }
        },
        tooltip: {
          backgroundColor: 'rgba(15, 23, 42, 0.92)',
          padding: 10,
          cornerRadius: 8
        }
      }
    }
  });
}

export function refreshChartsTheme() {
  if (radarChartInstance) {
    const theme = getThemeColors();
    radarChartInstance.options.scales.r.grid.color = theme.gridColor;
    radarChartInstance.options.scales.r.angleLines.color = theme.gridColor;
    radarChartInstance.options.scales.r.ticks.color = theme.tickColor;
    radarChartInstance.options.scales.r.pointLabels.color = theme.pointLabelColor;
    radarChartInstance.options.plugins.legend.labels.color = theme.labelColor;
    radarChartInstance.update();
  }
}
