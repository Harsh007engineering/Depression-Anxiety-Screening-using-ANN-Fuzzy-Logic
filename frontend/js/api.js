// Universal relative API base path: works on localhost, Railway, Docker, and any domain
const API_BASE = '';

export async function fetchQuestions() {
  const res = await fetch(`${API_BASE}/api/questions`);
  if (!res.ok) throw new Error(`Failed to load questions: ${res.statusText}`);
  return await res.json();
}

export async function fetchPersonas() {
  const res = await fetch(`${API_BASE}/api/personas`);
  if (!res.ok) throw new Error(`Failed to load personas: ${res.statusText}`);
  return await res.json();
}

export async function submitScreening(responses, contextualStress = 0.0) {
  const res = await fetch(`${API_BASE}/api/screen`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      responses: responses,
      contextual_stress: parseFloat(contextualStress)
    })
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || 'Failed to analyze assessment');
  }
  return await res.json();
}

export async function fetchFuzzyVisuals() {
  const res = await fetch(`${API_BASE}/api/fuzzy-visuals`);
  if (!res.ok) throw new Error(`Failed to load fuzzy visuals: ${res.statusText}`);
  return await res.json();
}

export async function fetchBenchmark() {
  const res = await fetch(`${API_BASE}/api/benchmark`);
  if (!res.ok) throw new Error(`Failed to load benchmark: ${res.statusText}`);
  return await res.json();
}
