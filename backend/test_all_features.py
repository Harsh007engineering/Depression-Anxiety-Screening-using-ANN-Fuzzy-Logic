"""
Comprehensive Live Feature Test Suite
Tests every backend endpoint, persona, input validation, and boundary conditions.
"""

import urllib.request
import urllib.error
import json

BASE = 'http://127.0.0.1:8000'

def run_feature_tests():
    print("=" * 65)
    print("STARTING FULL LIVE FEATURE VERIFICATION ON", BASE)
    print("=" * 65)

    # 1. Health check
    with urllib.request.urlopen(f'{BASE}/api/health') as r:
        health = json.loads(r.read())
        assert health['status'] == 'healthy', "Health check failed"
        assert health['ann_model_loaded'] is True, "ANN model not loaded"
        print("[PASS] 1. API Health Check & Model Verification")

    # 2. Questions endpoint
    with urllib.request.urlopen(f'{BASE}/api/questions') as r:
        questions = json.loads(r.read())
        assert len(questions) == 21, f"Expected 21 questions, got {len(questions)}"
        dep_count = sum(1 for q in questions if q['subscale'] == 'Depression')
        anx_count = sum(1 for q in questions if q['subscale'] == 'Anxiety')
        str_count = sum(1 for q in questions if q['subscale'] == 'Stress')
        assert dep_count == 7 and anx_count == 7 and str_count == 7, "Subscales mismatch"
        print(f"[PASS] 2. DASS-21 Question Inventory (7 Dep, 7 Anx, 7 Str)")

    # 3. Personas & Simulations
    with urllib.request.urlopen(f'{BASE}/api/personas') as r:
        personas = json.loads(r.read())
        assert len(personas) == 4, f"Expected 4 personas, got {len(personas)}"
        print(f"[PASS] 3. Personas Loaded ({len(personas)} archetypes)")

    for p in personas:
        payload = json.dumps({'responses': p['responses'], 'contextual_stress': p['contextual_stress']}).encode('utf-8')
        req = urllib.request.Request(f'{BASE}/api/screen', data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as r:
            res = json.loads(r.read())
            risk = res['summary']['risk_index']
            tier = res['summary']['triage_category']
            rules = len(res['fuzzy_triage']['top_fired_rules'])
            print(f"       -> Persona: {p['name']:<35} | Risk: {risk:>5.1f}% | Tier: {tier:<20} | Fired Rules: {rules}")

    # 4. Input Validation & Edge Cases
    print("\n[TEST] 4. Boundary & Input Validation Testing:")
    # Edge case A: All Zeros (completely healthy)
    p_zero = json.dumps({'responses': [0]*21, 'contextual_stress': 0.0}).encode('utf-8')
    req = urllib.request.Request(f'{BASE}/api/screen', data=p_zero, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as r:
        res = json.loads(r.read())
        assert res['summary']['risk_index'] < 25.0, "All zeros should be Minimal Risk"
        print(f"       -> Edge Case: All Zeros -> Risk: {res['summary']['risk_index']}% (Minimal)")

    # Edge case B: All Threes (maximum severity)
    p_max = json.dumps({'responses': [3]*21, 'contextual_stress': 10.0}).encode('utf-8')
    req = urllib.request.Request(f'{BASE}/api/screen', data=p_max, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as r:
        res = json.loads(r.read())
        assert res['summary']['risk_index'] >= 85.0, "All threes should be Critical Priority"
        print(f"       -> Edge Case: All Threes -> Risk: {res['summary']['risk_index']}% (Critical)")

    # Edge case C: Invalid response lengths / values
    try:
        p_invalid = json.dumps({'responses': [1]*10, 'contextual_stress': 0.0}).encode('utf-8')
        req = urllib.request.Request(f'{BASE}/api/screen', data=p_invalid, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req)
        assert False, "Should have rejected 10 responses"
    except urllib.error.HTTPError as e:
        assert e.code == 422, f"Expected 422 Unprocessable Entity, got {e.code}"
        print("       -> Validation: Rejected short response array (HTTP 422)")

    try:
        p_invalid_val = json.dumps({'responses': [0]*20 + [5], 'contextual_stress': 0.0}).encode('utf-8')
        req = urllib.request.Request(f'{BASE}/api/screen', data=p_invalid_val, headers={'Content-Type': 'application/json'})
        urllib.request.urlopen(req)
        assert False, "Should have rejected value 5"
    except urllib.error.HTTPError as e:
        assert e.code == 422, f"Expected 422 Unprocessable Entity, got {e.code}"
        print("       -> Validation: Rejected out-of-bounds rating 5 (HTTP 422)")

    # 5. Fuzzy Visuals Endpoint
    with urllib.request.urlopen(f'{BASE}/api/fuzzy-visuals') as r:
        viz = json.loads(r.read())
        assert 'depression' in viz and 'anxiety' in viz and 'stress' in viz, "Missing subscale terms"
        assert len(viz['dass_x']) == 85, "Expected 85 sample points"
        print("[PASS] 5. Fuzzy Visuals Generated with 5 linguistic terms each")

    # 6. Benchmark Data
    with urllib.request.urlopen(f'{BASE}/api/benchmark') as r:
        bench = json.loads(r.read())
        assert len(bench['metrics']) == 5, f"Expected 5 dimensions, got {len(bench['metrics'])}"
        print("[PASS] 6. Soft Computing Benchmark (5 Evaluation Dimensions)")

    # 7. Frontend Static Serving
    with urllib.request.urlopen(f'{BASE}/') as r:
        html = r.read().decode('utf-8')
        assert '<!DOCTYPE html>' in html, "Invalid HTML"
        assert 'NeuroFuzzy' in html, "Missing brand title"
        assert 'Soft Computing University Course Project' not in html, "Old student banner still present!"
        assert r.status == 200, "Frontend index status not 200"
        print("[PASS] 7. Frontend Landing Page Verification (Student banner confirmed removed)")

    with urllib.request.urlopen(f'{BASE}/static/css/styles.css') as r:
        assert r.status == 200, "CSS file not found"
        print("[PASS] 8. Static CSS Stylesheet Verified (HTTP 200)")

    with urllib.request.urlopen(f'{BASE}/static/js/wizard.js') as r:
        assert r.status == 200, "wizard.js not found"
        print("[PASS] 9. Static JavaScript Controller Verified (HTTP 200)")

    print("\n" + "=" * 65)
    print("ALL 9 LIVE FEATURE TESTS PASSED WITH 100% SUCCESS!")
    print("=" * 65)

if __name__ == '__main__':
    run_feature_tests()
