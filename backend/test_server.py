import uvicorn
import threading
import time
import urllib.request
import json

def run_server():
    uvicorn.run('app.main:app', host='127.0.0.1', port=8001, log_level='error')

if __name__ == '__main__':
    t = threading.Thread(target=run_server, daemon=True)
    t.start()
    time.sleep(2.5)

    try:
        # Test 1: Health
        with urllib.request.urlopen('http://127.0.0.1:8001/api/health') as res:
            health = json.loads(res.read().decode())
            print("1. Health check:", health)

        # Test 2: Questions
        with urllib.request.urlopen('http://127.0.0.1:8001/api/questions') as res:
            questions = json.loads(res.read().decode())
            print(f"2. Standard DASS-21 loaded: {len(questions)} items")

        # Test 3: Screening inference
        payload = json.dumps({'responses': [2, 1, 3, 1, 3, 2, 1, 2, 1, 3, 1, 2, 3, 1, 1, 3, 3, 1, 1, 1, 3], 'contextual_stress': 7.5}).encode('utf-8')
        req = urllib.request.Request('http://127.0.0.1:8001/api/screen', data=payload, headers={'Content-Type': 'application/json'})
        with urllib.request.urlopen(req) as res:
            screen_res = json.loads(res.read().decode())
            print("3. Screening Output:")
            print("   - Raw DASS-21 Scores:", screen_res['dass21_raw_scores'])
            print("   - ANN Projections:", {k: v['score'] for k, v in screen_res['ann_projections'].items()})
            print("   - Fuzzy Triage Summary:", screen_res['summary'])
            print(f"   - Fired Mamdani Rules: {len(screen_res['fuzzy_triage']['top_fired_rules'])} rules")

        # Test 4: Fuzzy Visuals
        with urllib.request.urlopen('http://127.0.0.1:8001/api/fuzzy-visuals') as res:
            fviz = json.loads(res.read().decode())
            print(f"4. Fuzzy Membership curves computed: {len(fviz['dass_x'])} sample points")

        # Test 5: Benchmark
        with urllib.request.urlopen('http://127.0.0.1:8001/api/benchmark') as res:
            bench = json.loads(res.read().decode())
            print(f"5. Benchmark metrics available: {len(bench['metrics'])} dimensions")

        # Test 6: Frontend Index HTML
        with urllib.request.urlopen('http://127.0.0.1:8001/') as res:
            html = res.read().decode()
            print(f"6. Frontend index served: HTTP {res.status}, length: {len(html)} characters")

        print("\nALL 6 END-TO-END SYSTEM INTEGRATION TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print("Test failed with error:", e)
        raise
