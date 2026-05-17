"""
M3 Task 4 — 20-query latency test against the live Azure API.
Run: python scripts/latency_test.py
     AZURE_APP_URL=https://... python scripts/latency_test.py
"""
import requests
import time
import csv
import os
from datetime import datetime

BASE_URL = os.getenv("AZURE_APP_URL", "https://healthcare-rag-app.azurewebsites.net")
API_KEY = os.getenv("API_KEY", "")   # set in .env when auth is enabled
VALID_CATS = {"Symptoms", "Diagnosis", "Treatment", "Medication", "Prevention", "General"}
LIMIT_MS = 5000

TEST_QUERIES = [
    ("What are the early symptoms of type 2 diabetes?", "Symptoms"),
    ("What symptoms indicate a possible heart attack?", "Symptoms"),
    ("What are warning signs of kidney disease?", "Symptoms"),
    ("What symptoms are linked to iron deficiency anaemia?", "Symptoms"),
    ("How is hypertension diagnosed?", "Diagnosis"),
    ("What tests diagnose thyroid disorders?", "Diagnosis"),
    ("How do doctors diagnose celiac disease?", "Diagnosis"),
    ("What treatments are available for moderate asthma?", "Treatment"),
    ("How is Crohn's disease treated in adults?", "Treatment"),
    ("What is the treatment for mild depression?", "Treatment"),
    ("What are the common side effects of metformin?", "Medication"),
    ("Is it safe to take ibuprofen and paracetamol together?", "Medication"),
    ("What are the main uses of corticosteroids?", "Medication"),
    ("How can I reduce the risk of cardiovascular disease?", "Prevention"),
    ("What lifestyle changes help prevent type 2 diabetes?", "Prevention"),
    ("How effective are vaccines in preventing influenza?", "Prevention"),
    ("What is the difference between a virus and a bacterium?", "General"),
    ("How does the immune system respond to infection?", "General"),
    ("What is BMI and how is it calculated?", "General"),
    ("What does peer-reviewed medical research mean?", "General"),
]


def warm_up():
    print(f"Target: {BASE_URL}")
    print("Sending warm-up request (not counted in results)...")
    try:
        r = requests.get(f"{BASE_URL}/health", timeout=120)
        print(f"  Warm-up: HTTP {r.status_code} in {r.elapsed.total_seconds():.1f}s")
    except Exception as e:
        print(f"  Warm-up failed: {e}")
        print("  Azure Free Tier may be idle. Wait 90s and retry.")


def query_once(question: str, n: int) -> dict:
    try:
        t0 = time.perf_counter()
        headers = {"X-API-Key": API_KEY} if API_KEY else {}
        r = requests.post(
            f"{BASE_URL}/query",
            json={"question": question},
            headers=headers,
            timeout=30,
        )
        ms = round((time.perf_counter() - t0) * 1000, 1)
        if r.status_code == 200:
            d = r.json()
            return {
                "n": n, "question": question, "ms": ms,
                "category": d.get("category", "MISSING"),
                "disc_ok": bool(d.get("disclaimer")),
                "lat_pass": ms <= LIMIT_MS,
                "cat_valid": d.get("category") in VALID_CATS,
                "error": None,
            }
        return {"n": n, "question": question, "ms": ms, "category": "ERR",
                "disc_ok": False, "lat_pass": False, "cat_valid": False,
                "error": f"HTTP {r.status_code}"}
    except requests.exceptions.Timeout:
        return {"n": n, "question": question, "ms": 30000, "category": "TIMEOUT",
                "disc_ok": False, "lat_pass": False, "cat_valid": False,
                "error": "Timeout after 30s"}


if __name__ == "__main__":
    warm_up()
    print("Waiting 5s for warm instance...\n")
    time.sleep(5)

    results = []
    for i, (q, expected) in enumerate(TEST_QUERIES, 1):
        r = query_once(q, i)
        r["expected"] = expected
        results.append(r)
        icon = "✅" if r["lat_pass"] and r["disc_ok"] else "❌"
        print(f"  [{i:02d}/20] {icon} {r['ms']:>7.1f}ms | {r['category']:<12} | {q[:50]}")
        time.sleep(0.5)

    lat_pass = sum(1 for r in results if r["lat_pass"])
    disc_pass = sum(1 for r in results if r["disc_ok"])
    avg_ms = sum(r["ms"] for r in results) / len(results)
    max_ms = max(r["ms"] for r in results)

    print(f"""
{'='*55}
Latency ≤5000ms : {lat_pass}/20  {'✅ PASS' if lat_pass == 20 else '❌ FAIL'}
Disclaimer OK   : {disc_pass}/20  {'✅ PASS' if disc_pass == 20 else '❌ FAIL'}
Average latency : {avg_ms:.0f} ms
Max latency     : {max_ms:.0f} ms
Run date        : {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*55}""")

    os.makedirs("reports", exist_ok=True)
    with open("reports/latency_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "n", "question", "expected", "category",
            "ms", "disc_ok", "lat_pass", "cat_valid", "error"
        ])
        writer.writeheader()
        writer.writerows(results)
    print("Saved: reports/latency_results.csv")
