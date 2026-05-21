#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# Healthcare RAG — Post-Deployment Test Script
#
# Run after deploy.sh to verify your Azure deployment is working.
# Usage:
#   bash azure/test_deployment.sh https://healthcare-rag-app.azurewebsites.net
#   bash azure/test_deployment.sh  (reads AZURE_APP_URL from .env)
# ──────────────────────────────────────────────────────────────────────────────
set -e

# Load .env if present
[ -f ".env" ] && source .env

BASE_URL="${1:-$AZURE_APP_URL}"

if [ -z "$BASE_URL" ]; then
    echo "❌ Usage: bash azure/test_deployment.sh <url>"
    echo "   Or set AZURE_APP_URL in .env"
    exit 1
fi

# Strip trailing slash
BASE_URL="${BASE_URL%/}"

echo "════════════════════════════════════════════════════════════════"
echo "  Healthcare RAG — Deployment Test"
echo "  Target: $BASE_URL"
echo "════════════════════════════════════════════════════════════════"
echo ""

# ── Test 1: Root endpoint ─────────────────────────────────────────────────────
echo "▶ Test 1/4 — Root endpoint (GET /)"
ROOT=$(curl -sf "${BASE_URL}/" 2>/dev/null || echo '{"error":"failed"}')
echo "  Response: $ROOT"
if echo "$ROOT" | grep -q "Healthcare RAG"; then
    echo "  ✅ Root endpoint OK"
else
    echo "  ❌ Root endpoint failed"
fi

echo ""

# ── Test 2: Health check ──────────────────────────────────────────────────────
echo "▶ Test 2/4 — Health check (GET /health)"
HEALTH=$(curl -sf "${BASE_URL}/health" 2>/dev/null || echo '{"status":"error"}')
echo "  Response: $HEALTH"
if echo "$HEALTH" | grep -q '"status":"ok"'; then
    echo "  ✅ Health check OK"
    echo "  $(echo $HEALTH | python3 -c "import sys,json; h=json.load(sys.stdin); print(f'  model_loaded={h.get(\"model_loaded\")}, groq_configured={h.get(\"groq_configured\")}, index_vectors={h.get(\"index_vectors\",0):,}')" 2>/dev/null || true)"
else
    echo "  ⚠️  Health endpoint reachable but status not ok — models may still be loading"
    echo "  Wait 2–3 more minutes and retry."
fi

echo ""

# ── Test 3: Medical query ─────────────────────────────────────────────────────
echo "▶ Test 3/4 — Medical query (POST /query)"
QUERY_BODY='{"question":"Does aspirin reduce the risk of cardiovascular events in high-risk patients?"}'

# Add API key header if set
if [ -n "$API_KEY" ]; then
    AUTH_HEADER='-H "X-API-Key: '"$API_KEY"'"'
else
    AUTH_HEADER=""
fi

QUERY_RESP=$(curl -sf \
    -X POST \
    -H "Content-Type: application/json" \
    ${API_KEY:+-H "X-API-Key: $API_KEY"} \
    -d "$QUERY_BODY" \
    "${BASE_URL}/query" 2>/dev/null || echo '{"error":"failed"}')

if echo "$QUERY_RESP" | grep -q '"answer"'; then
    ANSWER=$(echo "$QUERY_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('answer','')[:200])" 2>/dev/null || echo "parse error")
    CATEGORY=$(echo "$QUERY_RESP" | python3 -c "import sys,json; r=json.load(sys.stdin); print(r.get('category',''))" 2>/dev/null || echo "")
    echo "  ✅ Query endpoint OK"
    echo "  Category: $CATEGORY"
    echo "  Answer:   ${ANSWER}..."
else
    echo "  ❌ Query endpoint failed"
    echo "  Response: $QUERY_RESP"
fi

echo ""

# ── Test 4: Swagger UI ────────────────────────────────────────────────────────
echo "▶ Test 4/4 — Swagger UI (GET /docs)"
SWAGGER_STATUS=$(curl -so /dev/null -w "%{http_code}" "${BASE_URL}/docs" 2>/dev/null || echo "000")
if [ "$SWAGGER_STATUS" = "200" ]; then
    echo "  ✅ Swagger UI available at: ${BASE_URL}/docs"
else
    echo "  ❌ Swagger UI returned HTTP $SWAGGER_STATUS"
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Test complete."
echo "  Full Swagger UI: ${BASE_URL}/docs"
echo "════════════════════════════════════════════════════════════════"
