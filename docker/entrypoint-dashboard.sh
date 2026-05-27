#!/bin/sh
# entrypoint-dashboard.sh — Container startup script for Static HTML Dashboard
#
# Replaced Streamlit with nginx serving the standalone HTML SPA.
# The dashboard communicates with the FastAPI backend via HTTP fetch().
#
# At startup, this script:
#   1. Injects the API URL into index.html (replaces the apiBase default)
#   2. Replaces the __API_UPSTREAM__ placeholder in the nginx config
#   3. Starts nginx
#
# Environment variables:
#   AZURE_APP_URL  — Full URL to the FastAPI backend (e.g., https://api.example.com)
#                    If set, the dashboard HTML will use this as the API base.
set -e

echo "Starting Healthcare RAG Dashboard (static HTML SPA)..."
API_URL="${AZURE_APP_URL:-http://localhost:8000}"
echo "API endpoint: ${API_URL}"

INJECTED=false

# ── Step 1: Inject API URL into index.html ────────────────────────────────────
INDEX="/usr/share/nginx/html/index.html"
if [ -f "$INDEX" ]; then
    # Replace the apiBase default with the runtime value
    # Matches: apiBase: localStorage.getItem('rag_api_base') || 'http://localhost:8000'
    sed -i "s|apiBase: localStorage.getItem('rag_api_base') || 'http://localhost:8000'|apiBase: localStorage.getItem('rag_api_base') || '${API_URL}'|g" "$INDEX"
    echo "Index HTML configured with API URL: ${API_URL}"
    INJECTED=true
fi

# ── Step 2: Inject API upstream into nginx config ──────────────────────────────
NGINX_CONF="/etc/nginx/conf.d/dashboard.conf"
if [ -f "$NGINX_CONF" ]; then
    # Replace the __API_UPSTREAM__ placeholder with the actual upstream URL
    # This is needed because nginx does not support environment variables in config.
    sed -i "s|__API_UPSTREAM__|${API_URL}|g" "$NGINX_CONF"
    echo "Nginx config upstream set to: ${API_URL}"
    INJECTED=true
fi

if [ "$INJECTED" = false ]; then
    echo "WARNING: Could not find index.html or nginx config to inject API URL."
fi

# ── Step 3: Start nginx ────────────────────────────────────────────────────────
exec nginx -g "daemon off;"
