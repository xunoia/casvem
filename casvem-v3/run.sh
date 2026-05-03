#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

# ── Colors ────────────────────────────────────────────────────────────────────
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; CYAN='\033[0;36m'; NC='\033[0m'

echo -e "${CYAN}CaSVeM v3 — startup${NC}"
echo "────────────────────────────────"

# ── .env check ────────────────────────────────────────────────────────────────
if [ ! -f .env ]; then
    echo -e "${YELLOW}No .env found. Copying from .env.example...${NC}"
    cp .env.example .env
    echo -e "${RED}  → Open .env and set GEMINI_API_KEY before continuing.${NC}"
    exit 1
fi

source .env
if [ -z "$GEMINI_API_KEY" ] || [ "$GEMINI_API_KEY" = "your_gemini_api_key_here" ]; then
    echo -e "${RED}GEMINI_API_KEY not set in .env${NC}"
    echo "  Get a free key at: https://aistudio.google.com/apikey"
    exit 1
fi
echo -e "${GREEN}✓ .env loaded${NC}"

# ── Venv check ────────────────────────────────────────────────────────────────
if [ ! -d venv ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi
echo -e "${GREEN}✓ venv ready${NC}"

# ── Install / sync dependencies ───────────────────────────────────────────────
echo "Checking dependencies..."
venv/bin/pip install -r requirements.txt -q
echo -e "${GREEN}✓ dependencies up to date${NC}"

# ── Data directory ────────────────────────────────────────────────────────────
mkdir -p data
echo -e "${GREEN}✓ data/ directory ready${NC}"

echo "────────────────────────────────"
echo -e "${CYAN}Starting CaSVeM API on http://localhost:8000${NC}"
echo "  POST /memory  — add a memory"
echo "  POST /query   — query memories"
echo "  GET  /stats   — cache stats + cost"
echo "  DELETE /memory/{id}"
echo ""
echo "  Dashboard: runs in terminal alongside the server"
echo "  Ctrl-C to stop"
echo "────────────────────────────────"

venv/bin/uvicorn main:app --host 0.0.0.0 --port 8000 --reload
