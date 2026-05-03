#!/usr/bin/env bash
set -e

echo "=== CaSVeM Setup ==="
cd "$(dirname "$0")"

# Ollama lives in /usr/local/bin — ensure it's on PATH
export PATH="$PATH:/usr/local/bin"

# 1. Install Ollama (skip if already present)
if ! command -v ollama &>/dev/null; then
  echo "[1/5] Installing Ollama..."
  curl -fsSL https://ollama.com/install.sh | sh
  export PATH="$PATH:/usr/local/bin"
else
  echo "[1/5] Ollama already installed: $(ollama --version)"
fi

# 2. Install Docker
if ! command -v docker &>/dev/null; then
  echo "[2/5] Installing Docker..."
  curl -fsSL https://get.docker.com | sh
  sudo usermod -aG docker "$USER"
  echo "      Docker installed."
  echo "      IMPORTANT: run 'newgrp docker' or log out/in to use Docker without sudo."
  # Use sudo for this session since group change hasn't taken effect yet
  DOCKER="sudo docker"
else
  echo "[2/5] Docker already installed: $(docker --version)"
  DOCKER="docker"
fi

# 3. Python virtual environment + dependencies
echo "[3/5] Setting up Python virtual environment..."
python3 -m venv --clear .venv
source .venv/bin/activate
pip install --upgrade pip --quiet
pip install -r requirements.txt --quiet
echo "      Python deps installed in .venv/"

# 4. Pull Ollama models
echo "[4/5] Pulling Ollama models..."
echo "      qwen3:1.7b       (~1.4 GB) — fast LLM for binary tasks"
echo "      qwen3:4b         (~2.5 GB) — strong LLM for JSON extraction"
echo "      nomic-embed-text (~274 MB) — embedding model"
ollama pull qwen3:1.7b
ollama pull qwen3:4b
ollama pull nomic-embed-text
echo "      Models ready."

# 5. Start Weaviate
echo "[5/5] Starting Weaviate (graph + vector DB)..."
docker compose up -d 2>/dev/null || sudo docker compose up -d
echo "      Waiting for Weaviate..."
for i in $(seq 1 30); do
  if curl -sf http://localhost:8080/v1/.well-known/ready &>/dev/null; then
    echo "      Weaviate is ready at http://localhost:8080"
    break
  fi
  sleep 2
  if [ "$i" -eq 30 ]; then
    echo "      Weaviate did not start in time. Check: docker compose logs weaviate"
  fi
done

echo ""
echo "=== Setup complete ==="
echo ""
echo "To start CaSVeM:"
echo "  source .venv/bin/activate"
echo "  python main.py"
echo ""
echo "API docs: http://localhost:8000/docs"
