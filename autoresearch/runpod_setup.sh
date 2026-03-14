#!/usr/bin/env bash
# 🚀 RunPod setup for DWARF autoresearch (35M tier)
#
# Usage: bash autoresearch/runpod_setup.sh [git-branch]
#
# Assumes:
#   - RunPod H100 pod with CUDA drivers
#   - Network volume at /workspace/data/ (optional but recommended)
#   - No ANTHROPIC_API_KEY needed (35M loop is pure Python)
#
# Cache files needed on /workspace/data/:
#   fineweb_encoded_2048.pt          (~3-4 GB, training data)
#   condm_fineweb_edu_doc_cache.json (document cache)

set -euo pipefail

BRANCH="${1:-main}"
REPO_URL="https://github.com/Lanerra/wave-field-llm.git"
WORKSPACE="/workspace"
REPO_DIR="${WORKSPACE}/DWARF"
DATA_DIR="${WORKSPACE}/data"

echo "════════════════════════════════════════════════════════════════"
echo "  🚀 DWARF AutoResearch — RunPod Setup"
echo "  Branch: ${BRANCH}"
echo "════════════════════════════════════════════════════════════════"

# ── 1. Clone or pull repo ─────────────────────────────────────────
echo ""
echo "  📦 Setting up repository..."

if [ -d "${REPO_DIR}/.git" ]; then
    echo "  Repo exists, pulling latest..."
    cd "${REPO_DIR}"
    git fetch origin
    git checkout "${BRANCH}"
    git pull origin "${BRANCH}"
else
    echo "  Cloning ${REPO_URL}..."
    git clone "${REPO_URL}" "${REPO_DIR}"
    cd "${REPO_DIR}"
    git checkout "${BRANCH}"
fi

echo "  ✓ Repo ready at ${REPO_DIR}"

# ── 2. Install Python deps ───────────────────────────────────────
echo ""
echo "  📦 Installing Python dependencies..."

if command -v uv &>/dev/null; then
    echo "  Using uv..."
    uv pip install -e "${REPO_DIR}" 2>/dev/null || uv pip install -r "${REPO_DIR}/requirements.txt" 2>/dev/null || true
    uv pip install triton torch 2>/dev/null || true
elif command -v pip &>/dev/null; then
    echo "  Using pip..."
    pip install -e "${REPO_DIR}" 2>/dev/null || pip install -r "${REPO_DIR}/requirements.txt" 2>/dev/null || true
    pip install triton 2>/dev/null || true
else
    echo "  ✗ Neither uv nor pip found. Install Python package manager first."
    exit 1
fi

echo "  ✓ Dependencies installed"

# ── 3. Check for training data cache ─────────────────────────────
echo ""
echo "  📂 Checking training data cache..."

CACHE_FILE="fineweb_encoded_2048.pt"
DOC_CACHE_FILE="condm_fineweb_edu_doc_cache.json"
CACHE_OK=true

if [ ! -d "${DATA_DIR}" ]; then
    echo "  ⚠  Network volume not mounted at ${DATA_DIR}"
    echo "     Mount your RunPod network volume or create ${DATA_DIR} manually."
    echo "     Required files:"
    echo "       ${DATA_DIR}/${CACHE_FILE}"
    echo "       ${DATA_DIR}/${DOC_CACHE_FILE}"
    CACHE_OK=false
fi

if [ -d "${DATA_DIR}" ]; then
    if [ ! -f "${DATA_DIR}/${CACHE_FILE}" ]; then
        echo "  ⚠  Missing: ${DATA_DIR}/${CACHE_FILE}"
        echo "     Copy from your local machine or re-encode FineWeb-Edu."
        CACHE_OK=false
    else
        SIZE=$(du -h "${DATA_DIR}/${CACHE_FILE}" | cut -f1)
        echo "  ✓ ${CACHE_FILE} (${SIZE})"
    fi

    if [ ! -f "${DATA_DIR}/${DOC_CACHE_FILE}" ]; then
        echo "  ⚠  Missing: ${DATA_DIR}/${DOC_CACHE_FILE}"
        echo "     Copy from your local machine."
        CACHE_OK=false
    else
        echo "  ✓ ${DOC_CACHE_FILE}"
    fi
fi

# ── 4. Symlink cache into expected locations ─────────────────────
echo ""
echo "  🔗 Setting up symlinks..."

LOGS_DIR="${REPO_DIR}/benchmarks/logs"
mkdir -p "${LOGS_DIR}"

if [ -f "${DATA_DIR}/${CACHE_FILE}" ]; then
    ln -sf "${DATA_DIR}/${CACHE_FILE}" "${LOGS_DIR}/${CACHE_FILE}"
    echo "  ✓ ${CACHE_FILE} -> ${LOGS_DIR}/"
fi

if [ -f "${DATA_DIR}/${DOC_CACHE_FILE}" ]; then
    ln -sf "${DATA_DIR}/${DOC_CACHE_FILE}" "${LOGS_DIR}/${DOC_CACHE_FILE}"
    echo "  ✓ ${DOC_CACHE_FILE} -> ${LOGS_DIR}/"
fi

# ── 5. Create autoresearch output directories ────────────────────
echo ""
echo "  📁 Creating output directories..."

mkdir -p "${REPO_DIR}/autoresearch/runs"
mkdir -p "${REPO_DIR}/autoresearch/results"
mkdir -p "${REPO_DIR}/autoresearch/checkpoints"
echo "  ✓ autoresearch/{runs,results,checkpoints}"

# ── 6. Create tmux session ───────────────────────────────────────
echo ""
echo "  🖥  Setting up tmux session..."

if command -v tmux &>/dev/null; then
    if tmux has-session -t autoresearch 2>/dev/null; then
        echo "  tmux session 'autoresearch' already exists"
    else
        tmux new-session -d -s autoresearch -c "${REPO_DIR}"
        echo "  ✓ tmux session 'autoresearch' created (detached)"
    fi
else
    echo "  ⚠  tmux not found, installing..."
    apt-get update -qq && apt-get install -y -qq tmux
    tmux new-session -d -s autoresearch -c "${REPO_DIR}"
    echo "  ✓ tmux installed and session 'autoresearch' created"
fi

# ── 7. Verify GPU ────────────────────────────────────────────────
echo ""
echo "  🔧 GPU check..."
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)
    echo "  ✓ ${GPU_NAME} (${GPU_MEM})"
else
    echo "  ⚠  nvidia-smi not found"
fi

# ── 8. Print instructions ────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  ✓ Setup complete!"
echo ""
echo "  To run the 35M autoresearch loop:"
echo ""
echo "    tmux attach -t autoresearch"
echo ""
echo "    cd ${REPO_DIR}"
echo "    python autoresearch/loop_35m.py \\"
echo "      --candidates autoresearch/candidates_35m.json \\"
echo "      --steps 2000 \\"
echo "      --out autoresearch/results_35m.tsv"
echo ""
echo "  With Discord webhook:"
echo ""
echo "    python autoresearch/loop_35m.py \\"
echo "      --candidates autoresearch/candidates_35m.json \\"
echo "      --steps 2000 \\"
echo "      --webhook 'https://discord.com/api/webhooks/...' \\"
echo "      --out autoresearch/results_35m.tsv"
echo ""

if [ "${CACHE_OK}" = false ]; then
    echo "  ⚠  WARNING: Training data cache is missing or incomplete."
    echo "     The loop will fail until cache files are in ${DATA_DIR}/"
    echo ""
fi

echo "  Detach tmux with Ctrl+B, D"
echo "  Re-attach with: tmux attach -t autoresearch"
echo "════════════════════════════════════════════════════════════════"
