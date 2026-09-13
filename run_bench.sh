#!/usr/bin/env bash
# ==============================================================================
# llama.cpp Automated Server Build, Launch & Benchmark Script
# ==============================================================================
# Usage:
#   ./run_bench.sh [options]
#
# Examples:
#   ./run_bench.sh -c 8 -n 32 --cancel-rate 0.0
#   ./run_bench.sh -c 8 -n 32 --cancel-rate 0.2
#   ./run_bench.sh -c 16 -n 32 --cancel-rate 0.0
#   ./run_bench.sh --no-build  # Skip compilation step
# ==============================================================================

set -e

# Default settings
MODEL_PATH="${MODEL_PATH:-my_models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
SLOTS="${SLOTS:-8}"
CTX="${CTX:-4096}"
CONCURRENCY="${CONCURRENCY:-8}"
REQUESTS="${REQUESTS:-32}"
MAX_TOKENS="${MAX_TOKENS:-64}"
CANCEL_RATE="${CANCEL_RATE:-0.0}"
DO_BUILD=true
RESTART_SERVER=true

# Parse command-line flags
while [[ $# -gt 0 ]]; do
  case $1 in
    -c|--concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    -n|--requests)
      REQUESTS="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    -k|--cancel|--cancel-rate|-cr)
      CANCEL_RATE="$2"
      shift 2
      ;;
    -m|--model)
      MODEL_PATH="$2"
      shift 2
      ;;
    -s|--slots)
      SLOTS="$2"
      shift 2
      ;;
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    --ctx)
      CTX="$2"
      shift 2
      ;;
    --no-build)
      DO_BUILD=false
      shift
      ;;
    --keep-server)
      RESTART_SERVER=false
      shift
      ;;
    -h|--help)
      echo "Usage: ./run_bench.sh [-c CONCURRENCY] [-n REQUESTS] [-k|--cancel|--cancel-rate RATE] [--max-tokens TOKENS] [-m MODEL] [-s SLOTS] [-p PORT] [--ctx CTX] [--no-build] [--keep-server]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 1. Step 1: Incremental Build
if [ "$DO_BUILD" = true ]; then
  echo "=================================================================="
  echo "[1/3] Building llama-server (Release mode)..."
  echo "=================================================================="
  cmake --build build --config Release -j --target llama-server
  echo "Build complete."
  echo ""
fi

# 2. Step 2: Manage llama-server lifecycle
if [ "$RESTART_SERVER" = true ]; then
  echo "=================================================================="
  echo "[2/3] Starting llama-server on port $PORT ($SLOTS slots)..."
  echo "=================================================================="
  
  # Kill any existing llama-server instances
  pkill -9 -f "llama-server" 2>/dev/null || true
  sleep 1

  # Launch server in background
  ./build/bin/llama-server \
    -m "$MODEL_PATH" \
    -c "$CTX" \
    -np "$SLOTS" \
    --port "$PORT" > /tmp/llama_server_bench.log 2>&1 &
  
  SERVER_PID=$!
  echo "Server started with PID: $SERVER_PID (logs at /tmp/llama_server_bench.log)"

  # Wait for server health endpoint to be ready
  echo "Waiting for server to initialize..."
  READY=false
  for i in {1..30}; do
    if curl -s "http://localhost:${PORT}/health" | grep -q "ok"; then
      READY=true
      break
    fi
    sleep 1
  done

  if [ "$READY" = false ]; then
    echo "ERROR: Server failed to start within 30 seconds."
    echo "--- Server Logs ---"
    cat /tmp/llama_server_bench.log | tail -n 30
    exit 1
  fi
  echo "Server is READY."
  echo ""
else
  echo "[2/3] Using existing running llama-server."
  echo ""
fi

# 3. Step 3: Run Benchmark
echo "=================================================================="
echo "[3/3] Running Concurrency Benchmark..."
echo "  Concurrency : $CONCURRENCY clients"
echo "  Requests    : $REQUESTS total"
echo "  Max Tokens  : $MAX_TOKENS"
echo "  Cancel Rate : $CANCEL_RATE"
echo "=================================================================="
echo ""

python3 tools/server/bench/bench_concurrency.py \
  --url "http://localhost:${PORT}" \
  -c "$CONCURRENCY" \
  -n "$REQUESTS" \
  --max-tokens "$MAX_TOKENS" \
  --cancel-rate "$CANCEL_RATE"
