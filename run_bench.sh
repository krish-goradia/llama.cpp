#!/usr/bin/env bash
set -e

if [ -z "$MODEL_PATH" ]; then
  if [ -f "my_models/3b.gguf" ]; then
    MODEL_PATH="my_models/3b.gguf"
  else
    MODEL_PATH="my_models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
  fi
fi
PORT="${PORT:-8080}"
SLOTS="${SLOTS:-8}"
CTX="${CTX:-16384}"
CONCURRENCY="${CONCURRENCY:-8}"
REQUESTS="${REQUESTS:-32}"
MAX_TOKENS="${MAX_TOKENS:-64}"
CANCEL_RATE="${CANCEL_RATE:-0.0}"
DURATION="${DURATION:-0.0}"
RATE="${RATE:-0.0}"
JSON_OUT=""
DO_BUILD=true
RESTART_SERVER=true
SERVER_LOG="/tmp/llama_server_${USER:-bench}_$$.log"

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
    -d|--duration)
      DURATION="$2"
      shift 2
      ;;
    -r|--rate)
      RATE="$2"
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
    --seed)
      SEED="$2"
      shift 2
      ;;
    --json-out)
      JSON_OUT="$2"
      shift 2
      ;;
    --keep-server)
      RESTART_SERVER=false
      shift
      ;;
    -h|--help)
      echo "Usage: ./run_bench.sh [-c CONCURRENCY] [-n REQUESTS] [-k RATE] [--max-tokens TOKENS] [-m MODEL] [-s SLOTS] [-p PORT] [--ctx CTX] [--no-build] [--seed SEED]"
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

if [ "$DO_BUILD" = true ]; then
  cmake --build build --config Release -j --target llama-server
fi

if [ "$RESTART_SERVER" = true ]; then
  pkill -9 -f "llama-server" 2>/dev/null || true
  sleep 1

  SERVER_PREFIX=""
  if [ "$PIN_CORES" = "true" ] && command -v taskset &>/dev/null; then
    SERVER_PREFIX="taskset -c ${SERVER_CORES:-0-15}"
    echo "Core Pinning: Pinning server to cores ${SERVER_CORES:-0-15}"
  fi

  $SERVER_PREFIX ./build/bin/llama-server \
    -m "$MODEL_PATH" \
    -c "$CTX" \
    -np "$SLOTS" \
    --port "$PORT" > "$SERVER_LOG" 2>&1 &
  
  SERVER_PID=$!
  echo "Server started with PID: $SERVER_PID (logs at $SERVER_LOG)"

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
    cat "$SERVER_LOG" | tail -n 30
    exit 1
  fi
  echo "Server is READY."
fi

JSON_ARG=""
if [ -n "$JSON_OUT" ]; then
  JSON_ARG="--json-out $JSON_OUT"
fi

DURATION_ARG=""
if [ -n "$DURATION" ] && [ "$DURATION" != "0.0" ] && [ "$DURATION" != "0" ]; then
  DURATION_ARG="--duration $DURATION"
fi

RATE_ARG=""
if [ -n "$RATE" ] && [ "$RATE" != "0.0" ] && [ "$RATE" != "0" ]; then
  RATE_ARG="--rate $RATE"
fi

SEED_ARG=""
if [ -n "$SEED" ]; then
  SEED_ARG="--seed $SEED"
fi

CLIENT_PREFIX=""
if [ "$PIN_CORES" = "true" ] && command -v taskset &>/dev/null; then
  CLIENT_PREFIX="taskset -c ${CLIENT_CORES:-16-31}"
fi

$CLIENT_PREFIX python3 tools/server/bench/bench_concurrency.py \
  --url "http://localhost:${PORT}" \
  -c "$CONCURRENCY" \
  -n "$REQUESTS" \
  --max-tokens "$MAX_TOKENS" \
  --cancel-rate "$CANCEL_RATE" \
  $JSON_ARG \
  $DURATION_ARG \
  $RATE_ARG \
  $SEED_ARG
