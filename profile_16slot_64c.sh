#!/usr/bin/env bash
# ==============================================================================
# Standalone Profiling Runner: Upstream Master vs. Phase-2 Modernized Architecture
# ==============================================================================

set -e

MODEL_PATH="${MODEL_PATH:-my_models/3b.gguf}"
SLOTS=16
CTX=32768
CLIENTS=64
REQUESTS=512
MAX_TOKENS=64
CANCEL=0.0
TARGET="phase2"

while [[ $# -gt 0 ]]; do
  case $1 in
    -m|--model)
      MODEL_PATH="$2"
      shift 2
      ;;
    -s|--slots)
      SLOTS="$2"
      shift 2
      ;;
    -c|--clients)
      CLIENTS="$2"
      shift 2
      ;;
    -n|--requests)
      REQUESTS="$2"
      shift 2
      ;;
    -k|--cancel)
      CANCEL="$2"
      shift 2
      ;;
    --ctx)
      CTX="$2"
      shift 2
      ;;
    -t|--target)
      TARGET="$2"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [-m model] [-s slots] [-c clients] [-n requests] [-t phase2|master|all]"
      exit 0
      ;;
    *)
      echo "Unknown flag: $1"
      exit 1
      ;;
  esac
done

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: Model file $MODEL_PATH not found!"
    exit 1
fi

mkdir -p ./profiles
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

CURRENT_SRV_PID=""
CURRENT_WRAPPER_PID=""

kill_current_server() {
    if [ -n "$CURRENT_SRV_PID" ]; then
        kill -INT "$CURRENT_SRV_PID" 2>/dev/null || true
        for i in {1..20}; do
            kill -0 "$CURRENT_SRV_PID" 2>/dev/null || break
            sleep 0.5
        done
        kill -9 "$CURRENT_SRV_PID" 2>/dev/null || true
        wait "$CURRENT_SRV_PID" 2>/dev/null || true
        CURRENT_SRV_PID=""
    fi
    if [ -n "$CURRENT_WRAPPER_PID" ]; then
        kill -INT "$CURRENT_WRAPPER_PID" 2>/dev/null || true
        kill -9 "$CURRENT_WRAPPER_PID" 2>/dev/null || true
        wait "$CURRENT_WRAPPER_PID" 2>/dev/null || true
        CURRENT_WRAPPER_PID=""
    fi
}

wait_server_ready() {
    local port="$1"
    for r in {1..120}; do
        if curl -s "http://localhost:${port}/health" 2>/dev/null | grep -q "ok"; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

prime_server() {
    local port="$1"
    curl -s -X POST "http://localhost:${port}/completion" \
      -H "Content-Type: application/json" \
      -d '{"prompt":"Warmup request","n_predict":4}' > /dev/null 2>&1 || true
}

profile_run() {
    local label="$1"
    local bin_path="$2"

    if [ ! -f "$bin_path" ]; then
        echo "Binary $bin_path not found, skipping."
        return 0
    fi

    echo "================================================================================"
    echo " [1/3] Unprofiled Baseline for $label ($REQUESTS Requests, $CLIENTS Clients)"
    echo "================================================================================"
    kill_current_server
    PORT_BASE=$((10000 + RANDOM % 40000))
    "$bin_path" -m "$MODEL_PATH" -c "$CTX" -np "$SLOTS" --port "$PORT_BASE" > "./profiles/srv_${label}_base_${TIMESTAMP}.log" 2>&1 &
    CURRENT_SRV_PID=$!
    wait_server_ready "$PORT_BASE"
    prime_server "$PORT_BASE"

    python3 tools/server/bench/bench_concurrency.py \
      --url "http://localhost:${PORT_BASE}" \
      -c "$CLIENTS" \
      -n "$REQUESTS" \
      -k "$CANCEL" \
      --max-tokens "$MAX_TOKENS" \
      --json-out "./profiles/bench_${label}_base_${TIMESTAMP}.json"

    kill_current_server
    sleep 1

    echo "================================================================================"
    echo " [2/3] Nsight Systems Profiling for $label (CUDA + NVTX)"
    echo "================================================================================"
    PORT_NSYS=$((10000 + RANDOM % 40000))
    NSYS_OUT="./profiles/nsys_${label}_${TIMESTAMP}"
    
    if nsys launch --help >/dev/null 2>&1; then
        nsys launch \
          --trace=cuda,nvtx \
          --cuda-graph-trace=node \
          --sample=cpu \
          --output="$NSYS_OUT" \
          --force-overwrite=true \
          "$bin_path" -m "$MODEL_PATH" -c "$CTX" -np "$SLOTS" --port "$PORT_NSYS" > "./profiles/srv_${label}_nsys_${TIMESTAMP}.log" 2>&1 &
        CURRENT_WRAPPER_PID=$!
        for t in {1..30}; do
            CURRENT_SRV_PID=$(pgrep -P "$CURRENT_WRAPPER_PID" llama-server 2>/dev/null || pgrep -f "^${bin_path}.*--port ${PORT_NSYS}" 2>/dev/null | head -n 1 || echo "")
            [ -n "$CURRENT_SRV_PID" ] && break
            sleep 0.5
        done
        wait_server_ready "$PORT_NSYS"
        prime_server "$PORT_NSYS"
        nsys start || true
        python3 tools/server/bench/bench_concurrency.py \
          --url "http://localhost:${PORT_NSYS}" \
          -c "$CLIENTS" \
          -n "$REQUESTS" \
          -k "$CANCEL" \
          --max-tokens "$MAX_TOKENS" \
          --json-out "./profiles/bench_${label}_nsys_${TIMESTAMP}.json"
        nsys stop || true
        kill_current_server
    else
        nsys profile \
          --trace=cuda,nvtx \
          --cuda-graph-trace=node \
          --sample=cpu \
          --delay=10 \
          --stop-on-exit=true \
          --output="$NSYS_OUT" \
          --force-overwrite=true \
          "$bin_path" -m "$MODEL_PATH" -c "$CTX" -np "$SLOTS" --port "$PORT_NSYS" > "./profiles/srv_${label}_nsys_${TIMESTAMP}.log" 2>&1 &
        CURRENT_WRAPPER_PID=$!
        CURRENT_SRV_PID=$(pgrep -P "$CURRENT_WRAPPER_PID" llama-server 2>/dev/null || echo "$CURRENT_WRAPPER_PID")
        wait_server_ready "$PORT_NSYS"
        prime_server "$PORT_NSYS"
        python3 tools/server/bench/bench_concurrency.py \
          --url "http://localhost:${PORT_NSYS}" \
          -c "$CLIENTS" \
          -n "$REQUESTS" \
          -k "$CANCEL" \
          --max-tokens "$MAX_TOKENS" \
          --json-out "./profiles/bench_${label}_nsys_${TIMESTAMP}.json"
        kill_current_server
    fi

    if [ -f "${NSYS_OUT}.nsys-rep" ]; then
        echo "Running GPU Busy/Idle & Bubble Gap Analysis (${NSYS_OUT}.nsys-rep)..."
        python3 gpu_gap_analysis.py "${NSYS_OUT}.nsys-rep" --bubble-us 50 | tee "./profiles/gpu_gap_${label}_${TIMESTAMP}.txt" || true
    fi

    echo "================================================================================"
    echo " [3/3] Linux perf for $label (Process-Wide, 499Hz DWARF, Context Switches)"
    echo "================================================================================"
    PORT_PERF=$((10000 + RANDOM % 40000))
    PERF_DATA="./profiles/perf_${label}_${TIMESTAMP}.data"
    STAT_OUT="./profiles/perf_stat_${label}_${TIMESTAMP}.txt"

    "$bin_path" -m "$MODEL_PATH" -c "$CTX" -np "$SLOTS" --port "$PORT_PERF" > "./profiles/srv_${label}_perf_${TIMESTAMP}.log" 2>&1 &
    CURRENT_SRV_PID=$!
    wait_server_ready "$PORT_PERF"
    prime_server "$PORT_PERF"

    perf record -p "$CURRENT_SRV_PID" -F 499 --call-graph dwarf,4096 -o "$PERF_DATA" &
    PERF_REC_PID=$!

    perf stat -p "$CURRENT_SRV_PID" -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions -o "$STAT_OUT" &
    PERF_STAT_PID=$!

    python3 tools/server/bench/bench_concurrency.py \
      --url "http://localhost:${PORT_PERF}" \
      -c "$CLIENTS" \
      -n "$REQUESTS" \
      -k "$CANCEL" \
      --max-tokens "$MAX_TOKENS" \
      --json-out "./profiles/bench_${label}_perf_${TIMESTAMP}.json"

    kill -INT "$PERF_REC_PID" 2>/dev/null || true
    wait "$PERF_REC_PID" 2>/dev/null || true
    kill -INT "$PERF_STAT_PID" 2>/dev/null || true
    wait "$PERF_STAT_PID" 2>/dev/null || true
    kill_current_server

    perf report -i "$PERF_DATA" --stdio --sort dso,sym --percent-limit 0.5 > "./profiles/perf_dso_sym_${label}_${TIMESTAMP}.txt" || true
    perf report -i "$PERF_DATA" --stdio --sort comm,tid,dso,sym --percent-limit 0.5 > "./profiles/perf_comm_tid_${label}_${TIMESTAMP}.txt" || true
}

if [ "$TARGET" = "phase2" ] || [ "$TARGET" = "all" ]; then
    profile_run "phase2" "./build-phase2/bin/llama-server"
fi

if [ "$TARGET" = "master" ] || [ "$TARGET" = "all" ]; then
    profile_run "master" "./build-master/bin/llama-server"
fi

echo "================================================================================"
echo " Profiling Run Completed. Outputs in ./profiles/"
echo "================================================================================"
