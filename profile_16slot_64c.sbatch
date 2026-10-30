#!/bin/bash
#SBATCH --job-name=llama_profile_16s_64c
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=profile_16s_64c_%j.log
#SBATCH --error=profile_16s_64c_%j.err

set -e

# ==============================================================================
# Systems Profiling Pipeline: Upstream Master vs. Phase-2 Modernized Architecture
# Workload: 16 Slots (-np 16), 32K Context (-c 32768 -> 2048 ctx/slot)
# Traffic : 64 Concurrent Clients, 512 Requests (Steady-State Window)
#
# Instruments:
#  1. Unprofiled Baseline (Observer Effect Measurement)
#  2. NVIDIA Nsight Systems (nsys launch/start/stop, NVTX Phase Breakdown)
#  3. Linux perf (Process-wide -p, DWARF call-graph, dso,sym libcuda separation)
#  4. Context-Switch & Migration Efficiency (Thundering Herd Quantification)
# ==============================================================================

# 1. Environment & Module Setup
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc-13.1.0 cuda-12.8 2>/dev/null || module load gcc/13.1.0 cuda/12.8 2>/dev/null || true

export CC=gcc
export CXX=g++
export LLAMA_ARG_N_GPU_LAYERS=99

echo "================================================================================"
echo " [Step 0] Node Diagnostics & Tool Verification"
echo "================================================================================"
hostname
gcc --version | head -n 1
nvcc --version | grep "release" || true
nvidia-smi

echo -n "Checking nsys: "
which nsys || echo "WARNING: nsys not found in PATH"
echo -n "Checking perf: "
which perf || echo "WARNING: perf not found in PATH"
echo -n "perf_event_paranoid: "
cat /proc/sys/kernel/perf_event_paranoid 2>/dev/null || echo "N/A"

mkdir -p ./profiles
JOB_ID="${SLURM_JOB_ID:-manual}"
MODEL_FILE="my_models/3b.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found! Existing models in my_models/:"
    ls -la my_models/ || true
    exit 1
fi

# ==============================================================================
# Step 1: Compile Both Architectures (RelWithDebInfo + DWARF)
# ==============================================================================
echo "================================================================================"
echo " [Step 1] Compiling Modernized phase-2 (build-phase2)"
echo "================================================================================"
cmake -B build-phase2 \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX2=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" \
  -DCMAKE_BUILD_TYPE=RelWithDebInfo \
  -DLLAMA_BUILD_TESTS=OFF \
  -DLLAMA_BUILD_UI=OFF
cmake --build build-phase2 --config RelWithDebInfo -j 32 --target llama-server

echo "================================================================================"
echo " [Step 1] Compiling Upstream Master Baseline (build-master)"
echo "================================================================================"
if [ -d "llama.cpp-master" ]; then
    cmake -B build-master \
      -DGGML_CUDA=ON \
      -DGGML_NATIVE=OFF \
      -DGGML_AVX2=ON \
      -DCMAKE_CUDA_ARCHITECTURES="89" \
      -DCMAKE_BUILD_TYPE=RelWithDebInfo \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_UI=OFF \
      -S llama.cpp-master
    cmake --build build-master --config RelWithDebInfo -j 32 --target llama-server
else
    echo "WARNING: llama.cpp-master directory not found! Only phase-2 will be evaluated."
fi

# Thermal & Power Quiescence Barrier
wait_for_quiescence() {
    echo -n "Waiting for GPU thermal and power quiescence..."
    for i in {1..30}; do
        TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo "0")
        POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1 | cut -d'.' -f1 || echo "0")
        if [ "$TEMP" -le 50 ] && [ "$POWER" -le 60 ]; then
            echo " Ready (${TEMP}C, ${POWER}W)"
            return 0
        fi
        sleep 2
    done
    echo " Proceeding (${TEMP}C, ${POWER}W)"
}

# Process Manager Helpers (Strict PID-based tracking, NO broad pkill)
CURRENT_SRV_PID=""
CURRENT_WRAPPER_PID=""

kill_current_server() {
    if [ -n "$CURRENT_SRV_PID" ]; then
        echo "Stopping server process (PID: $CURRENT_SRV_PID)..."
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
        echo "Stopping wrapper process (PID: $CURRENT_WRAPPER_PID)..."
        kill -INT "$CURRENT_WRAPPER_PID" 2>/dev/null || true
        kill -9 "$CURRENT_WRAPPER_PID" 2>/dev/null || true
        wait "$CURRENT_WRAPPER_PID" 2>/dev/null || true
        CURRENT_WRAPPER_PID=""
    fi
    nsys shutdown 2>/dev/null || true
}

wait_server_ready() {
    local port="$1"
    for r in {1..120}; do # allow up to 60s for 16-slot CUDA allocation
        if curl -s "http://localhost:${port}/health" 2>/dev/null | grep -q "ok"; then
            return 0
        fi
        sleep 0.5
    done
    return 1
}

# Warmup helper: Prime CUDA context, page tables, and KV scratchpads
prime_server() {
    local port="$1"
    echo "Priming server with warmup request..."
    curl -s -X POST "http://localhost:${port}/completion" \
      -H "Content-Type: application/json" \
      -d '{"prompt":"Warmup request to initialize GPU memory buffers","n_predict":4}' > /dev/null 2>&1 || true
}

# ==============================================================================
# PROFILING HARNESS FUNCTION
# Evaluates a single binary target across:
#   1. Unprofiled Baseline (512 requests, 64 concurrency)
#   2. Nsight Systems (nsys launch/start/stop, steady-state capture)
#   3. Linux perf (Process-wide -p, 499Hz DWARF, dso,sym separation)
# ==============================================================================
profile_target() {
    local label="$1"
    local bin_path="$2"

    if [ ! -f "$bin_path" ]; then
        echo "Skipping $label: binary $bin_path does not exist."
        return 0
    fi

    echo "################################################################################"
    echo " STARTING PROFILING PIPELINE: $label ($bin_path)"
    echo "################################################################################"

    # --------------------------------------------------------------------------
    # Sub-Step A: Unprofiled Baseline (Reference Throughput)
    # --------------------------------------------------------------------------
    echo "--------------------------------------------------------------------------------"
    echo " [$label] Sub-Step A: Unprofiled Steady-State Baseline (512 Requests, 64 Clients)"
    echo "--------------------------------------------------------------------------------"
    kill_current_server
    wait_for_quiescence

    PORT_BASE=$((10000 + RANDOM % 40000))
    LOG_BASE="./profiles/server_${label}_base_${JOB_ID}.log"
    JSON_BASE="./profiles/bench_${label}_base_${JOB_ID}.json"

    "$bin_path" \
      -m "$MODEL_FILE" \
      -c 32768 \
      -np 16 \
      --port "$PORT_BASE" > "$LOG_BASE" 2>&1 &
    CURRENT_SRV_PID=$!

    if ! wait_server_ready "$PORT_BASE"; then
        echo "ERROR: Server failed to start for $label baseline!"
        tail -n 30 "$LOG_BASE"
        kill_current_server
        return 1
    fi

    prime_server "$PORT_BASE"

    echo "Running unprofiled benchmark: 64 clients, 512 requests..."
    python3 tools/server/bench/bench_concurrency.py \
      --url "http://localhost:${PORT_BASE}" \
      -c 64 \
      -n 512 \
      -k 0.0 \
      --max-tokens 64 \
      --json-out "$JSON_BASE"

    kill_current_server
    sleep 1

    # --------------------------------------------------------------------------
    # Sub-Step B: NVIDIA Nsight Systems (Targeted Steady-State Capture)
    # --------------------------------------------------------------------------
    echo "--------------------------------------------------------------------------------"
    echo " [$label] Sub-Step B: NVIDIA Nsight Systems (CUDA + NVTX Steady-State Capture)"
    echo "--------------------------------------------------------------------------------"
    kill_current_server
    wait_for_quiescence

    PORT_NSYS=$((10000 + RANDOM % 40000))
    NSYS_OUT="./profiles/nsys_${label}_${JOB_ID}"
    LOG_NSYS="./profiles/server_${label}_nsys_${JOB_ID}.log"
    JSON_NSYS="./profiles/bench_${label}_nsys_${JOB_ID}.json"

    # Use standalone nsys profile with targeted delay to avoid daemon session state locks
    NSYS_MODE="profile_delay"
    nsys shutdown 2>/dev/null || true
    pkill -9 -f "nsys" 2>/dev/null || true

    echo "Launching $bin_path with nsys profile (targeted delay)..."
    nsys profile \
      --trace=cuda,nvtx \
      --cuda-graph-trace=node \
      --sample=cpu \
      --delay=10 \
      --stop-on-exit=true \
      --output="$NSYS_OUT" \
      --force-overwrite=true \
      "$bin_path" \
        -m "$MODEL_FILE" \
        -c 16384 \
        -np 16 \
        --port "$PORT_NSYS" > "$LOG_NSYS" 2>&1 &
    CURRENT_WRAPPER_PID=$!
    for t in {1..30}; do
        CURRENT_SRV_PID=$(pgrep -P "$CURRENT_WRAPPER_PID" llama-server 2>/dev/null || pgrep -f "^${bin_path}.*--port ${PORT_NSYS}" 2>/dev/null | head -n 1 || echo "")
        [ -n "$CURRENT_SRV_PID" ] && break
        sleep 0.5
    done
    [ -z "$CURRENT_SRV_PID" ] && CURRENT_SRV_PID="$CURRENT_WRAPPER_PID"

    if ! wait_server_ready "$PORT_NSYS"; then
        echo "ERROR: Server failed to start under nsys for $label!"
        tail -n 30 "$LOG_NSYS"
        kill_current_server
        return 1
    fi

    prime_server "$PORT_NSYS"

    # Trigger profiler start explicitly if launch session is active
    if [ "$NSYS_MODE" = "launch_session" ]; then
        echo "Starting nsys capture right before steady-state client load..."
        nsys start --output="$NSYS_OUT" --force-overwrite=true || true
    fi

    echo "Running profiled benchmark under nsys: 64 clients, 512 requests..."
    python3 tools/server/bench/bench_concurrency.py \
      --url "http://localhost:${PORT_NSYS}" \
      -c 64 \
      -n 512 \
      -k 0.0 \
      --max-tokens 64 \
      --json-out "$JSON_NSYS"

    if [ "$NSYS_MODE" = "launch_session" ]; then
        echo "Stopping nsys capture..."
        nsys stop || true
    fi

    echo "Stopping server process cleanly (PID: $CURRENT_SRV_PID) to flush nsys trace..."
    if [ -n "$CURRENT_SRV_PID" ]; then
        kill -INT "$CURRENT_SRV_PID" 2>/dev/null || true
    fi
    if [ -n "$CURRENT_WRAPPER_PID" ]; then
        echo "Waiting for nsys wrapper (PID: $CURRENT_WRAPPER_PID) to finalize..."
        for w in {1..60}; do
            kill -0 "$CURRENT_WRAPPER_PID" 2>/dev/null || break
            sleep 1
        done
        wait "$CURRENT_WRAPPER_PID" 2>/dev/null || true
        CURRENT_WRAPPER_PID=""
    fi
    CURRENT_SRV_PID=""
    sleep 2

    if [ -f "${NSYS_OUT}.nsys-rep" ]; then
        echo "Extracting Nsight Systems Summary Statistics (${NSYS_OUT}.nsys-rep)..."
        nsys stats \
          --report cuda_api_sum,cuda_gpu_kern_sum,nvtx_sum \
          "${NSYS_OUT}.nsys-rep" > "./profiles/nsys_stats_${label}_${JOB_ID}.txt" 2>&1 || true
        head -n 45 "./profiles/nsys_stats_${label}_${JOB_ID}.txt" || true

        echo "Exporting SQLite database: ${NSYS_OUT}.sqlite..."
        nsys export --type=sqlite --force-overwrite=true -o "${NSYS_OUT}.sqlite" "${NSYS_OUT}.nsys-rep" 2>/dev/null || true

        echo "--------------------------------------------------------------------------------"
        echo " [$label] GPU Busy/Idle & Bubble Gap Analysis"
        echo "--------------------------------------------------------------------------------"
        python3 gpu_gap_analysis.py "${NSYS_OUT}.sqlite" --bubble-us 50 | tee "./profiles/gpu_gap_${label}_${JOB_ID}.txt" || true
    fi

    # --------------------------------------------------------------------------
    # Sub-Step C: Linux perf (Process-Wide, 499Hz DWARF, dso,sym separation)
    # --------------------------------------------------------------------------
    echo "--------------------------------------------------------------------------------"
    echo " [$label] Sub-Step C: Linux perf (Process-Wide, 499Hz DWARF, Context Switches)"
    echo "--------------------------------------------------------------------------------"
    kill_current_server
    wait_for_quiescence

    PORT_PERF=$((10000 + RANDOM % 40000))
    LOG_PERF="./profiles/server_${label}_perf_${JOB_ID}.log"
    JSON_PERF="./profiles/bench_${label}_perf_${JOB_ID}.json"
    PERF_DATA="./profiles/perf_${label}_${JOB_ID}.data"
    STAT_OUT="./profiles/perf_stat_${label}_${JOB_ID}.txt"

    "$bin_path" \
      -m "$MODEL_FILE" \
      -c 32768 \
      -np 16 \
      --port "$PORT_PERF" > "$LOG_PERF" 2>&1 &
    CURRENT_SRV_PID=$!

    if ! wait_server_ready "$PORT_PERF"; then
        echo "ERROR: Server failed to start for $label perf profiling!"
        tail -n 30 "$LOG_PERF"
        kill_current_server
        return 1
    fi

    # Verify Scheduler TID vs PID
    echo "Server PIDs & Thread Hierarchy:"
    ps -T -p "$CURRENT_SRV_PID" || true

    prime_server "$PORT_PERF"

    PERF_REC_PID=""
    PERF_STAT_PID=""
    if command -v perf &>/dev/null; then
        echo "Attaching process-wide Linux perf (-p $CURRENT_SRV_PID, 499Hz DWARF)..."
        perf record \
          -p "$CURRENT_SRV_PID" \
          -F 499 \
          --call-graph dwarf,4096 \
          -o "$PERF_DATA" &
        PERF_REC_PID=$!

        echo "Attaching process-wide perf stat for context switches & migrations..."
        perf stat \
          -p "$CURRENT_SRV_PID" \
          -e task-clock,context-switches,cpu-migrations,page-faults,cycles,instructions \
          -o "$STAT_OUT" &
        PERF_STAT_PID=$!
    else
        echo "WARNING: perf command not found in PATH on this compute node. Skipping PMU capture."
    fi

    sleep 0.5

    echo "Running profiled benchmark under perf: 64 clients, 512 requests..."
    python3 tools/server/bench/bench_concurrency.py \
      --url "http://localhost:${PORT_PERF}" \
      -c 64 \
      -n 512 \
      -k 0.0 \
      --max-tokens 64 \
      --json-out "$JSON_PERF"

    # Stop collectors
    kill -INT "$PERF_REC_PID" 2>/dev/null || true
    wait "$PERF_REC_PID" 2>/dev/null || true

    kill -INT "$PERF_STAT_PID" 2>/dev/null || true
    wait "$PERF_STAT_PID" 2>/dev/null || true

    kill_current_server
    sleep 1

    # Generate Reports
    if [ -f "$STAT_OUT" ]; then
        echo "--------------------------------------------------------------------------------"
        echo " [$label] Process-Wide Hardware Counters & Context Switches"
        echo "--------------------------------------------------------------------------------"
        cat "$STAT_OUT"
    fi

    if [ -f "$PERF_DATA" ]; then
        echo "--------------------------------------------------------------------------------"
        echo " [$label] Perf Breakdown sorted by DSO and Symbol (Distinguishing libcuda vs CPU)"
        echo "--------------------------------------------------------------------------------"
        REPORT_DSO="./profiles/perf_dso_sym_${label}_${JOB_ID}.txt"
        perf report \
          -i "$PERF_DATA" \
          --stdio \
          --sort dso,sym \
          --percent-limit 0.5 > "$REPORT_DSO" || true
        head -n 45 "$REPORT_DSO" || true

        echo "--------------------------------------------------------------------------------"
        echo " [$label] Perf Breakdown sorted by Thread (Scheduler vs HTTP Workers)"
        echo "--------------------------------------------------------------------------------"
        REPORT_COMM="./profiles/perf_comm_tid_${label}_${JOB_ID}.txt"
        perf report \
          -i "$PERF_DATA" \
          --stdio \
          --sort comm,tid,dso,sym \
          --percent-limit 0.5 > "$REPORT_COMM" || true
        head -n 45 "$REPORT_COMM" || true
    fi
}

# ==============================================================================
# Execute Profiling Across Both Architectures
# ==============================================================================
profile_target "phase2" "./build-phase2/bin/llama-server"

if [ -f "./build-master/bin/llama-server" ]; then
    profile_target "master" "./build-master/bin/llama-server"
fi

# ==============================================================================
# Final Comparative Summary: Observer Overhead & Context Switches per Token
# ==============================================================================
echo "================================================================================"
echo " FINAL SYSTEMS COMPARISON MATRIX"
echo "================================================================================"

python3 - << 'EOF'
import json, os, glob, re

def parse_summary(label):
    base_json = glob.glob(f"./profiles/bench_{label}_base_*.json")
    perf_json = glob.glob(f"./profiles/bench_{label}_perf_*.json")
    stat_txt  = glob.glob(f"./profiles/perf_stat_{label}_*.txt")
    
    res = {"label": label, "base_tok_s": 0.0, "perf_tok_s": 0.0, "tokens": 0, "cs": 0, "cs_per_tok": 0.0}
    
    if base_json and os.path.exists(base_json[-1]):
        try:
            d = json.load(open(base_json[-1]))
            res["base_tok_s"] = d.get("tok_per_sec", 0.0)
        except: pass

    if perf_json and os.path.exists(perf_json[-1]):
        try:
            d = json.load(open(perf_json[-1]))
            res["perf_tok_s"] = d.get("tok_per_sec", 0.0)
            res["tokens"] = d.get("total_tokens", 0)
        except: pass

    if stat_txt and os.path.exists(stat_txt[-1]):
        try:
            for line in open(stat_txt[-1]):
                if "context-switches" in line:
                    match = re.search(r'([0-9,]+)\s+context-switches', line)
                    if match:
                        res["cs"] = int(match.group(1).replace(',', ''))
        except: pass

    if res["tokens"] > 0 and res["cs"] > 0:
        res["cs_per_tok"] = res["cs"] / float(res["tokens"])

    return res

print(f"{'Target':<10} | {'Unprofiled (tok/s)':<18} | {'Profiled (tok/s)':<16} | {'Observer Overhead':<18} | {'Context Switches/Tok':<20}")
print("-" * 92)

for target in ["master", "phase2"]:
    s = parse_summary(target)
    if s["base_tok_s"] > 0:
        overhead = ((s["base_tok_s"] - s["perf_tok_s"]) / s["base_tok_s"]) * 100.0 if s["base_tok_s"] > 0 else 0.0
        print(f"{s['label']:<10} | {s['base_tok_s']:<18.2f} | {s['perf_tok_s']:<16.2f} | {overhead:<17.1f}% | {s['cs_per_tok']:<20.2f}")

def parse_gap(label):
    files = sorted(glob.glob(f"./profiles/gpu_gap_{label}_*.txt"))
    if not files: return None
    t = open(files[-1]).read()
    busy = re.search(r"GPU busy\s*:\s*([\d\.]+)%", t)
    bubbles = re.search(r"Bubbles >= 50 us\s*:\s*(\d+)\s*gaps,\s*([\d\.]+)%", t)
    mean_b = re.search(r"mean bubble\s*:\s*(\d+)\s*us", t)
    step_t = re.search(r"implied step time\s*:\s*([\d\.]+)\s*ms", t)
    return {
        "busy": float(busy.group(1)) if busy else 0.0,
        "bubble_pct": float(bubbles.group(2)) if bubbles else 0.0,
        "mean_bubble_us": int(mean_b.group(1)) if mean_b else 0,
        "step_ms": float(step_t.group(1)) if step_t else 0.0
    }

m = parse_gap("master")
p = parse_gap("phase2")
if m and p:
    print("\n" + "=" * 92)
    print(" COMPARATIVE GPU BUBBLE & IDLE GAP MATRIX (Nsight Systems SQLite)")
    print("=" * 92)
    print(f"{'Metric':<30} | {'Upstream Master':<18} | {'Phase-2 Lock-Free':<18} | {'Delta':<15}")
    print("-" * 92)
    print(f"{'GPU Busy Share':<30} | {m['busy']:<18.1f}% | {p['busy']:<18.1f}% | {p['busy'] - m['busy']:+.1f}%")
    print(f"{'GPU Bubble Share (>=50us)':<30} | {m['bubble_pct']:<18.1f}% | {p['bubble_pct']:<18.1f}% | {p['bubble_pct'] - m['bubble_pct']:+.1f}%")
    print(f"{'Mean Bubble Duration':<30} | {m['mean_bubble_us']:<18} us | {p['mean_bubble_us']:<18} us | {p['mean_bubble_us'] - m['mean_bubble_us']:+d} us")
    print(f"{'Implied Step Time':<30} | {m['step_ms']:<18.2f} ms | {p['step_ms']:<18.2f} ms | {p['step_ms'] - m['step_ms']:+.2f} ms")

EOF

echo "================================================================================"
echo " Profiling Job Complete! All reports and raw profiles stored in ./profiles/"
echo "================================================================================"
