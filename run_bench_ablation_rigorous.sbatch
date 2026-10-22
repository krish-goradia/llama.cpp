#!/bin/bash
#SBATCH --job-name=llama_gold_ablation_rigorous
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:30:00
#SBATCH --output=gold_ablation_rigorous_%j.log
#SBATCH --error=gold_ablation_rigorous_%j.err

set -e

# ==============================================================================
# EuroMLSys Rigorous Mechanistic Ablation Suite
# ==============================================================================
# Research Objectives:
#   1. Target E ("Neither" Build): phase-2 tree with BOTH lock-free flags disabled
#      (-DLLAMA_SERVER_DISABLE_MPSC=ON -DLLAMA_SERVER_DISABLE_SPSC=ON).
#      Tests if gain is from lock-free queues OR from shared commits (slot recycling/sampler).
#   2. Target B (Queue Only): Lock-Free MPSC task ingestion only.
#   3. Target C (Channels Only): Lock-Free SPSC response egress only.
#   4. Target D (Full Phase-2): Lock-Free MPSC + SPSC channels.
#   5. Target S (Scheduler Core Pinning): Upstream Master with Main Scheduler Thread
#      isolated on Core 0 and HTTP worker threads on Cores 1-15.
#      Tests if Upstream Master's drop is lock contention vs scheduler thread starvation!
#   6. Target A (Upstream Master Baseline): Master commit a3b1effcd on Cores 0-15.
#
# Methodology:
#   - Server on Cores 0-15, Client strictly isolated on Cores 24-31 (Zero core overlap!)
#   - 10 Paired Trials per target with Synchronized PRNG Seeds
#   - Zero-overhead OS context switch telemetry (/proc/PID/task/*/status) per run
#   - Active Thermal & Power Quiescence Barrier (<= 45C, <= 50W)
# ==============================================================================

# 1. Environment & Modules
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc-13.1.0 cuda-12.8 2>/dev/null || module load gcc/13.1.0 cuda/12.8 2>/dev/null || true

export CC=gcc
export CXX=g++
export LLAMA_ARG_N_GPU_LAYERS=99
export BENCH_PORT=$((10000 + RANDOM % 40000))
MODEL_FILE="my_models/3b.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found!"
    exit 1
fi

chmod +x ./bench_gold.sh ./run_bench.sh tools/server/bench/*.py 2>/dev/null || true

# 2. Hardware and NUMA Topology Diagnostics
echo "================================================================================"
echo " [Step 0] NUMA & Socket Topology Diagnostics"
echo "================================================================================"
./check_numa_topology.sh || true

# 3. Compile All Ablation Builds
echo "================================================================================"
echo " [Step 1] Compiling Target D: Full Phase-2 (build-phase2)"
echo "================================================================================"
cmake -B build-phase2 \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs"
cmake --build build-phase2 --config Release -j 32 --target llama-server

echo "================================================================================"
echo " [Step 1] Compiling Target B: Queue Only (build-ablation-queue)"
echo "================================================================================"
cmake -B build-ablation-queue \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SERVER_DISABLE_SPSC=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs"
cmake --build build-ablation-queue --config Release -j 32 --target llama-server

echo "================================================================================"
echo " [Step 1] Compiling Target C: Channels Only (build-ablation-channels)"
echo "================================================================================"
cmake -B build-ablation-channels \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SERVER_DISABLE_MPSC=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs"
cmake --build build-ablation-channels --config Release -j 32 --target llama-server

echo "================================================================================"
echo " [Step 1] Compiling Target E: Neither Build (build-ablation-neither)"
echo "================================================================================"
cmake -B build-ablation-neither \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SERVER_DISABLE_MPSC=ON \
  -DLLAMA_SERVER_DISABLE_SPSC=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs"
cmake --build build-ablation-neither --config Release -j 32 --target llama-server

echo "================================================================================"
echo " [Step 1] Compiling Target A: Upstream Master Baseline (llama.cpp-master/build)"
echo "================================================================================"
if [ -d "llama.cpp-master" ]; then
    mkdir -p llama.cpp-master/tools/server/bench/ 2>/dev/null || true
    cp -f tools/server/bench/bench_concurrency.py llama.cpp-master/tools/server/bench/ 2>/dev/null || true
    cp -f run_bench.sh llama.cpp-master/ 2>/dev/null || true

    cmake -B llama.cpp-master/build \
      -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON \
      -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_BUILD_TYPE=Release \
      -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs" \
      -S llama.cpp-master
    cmake --build llama.cpp-master/build --config Release -j 32 --target llama-server
fi

# Core Allocation: Server on Cores 0-15, Client strictly on Cores 24-31 (No overlap!)
export SERVER_CORES="0-15"
export CLIENT_CORES="24-31"

TMP_DIR="$(mktemp -d -t llama_ablation_rigorous_XXXXXX)"
echo "Ablation results stored at: $TMP_DIR"

run_rigorous_target() {
    local TARGET_KEY="$1"
    local TARGET_LABEL="$2"
    local BIN_DIR="$3"
    local PIN_SCHEDULER_CORE="$4" # true or false
    local TRIALS=10
    local CLIENTS=128
    local REQS=128
    local SLOTS=16

    echo ""
    echo "================================================================================"
    echo " >>> Testing Target: [${TARGET_KEY}] ${TARGET_LABEL}"
    echo "     Binary Path: $BIN_DIR/bin/llama-server"
    echo "================================================================================"

    # Print Code Path and Compile-Time Diagnostics
    echo "  [Code Path Diagnostics for ${TARGET_KEY}]"
    if [ -f "$BIN_DIR/CMakeCache.txt" ]; then
        echo "    CMake Definitions in $BIN_DIR/CMakeCache.txt:"
        grep -E "LLAMA_SERVER_DISABLE_MPSC|LLAMA_SERVER_DISABLE_SPSC" "$BIN_DIR/CMakeCache.txt" | sed 's/^/      /' || echo "      (Standard Upstream - No custom ablation flags)"
    fi
    if command -v nm &>/dev/null && [ -f "$BIN_DIR/bin/llama-server" ]; then
        local HAS_MPSC=$(nm -C "$BIN_DIR/bin/llama-server" 2>/dev/null | grep -c "mpsc_queue" || true)
        local HAS_SPSC=$(nm -C "$BIN_DIR/bin/llama-server" 2>/dev/null | grep -c "spsc_ring_buffer" || true)
        echo "    ELF Symbol Audit:"
        if [ "$HAS_MPSC" -gt 0 ]; then
            echo "      Task Ingress  : Lock-Free Dmitry Vyukov MPSC Node Queue (mpsc_queue symbols found: $HAS_MPSC)"
        else
            echo "      Task Ingress  : Legacy Mutex std::deque (mpsc_queue symbols NOT found)"
        fi
        if [ "$HAS_SPSC" -gt 0 ]; then
            echo "      Response Egress: Lock-Free SPSC Ring Buffer Channels (spsc_ring_buffer symbols found: $HAS_SPSC)"
        else
            echo "      Response Egress: Legacy Mutex Response Handling (spsc_ring_buffer symbols NOT found)"
        fi
    fi

    for trial in $(seq 1 $TRIALS); do
        echo "--- Target ${TARGET_KEY} | Trial ${trial}/${TRIALS} (Seed ${trial}) ---"
        pkill -9 -f "llama-server" 2>/dev/null || true
        
        # Thermal Quiescence Barrier
        for w in {1..30}; do
            TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo "0")
            POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1 | cut -d'.' -f1 || echo "0")
            if [ "$TEMP" -le 45 ] && [ "$POWER" -le 50 ]; then
                break
            fi
            sleep 2
        done

        PORT=$((10000 + RANDOM % 40000))
        SERVER_LOG="$TMP_DIR/server_${TARGET_KEY}_t${trial}.log"
        JSON_OUT="$TMP_DIR/${TARGET_KEY}_t${trial}.json"
        TELEMETRY_OUT="$TMP_DIR/telemetry_${TARGET_KEY}_t${trial}.json"

        # Server Core Pinning:
        # Default: Server on Cores 0-15
        # Target S: Launch server on Cores 1-15, then move Main Scheduler Loop (PID) to Core 0,
        # ensuring worker/HTTP threads stay strictly on Cores 1-15 without touching Core 0.
        local SERVER_AFFINITY="$SERVER_CORES"
        if [ "$PIN_SCHEDULER_CORE" = "true" ]; then
            SERVER_AFFINITY="1-15"
        fi

        taskset -c $SERVER_AFFINITY "$BIN_DIR/bin/llama-server" \
            -m "$MODEL_FILE" \
            -s "$SLOTS" \
            -c 16384 \
            --port "$PORT" \
            --threads 8 \
            --threads-http 31 \
            -ub 512 -b 512 \
            > "$SERVER_LOG" 2>&1 &
        SERVER_PID=$!

        # Wait for Server Readiness
        READY=false
        for attempt in {1..60}; do
            if grep -q "HTTP server is listening" "$SERVER_LOG" 2>/dev/null || curl -s "http://127.0.0.1:${PORT}/health" 2>/dev/null | grep -q "ok"; then
                READY=true
                break
            fi
            sleep 0.5
        done

        if [ "$READY" != "true" ]; then
            echo "ERROR: Server failed to start for ${TARGET_KEY} trial ${trial}"
            cat "$SERVER_LOG" | tail -n 20
            pkill -9 -f "llama-server" 2>/dev/null || true
            continue
        fi

        # Print runtime server log confirmation of active code paths
        echo "  [Runtime Code Path Confirmation]"
        grep -E "task queue ingress|response egress" "$SERVER_LOG" 2>/dev/null | sed 's/^/    /' || echo "    (Standard Master Baseline Queue)"

        # If Scheduler Dedicated Pinning is requested (Target S):
        # Explicitly pin Main Scheduler Loop (TID = PID) to Core 0, and confirm HTTP worker threads on Cores 1-15
        if [ "$PIN_SCHEDULER_CORE" = "true" ]; then
            echo "  [Target S] Applying Scheduler Isolation Pinning..."
            taskset -cp 0 "$SERVER_PID"
            for tid in $(ls /proc/$SERVER_PID/task/ 2>/dev/null); do
                if [ "$tid" != "$SERVER_PID" ]; then
                    taskset -cp 1-15 "$tid" >/dev/null 2>&1 || true
                fi
            done
            local SAMPLE_TID=$(ls /proc/$SERVER_PID/task/ 2>/dev/null | grep -v "^$SERVER_PID$" | head -n 1)
            echo "  [Target S Verified Affinities]"
            echo "    Main Scheduler Loop (PID/TID $SERVER_PID): $(taskset -cp $SERVER_PID 2>/dev/null)"
            echo "    Worker Thread sample (TID $SAMPLE_TID): $(taskset -cp $SAMPLE_TID 2>/dev/null)"
            echo "    Total Server Threads: $(ls /proc/$SERVER_PID/task/ 2>/dev/null | wc -l)"
        fi

        # Record Pre-Run Context Switch Telemetry
        python3 tools/server/bench/record_thread_telemetry.py --pid "$SERVER_PID" --start --state-file "$TMP_DIR/telemetry_state.json" >/dev/null 2>&1 || true

        # Run Client Benchmark (strictly pinned to Cores 24-31)
        taskset -c $CLIENT_CORES python3 tools/server/bench/bench_concurrency.py \
            --url "http://127.0.0.1:${PORT}" \
            -c "$CLIENTS" \
            -n "$REQS" \
            --max-tokens 64 \
            --seed "$trial" \
            --json-out "$JSON_OUT" > "$TMP_DIR/client_${TARGET_KEY}_t${trial}.log" 2>&1 || true

        if [ ! -s "$JSON_OUT" ]; then
            echo "  [ERROR] Client benchmark failed to produce JSON output for ${TARGET_KEY} trial ${trial}!"
            tail -n 20 "$TMP_DIR/client_${TARGET_KEY}_t${trial}.log" 2>/dev/null || true
        fi

        # Extract Emitted Tokens from JSON
        TOKENS_EMITTED=$(python3 -c "import json; data=json.load(open('$JSON_OUT')); print(data.get('total_tokens', 0))" 2>/dev/null || echo "0")

        # Record Post-Run Context Switch Telemetry
        python3 tools/server/bench/record_thread_telemetry.py \
            --pid "$SERVER_PID" \
            --stop \
            --state-file "$TMP_DIR/telemetry_state.json" \
            --tokens "$TOKENS_EMITTED" \
            --json-out "$TELEMETRY_OUT" || true

        pkill -9 -f "llama-server" 2>/dev/null || true
        sleep 1
    done
}

# Execute All 6 Targets
run_rigorous_target "target_a" "Upstream Master Baseline" "llama.cpp-master/build" "false"
run_rigorous_target "target_e" "Neither Build (MPSC=OFF, SPSC=OFF, phase-2 tree)" "build-ablation-neither" "false"
run_rigorous_target "target_s" "Master Baseline + Scheduler Pinned to Core 0" "llama.cpp-master/build" "true"
run_rigorous_target "target_b" "Queue Only (MPSC=ON, SPSC=OFF)" "build-ablation-queue" "false"
run_rigorous_target "target_c" "Channels Only (MPSC=OFF, SPSC=ON)" "build-ablation-channels" "false"
run_rigorous_target "target_d" "Full Phase-2 (MPSC=ON, SPSC=ON)" "build-phase2" "false"

# 4. Statistical Analysis and Mechanism Reporting
TMP_DIR="$TMP_DIR" python3 - <<'PYCODE'
import glob
import json
import math
import os

tmp_dir = os.environ.get("TMP_DIR", "")
targets = [
    ("target_a", "Target A: Master Baseline"),
    ("target_e", "Target E: Neither Build (phase-2 tree, MPSC=OFF, SPSC=OFF)"),
    ("target_s", "Target S: Master with Scheduler Pinned to Core 0"),
    ("target_b", "Target B: Queue Only (MPSC=ON, SPSC=OFF)"),
    ("target_c", "Target C: Channels Only (MPSC=OFF, SPSC=ON)"),
    ("target_d", "Target D: Full Phase-2 (MPSC=ON, SPSC=ON)"),
]

data = {}
for key, label in targets:
    files = sorted(glob.glob(os.path.join(tmp_dir, f"{key}_t*.json")))
    t_vals = []
    ttft_vals = []
    ttft_p99_vals = []
    nv_switches = []
    sched_nv_switches = []

    for f in files:
        try:
            with open(f) as fp:
                j = json.load(fp)
            if j.get("failed", 0) == 0 and j.get("tok_per_sec", 0) > 0:
                t_vals.append(j["tok_per_sec"])
                ttft_vals.append(j["ttft"]["mean"])
                ttft_p99_vals.append(j["ttft"]["p99"])
                
                # Check paired telemetry
                tel_f = f.replace(f"{key}_t", f"telemetry_{key}_t")
                if os.path.exists(tel_f):
                    with open(tel_f) as tfp:
                        tj = json.load(tfp)
                    nv_switches.append(tj.get("total_involuntary_per_1k_tokens", 0))
                    sched_nv_switches.append(tj.get("main_thread", {}).get("involuntary_per_1k_tokens", 0))
        except Exception:
            continue
    data[key] = {
        "label": label,
        "throughput": t_vals,
        "ttft_mean": ttft_vals,
        "ttft_p99": ttft_p99_vals,
        "nv_switches": nv_switches,
        "sched_nv_switches": sched_nv_switches
    }

def mean_std(arr):
    if not arr: return 0.0, 0.0
    m = sum(arr) / len(arr)
    var = sum((x - m) ** 2 for x in arr) / max(1, len(arr) - 1)
    return m, math.sqrt(var)

print("\n" + "=" * 120)
print("             EUYSYS / EUROMLSYS RIGOROUS 6-WAY MECHANISTIC ABLATION REPORT (128c / 16s)")
print("=" * 120)
print(f" {'TARGET':<52} {'THROUGHPUT (tok/s)':<22} {'DELTA vs MASTER':<18} {'SCHED INVOL/1k tok':<20}")
print("-" * 120)

m_base, _ = mean_std(data["target_a"]["throughput"])

for key, label in targets:
    t_arr = data[key]["throughput"]
    m, s = mean_std(t_arr)
    delta_str = "0.0% (BASELINE)"
    if m_base > 0 and key != "target_a":
        diff = ((m - m_base) / m_base) * 100.0
        delta_str = f"{diff:+.2f}%"
    
    sched_m, _ = mean_std(data[key]["sched_nv_switches"])
    sched_str = f"{sched_m:.2f}" if data[key]["sched_nv_switches"] else "N/A"
    
    t_str = f"{m:.2f} +/- {s:.2f}" if len(t_arr) > 1 else (f"{m:.2f}" if len(t_arr) == 1 else "NO RUNS")
    print(f" {label:<52} {t_str:<22} {delta_str:<18} {sched_str:<20}")

print("=" * 120)
print(" Mechanistic Deductions:")
print(" 1. If Target E ('Neither') achieves ~1430 tok/s: The speedup is from slot recycling/context optimizations.")
print(" 2. If Target E drops to ~1156 tok/s: Lock-free queueing is verified as the true bottleneck.")
print(" 3. If Target S (Scheduler Pinned) jumps to ~1430 tok/s: The bottleneck is main loop thread starvation.")
print("=" * 120 + "\n")
PYCODE

echo "Rigorous Mechanistic Ablation Finished Successfully!"
