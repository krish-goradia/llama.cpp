#!/bin/bash
#SBATCH --job-name=llama_gold_ablation_4way
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:30:00
#SBATCH --output=gold_ablation_%j.log
#SBATCH --error=gold_ablation_%j.err

set -e

# ==============================================================================
# EuroMLSys Gold-Standard Benchmark: 4-Way Mechanistic Ablation Suite
# ==============================================================================
# Research Objective:
#   Dissect the +9.6% speedup under high concurrency (128 clients, 16 slots)
#   by isolating the independent contribution of each architectural component:
#     1. Master Baseline (pure mutex queue & response)
#     2. Target B: +Queue Only (Lock-free MPSC Task Queue, Mutex Response Egress)
#     3. Target C: +Channels Only (Mutex Task Queue, Lock-free SPSC Response Egress)
#     4. Target D: Full Phase-2 (Lock-free MPSC Queue + Lock-free SPSC Channels)
# Methodology:
#   - 10-Trial Balanced Testing per Target with Synchronized PRNG Seeds
#   - Hardware CPU Pinning: Server on Cores 0-15, Client on Cores 16-31
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

chmod +x ./bench_gold.sh ./bench_ab.sh ./ab_bench.sh ./run_bench.sh 2>/dev/null || true

# 2. Compile All 4 Targets
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
echo " [Step 1] Compiling Target B: +Queue Only (build-ablation-queue)"
echo "================================================================================"
cmake -B build-ablation-queue \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SERVER_DISABLE_SPSC=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs"
cmake --build build-ablation-queue --config Release -j 32 --target llama-server

echo "================================================================================"
echo " [Step 1] Compiling Target C: +Channels Only (build-ablation-channels)"
echo "================================================================================"
cmake -B build-ablation-channels \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_SERVER_DISABLE_MPSC=ON \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs"
cmake --build build-ablation-channels --config Release -j 32 --target llama-server

echo "================================================================================"
echo " [Step 1] Compiling Target A: Upstream Master (llama.cpp-master/build)"
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

# Enable Hardware CPU Core Pinning
export PIN_CORES=true
export SERVER_CORES="0-15"
export CLIENT_CORES="16-31"

mkdir -p ./ablation_results
echo "================================================================================"
echo " [Step 2] Executing 4-Way Ablation Benchmark (128 Clients, 16 Slots, 10 Trials)"
echo "================================================================================"

run_ablation_point() {
    local TARGET_NAME="$1"
    local BIN_PATH="$2"
    local CLIENTS="$3"
    local REQS="$4"
    local SLOTS="$5"
    local CANCEL="$6"
    local TRIALS=10

    echo ">>> Running Target: ${TARGET_NAME} (${CLIENTS} Clients, ${SLOTS} Slots, ${TRIALS} Trials)..."
    for trial in $(seq 1 $TRIALS); do
        pkill -9 -f "llama-server" 2>/dev/null || true
        for w in {1..30}; do
            TEMP=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 || echo "0")
            POWER=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1 | cut -d'.' -f1 || echo "0")
            if [ "$TEMP" -le 45 ] && [ "$POWER" -le 50 ]; then
                break
            fi
            sleep 1
        done

        taskset -c 0-15 "$BIN_PATH" -m "$MODEL_FILE" -c 16384 -np "$SLOTS" --port "$BENCH_PORT" > "/tmp/llama_srv_$$.log" 2>&1 &
        SRV_PID=$!
        for w in {1..30}; do
            if curl -s "http://localhost:${BENCH_PORT}/health" | grep -q "ok"; then
                break
            fi
            sleep 1
        done

        OUT_JSON="./ablation_results/${TARGET_NAME}_c${CLIENTS}_t${trial}.json"
        taskset -c 16-31 python3 tools/server/bench/bench_concurrency.py \
          --url "http://localhost:${BENCH_PORT}" \
          -c "$CLIENTS" -n "$REQS" --cancel-rate "$CANCEL" --max-tokens 64 \
          --seed "$trial" --json-out "$OUT_JSON" > /dev/null 2>&1 || true

        pkill -9 -f "llama-server" 2>/dev/null || true
        sleep 1
    done
}

# Run 4 targets at 128 clients on 16 slots
run_ablation_point "master" "./llama.cpp-master/build/bin/llama-server" 128 128 16 0.0
run_ablation_point "queue_only" "./build-ablation-queue/bin/llama-server" 128 128 16 0.0
run_ablation_point "channels_only" "./build-ablation-channels/bin/llama-server" 128 128 16 0.0
run_ablation_point "phase2_both" "./build-phase2/bin/llama-server" 128 128 16 0.0

echo "================================================================================"
echo " [Step 3] Statistical Analysis of 4-Way Ablation"
echo "================================================================================"

python3 - << 'EOF'
import json, glob, math, os

targets = ["master", "queue_only", "channels_only", "phase2_both"]
results = {}

for t in targets:
    files = glob.glob(f"./ablation_results/{t}_c128_t*.json")
    tok_s = []
    ttft = []
    for f in sorted(files):
        try:
            d = json.load(open(f))
            if d.get("valid", True) and d.get("tok_per_sec", 0) > 0:
                tok_s.append(d["tok_per_sec"])
                ttft.append(d["ttft_mean_ms"])
        except: pass
    
    n = len(tok_s)
    if n > 0:
        mean_tp = sum(tok_s) / n
        std_tp = math.sqrt(sum((x - mean_tp)**2 for x in tok_s) / (n - 1)) if n > 1 else 0
        ci_tp = 2.262 * (std_tp / math.sqrt(n)) if n == 10 else 1.96 * (std_tp / math.sqrt(n))
        mean_ttft = sum(ttft) / n
        results[t] = {"n": n, "tp": mean_tp, "ci": ci_tp, "ttft": mean_ttft}

print(f"{'Target Architecture':<22} | {'Trials':<6} | {'Throughput (tok/s)':<22} | {'Delta vs Master':<16} | {'TTFT Mean (ms)':<16}")
print("-" * 90)

master_tp = results.get("master", {}).get("tp", 1.0)
for t in targets:
    if t in results:
        r = results[t]
        delta = ((r["tp"] - master_tp) / master_tp) * 100.0
        delta_str = f"{delta:+.2f}%" if t != "master" else "BASELINE"
        print(f"{t:<22} | {r['n']:<6} | {r['tp']:>8.2f} +- {r['ci']:<10.2f} | {delta_str:<16} | {r['ttft']:>8.2f} ms")

EOF

echo "================================================================================"
echo " 4-Way Ablation Benchmark Complete!"
echo "================================================================================"
