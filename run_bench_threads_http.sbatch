#!/bin/bash
#SBATCH --job-name=llama_gold_threads_http
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=gold_threads_http_%j.log
#SBATCH --error=gold_threads_http_%j.err

set -e

# ==============================================================================
# EuroMLSys Gold-Standard Benchmark: HTTP Thread Pool Sizing Sweep
# ==============================================================================
# Research Objective:
#   Dissect whether the +18% to +23% speedup under 16-core CPU pinning is caused
#   by Master running 31 threads oversubscribed, or by true lock-free synchronization.
#
# Reviewer Scrutiny Addressed:
#   "If you set --threads-http to something sane for master (e.g. 16 or slots+2),
#    how much of the gap remains?"
#
# Methodology:
#   - 10-Trial Balanced ABBA Paired Testing per Case with Common Random Numbers
#   - Hardware CPU Pinning: Server on Cores 0-15, Client on Cores 16-31
#   - Active Thermal & Power Quiescence Barrier (<= 45C, <= 50W)
#   - Identical --threads-http settings applied symmetrically to both Master & Phase-2
#   - 4-Tier Equivalence Labels: [WIN], [REGRESSION], [EQUIVALENT], [INCONCLUSIVE]
# ==============================================================================

# 1. Environment & Modules
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc-13.1.0 cuda-12.8 2>/dev/null || module load gcc/13.1.0 cuda/12.8 2>/dev/null || true

export CC=gcc
export CXX=g++
export LLAMA_ARG_N_GPU_LAYERS=99
export BENCH_PORT=$((10000 + RANDOM % 40000))

echo "================================================================================"
echo " [Step 0] Node Diagnostics: NUMA & Hyperthread Topology"
echo "================================================================================"
hostname
lscpu | grep -E "Model name|Socket|Thread|Core\(s\) per socket|NUMA node" || true
nvidia-smi topo -m 2>/dev/null || true

MODEL_FILE="my_models/3b.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found in my_models/!"
    ls -la my_models/ || true
    exit 1
fi
echo "Using Model: $MODEL_FILE"

chmod +x ./bench_gold.sh ./bench_ab.sh ./ab_bench.sh ./run_bench.sh 2>/dev/null || true

# 2. Build Both Architectures
echo "================================================================================"
echo " [Step 1] Compiling Modernized phase-2 (build)"
echo "================================================================================"
cmake -B build \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs"
cmake --build build --config Release -j 32 --target llama-server

echo "================================================================================"
echo " [Step 1] Compiling Upstream Master Baseline (llama.cpp-master/build)"
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
else
    echo "ERROR: llama.cpp-master directory not found!"
    exit 1
fi

# Enable Hardware CPU Core Pinning
export PIN_CORES=true
export SERVER_CORES="0-15"
export CLIENT_CORES="16-31"

echo "================================================================================"
echo " [Step 2] Executing Controlled --threads-http Benchmark (128 Clients, 16 Slots)"
echo "================================================================================"

# Case 1: Exactly 16 HTTP Threads (1 Thread per Core, Zero Oversubscription)
echo ">>> [1/4] Controlled Threads: --threads-http 16 (1 Thread per Core, 0% Cancel)..."
./bench_gold.sh -c 128 -n 128 -k 0.0 -s 16 --threads-http 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 2: 18 HTTP Threads (slots + 2 formula, standard sizing)
echo ">>> [2/4] Controlled Threads: --threads-http 18 (slots+2 formula, 0% Cancel)..."
./bench_gold.sh -c 128 -n 128 -k 0.0 -s 16 --threads-http 18 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 3: 8 HTTP Threads (Undersubscribed, 8 threads on 16 cores)
echo ">>> [3/4] Controlled Threads: --threads-http 8 (Undersubscribed, 0% Cancel)..."
./bench_gold.sh -c 128 -n 128 -k 0.0 -s 16 --threads-http 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 4: Exactly 16 HTTP Threads with 20% Cancellation Churn
echo ">>> [4/4] Controlled Threads: --threads-http 16 with 20% Churn..."
./bench_gold.sh -c 128 -n 128 -k 0.2 -s 16 --threads-http 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo "================================================================================"
echo " HTTP Thread Pool Sizing Sweep Finished Successfully!"
echo "================================================================================"
