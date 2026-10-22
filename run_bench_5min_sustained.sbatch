#!/bin/bash
#SBATCH --job-name=llama_gold_5min_sustained
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:30:00
#SBATCH --output=gold_5min_%j.log
#SBATCH --error=gold_5min_%j.err

set -e

# ==============================================================================
# EuroMLSys Gold-Standard Benchmark: 5-Minute Continuous Sustained Stress
# ==============================================================================
# Research Objective:
#   Evaluate long-term operational stability, memory leak resistance (RSS growth),
#   and steady-state throughput under prolonged continuous load (300 seconds per run).
#
# Reviewer Scrutiny Addressed:
#   - "Do lock-free freelists and circular ring buffers fragment or leak over time?"
#   - "Does throughput decay or tail latency drift across hundreds of seconds?"
#
# Methodology:
#   - 3 Paired ABBA Trials (300s per trial = 30 mins per case)
#   - Hardware CPU Pinning: Server on Cores 0-15, Client on Cores 16-31
#   - Active Thermal & Power Quiescence Barrier (<= 45C, <= 50W)
#   - Track RSS memory and steady-state token generation
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
echo " [Step 2] Executing 5-Minute Sustained Continuous Benchmarks (16 Slots)"
echo "================================================================================"

# Case 1: 5-Minute Continuous Closed-Loop Load (64 Clients, 16 Slots, 0% Cancel)
echo ">>> [1/3] 300s Continuous Saturated Load: 64 Clients, 16 Slots, 0% Cancel..."
./bench_gold.sh -c 64 -d 300 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

# Case 2: 5-Minute Continuous Closed-Loop Load with 20% Churn (64 Clients, 16 Slots, 20% Cancel)
echo ">>> [2/3] 300s Continuous Churn Load: 64 Clients, 16 Slots, 20% Cancel..."
./bench_gold.sh -c 64 -d 300 -k 0.2 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

# Case 3: 5-Minute Continuous Open-Loop Poisson Stream (16.0 req/s, 16 Slots, 10% Cancel)
echo ">>> [3/3] 300s Continuous Open-Loop Poisson: 16.0 req/s, 16 Slots, 10% Cancel..."
./bench_gold.sh -r 16.0 -d 300 -k 0.1 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo "================================================================================"
echo " 5-Minute Sustained Benchmark Suite Finished Successfully!"
echo "================================================================================"
