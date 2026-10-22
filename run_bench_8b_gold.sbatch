#!/bin/bash
#SBATCH --job-name=llama_gold_8b_scale
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:30:00
#SBATCH --output=gold_8b_%j.log
#SBATCH --error=gold_8b_%j.err

set -e

# ==============================================================================
# EuroMLSys Gold-Standard Benchmark: Model Scale Generalization (8B Model)
# ==============================================================================
# Research Objective:
#   Validate whether the Lock-Free Server Architecture generalizes beyond 3B
#   to standard production models (Llama-3-8B-Instruct, Q4_K_M, 16 Slots).
# Methodology:
#   - 10-Trial Balanced ABBA Paired Testing with Synchronized PRNG Seeds
#   - Hardware CPU Pinning: Server on Cores 0-15, Client on Cores 16-31
#   - Active Thermal & Power Quiescence Barrier (<= 45C, <= 50W)
#   - Concurrency Scale: 32, 64, 128 Clients on 16 Slots (0% and 20% Cancel)
# ==============================================================================

# 1. Environment & Modules
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc-13.1.0 cuda-12.8 2>/dev/null || module load gcc/13.1.0 cuda/12.8 2>/dev/null || true

export CC=gcc
export CXX=g++
export LLAMA_ARG_N_GPU_LAYERS=99
export BENCH_PORT=$((10000 + RANDOM % 40000))

# Target 8B model
MODEL_FILE="my_models/8b.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    MODEL_FILE="my_models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
fi

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: 8B Model file not found in my_models/! Available models:"
    ls -la my_models/ || true
    exit 1
fi
echo "Using 8B Model: $MODEL_FILE"

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
echo " [Step 2] Executing 8B Generalization Benchmark (16 Slots, CPU-Pinned)"
echo "================================================================================"

# Case 1: 32 Clients, 16 Slots, 0% Cancel (10 Paired Trials)
echo ">>> [1/4] 8B Scaling: 32 Clients, 64 Reqs, 16 Slots, 0% Cancel (10 Trials)..."
./bench_gold.sh -c 32 -n 64 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 2: 64 Clients, 16 Slots, 0% Cancel (10 Paired Trials)
echo ">>> [2/4] 8B Scaling: 64 Clients, 64 Reqs, 16 Slots, 0% Cancel (10 Trials)..."
./bench_gold.sh -c 64 -n 64 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 3: 128 Clients, 16 Slots, 0% Cancel (10 Paired Trials) - The Peak Test on 8B
echo ">>> [3/4] 8B Scaling: 128 Clients, 128 Reqs, 16 Slots, 0% Cancel (10 Trials)..."
./bench_gold.sh -c 128 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 4: 128 Clients, 16 Slots, 20% Cancel (10 Paired Trials) - Churn on 8B
echo ">>> [4/4] 8B Scaling: 128 Clients, 128 Reqs, 16 Slots, 20% Cancel (10 Trials)..."
./bench_gold.sh -c 128 -n 128 -k 0.2 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo "================================================================================"
echo " 8B Model Generalization Benchmark Complete!"
echo "================================================================================"
