#!/bin/bash
#SBATCH --job-name=llama_gold_bench2
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=gold_bench2_%j.log
#SBATCH --error=gold_bench2_%j.err

set -e

# ==============================================================================
# Systems Gold-Standard Benchmark Suite 2: Extreme Scaling & Slot Saturation
# ==============================================================================
# 1. Active GPU Thermal & Power Quiescence Barrier (polls nvidia-smi <=45C, <=50W)
# 2. Interleaved ABBA Multi-Trial Protocol (3 trials per case, alternating start)
# 3. Discarded Warmup Requests (primes CUDA context and KV-cache scratchpads)
# 4. Synchronized Arrival Barrier (concurrent clients launch at identical microsecond)
# 5. Full Statistical Distribution Reporting (Median, Mean ± StdDev, CV% stability)
# ==============================================================================

# 1. Source module environment and load compilers
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc-13.1.0 cuda-12.8 2>/dev/null || module load gcc/13.1.0 cuda/12.8 2>/dev/null || true

export CC=gcc
export CXX=g++

echo "================================================================================"
echo " [Step 0] Node Diagnostics"
echo "================================================================================"
hostname
gcc --version | head -n 1
nvcc --version | grep "release" || true
nvidia-smi

export LLAMA_ARG_N_GPU_LAYERS=99
export BENCH_PORT=$((10000 + RANDOM % 40000))

# Ensure benchmark scripts are executable
chmod +x ./bench_gold.sh ./bench_ab.sh ./ab_bench.sh ./run_bench.sh 2>/dev/null || true

# 2. Build Check
if [ ! -f "build/bin/llama-server" ] || [ ! -f "llama.cpp-master/build/bin/llama-server" ]; then
  echo "================================================================================"
  echo " [Step 1] Compiling Modernized phase-2 (CUDA sm_89 + AVX2, No-AMX)"
  echo "================================================================================"
  cmake -B build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX2=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_UI=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" \
    -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs"
  cmake --build build --config Release -j 32 --target llama-server test-server-queue

  # Synchronize benchmark harness to master
  mkdir -p llama.cpp-master/tools/server/bench/ 2>/dev/null || true
  cp -f tools/server/bench/bench_concurrency.py llama.cpp-master/tools/server/bench/ 2>/dev/null || true
  cp -f run_bench.sh llama.cpp-master/ 2>/dev/null || true

  echo "================================================================================"
  echo " [Step 2] Compiling Upstream Master Baseline (CUDA sm_89 + AVX2, No-AMX)"
  echo "================================================================================"
  cmake -B llama.cpp-master/build \
    -DGGML_CUDA=ON \
    -DGGML_NATIVE=OFF \
    -DGGML_AVX2=ON \
    -DCMAKE_CUDA_ARCHITECTURES="89" \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_UI=OFF \
    -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" \
    -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs" \
    -S llama.cpp-master
  cmake --build llama.cpp-master/build --config Release -j 32 --target llama-server
fi

echo "================================================================================"
echo " [Step 3] Running Lock-Free Unit Tests"
echo "================================================================================"
./build/bin/test-server-queue

echo "================================================================================"
echo " [Step 4] Running Extreme Scaling & Slot Saturation Suite (Gold-Standard)"
echo "================================================================================"

MODEL_FILE="my_models/3b.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found! Existing models in my_models/:"
    ls -la my_models/ || true
    exit 1
fi

# ------------------------------------------------------------------------------
# PART 1: Slot Scaling (16 and 24 Slots on 48GB L40S)
# ------------------------------------------------------------------------------
echo ">>> [1/10] 32 Clients on 16 Slots (2.0x Load, 0% Cancel)..."
./bench_gold.sh -c 32 -n 64 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [2/10] 32 Clients on 16 Slots (2.0x Load, 20% Cancel)..."
./bench_gold.sh -c 32 -n 64 -k 0.2 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [3/10] 64 Clients on 16 Slots (4.0x Load, 0% Cancel)..."
./bench_gold.sh -c 64 -n 64 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [4/10] 64 Clients on 16 Slots (4.0x Load, 25% Cancel)..."
./bench_gold.sh -c 64 -n 64 -k 0.25 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [5/10] 64 Clients on 24 Slots (High VRAM Slot Saturation, 0% Cancel)..."
./bench_gold.sh -c 64 -n 64 -k 0.0 -s 24 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

# ------------------------------------------------------------------------------
# PART 2: Ultra-High Concurrency Storm (96 & 128 Clients)
# ------------------------------------------------------------------------------
echo ">>> [6/10] 96 Clients on 16 Slots (0% Cancel)..."
./bench_gold.sh -c 96 -n 96 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [7/10] 96 Clients on 16 Slots (20% Cancel)..."
./bench_gold.sh -c 96 -n 96 -k 0.2 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [8/10] 128 Clients Massive Concurrency Storm (0% Cancel)..."
./bench_gold.sh -c 128 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [9/10] 128 Clients Massive Concurrency Storm (25% Cancel)..."
./bench_gold.sh -c 128 -n 128 -k 0.25 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

# ------------------------------------------------------------------------------
# PART 3: Reverse Execution Order (Modernized FIRST, Master SECOND)
# Proves results are not biased by GPU thermal warmup or kernel caching
# ------------------------------------------------------------------------------
echo ">>> [10/10] Bias-Check: 64 Clients REAL REVERSE Order (Phase-2 First on Odd, Master First on Even, 20% Cancel)..."
./bench_gold.sh -c 64 -n 64 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --reverse --no-build

echo "================================================================================"
echo " Extreme Scaling & Bias-Check Suite 2 Finished Successfully!"
echo "================================================================================"
