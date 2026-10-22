#!/bin/bash
#SBATCH --job-name=llama_gold_bench3_poisson
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:30:00
#SBATCH --output=gold_bench3_%j.log
#SBATCH --error=gold_bench3_%j.err

set -e

# ==============================================================================
# Systems Gold-Standard Benchmark Suite 3: Open-Loop Poisson & Sustained Load
# ==============================================================================
# Focus:
#   1. Open-Loop Poisson Arrival Rate Sweep (2.0 to 32.0 req/s)
#   2. Open-Loop Churn Under Burst Arrival (Poisson + 10%/20% Cancel)
#   3. Sustained Duration Steady-State Stability (60s and 120s Continuous Traffic)
# Methodology:
#   - Active GPU Thermal & Power Quiescence Barrier (nvidia-smi <= 45C, <= 50W)
#   - Common Random Numbers (CRN): Synchronized random seed per trial pair
#   - Interleaved ABBA Multi-Trial Protocol (10 trials per point, exact 5:5 balance)
#   - Discarded Warmup Requests (prime CUDA context and KV scratchpads)
#   - Paired Statistical Hypothesis Testing (Mean +- 95% CI, p-value, delta)
# ==============================================================================

# 1. Environment & Modules
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

# 2. Build Modernized Phase-2 and Upstream Master
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

echo "================================================================================"
echo " [Step 3] Running Lock-Free Unit Tests"
echo "================================================================================"
./build/bin/test-server-queue

MODEL_FILE="my_models/3b.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found! Existing models in my_models/:"
    ls -la my_models/ || true
    exit 1
fi

echo "================================================================================"
echo " [Step 4] Starting Open-Loop Poisson & Sustained Load Suite"
echo "================================================================================"

# PART 1: Open-Loop Poisson Arrival Rate Sweep (8 Slots, 0% Cancel)
echo ">>> [1/14] Poisson Rate: 2.0 req/s (Light Load, 64 reqs)..."
./bench_gold.sh -r 2.0 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [2/14] Poisson Rate: 4.0 req/s (0.5x Saturation, 64 reqs)..."
./bench_gold.sh -r 4.0 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [3/14] Poisson Rate: 8.0 req/s (1.0x Saturation Boundary, 64 reqs)..."
./bench_gold.sh -r 8.0 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [4/14] Poisson Rate: 12.0 req/s (1.5x Over-Capacity, 96 reqs)..."
./bench_gold.sh -r 12.0 -n 96 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [5/14] Poisson Rate: 16.0 req/s (2.0x Heavy Backlog, 96 reqs)..."
./bench_gold.sh -r 16.0 -n 96 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [6/14] Poisson Rate: 20.0 req/s (2.5x Queue Saturation, 128 reqs)..."
./bench_gold.sh -r 20.0 -n 128 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [7/14] Poisson Rate: 24.0 req/s (3.0x Burst Contention Storm, 128 reqs)..."
./bench_gold.sh -r 24.0 -n 128 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [8/14] Poisson Rate: 32.0 req/s (4.0x Extreme Spike Storm, 128 reqs)..."
./bench_gold.sh -r 32.0 -n 128 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# PART 2: Open-Loop Poisson with Cancellation Churn
echo ">>> [9/14] Poisson 8.0 req/s with 10% Cancellation..."
./bench_gold.sh -r 8.0 -n 64 -k 0.1 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [10/14] Poisson 8.0 req/s with 20% Cancellation..."
./bench_gold.sh -r 8.0 -n 64 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [11/14] Poisson 16.0 req/s with 20% Cancellation..."
./bench_gold.sh -r 16.0 -n 96 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# PART 3: Sustained Steady-State Duration Testing (60s and 120s, 10 Trials)
echo ">>> [12/14] Sustained 60s Steady-State Load (8 Clients, 0% Cancel)..."
./bench_gold.sh -c 8 -d 60 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [13/14] Sustained 60s Steady-State Load (16 Clients, 20% Cancel)..."
./bench_gold.sh -c 16 -d 60 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [14/14] Sustained 120s Deep Steady-State Stress (32 Clients, 20% Cancel)..."
./bench_gold.sh -c 32 -d 120 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo "================================================================================"
echo " Suite 3 (Open-Loop Poisson & Sustained Load) Finished Successfully!"
echo "================================================================================"
