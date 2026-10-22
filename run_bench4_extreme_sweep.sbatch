#!/bin/bash
#SBATCH --job-name=llama_gold_bench4_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=gold_bench4_%j.log
#SBATCH --error=gold_bench4_%j.err

set -e

# ==============================================================================
# Systems Gold-Standard Benchmark Suite 4: High-Resolution Concurrency & Churn Sweep
# ==============================================================================
# Focus: Dense Data-Grid for Publication-Quality Graphs
#   1. Fine-Grained Concurrency Saturation Sweep (4 to 128 Clients)
#   2. Fine-Grained Cancellation Churn Sweep (0% to 40% Cancel Rate)
#   3. Multi-Slot Scaling Efficiency (4, 8, 16, 24 Slots @ 4x Load)
# Methodology:
#   - Active GPU Thermal & Power Quiescence Barrier (nvidia-smi <= 45C, <= 50W)
#   - Common Random Numbers (CRN): Synchronized random seed per trial pair
#   - Interleaved ABBA Multi-Trial Protocol (10 trials per point, exact 5:5 balance)
#   - Synchronized Client Arrival Barrier (threading.Barrier)
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
echo " [Step 4] Starting High-Resolution Concurrency & Churn Sweep"
echo "================================================================================"

# PART 1: Fine-Grained Concurrency Saturation Sweep (8 Slots, 0% Cancel)
echo ">>> [1/21] Concurrency: 4 Clients (0.5x Slot Load)..."
./bench_gold.sh -c 4 -n 16 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [2/21] Concurrency: 8 Clients (1.0x Exact Saturation)..."
./bench_gold.sh -c 8 -n 32 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [3/21] Concurrency: 12 Clients (1.5x Mild Backlog)..."
./bench_gold.sh -c 12 -n 36 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [4/21] Concurrency: 16 Clients (2.0x Moderate Queue)..."
./bench_gold.sh -c 16 -n 48 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [5/21] Concurrency: 24 Clients (3.0x Queue Buildup)..."
./bench_gold.sh -c 24 -n 48 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [6/21] Concurrency: 32 Clients (4.0x Heavy Backlog)..."
./bench_gold.sh -c 32 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [7/21] Concurrency: 48 Clients (6.0x Severe Contention)..."
./bench_gold.sh -c 48 -n 96 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [8/21] Concurrency: 64 Clients (8.0x Contention Storm)..."
./bench_gold.sh -c 64 -n 128 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [9/21] Concurrency: 96 Clients (12.0x Massive Burst)..."
./bench_gold.sh -c 96 -n 128 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [10/21] Concurrency: 128 Clients (16.0x Extreme Saturation Storm)..."
./bench_gold.sh -c 128 -n 128 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# PART 2: Fine-Grained Cancellation Churn Sweep (32 Clients, 64 Reqs, 8 Slots)
echo ">>> [11/21] Cancel Sweep: 0% Cancellation..."
./bench_gold.sh -c 32 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [12/21] Cancel Sweep: 5% Cancellation..."
./bench_gold.sh -c 32 -n 64 -k 0.05 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [13/21] Cancel Sweep: 10% Cancellation..."
./bench_gold.sh -c 32 -n 64 -k 0.10 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [14/21] Cancel Sweep: 15% Cancellation..."
./bench_gold.sh -c 32 -n 64 -k 0.15 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [15/21] Cancel Sweep: 20% Cancellation..."
./bench_gold.sh -c 32 -n 64 -k 0.20 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [16/21] Cancel Sweep: 30% Cancellation..."
./bench_gold.sh -c 32 -n 64 -k 0.30 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [17/21] Cancel Sweep: 40% Heavy Churn..."
./bench_gold.sh -c 32 -n 64 -k 0.40 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# PART 3: Multi-Slot Scaling Efficiency Sweep (Constant 4x Load Factor)
echo ">>> [18/21] Slot Scaling: 4 Slots (16 Clients, 32 Reqs)..."
./bench_gold.sh -c 16 -n 32 -k 0.0 -s 4 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [19/21] Slot Scaling: 8 Slots (32 Clients, 64 Reqs)..."
./bench_gold.sh -c 32 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [20/21] Slot Scaling: 16 Slots (64 Clients, 128 Reqs)..."
./bench_gold.sh -c 64 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [21/21] Slot Scaling: 24 Slots (96 Clients, 128 Reqs)..."
./bench_gold.sh -c 96 -n 128 -k 0.0 -s 24 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo "================================================================================"
echo " Suite 4 (High-Resolution Concurrency & Churn Sweep) Finished Successfully!"
echo "================================================================================"
