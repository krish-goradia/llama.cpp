#!/bin/bash
#SBATCH --job-name=llama_regression_check
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=regression_check_%j.log
#SBATCH --error=regression_check_%j.err

set -e

# ==============================================================================
# Targeted High-Churn & Tail-Latency Regression Verification Suite
# ==============================================================================
# Focus: Validating the FIFO queue fix & 3-Tier priority polling under
# high cancellation rates where the previous LIFO push_front tail spike occurred.
# ==============================================================================

# 1. Environment & Compiler Setup
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc-13.1.0 cuda-12.8 2>/dev/null || module load gcc/13.1.0 cuda/12.8 2>/dev/null || true

export CC=gcc
export CXX=g++

echo "================================================================================"
echo " [Step 0] Node Diagnostics & Hardware Quiescence State"
echo "================================================================================"
hostname
gcc --version | head -n 1
nvcc --version | grep "release" || true
nvidia-smi

export LLAMA_ARG_N_GPU_LAYERS=99
export BENCH_PORT=$((10000 + RANDOM % 40000))

# Ensure benchmark scripts are executable
chmod +x ./bench_gold.sh ./bench_ab.sh ./ab_bench.sh ./run_bench.sh 2>/dev/null || true

# 2. Always Force Rebuild Modernized phase-2 with Latest FIFO Fix
echo "================================================================================"
echo " [Step 1] Compiling Modernized phase-2 with FIFO Fix (CUDA sm_89 + AVX2)"
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

# 3. Synchronize Harness and Compile Master Baseline (if needed)
mkdir -p llama.cpp-master/tools/server/bench/ 2>/dev/null || true
cp -f tools/server/bench/bench_concurrency.py llama.cpp-master/tools/server/bench/ 2>/dev/null || true
cp -f run_bench.sh llama.cpp-master/ 2>/dev/null || true

if [ ! -f "llama.cpp-master/build/bin/llama-server" ]; then
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
echo " [Step 3] Running Lock-Free Queue Unit & Stress Tests"
echo "================================================================================"
./build/bin/test-server-queue

echo "================================================================================"
echo " [Step 4] Running High-Churn Regression Suite (10 Interleaved Trials, Paired 95% CI)"
echo "================================================================================"

MODEL_FILE="my_models/3b.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found! Existing models in my_models/:"
    ls -la my_models/ || true
    exit 1
fi

# ------------------------------------------------------------------------------
# PART 1: Core High-Churn & Tail-Latency Cases (10 Trials Each)
# ------------------------------------------------------------------------------

# CASE 0: Control Anchor (16 Clients, 0% Cancel - verify clean baseline parity)
echo ">>> [1/7] CONTROL ANCHOR: 16 Clients on 8 Slots (0% Cancel, 10 Trials)..."
./bench_gold.sh -c 16 -n 32 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# CASE 1: 16 Clients, 40% Cancel
echo ">>> [2/7] REGRESSION CASE 1: 16 Clients (40% Cancel Chaos, 10 Trials)..."
./bench_gold.sh -c 16 -n 32 -k 0.4 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# CASE 2: 32 Clients, 20% Cancel (Previously showed TTFT p99 jitter in old LIFO code)
echo ">>> [3/7] REGRESSION CASE 2: 32 Clients (20% Cancel - Tail Latency Jitter Focus, 10 Trials)..."
./bench_gold.sh -c 32 -n 64 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# CASE 3: 32 Clients, 40% Cancel
echo ">>> [4/7] REGRESSION CASE 3: 32 Clients (40% Cancel Chaos Storm, 10 Trials)..."
./bench_gold.sh -c 32 -n 64 -k 0.4 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# CASE 4: 64 Clients, 20% Cancel (Concurrency storm with churn)
echo ">>> [5/7] REGRESSION CASE 4: 64 Clients (20% Cancel Storm, 10 Trials)..."
./bench_gold.sh -c 64 -n 64 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# ------------------------------------------------------------------------------
# PART 2: Sustained Load (60s+) & Open-Loop Poisson Arrival Tests
# ------------------------------------------------------------------------------

# CASE 5: Sustained 60-Second Closed-Loop Load with 20% Cancellations
echo ">>> [6/7] SUSTAINED LOAD: 16 Clients, 60s Duration (20% Cancel)..."
./bench_gold.sh -c 16 -d 60 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 5 --no-build

# CASE 6: Open-Loop Poisson Arrival Test (15 req/s fixed rate for 60 seconds)
# Exposes genuine queue backlog delay and tail latency independent of client-wait throttling
echo ">>> [7/7] OPEN-LOOP POISSON: 15 req/s Poisson Arrival, 60s Duration (20% Cancel)..."
./bench_gold.sh -r 15.0 -d 60 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 5 --no-build

echo "================================================================================"
echo " Targeted High-Churn Regression Suite Finished Successfully!"
echo "================================================================================"
