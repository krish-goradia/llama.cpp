#!/bin/bash
#SBATCH --job-name=llama_gold_poisson_knee
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:30:00
#SBATCH --output=gold_poisson_%j.log
#SBATCH --error=gold_poisson_%j.err

set -e

# ==============================================================================
# EuroMLSys Gold-Standard Benchmark: Open-Loop Poisson Knee-Point Sweep
# ==============================================================================
# Methodological Standards (Strict Reviewer Compliance):
#   1. Dense Knee Sweep: 10.0 to 16.0 req/s in 1.0 req/s increments (n=10 paired trials)
#   2. Baseline Control (4.0 req/s) & Saturated Overload (24.0 & 32.0 req/s)
#   3. Empirical Noise Floor: Master-vs-Master (A/A) at the 12.0 req/s knee (n=10 paired trials)
#   4. Hardware CPU Pinning: Server on Cores 0-15, Client on Cores 16-31
#   5. Synchronized PRNG Seeds (--seed $trial) for Common Random Numbers (CRN)
#   6. Complete Metric Uniformity: Throughput, TTFT p50, TTFT p99, and Goodput reported at every rate
#   7. 4-Tier TOST Equivalence Labels: [WIN], [REGRESSION], [EQUIVALENT], [INCONCLUSIVE]
#   8. Multiplicity Control: Holm-Bonferroni FWER step-down adjustment across all rates
#   9. Max Sustainable Rate under TTFT SLO Evaluation (SLO <= 100ms, <= 200ms, <= 500ms)
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
echo " [Step 2] Executing Dense Poisson Knee Sweep (10-16 req/s in 1 req/s steps)"
echo "================================================================================"

# Baseline Arrival-Limited Control
echo ">>> [1/12] Baseline Control: 4.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 4.0 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# The Knee Sweep (10.0 to 16.0 req/s in 1.0 req/s steps)
echo ">>> [2/12] Knee Sweep: 10.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 10.0 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [3/12] Knee Sweep: 11.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 11.0 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [4/12] Knee Sweep: 12.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 12.0 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Empirical A/A Noise Floor right at the 12.0 req/s knee
echo ">>> [5/12] Empirical A/A Noise Floor: 12.0 req/s Master vs Master (10 Trials)..."
./bench_gold.sh --aa -r 12.0 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [6/12] Knee Sweep: 13.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 13.0 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [7/12] Knee Sweep: 14.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 14.0 -n 96 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [8/12] Knee Sweep: 15.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 15.0 -n 96 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [9/12] Knee Sweep: 16.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 16.0 -n 96 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Saturated Overload Points
echo ">>> [10/12] Saturated Overload: 24.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 24.0 -n 128 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [11/12] Saturated Overload: 32.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 32.0 -n 128 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Sustained Deep Stress Test
echo ">>> [12/12] Sustained 120s Continuous Load: 32 Clients, 20% Churn (5 Paired Trials)..."
./bench_gold.sh -c 32 -d 120 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 5 --no-build

echo "================================================================================"
echo " Dense Poisson Knee Benchmark Suite Finished Successfully!"
echo "================================================================================"
