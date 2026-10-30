#!/bin/bash
#SBATCH --job-name=llama_gold_poisson_knee
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=09:30:00
#SBATCH --output=gold_poisson_%j.log
#SBATCH --error=gold_poisson_%j.err

set -e

# ==============================================================================
# EuroMLSys Gold-Standard Benchmark: Open-Loop Poisson Knee-Point Sweep
# ==============================================================================
# Methodological Standards (Strict Reviewer Compliance):
#   1. Dense Knee Sweep: 12.0 to 26.0 req/s on 16 Slots in 2.0 req/s increments (n=10 paired trials)
#   2. Baseline Control (12.0 req/s) & Saturated Overload (24.0 & 26.0 req/s)
#   3. Empirical Noise Floor: Master-vs-Master (A/A) at the 20.0 req/s knee (n=10 paired trials)
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
echo " [Step 2] Executing Dense Poisson Knee Sweep (12 to 26 req/s on 16 Slots)"
echo "================================================================================"

# Baseline Arrival-Limited Control (12.0 req/s)
echo ">>> [1/8] Baseline Control: 12.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 12.0 -n 96 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Approaching Saturation (16.0 req/s)
echo ">>> [2/8] Approaching Knee: 16.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 16.0 -n 96 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# The Critical Knee Sweep (18.0, 20.0, 22.0 req/s)
echo ">>> [3/8] Knee Point: 18.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 18.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [4/8] Knee Point: 20.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 20.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Empirical A/A Noise Floor right at 20.0 req/s
echo ">>> [5/8] Empirical A/A Noise Floor at 20.0 req/s (Master vs Master, 10 Trials)..."
./bench_gold.sh --aa -r 20.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [6/8] Knee Point: 22.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 22.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Saturated Overload Points (24.0 and 26.0 req/s)
echo ">>> [7/8] Saturated Overload: 24.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 24.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo ">>> [8/8] Severe Overload: 26.0 req/s (10 Paired Trials)..."
./bench_gold.sh -r 26.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

./bench_gold.sh -r 28.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

./bench_gold.sh -r 30.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

./bench_gold.sh -r 32.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

./bench_gold.sh -r 34.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

./bench_gold.sh -r 36.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

./bench_gold.sh -r 38.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

./bench_gold.sh -r 40.0 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build
echo "================================================================================"
echo " Dense Poisson Knee Benchmark Suite Finished Successfully!"
echo "================================================================================"
