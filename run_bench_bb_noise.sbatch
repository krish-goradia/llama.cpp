#!/bin/bash
#SBATCH --job-name=llama_gold_bench_bb_noise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --output=gold_bench_bb_%j.log
#SBATCH --error=gold_bench_bb_%j.err

set -e

# ==============================================================================
# Systems Gold-Standard B/B Baseline: Lock-Free Internal Noise & Variance Floor
# ==============================================================================
# Focus: Run Lock-Free vs Lock-Free (Identical Binary on Same Node)
#   1. Establishes empirical internal variance of lock-free architecture
#   2. Directly addresses reviewer critique: Lock-free variance under churn
#   3. Evaluates 16-slot peak concurrency and churn noise bounds
# Protocol:
#   - 10 Paired Trials (Balanced ABBA crossover, exactly 5 starts per instance)
#   - Common Random Numbers (CRN): Synchronized random seed per trial pair
#   - Active GPU Thermal & Power Quiescence Barrier (<= 45C, <= 50W)
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

echo "================================================================================"
echo " [Step 1] Compiling Modernized Phase-2 Lock-Free (CUDA sm_89 + AVX2, No-AMX)"
echo "================================================================================"
cmake -B build \
  -DGGML_CUDA=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_AVX2=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" \
  -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs" \
  -S .
cmake --build build --config Release -j 32 --target llama-server

MODEL_FILE="my_models/3b.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found! Existing models in my_models/:"
    ls -la my_models/ || true
    exit 1
fi

# Enable Hardware CPU Core Pinning
export PIN_CORES=true
export SERVER_CORES="0-15"
export CLIENT_CORES="16-31"

echo "================================================================================"
echo " [Step 2] Executing B/B Noise Floor Benchmark (Lock-Free vs Lock-Free, 10 Trials)"
echo "================================================================================"

# Case 1: Closed-Loop High Concurrency on 8 Slots (64 clients, 64 reqs, 0% cancel)
echo ">>> [1/5] B/B Test: 64 Clients, 64 Reqs, 8 Slots, 0% Cancel (10 Trials)..."
./bench_gold.sh --bb -c 64 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 2: Closed-Loop High Concurrency with Churn on 8 Slots (64 clients, 64 reqs, 20% cancel)
echo ">>> [2/5] B/B Test: 64 Clients, 64 Reqs, 8 Slots, 20% Cancel (10 Trials)..."
./bench_gold.sh --bb -c 64 -n 64 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 3: Headline Concurrency Peak on 16 Slots (128 clients, 128 reqs, 0% cancel)
echo ">>> [3/5] B/B Test: 128 Clients, 128 Reqs, 16 Slots, 0% Cancel (10 Trials)..."
./bench_gold.sh --bb -c 128 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 4: Headline Churn Peak on 16 Slots (128 clients, 128 reqs, 20% cancel)
echo ">>> [4/5] B/B Test: 128 Clients, 128 Reqs, 16 Slots, 20% Cancel (10 Trials)..."
./bench_gold.sh --bb -c 128 -n 128 -k 0.2 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 5: Open-Loop Poisson Traffic (16 req/s, 96 reqs, 8 slots, 0% cancel)
echo ">>> [5/5] B/B Test: Poisson 16.0 req/s, 96 Reqs, 8 Slots, 0% Cancel (10 Trials)..."
./bench_gold.sh --bb -r 16.0 -n 96 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo "================================================================================"
echo " B/B Noise Floor Benchmark Complete!"
echo "================================================================================"
