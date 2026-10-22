#!/bin/bash
#SBATCH --job-name=llama_ab_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:00:00
#SBATCH --output=bench_%j.log
#SBATCH --error=bench_%j.err

set -e

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
chmod +x ./bench_ab.sh ./ab_bench.sh ./run_bench.sh 2>/dev/null || true

# Preserve cached build artifacts so we do not recompile CUDA from scratch
# rm -rf build llama.cpp-master/build

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

echo "================================================================================"
echo " [Step 4] Running 15-Case Concurrency Benchmark Suite"
echo "================================================================================"

# Verify model existence
MODEL_FILE="my_models/3b.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found! Existing models in my_models/:"
    ls -la my_models/ || true
    exit 1
fi

# GROUP A: Baseline & Saturation Sweep (0% Cancel)
echo ">>> [1/15] Case 1: 4 Clients (0.5x Slot Load)..."
./bench_ab.sh -c 4 -n 16 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo ">>> [2/15] Case 2: 8 Clients (1.0x Exact Saturation)..."
./bench_ab.sh -c 8 -n 32 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo ">>> [3/15] Case 3: 16 Clients (2.0x Moderate Backlog)..."
./bench_ab.sh -c 16 -n 32 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo ">>> [4/15] Case 4: 32 Clients (4.0x Heavy Contention)..."
./bench_ab.sh -c 32 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo ">>> [5/15] Case 5: 64 Clients (8.0x Burst Contention Storm)..."
./bench_ab.sh -c 64 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

# GROUP B: Real-World Churn Sweep (10% - 20% Cancel)
echo ">>> [6/15] Case 6: 8 Clients (10% Cancel)..."
./bench_ab.sh -c 8 -n 32 -k 0.1 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo ">>> [7/15] Case 7: 16 Clients (20% Cancel)..."
./bench_ab.sh -c 16 -n 32 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo ">>> [8/15] Case 8: 32 Clients (20% Cancel)..."
./bench_ab.sh -c 32 -n 64 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo ">>> [9/15] Case 9: 64 Clients (20% Cancel)..."
./bench_ab.sh -c 64 -n 64 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

# GROUP C: Chaos & Rapid Abort Storm (40% - 50% Cancel)
echo ">>> [10/15] Case 10: 16 Clients (40% Cancel Chaos)..."
./bench_ab.sh -c 16 -n 32 -k 0.4 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo ">>> [11/15] Case 11: 32 Clients (40% Cancel Chaos)..."
./bench_ab.sh -c 32 -n 64 -k 0.4 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo ">>> [12/15] Case 12: 64 Clients (50% Abort Storm)..."
./bench_ab.sh -c 64 -n 64 -k 0.5 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

# GROUP D: Token Length & Deep Streaming Variations
echo ">>> [13/15] Case 13: 16 Clients (Short Burst, Max 32 Tokens)..."
./bench_ab.sh -c 16 -n 48 -k 0.0 --max-tokens 32 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo ">>> [14/15] Case 14: 16 Clients (Deep Stream, Max 128 Tokens)..."
./bench_ab.sh -c 16 -n 32 -k 0.0 --max-tokens 128 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo ">>> [15/15] Case 15: 32 Clients (Full Stress: Deep Stream 128 Tok + 20% Cancel)..."
./bench_ab.sh -c 32 -n 64 -k 0.2 --max-tokens 128 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --cooldown 10 --no-build

echo "================================================================================"
echo " All 15 Benchmark Cases Finished Successfully!"
echo "================================================================================"
