#!/bin/bash
#SBATCH --job-name=llama_gold_headline_10tr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=gold_headline_10tr_%j.log
#SBATCH --error=gold_headline_10tr_%j.err

set -e

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc-13.1.0 cuda-12.8 2>/dev/null || module load gcc/13.1.0 cuda/12.8 2>/dev/null || true

export CC=gcc
export CXX=g++
export LLAMA_ARG_N_GPU_LAYERS=99
export BENCH_PORT=$((10000 + RANDOM % 40000))
MODEL_FILE="my_models/3b.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found!"
    exit 1
fi

chmod +x ./bench_gold.sh ./bench_ab.sh ./ab_bench.sh ./run_bench.sh 2>/dev/null || true

# 1. Compile Phase-2
cmake -B build \
  -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DGGML_AVX2=ON \
  -DCMAKE_CUDA_ARCHITECTURES="89" -DCMAKE_BUILD_TYPE=Release \
  -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_UI=OFF \
  -DCMAKE_EXE_LINKER_FLAGS="-lstdc++fs" -DCMAKE_SHARED_LINKER_FLAGS="-lstdc++fs"
cmake --build build --config Release -j 32 --target llama-server

# 2. Compile Upstream Master Baseline
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

# Enable CPU Core Pinning: Server on Cores 0-15, Client on Cores 16-31
export PIN_CORES=true
export SERVER_CORES="0-15"
export CLIENT_CORES="16-31"

echo "================================================================================"
echo " EXECUTING HEADLINE 10-TRIAL REPLICATIONS (CPU PINNED: SRV 0-15, CLI 16-31)"
echo "================================================================================"

# Case 1: 64 Clients, 16 Slots, 0% Cancel (10 Paired ABBA Trials)
echo ">>> [1/5] Headline Test: 64 Clients, 64 Reqs, 16 Slots, 0% Cancel (10 Trials)..."
./bench_gold.sh -c 64 -n 64 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 2: 96 Clients, 16 Slots, 0% Cancel (10 Paired ABBA Trials)
echo ">>> [2/5] Headline Test: 96 Clients, 96 Reqs, 16 Slots, 0% Cancel (10 Trials)..."
./bench_gold.sh -c 96 -n 96 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 3: 128 Clients, 16 Slots, 0% Cancel (10 Paired ABBA Trials)
echo ">>> [3/5] Headline Test: 128 Clients, 128 Reqs, 16 Slots, 0% Cancel (10 Trials)..."
./bench_gold.sh -c 128 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 4: 128 Clients, 16 Slots, 20% Cancel (10 Paired ABBA Trials)
echo ">>> [4/5] Headline Test: 128 Clients, 128 Reqs, 16 Slots, 20% Cancel (10 Trials)..."
./bench_gold.sh -c 128 -n 128 -k 0.2 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 5: Master-vs-Master (A/A) Control at 128 Clients, 16 Slots (10 Paired Trials)
echo ">>> [5/5] A/A Noise Floor Control: Master vs Master @ 128 Clients, 16 Slots (10 Trials)..."
./bench_gold.sh --aa -c 128 -n 128 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo "================================================================================"
echo " Headline 10-Trial Benchmark Suite Complete!"
echo "================================================================================"
