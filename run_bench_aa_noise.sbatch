#!/bin/bash
#SBATCH --job-name=llama_gold_bench_aa_noise
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=gold_bench_aa_%j.log
#SBATCH --error=gold_bench_aa_%j.err

set -e

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc-13.1.0 cuda-12.8 2>/dev/null || module load gcc/13.1.0 cuda/12.8 2>/dev/null || true

export CC=gcc
export CXX=g++
export LLAMA_ARG_N_GPU_LAYERS=99
export BENCH_PORT=$((10000 + RANDOM % 40000))

chmod +x ./bench_gold.sh ./run_bench.sh 2>/dev/null || true

mkdir -p llama.cpp-master/tools/server/bench/ 2>/dev/null || true
cp -f tools/server/bench/bench_concurrency.py llama.cpp-master/tools/server/bench/ 2>/dev/null || true
cp -f run_bench.sh llama.cpp-master/ 2>/dev/null || true

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

MODEL_FILE="my_models/3b.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found!"
    exit 1
fi

echo "================================================================================"
echo " Executing A/A Noise Floor Benchmark (Master vs Master, 10 Trials per case)"
echo "================================================================================"

# Case 1: Closed-Loop High Concurrency (64 clients, 64 reqs, 0% cancel)
echo ">>> [1/3] A/A Test: 64 Clients, 64 Reqs, 8 Slots, 0% Cancel (10 Trials)..."
./bench_gold.sh --aa -c 64 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 2: Closed-Loop High Concurrency with Churn (64 clients, 64 reqs, 20% cancel)
echo ">>> [2/3] A/A Test: 64 Clients, 64 Reqs, 8 Slots, 20% Cancel (10 Trials)..."
./bench_gold.sh --aa -c 64 -n 64 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

# Case 3: Open-Loop Poisson Traffic (16 req/s, 96 reqs, 8 slots, 0% cancel)
echo ">>> [3/3] A/A Test: Poisson 16.0 req/s, 96 Reqs, 8 Slots, 0% Cancel (10 Trials)..."
./bench_gold.sh --aa -r 16.0 -n 96 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build

echo "================================================================================"
echo " A/A Noise Floor Benchmark Complete!"
echo "================================================================================"
