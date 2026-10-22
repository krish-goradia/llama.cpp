#!/bin/bash
#SBATCH --job-name=llama_gold_bench
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=01:30:00
#SBATCH --output=gold_%j.log
#SBATCH --error=gold_%j.err

set -e

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

chmod +x ./bench_gold.sh ./bench_ab.sh ./ab_bench.sh ./run_bench.sh 2>/dev/null || true

MODEL_FILE="my_models/3b.gguf"
if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found!"
    exit 1
fi

echo "================================================================================"
echo " Running Systems Gold-Standard Interleaved Benchmark Matrix (3 Trials/Case)"
echo "================================================================================"

echo ">>> [1/7] Saturation: 8 Clients on 8 Slots (0% Cancel)..."
./bench_gold.sh -c 8 -n 32 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [2/7] Moderate Backlog: 16 Clients on 8 Slots (0% Cancel)..."
./bench_gold.sh -c 16 -n 32 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [3/7] Heavy Contention: 32 Clients on 8 Slots (0% Cancel)..."
./bench_gold.sh -c 32 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [4/7] Burst Contention Storm: 64 Clients on 8 Slots (0% Cancel)..."
./bench_gold.sh -c 64 -n 64 -k 0.0 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [5/7] Real-World Churn: 8 Clients (10% Cancel)..."
./bench_gold.sh -c 8 -n 32 -k 0.1 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [6/7] High Cancellation Backlog: 32 Clients (20% Cancel)..."
./bench_gold.sh -c 32 -n 64 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo ">>> [7/7] Chaos Abort Storm: 64 Clients (20% Cancel)..."
./bench_gold.sh -c 64 -n 64 -k 0.2 -s 8 -p $BENCH_PORT -m "$MODEL_FILE" --trials 3 --no-build

echo "================================================================================"
echo " Systems Gold-Standard Benchmark Suite Finished Successfully!"
echo "================================================================================"
