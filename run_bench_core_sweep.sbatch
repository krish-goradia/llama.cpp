#!/bin/bash
#SBATCH --job-name=llama_gold_core_sweep
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=04:00:00
#SBATCH --output=gold_core_sweep_%j.log
#SBATCH --error=gold_core_sweep_%j.err

set -e

# ==============================================================================
# EuroMLSys Host CPU Budget & Core Scarcity Scaling Sweep
# ==============================================================================
# Research Objective:
#   Map the exact dose-response curve of server throughput as server CPU cores
#   scale from extreme scarcity to parity: 4, 8, 12, 16, 20, 24 cores.
#
# Methodological Guarantees:
#   - Client is permanently fixed to Cores 24-31 (8 isolated cores, ZERO overlap!)
#   - Server cores swept across Cores 0 through (N-1)
#   - Evaluated at high concurrency (64 clients, 16 slots, 10 paired trials)
#   - Explicitly records both Master and Phase-2 at matched vs default HTTP threads
# ==============================================================================

# 1. Environment & Modules
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc-13.1.0 cuda-12.8 2>/dev/null || module load gcc/13.1.0 cuda/12.8 2>/dev/null || true

export CC=gcc
export CXX=g++
export LLAMA_ARG_N_GPU_LAYERS=99
MODEL_FILE="my_models/3b.gguf"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Model file $MODEL_FILE not found!"
    exit 1
fi

chmod +x ./bench_gold.sh ./run_bench.sh tools/server/bench/*.py 2>/dev/null || true

# 2. Build Both Architectures
echo "================================================================================"
echo " [Step 1] Compiling Phase-2 Lock-Free (build)"
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
fi

# Fixed Client Isolation (Cores 24-31)
export PIN_CORES=true
export CLIENT_CORES="24-31"

CORE_TIERS=("0-3" "0-7" "0-11" "0-15" "0-19" "0-23")
CORE_COUNTS=(4 8 12 16 20 24)

echo "================================================================================"
echo " [Step 2] Executing Server Core Scaling Sweep (Client Fixed on Cores 24-31)"
echo "================================================================================"

for i in "${!CORE_TIERS[@]}"; do
    TIER="${CORE_TIERS[$i]}"
    COUNT="${CORE_COUNTS[$i]}"
    BENCH_PORT=$((10000 + RANDOM % 40000))

    echo ""
    echo "================================================================================"
    echo " >>> Core Sweep Point: Server on $COUNT Cores ($TIER) | Client on 24-31"
    echo "================================================================================"

    export SERVER_CORES="$TIER"
    ./bench_gold.sh -c 64 -n 64 -k 0.0 -s 16 -p $BENCH_PORT -m "$MODEL_FILE" --trials 10 --no-build
done

echo "================================================================================"
echo " Server Core Scaling Sweep Completed Successfully!"
echo "================================================================================"
