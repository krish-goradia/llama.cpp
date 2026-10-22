#!/bin/bash
#SBATCH --job-name=llama_reproduce_case10
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --time=02:00:00
#SBATCH --output=reproduce_case10_%j.log
#SBATCH --error=reproduce_case10_%j.err

set -e

# ==============================================================================
# Step 1: Inspect the Historical Failed Log (Trial 2, PID 3603206 / Log 3603200)
# ==============================================================================
echo "================================================================================"
echo " [Step 1] Inspecting Historical Log: /tmp/llama_server_krish.goradia_3603200.log"
echo "================================================================================"

HIST_LOG="/tmp/llama_server_krish.goradia_3603200.log"
if [ -f "$HIST_LOG" ]; then
    echo "Found $HIST_LOG! Saving copy to ./failed_run_3603200.log..."
    cp "$HIST_LOG" ./failed_run_3603200.log
    echo "--- Last 50 Lines of Failed Run Server Log ---"
    tail -n 50 "$HIST_LOG"
    echo "--- Checking for Fatal Signals / Errors ---"
    grep -i -E "abort|assert|signal|segmentation|fault|crash|fatal|terminate" "$HIST_LOG" || echo "No explicit crash keyword found in log."
else
    echo "Log $HIST_LOG not present in /tmp on this node."
fi

# ==============================================================================
# Step 2: Environment & Compiler Setup
# ==============================================================================
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/Modules/init/bash 2>/dev/null || true
module purge 2>/dev/null || true
module load gcc-13.1.0 cuda-12.8 2>/dev/null || module load gcc/13.1.0 cuda/12.8 2>/dev/null || true

export CC=gcc
export CXX=g++
export LLAMA_ARG_N_GPU_LAYERS=99

echo "================================================================================"
echo " [Step 2] Compiling Modernized phase-2 (CUDA sm_89 + AVX2)"
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

./build/bin/test-server-queue

# ==============================================================================
# Step 3: Run Case 10 Reproduction Loop (50 Iterations)
# Case 10: 64 Clients | 64 Requests | 8 Slots | 20% Cancel Rate
# ==============================================================================
MODEL_FILE="my_models/3b.gguf"
PORT=$((10000 + RANDOM % 40000))
TOTAL_ITERS=50
FAIL_COUNT=0
PASS_COUNT=0

echo "================================================================================"
echo " [Step 3] Running 50-Iteration Reproduction Loop on Lock-Free Build"
echo " Configuration: 64 Clients | 64 Requests | 8 Slots | 20% Cancel"
echo "================================================================================"

mkdir -p ./reproduce_failures

for i in $(seq 1 "$TOTAL_ITERS"); do
    echo -ne ">>> Iteration [$i/$TOTAL_ITERS] ... "
    ITER_LOG="/tmp/llama_server_reproduce_${i}.log"
    JSON_OUT="/tmp/reproduce_iter_${i}.json"

    # Clean previous instances
    pkill -9 -f "llama-server" 2>/dev/null || true
    sleep 0.5

    # Launch server
    ./build/bin/llama-server \
        -m "$MODEL_FILE" \
        -c 16384 \
        -np 8 \
        --port "$PORT" > "$ITER_LOG" 2>&1 &
    SRV_PID=$!

    # Wait for ready
    READY=false
    for r in {1..20}; do
        if curl -s "http://localhost:${PORT}/health" | grep -q "ok"; then
            READY=true
            break
        fi
        sleep 0.5
    done

    if [ "$READY" = false ]; then
        echo "FAILED TO INITIALIZE SERVER (PID: $SRV_PID)!"
        cp "$ITER_LOG" "./reproduce_failures/server_init_fail_${i}.log"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        continue
    fi

    # Run client benchmark
    python3 tools/server/bench/bench_concurrency.py \
        --url "http://localhost:${PORT}" \
        -c 64 \
        -n 64 \
        -k 0.2 \
        --max-tokens 64 \
        --json-out "$JSON_OUT" > "/tmp/bench_out_${i}.txt" 2>&1 || true

    # Check whether server is still alive
    if ps -p "$SRV_PID" > /dev/null; then
        SERVER_ALIVE="ALIVE"
    else
        SERVER_ALIVE="DIED"
    fi

    # Parse failure count from JSON
    FAILS=$(python3 -c "import json; data=json.load(open('$JSON_OUT')); print(data.get('failed', data.get('summary', {}).get('requests_failed', -1)))" 2>/dev/null || echo "-1")

    if [ "$FAILS" = "0" ] && [ "$SERVER_ALIVE" = "ALIVE" ]; then
        echo "PASSED (0 fails, Server: $SERVER_ALIVE)"
        PASS_COUNT=$((PASS_COUNT + 1))
        rm -f "$ITER_LOG" "$JSON_OUT" "/tmp/bench_out_${i}.txt"
    else
        echo "FAILED (Requests Failed: $FAILS, Server: $SERVER_ALIVE)!"
        FAIL_COUNT=$((FAIL_COUNT + 1))
        cp "$ITER_LOG" "./reproduce_failures/server_fail_iter_${i}.log"
        cp "/tmp/bench_out_${i}.txt" "./reproduce_failures/client_fail_iter_${i}.txt"
        echo "    Saved crash/failure logs to ./reproduce_failures/"
    fi

    pkill -9 -f "llama-server" 2>/dev/null || true
done

echo "================================================================================"
echo " REPRODUCTION LOOP COMPLETE"
echo " Total Iterations : $TOTAL_ITERS"
echo " Passed (0 fails) : $PASS_COUNT"
echo " Failed           : $FAIL_COUNT"
echo "================================================================================"
