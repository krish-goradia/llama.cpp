#!/usr/bin/env bash
# ==============================================================================
# Unified Master vs Current (Lock-Free) A/B Concurrency Benchmark
# ==============================================================================
# Automatically runs interleaved benchmarks across Legacy Master and Current
# codebases, collects full percentile metrics, and renders expressive,
# side-by-side comparison tables.
# ==============================================================================

set -e

DIR_CURRENT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR_MASTER="${DIR_CURRENT}/llama.cpp-master"

# Default configuration
MODEL_PATH="${MODEL_PATH:-my_models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf}"
PORT="${PORT:-8080}"
SLOTS="${SLOTS:-8}"
CTX="${CTX:-4096}"
MAX_TOKENS="${MAX_TOKENS:-64}"
COOLDOWN="${COOLDOWN:-15}"
DO_BUILD=true
REVERSE=false
SUITE="all"
CUSTOM_CONCURRENCY=""
CUSTOM_REQUESTS=""
CUSTOM_CANCEL=""

# ANSI Colors
BOLD="\033[1m"
GREEN="\033[32m"
RED="\033[31m"
CYAN="\033[36m"
YELLOW="\033[33m"
RESET="\033[0m"

# Parse CLI flags
while [[ $# -gt 0 ]]; do
  case $1 in
    --suite)
      SUITE="$2"
      shift 2
      ;;
    -c|--concurrency)
      CUSTOM_CONCURRENCY="$2"
      SUITE="custom"
      shift 2
      ;;
    -n|--requests)
      CUSTOM_REQUESTS="$2"
      SUITE="custom"
      shift 2
      ;;
    -k|--cancel|--cancel-rate)
      CUSTOM_CANCEL="$2"
      SUITE="custom"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    -s|--slots)
      SLOTS="$2"
      shift 2
      ;;
    -m|--model)
      MODEL_PATH="$2"
      shift 2
      ;;
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    --cooldown)
      COOLDOWN="$2"
      shift 2
      ;;
    --reverse)
      REVERSE=true
      shift
      ;;
    --no-build)
      DO_BUILD=false
      shift
      ;;
    -h|--help)
      echo "Usage: ./bench_ab.sh [options]"
      echo ""
      echo "Options:"
      echo "  --suite <all|16|64|custom>   Benchmark suite to run (default: all)"
      echo "                               - all: 16c (0% & 20% cancel) + 64c (0% & 20% cancel)"
      echo "                               - 16 : 16 clients (0% & 20% cancel)"
      echo "                               - 64 : 64 clients (0% & 20% cancel)"
      echo "  -c, --concurrency <int>      Custom client concurrency"
      echo "  -n, --requests <int>         Custom total request count"
      echo "  -k, --cancel <float>         Custom cancel rate (0.0 to 1.0)"
      echo "  -s, --slots <int>            Number of parallel server slots (default: 8)"
      echo "  -m, --model <path>           Path to GGUF model"
      echo "  -p, --port <int>             Server port (default: 8080)"
      echo "  --cooldown <seconds>         Cooldown sleep between benchmark runs (default: 15)"
      echo "  --reverse                    Run Current modernized server first, then Master baseline"
      echo "  --no-build                   Skip compilation of binaries"
      echo "  -h, --help                   Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

if [ ! -d "$DIR_MASTER" ]; then
  echo -e "${RED}ERROR: Master directory not found at $DIR_MASTER${RESET}"
  exit 1
fi

if [ ! -f "$DIR_CURRENT/$MODEL_PATH" ]; then
  echo -e "${RED}ERROR: Model file not found at $DIR_CURRENT/$MODEL_PATH${RESET}"
  exit 1
fi

TMP_DIR="/tmp/llama_ab_bench_$$"
mkdir -p "$TMP_DIR"
trap 'rm -rf "$TMP_DIR"; pkill -9 -f "llama-server" 2>/dev/null || true' EXIT

echo -e "${BOLD}================================================================================${RESET}"
echo -e "${BOLD}          llama.cpp Automated A/B Performance Comparison Benchmark             ${RESET}"
echo -e "${BOLD}================================================================================${RESET}"
echo -e "  Current (Modernized) : ${CYAN}$DIR_CURRENT${RESET}"
echo -e "  Legacy (Master Base) : ${CYAN}$DIR_MASTER${RESET}"
echo -e "  Model                : ${YELLOW}$MODEL_PATH${RESET}"
echo -e "  Parallel Slots       : ${YELLOW}$SLOTS${RESET}"
echo -e "  Suite Selected       : ${GREEN}$SUITE${RESET}"
echo -e "${BOLD}================================================================================${RESET}"
echo ""

# 1. Build Phase
if [ "$DO_BUILD" = true ]; then
  echo -e "${BOLD}>>> [1/2] Building Current Modernized Server (Lock-Free)...${RESET}"
  echo -e "    Directory : ${CYAN}$DIR_CURRENT${RESET}"
  echo -e "    Target    : ${YELLOW}$DIR_CURRENT/build/bin/llama-server${RESET}"
  cd "$DIR_CURRENT"
  cmake --build build --config Release -j --target llama-server
  echo -e "    Binary    : $(ls -lh "$DIR_CURRENT/build/bin/llama-server" 2>/dev/null | awk '{print $9, "(" $5 ")"}')\n"

  echo -e "${BOLD}>>> [2/2] Building Legacy Master Server (Baseline)...${RESET}"
  echo -e "    Directory : ${CYAN}$DIR_MASTER${RESET}"
  echo -e "    Target    : ${YELLOW}$DIR_MASTER/build/bin/llama-server${RESET}"
  cd "$DIR_MASTER"
  cmake --build build --config Release -j --target llama-server
  echo -e "    Binary    : $(ls -lh "$DIR_MASTER/build/bin/llama-server" 2>/dev/null | awk '{print $9, "(" $5 ")"}')\n"
  echo ""
fi

# Function to run a single comparison case
run_ab_case() {
  local CASE_NAME="$1"
  local CLIENTS="$2"
  local REQS="$3"
  local CANCEL="$4"

  local MASTER_JSON="$TMP_DIR/master_${CLIENTS}c_${CANCEL}k.json"
  local CURRENT_JSON="$TMP_DIR/current_${CLIENTS}c_${CANCEL}k.json"

  echo -e "\n${BOLD}${CYAN}################################################################################${RESET}"
  echo -e "${BOLD}${CYAN} RUNNING CASE: ${CASE_NAME} (${CLIENTS} Clients | ${REQS} Requests | $(python3 -c "print(int(${CANCEL}*100))")% Cancel Rate)${RESET}"
  echo -e "${BOLD}${CYAN}################################################################################${RESET}\n"

  if [ "$REVERSE" = true ]; then
    # Step 1: Run Current Modernized Server FIRST
    echo -e "${BOLD}>>> [Step 1/2] Executing Current Modernized Server (Lock-Free) [FIRST]...${RESET}"
    echo -e "    Directory : ${CYAN}$DIR_CURRENT${RESET}"
    echo -e "    Binary    : ${YELLOW}$DIR_CURRENT/build/bin/llama-server${RESET}"
    pkill -9 -f "llama-server" 2>/dev/null || true
    sleep 2

    cd "$DIR_CURRENT"
    ./run_bench.sh \
      -c "$CLIENTS" \
      -n "$REQS" \
      --cancel "$CANCEL" \
      -s "$SLOTS" \
      -p "$PORT" \
      -m "$MODEL_PATH" \
      --ctx "$CTX" \
      --max-tokens "$MAX_TOKENS" \
      --json-out "$CURRENT_JSON" \
      --no-build

    echo -e "\n>>> Cooldown for ${COOLDOWN}s to stabilize CPU/GPU thermals..."
    pkill -9 -f "llama-server" 2>/dev/null || true
    sleep "$COOLDOWN"

    # Step 2: Run Legacy Master Baseline SECOND
    echo -e "${BOLD}>>> [Step 2/2] Executing Legacy Master Baseline [SECOND]...${RESET}"
    echo -e "    Directory : ${CYAN}$DIR_MASTER${RESET}"
    echo -e "    Binary    : ${YELLOW}$DIR_MASTER/build/bin/llama-server${RESET}"
    cd "$DIR_MASTER"
    ./run_bench.sh \
      -c "$CLIENTS" \
      -n "$REQS" \
      --cancel "$CANCEL" \
      -s "$SLOTS" \
      -p "$PORT" \
      -m "$MODEL_PATH" \
      --ctx "$CTX" \
      --max-tokens "$MAX_TOKENS" \
      --json-out "$MASTER_JSON" \
      --no-build

    echo -e "\n>>> Cooldown for ${COOLDOWN}s..."
    pkill -9 -f "llama-server" 2>/dev/null || true
    sleep "$COOLDOWN"
  else
    # Standard: Run Legacy Master Baseline FIRST
    echo -e "${BOLD}>>> [Step 1/2] Executing Legacy Master Baseline [FIRST]...${RESET}"
    echo -e "    Directory : ${CYAN}$DIR_MASTER${RESET}"
    echo -e "    Binary    : ${YELLOW}$DIR_MASTER/build/bin/llama-server${RESET}"
    pkill -9 -f "llama-server" 2>/dev/null || true
    sleep 2

    cd "$DIR_MASTER"
    ./run_bench.sh \
      -c "$CLIENTS" \
      -n "$REQS" \
      --cancel "$CANCEL" \
      -s "$SLOTS" \
      -p "$PORT" \
      -m "$MODEL_PATH" \
      --ctx "$CTX" \
      --max-tokens "$MAX_TOKENS" \
      --json-out "$MASTER_JSON" \
      --no-build

    echo -e "\n>>> Cooldown for ${COOLDOWN}s to stabilize CPU/GPU thermals..."
    pkill -9 -f "llama-server" 2>/dev/null || true
    sleep "$COOLDOWN"

    # Step 2: Run Current Modernized Server SECOND
    echo -e "${BOLD}>>> [Step 2/2] Executing Current Modernized Server (Lock-Free) [SECOND]...${RESET}"
    echo -e "    Directory : ${CYAN}$DIR_CURRENT${RESET}"
    echo -e "    Binary    : ${YELLOW}$DIR_CURRENT/build/bin/llama-server${RESET}"
    cd "$DIR_CURRENT"
    ./run_bench.sh \
      -c "$CLIENTS" \
      -n "$REQS" \
      --cancel "$CANCEL" \
      -s "$SLOTS" \
      -p "$PORT" \
      -m "$MODEL_PATH" \
      --ctx "$CTX" \
      --max-tokens "$MAX_TOKENS" \
      --json-out "$CURRENT_JSON" \
      --no-build

    echo -e "\n>>> Cooldown for ${COOLDOWN}s..."
    pkill -9 -f "llama-server" 2>/dev/null || true
    sleep "$COOLDOWN"
  fi

  # 3. Render Formatted Expressive Comparison Table via Python Helper
  CASE_NAME="$CASE_NAME" CLIENTS="$CLIENTS" REQS="$REQS" SLOTS="$SLOTS" CANCEL="$CANCEL" MASTER_JSON="$MASTER_JSON" CURRENT_JSON="$CURRENT_JSON" python3 - <<'PYCODE'
import json, sys, os

master_file = os.environ["MASTER_JSON"]
current_file = os.environ["CURRENT_JSON"]
case_name = os.environ["CASE_NAME"]
clients = os.environ["CLIENTS"]
reqs = os.environ["REQS"]
slots = os.environ["SLOTS"]
cancel_rate_pct = int(round(float(os.environ["CANCEL"]) * 100))

with open(master_file) as f:
    m = json.load(f)
with open(current_file) as f:
    c = json.load(f)

# ANSI Color codes
BOLD = "\033[1m"
GREEN = "\033[32m"
RED = "\033[31m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
GRAY = "\033[90m"
RESET = "\033[0m"

def calc_delta(m_val, c_val, higher_is_better=False):
    if m_val == 0:
        return 0.0, "[SAME]", RESET
    pct = ((c_val - m_val) / m_val) * 100.0
    if abs(pct) < 0.05:
        return pct, "[SAME]", GRAY
    if higher_is_better:
        if pct > 0:
            return pct, "[FASTER]", GREEN
        else:
            return pct, "[SLOWER]", RED
    else:
        if pct < 0:
            return pct, "[IMPROVED]", GREEN
        else:
            return pct, "[REGRESSION]", RED

def fmt_row(label, m_str, c_str, delta_pct, status, color):
    sign = "+" if delta_pct > 0 else ""
    delta_str = f"{sign}{delta_pct:6.2f}%" if status != "[SAME]" else "  0.00%"
    return f" {label:<32} {m_str:>16} {c_str:>18}   {color}{delta_str:>9}  {status:<12}{RESET}"

print(f"{BOLD}========================================================================================================{RESET}")
print(f"{BOLD}                       A/B BENCHMARK COMPARISON REPORT: {case_name}{RESET}")
print(f"{BOLD}========================================================================================================{RESET}")
print(f" Concurrency: {BOLD}{clients} clients{RESET} | Requests: {BOLD}{reqs} total{RESET} | Slots: {BOLD}{slots}{RESET} | Cancel Rate: {BOLD}{cancel_rate_pct}%{RESET}")
print(f"{BOLD}--------------------------------------------------------------------------------------------------------{RESET}")
print(f" {BOLD}{'METRIC':<32} {'LEGACY (MASTER)':>16} {'CURRENT (LOCKFREE)':>18}   {'DELTA (%)':>9}  {'STATUS':<12}{RESET}")
print(f"{BOLD}--------------------------------------------------------------------------------------------------------{RESET}")

# 1. Throughput & Overview
t_pct, t_stat, t_col = calc_delta(m['tok_per_sec'], c['tok_per_sec'], higher_is_better=True)
print(fmt_row("Throughput (tokens/sec)", f"{m['tok_per_sec']:.2f} tok/s", f"{c['tok_per_sec']:.2f} tok/s", t_pct, t_stat, t_col))

w_pct, w_stat, w_col = calc_delta(m['total_time_s'], c['total_time_s'], higher_is_better=False)
print(fmt_row("Total Wall Time (s)", f"{m['total_time_s']:.2f} s", f"{c['total_time_s']:.2f} s", w_pct, w_stat, w_col))

print(f" {'Total Tokens Emitted':<32} {m['total_tokens']:>16} {c['total_tokens']:>18}")
m_req_str = f"{m['successful']}/{m['cancelled']}/{m['failed']}"
c_req_str = f"{c['successful']}/{c['cancelled']}/{c['failed']}"
print(f" {'Requests (Succ / Canc / Fail)':<32} {m_req_str:>16} {c_req_str:>18}")

# 2. TTFT
print(f"{BOLD}--------------------------------------------------------------------------------------------------------{RESET}")
print(f"{BOLD} TIME TO FIRST TOKEN (TTFT in ms){RESET}")
for k, name in [('mean', 'Mean'), ('p50', 'p50 (Median)'), ('p95', 'p95 (Tail)'), ('p99', 'p99 (Worst Tail)'), ('max', 'Max Spike')]:
    if k in m['ttft'] and k in c['ttft']:
        pct, stat, col = calc_delta(m['ttft'][k], c['ttft'][k], higher_is_better=False)
        print(fmt_row(f"   - {name}", f"{m['ttft'][k]:.2f} ms", f"{c['ttft'][k]:.2f} ms", pct, stat, col))

# 3. ITL
print(f"{BOLD}--------------------------------------------------------------------------------------------------------{RESET}")
print(f"{BOLD} INTER-TOKEN LATENCY (ITL in ms){RESET}")
for k, name in [('mean', 'Mean'), ('p50', 'p50 (Median)'), ('p95', 'p95 (Tail Jitter)'), ('p99', 'p99 (Worst Tail)'), ('max', 'Max Spike')]:
    if k in m['itl'] and k in c['itl']:
        pct, stat, col = calc_delta(m['itl'][k], c['itl'][k], higher_is_better=False)
        print(fmt_row(f"   - {name}", f"{m['itl'][k]:.2f} ms", f"{c['itl'][k]:.2f} ms", pct, stat, col))

# 4. Request Duration
print(f"{BOLD}--------------------------------------------------------------------------------------------------------{RESET}")
print(f"{BOLD} REQUEST DURATION (Total Time in ms){RESET}")
for k, name in [('mean', 'Mean'), ('p50', 'p50 (Median)'), ('p95', 'p95 (Tail)'), ('p99', 'p99 (Worst Tail)')]:
    if k in m['duration'] and k in c['duration']:
        pct, stat, col = calc_delta(m['duration'][k], c['duration'][k], higher_is_better=False)
        print(fmt_row(f"   - {name}", f"{m['duration'][k]:.2f} ms", f"{c['duration'][k]:.2f} ms", pct, stat, col))

print(f"{BOLD}========================================================================================================{RESET}\n")
PYCODE
}

# Execute selected suite
case "$SUITE" in
  all)
    run_ab_case "16-Client Baseline" 16 32 0.0
    run_ab_case "16-Client 20% Cancel" 16 32 0.2
    run_ab_case "64-Client High Concurrency" 64 64 0.0
    run_ab_case "64-Client Concurrency + 20% Cancel" 64 64 0.2
    ;;
  16)
    run_ab_case "16-Client Baseline" 16 32 0.0
    run_ab_case "16-Client 20% Cancel" 16 32 0.2
    ;;
  64)
    run_ab_case "64-Client High Concurrency" 64 64 0.0
    run_ab_case "64-Client Concurrency + 20% Cancel" 64 64 0.2
    ;;
  custom)
    C_CLI="${CUSTOM_CONCURRENCY:-8}"
    C_REQ="${CUSTOM_REQUESTS:-32}"
    C_CAN="${CUSTOM_CANCEL:-0.0}"
    run_ab_case "Custom A/B Case" "$C_CLI" "$C_REQ" "$C_CAN"
    ;;
  *)
    echo "Unknown suite: $SUITE (must be all, 16, 64, or custom)"
    exit 1
    ;;
esac

echo -e "${BOLD}${GREEN}================================================================================${RESET}"
echo -e "${BOLD}${GREEN}              All A/B Benchmarks Finished Successfully!                        ${RESET}"
echo -e "${BOLD}${GREEN}================================================================================${RESET}"
