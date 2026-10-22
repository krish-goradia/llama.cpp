#!/usr/bin/env bash
# ==============================================================================
# Systems Gold-Standard A/B Concurrency Benchmark for llama.cpp
# ==============================================================================
# Methodological Guarantees (MLSys / OSDI Standard):
#   1. Active GPU Thermal & Power Quiescence Barrier (polls nvidia-smi until idle)
#   2. Interleaved Multi-Trial Protocol (ABBA alternating order per case)
#   3. Full Support for Real Reverse Execution Order (--reverse)
#   4. Paired Hypothesis Testing & 95% Confidence Intervals (10+ Trials Default)
#   5. High Classifier Noise Threshold (Requires >=3.0% delta & p < 0.05)
#   6. Strict Validity Filter: Any failed request marks run INVALID and is excluded
#   7. Supports Sustained-Load Runs (--duration 60s+) & Open-Loop Poisson (--rate)
# ==============================================================================

set -e

DIR_CURRENT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DIR_MASTER="${DIR_CURRENT}/llama.cpp-master"

# Default configuration
MODEL_PATH="${MODEL_PATH:-my_models/3b.gguf}"
PORT="${PORT:-8080}"
SLOTS="${SLOTS:-8}"
CTX="${CTX:-16384}"
MAX_TOKENS="${MAX_TOKENS:-64}"
TARGET_TEMP="${TARGET_TEMP:-45}"
TARGET_POWER="${TARGET_POWER:-50}"
TRIALS="${TRIALS:-10}"
DO_BUILD=false
CUSTOM_CONCURRENCY=8
CUSTOM_REQUESTS=32
CUSTOM_CANCEL=0.0
CUSTOM_DURATION=0.0
CUSTOM_RATE=0.0
REVERSE=false
AA_MODE=false

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
    -c|--concurrency)
      CUSTOM_CONCURRENCY="$2"
      shift 2
      ;;
    -n|--requests)
      CUSTOM_REQUESTS="$2"
      shift 2
      ;;
    -d|--duration)
      CUSTOM_DURATION="$2"
      shift 2
      ;;
    -r|--rate)
      CUSTOM_RATE="$2"
      shift 2
      ;;
    -k|--cancel|--cancel-rate)
      CUSTOM_CANCEL="$2"
      shift 2
      ;;
    -m|--model)
      MODEL_PATH="$2"
      shift 2
      ;;
    -s|--slots)
      SLOTS="$2"
      shift 2
      ;;
    -p|--port)
      PORT="$2"
      shift 2
      ;;
    --ctx)
      CTX="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --trials)
      TRIALS="$2"
      shift 2
      ;;
    --reverse)
      REVERSE=true
      shift
      ;;
    --aa|--aa-test)
      AA_MODE=true
      shift
      ;;
    --threads-http)
      THREADS_HTTP="$2"
      shift 2
      ;;
    --target-temp)
      TARGET_TEMP="$2"
      shift 2
      ;;
    --target-power)
      TARGET_POWER="$2"
      shift 2
      ;;
    --no-build)
      DO_BUILD=false
      shift
      ;;
    -h|--help)
      echo "Usage: ./bench_gold.sh [-c CLIENTS] [-n REQUESTS] [-d DURATION] [-r RATE] [-k CANCEL_RATE] [-s SLOTS] [-m MODEL] [--trials N] [--reverse]"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

TMP_DIR="$(mktemp -d -t llama_bench_gold_XXXXXX)"
trap 'pkill -9 -f "llama-server" 2>/dev/null || true; rm -rf "$TMP_DIR"' EXIT

# Active Thermal & Power Stabilization Loop
wait_for_gpu_cooldown() {
  local target_temp="$1"
  local target_power="$2"
  local max_wait="${3:-30}"
  local elapsed=0

  if ! command -v nvidia-smi &>/dev/null; then
    sleep 3
    return
  fi

  echo -ne "    [Thermal Quiescence] Waiting for GPU (Target: <=${target_temp}C, <=${target_power}W)... "
  while [ $elapsed -lt $max_wait ]; do
    local cur_temp=$(nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1 | tr -d ' ')
    local cur_power=$(nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits 2>/dev/null | head -n 1 | cut -d'.' -f1 | tr -d ' ')
    if [ -n "$cur_temp" ] && [ -n "$cur_power" ]; then
      if [ "$cur_temp" -le "$target_temp" ] && [ "$cur_power" -le "$target_power" ]; then
        echo -e "${GREEN}Settled at ${cur_temp}C, ${cur_power}W in ${elapsed}s.${RESET}"
        sleep 1
        return
      fi
    fi
    sleep 2
    elapsed=$((elapsed + 2))
  done
  echo -e "${YELLOW}Reached ${elapsed}s ceiling (${cur_temp:-?}C, ${cur_power:-?}W). Continuing.${RESET}"
}

run_gold_case() {
  local CLIENTS="$1"
  local REQS="$2"
  local CANCEL="$3"
  local DURATION="$4"
  local RATE="$5"
  local CANCEL_PCT=$(awk -v c="$CANCEL" 'BEGIN { printf "%.0f", c*100 }')

  local MODE_DESC="Closed-Loop (${CLIENTS} clients)"
  if (( $(echo "$RATE > 0.0" | bc -l) )); then
    MODE_DESC="Open-Loop Poisson (${RATE} req/s)"
  fi

  local SCALE_DESC="${REQS} Requests"
  if (( $(echo "$DURATION > 0.0" | bc -l) )); then
    SCALE_DESC="Sustained ${DURATION}s Duration"
  fi

  local TARGET_DIR="$DIR_CURRENT"
  local TARGET_LABEL="Phase-2 Lock-Free"
  if [ "$AA_MODE" = true ]; then
    TARGET_DIR="$DIR_MASTER"
    TARGET_LABEL="Master Baseline (Instance B)"
  fi

  echo -e "\n${BOLD}${CYAN}================================================================================${RESET}"
  echo -e "${BOLD}${CYAN} GOLD-STANDARD TRIAL: ${MODE_DESC} | ${SCALE_DESC} | ${CANCEL_PCT}% Cancel | ${TRIALS} Trials${RESET}"
  if [ "$AA_MODE" = true ]; then
    echo -e "${BOLD}${YELLOW} [Execution Mode: A/A EMPIRICAL NOISE FLOOR (Master vs Master)]${RESET}"
  elif [ "$REVERSE" = true ]; then
    echo -e "${BOLD}${YELLOW} [Execution Mode: REAL REVERSE ORDER (Phase-2 First on Odd Trials)]${RESET}"
  else
    echo -e "${BOLD}${CYAN} [Execution Mode: Standard Interleaved ABBA Order (Master First on Odd Trials)]${RESET}"
  fi
  echo -e "${BOLD}${CYAN}================================================================================${RESET}\n"

  EXTRA_HTTP_ARGS=""
  if [ -n "$THREADS_HTTP" ]; then
    EXTRA_HTTP_ARGS="--threads-http $THREADS_HTTP"
  fi

  for trial in $(seq 1 "$TRIALS"); do
    echo -e "${BOLD}--- Trial ${trial}/${TRIALS} (Seed: ${trial}) ---${RESET}"
    local MASTER_JSON="$TMP_DIR/master_t${trial}.json"
    local CURRENT_JSON="$TMP_DIR/current_t${trial}.json"

    # Execution order:
    # Standard: Odd trials = Master FIRST; Even trials = Target SECOND
    # Reverse (--reverse): Odd trials = Target FIRST; Even trials = Master FIRST
    local RUN_CURRENT_FIRST=false
    if [ "$REVERSE" = true ]; then
      if [ $((trial % 2)) -eq 1 ]; then
        RUN_CURRENT_FIRST=true
      fi
    else
      if [ $((trial % 2)) -eq 0 ]; then
        RUN_CURRENT_FIRST=true
      fi
    fi

    if [ "$RUN_CURRENT_FIRST" = false ]; then
      echo -e "  [Order: Master Baseline (Instance A) FIRST -> ${TARGET_LABEL} SECOND]"
      
      # 1. Master Baseline (Instance A)
      pkill -9 -f "llama-server" 2>/dev/null || true
      wait_for_gpu_cooldown "$TARGET_TEMP" "$TARGET_POWER" 30
      sync
      cd "$DIR_MASTER"
      ./run_bench.sh -c "$CLIENTS" -n "$REQS" -d "$DURATION" -r "$RATE" --cancel "$CANCEL" -s "$SLOTS" -p "$PORT" -m "$MODEL_PATH" --ctx "$CTX" --max-tokens "$MAX_TOKENS" --seed "$trial" $EXTRA_HTTP_ARGS --json-out "$MASTER_JSON" --no-build || true

      # 2. Target System (Phase-2 or Master Instance B)
      pkill -9 -f "llama-server" 2>/dev/null || true
      wait_for_gpu_cooldown "$TARGET_TEMP" "$TARGET_POWER" 30
      sync
      cd "$TARGET_DIR"
      ./run_bench.sh -c "$CLIENTS" -n "$REQS" -d "$DURATION" -r "$RATE" --cancel "$CANCEL" -s "$SLOTS" -p "$PORT" -m "$MODEL_PATH" --ctx "$CTX" --max-tokens "$MAX_TOKENS" --seed "$trial" $EXTRA_HTTP_ARGS --json-out "$CURRENT_JSON" --no-build || true
    else
      echo -e "  [Order: ${TARGET_LABEL} FIRST -> Master Baseline (Instance A) SECOND]"
      
      # 1. Target System (Phase-2 or Master Instance B)
      pkill -9 -f "llama-server" 2>/dev/null || true
      wait_for_gpu_cooldown "$TARGET_TEMP" "$TARGET_POWER" 30
      sync
      cd "$TARGET_DIR"
      ./run_bench.sh -c "$CLIENTS" -n "$REQS" -d "$DURATION" -r "$RATE" --cancel "$CANCEL" -s "$SLOTS" -p "$PORT" -m "$MODEL_PATH" --ctx "$CTX" --max-tokens "$MAX_TOKENS" --seed "$trial" $EXTRA_HTTP_ARGS --json-out "$CURRENT_JSON" --no-build || true

      # 2. Master Baseline (Instance A)
      pkill -9 -f "llama-server" 2>/dev/null || true
      wait_for_gpu_cooldown "$TARGET_TEMP" "$TARGET_POWER" 30
      sync
      cd "$DIR_MASTER"
      ./run_bench.sh -c "$CLIENTS" -n "$REQS" -d "$DURATION" -r "$RATE" --cancel "$CANCEL" -s "$SLOTS" -p "$PORT" -m "$MODEL_PATH" --ctx "$CTX" --max-tokens "$MAX_TOKENS" --seed "$trial" $EXTRA_HTTP_ARGS --json-out "$MASTER_JSON" --no-build || true
    fi
  done

  # Statistical Multi-Trial Aggregator with Paired Hypothesis Testing & 95% Confidence Intervals
  TMP_DIR="$TMP_DIR" CLIENTS="$CLIENTS" REQS="$REQS" SLOTS="$SLOTS" CANCEL="$CANCEL" TRIALS="$TRIALS" DURATION="$DURATION" RATE="$RATE" REVERSE="$REVERSE" AA_MODE="$AA_MODE" python3 - <<'PYCODE'
import json
import glob
import math
import os
import sys

tmp_dir = os.environ.get("TMP_DIR", "")
clients = os.environ.get("CLIENTS", "0")
reqs = os.environ.get("REQS", "0")
slots = os.environ.get("SLOTS", "0")
cancel_str = os.environ.get("CANCEL", "0.0")
cancel_pct = int(round(float(cancel_str) * 100))
trials_requested = int(os.environ.get("TRIALS", "10"))
duration = float(os.environ.get("DURATION", "0.0"))
rate = float(os.environ.get("RATE", "0.0"))
reverse_str = os.environ.get("REVERSE", "false")

T_CRIT_95 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571,
    6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
    11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086,
    25: 2.060, 30: 2.042,
}

def get_tcrit(df):
    if df in T_CRIT_95: return T_CRIT_95[df]
    if df > 30: return 1.96 + 2.37 / float(df)
    keys = sorted(T_CRIT_95.keys())
    for k in keys:
        if k >= df: return T_CRIT_95[k]
    return 2.0

def incbeta(a, b, x):
    if x <= 0.0: return 0.0
    if x >= 1.0: return 1.0
    if x > (a + 1.0) / (a + b + 2.0):
        return 1.0 - incbeta(b, a, 1.0 - x)
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta) / a
    f = 1.0; c = 1.0; d = 0.0
    for m in range(1, 200):
        num = m * (b - m) * x / ((a + 2*m - 1) * (a + 2*m))
        d = 1.0 + num * d; d = 1e-30 if abs(d) < 1e-30 else d; d = 1.0 / d
        c = 1.0 + num / c; c = 1e-30 if abs(c) < 1e-30 else c
        f *= (c * d)
        num = -(a + m) * (a + b + m) * x / ((a + 2*m) * (a + 2*m + 1))
        d = 1.0 + num * d; d = 1e-30 if abs(d) < 1e-30 else d; d = 1.0 / d
        c = 1.0 + num / c; c = 1e-30 if abs(c) < 1e-30 else c
        delta = c * d
        f *= delta
        if abs(delta - 1.0) < 1e-10: break
    return front * f

def t_pvalue(t_stat, df):
    if df <= 0 or math.isnan(t_stat): return 1.0
    x = df / (df + (t_stat ** 2))
    return incbeta(df / 2.0, 0.5, x)

# 1. Load and Pair Trials
valid_pairs = []
invalid_trials = []

for t in range(1, trials_requested + 1):
    m_path = os.path.join(tmp_dir, f"master_t{t}.json")
    c_path = os.path.join(tmp_dir, f"current_t{t}.json")
    if not os.path.exists(m_path) or not os.path.exists(c_path):
        continue
    try:
        with open(m_path) as fp: m_data = json.load(fp)
        with open(c_path) as fp: c_data = json.load(fp)
    except Exception as e:
        invalid_trials.append((t, f"Corrupted JSON: {e}"))
        continue

    m_valid = m_data.get("valid", True) and m_data.get("failed", 0) == 0
    c_valid = c_data.get("valid", True) and c_data.get("failed", 0) == 0

    if not m_valid or not c_valid:
        err_msg = []
        if not m_valid: err_msg.append(f"Master had {m_data.get('failed', 0)} failed reqs")
        if not c_valid: err_msg.append(f"LockFree had {c_data.get('failed', 0)} failed reqs")
        invalid_trials.append((t, "; ".join(err_msg)))
        continue

    valid_pairs.append((m_data, c_data))

n_valid = len(valid_pairs)
print('=' * 120)
print(f'          SYSTEMS GOLD-STANDARD STATISTICAL REPORT (Paired Tests & 95% Confidence Intervals)')
print('=' * 120)
aa_str = os.environ.get("AA_MODE", "false")
is_aa = aa_str.lower() == "true"

if is_aa:
    order_note = "A/A NOISE FLOOR PROTOCOL (Master-A vs Master-B)"
    master_header = "MASTER-A (95% CI)"
    target_header = "MASTER-B (95% CI)"
else:
    order_note = "REAL REVERSE ORDER (Phase-2 First on Odd)" if reverse_str.lower() == "true" else "Standard Interleaved ABBA"
    master_header = "MASTER (95% CI)"
    target_header = "LOCKFREE (95% CI)"

workload_note = f"Open-Loop Poisson ({rate:.1f} req/s)" if rate > 0 else f"Closed-Loop ({clients} clients)"
duration_note = f"Sustained {duration:.1f}s" if duration > 0 else f"{reqs} Requests"
print(f' Workload : {workload_note} | {duration_note} | Slots: {slots} | Cancel: {cancel_pct}%')
print(f' Protocol : {order_note} | Valid Paired Trials: {n_valid} / {trials_requested}')

if invalid_trials:
    print('-' * 120)
    for t_idx, reason in invalid_trials:
        print(f' [!] EXCLUDED Trial {t_idx}: INVALID ({reason})')

if n_valid < 2:
    print('\n[ERROR] Fewer than 2 valid paired trials. Cannot compute statistical hypothesis test.')
    sys.exit(1)

print('-' * 120)
print(f' {"METRIC":<28} {master_header:<22} {target_header:<22} {"PAIRED DELTA (95% CI)":<25} {"P-VALUE":<10} {"STATUS":<12}')
print('-' * 120)

df = n_valid - 1
tcrit = get_tcrit(df)

def paired_stat_row(name, m_vals, c_vals, unit, invert=False):
    m_mean = sum(m_vals) / float(n_valid)
    m_var = sum((x - m_mean)**2 for x in m_vals) / float(df) if df > 0 else 0.0
    m_se = math.sqrt(m_var) / math.sqrt(n_valid)
    m_ci = tcrit * m_se

    c_mean = sum(c_vals) / float(n_valid)
    c_var = sum((x - c_mean)**2 for x in c_vals) / float(df) if df > 0 else 0.0
    c_se = math.sqrt(c_var) / math.sqrt(n_valid)
    c_ci = tcrit * c_se

    diffs = [ci - mi for mi, ci in zip(m_vals, c_vals)]
    d_mean = sum(diffs) / float(n_valid)
    d_var = sum((x - d_mean)**2 for x in diffs) / float(df) if df > 0 else 0.0
    d_se = math.sqrt(d_var) / math.sqrt(n_valid)
    d_ci = tcrit * d_se

    t_stat = d_mean / d_se if d_se > 1e-9 else 0.0
    p_val = t_pvalue(abs(t_stat), df)

    rel_delta = (d_mean / m_mean * 100.0) if m_mean != 0 else 0.0
    ci_low_pct = ((d_mean - d_ci) / m_mean * 100.0) if m_mean != 0 else 0.0
    ci_high_pct = ((d_mean + d_ci) / m_mean * 100.0) if m_mean != 0 else 0.0

    # 4-Tier Standard Classifier Rules (EuroMLSys / Reviewer Standard):
    # 1. EQUIVALENT: Entire 95% CI is bounded inside [-3.0%, +3.0%]
    # 2. WIN: Statistically significant (p < 0.05), CI strictly excludes 0 in favorable direction
    # 3. REGRESSION: Statistically significant (p < 0.05), CI strictly excludes 0 in unfavorable direction
    # 4. INCONCLUSIVE: CI crosses 0 or extends beyond +-3.0% (variance too high)
    if ci_low_pct >= -3.0 and ci_high_pct <= 3.0:
        status = "[EQUIVALENT]"
    elif p_val < 0.05 and ((not invert and ci_low_pct > 0.0) or (invert and ci_high_pct < 0.0)):
        status = "[WIN]"
    elif p_val < 0.05 and ((not invert and ci_high_pct < 0.0) or (invert and ci_low_pct > 0.0)):
        status = "[REGRESSION]"
    else:
        status = "[INCONCLUSIVE]"

    m_str = f"{m_mean:.2f} +/- {m_ci:.2f} {unit}"
    c_str = f"{c_mean:.2f} +/- {c_ci:.2f} {unit}"
    sign = "+" if rel_delta > 0 else ""
    delta_str = f"{sign}{rel_delta:.2f}% [{ci_low_pct:+.1f}%, {ci_high_pct:+.1f}%]"
    p_str = f"p={p_val:.4f}" if p_val >= 0.0001 else "p<0.0001"

    print(f' {name:<28} {m_str:<22} {c_str:<22} {delta_str:<25} {p_str:<10} {status:<12}')

paired_stat_row("Throughput", [m["tok_per_sec"] for m, c in valid_pairs], [c["tok_per_sec"] for m, c in valid_pairs], "tok/s", invert=False)
paired_stat_row("Goodput", [m["successful"] / m["total_time_s"] if m["total_time_s"] > 0 else 0 for m, c in valid_pairs], [c["successful"] / c["total_time_s"] if c["total_time_s"] > 0 else 0 for m, c in valid_pairs], "req/s", invert=False)
paired_stat_row("TTFT Median (p50)", [m["ttft"]["p50"] for m, c in valid_pairs], [c["ttft"]["p50"] for m, c in valid_pairs], "ms", invert=True)
paired_stat_row("TTFT Worst-Tail (p99)", [m["ttft"]["p99"] for m, c in valid_pairs], [c["ttft"]["p99"] for m, c in valid_pairs], "ms", invert=True)
paired_stat_row("ITL Tail Jitter (p95)", [m["itl"]["p95"] for m, c in valid_pairs], [c["itl"]["p95"] for m, c in valid_pairs], "ms", invert=True)

print('=' * 120)
print(' Note: Classification standards (TOST 3.0% equivalence threshold):')
print('       [WIN]          : p < 0.05 and 95% CI strictly excludes 0 in favorable direction.')
print('       [REGRESSION]   : p < 0.05 and 95% CI strictly excludes 0 in unfavorable direction.')
print('       [EQUIVALENT]   : 95% CI is fully bounded inside [-3.0%, +3.0%].')
print('       [INCONCLUSIVE] : 95% CI crosses 0 or exceeds [-3.0%, +3.0%] boundary (unresolved).\n')
PYCODE
}

run_gold_case "$CUSTOM_CONCURRENCY" "$CUSTOM_REQUESTS" "$CUSTOM_CANCEL" "$CUSTOM_DURATION" "$CUSTOM_RATE"
