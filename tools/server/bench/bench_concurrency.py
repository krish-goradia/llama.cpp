#!/usr/bin/env python3
"""
Concurrency & Streaming Latency Benchmark for llama-server.
Systems Gold-Standard (MLSys / OSDI):
- Supports Closed-Loop Concurrency and Open-Loop Poisson Arrivals (--rate)
- Supports Sustained Duration Steady-State Benchmarks (--duration >= 60s)
- Discards cold-start warmup requests to isolate pure runtime inference
- Synchronized arrival barrier (threading.Barrier) for simultaneous burst testing
- High-resolution monotonic timing (time.perf_counter)
- Strict validity: Any failed request marks the run INVALID and exits with error
- Computes full distribution metrics (mean, p50, p90, p95, p99, min, max)
"""

import argparse
import json
import math
import random
import subprocess
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List


class RequestMetric:
    def __init__(self, request_id: int, scheduled_time: float = 0.0):
        self.request_id: int = request_id
        self.success: bool = False
        self.cancelled: bool = False
        self.error: Optional[str] = None
        self.scheduled_time: float = scheduled_time
        self.start_time: float = 0.0
        self.first_token_time: float = 0.0
        self.end_time: float = 0.0
        self.prompt_tokens: int = 0
        self.completion_tokens: int = 0
        self.inter_token_intervals_ms: List[float] = []

    @property
    def ttft_ms(self) -> float:
        # Client perceived TTFT from scheduled arrival
        ref = self.scheduled_time if self.scheduled_time > 0 else self.start_time
        if self.first_token_time > 0 and ref > 0:
            return (self.first_token_time - ref) * 1000.0
        return 0.0

    @property
    def queue_delay_ms(self) -> float:
        if self.scheduled_time > 0 and self.start_time > 0:
            return max(0.0, (self.start_time - self.scheduled_time) * 1000.0)
        return 0.0

    @property
    def total_latency_ms(self) -> float:
        ref = self.scheduled_time if self.scheduled_time > 0 else self.start_time
        if self.end_time > 0 and ref > 0:
            return (self.end_time - ref) * 1000.0
        return 0.0


def calculate_percentiles(values):
    if not values:
        return {
            "count": 0,
            "mean": 0.0,
            "std": 0.0,
            "p50": 0.0,
            "p90": 0.0,
            "p95": 0.0,
            "p99": 0.0,
            "min": 0.0,
            "max": 0.0,
        }
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    mean_val = sum(sorted_vals) / float(n)
    variance = sum((x - mean_val) ** 2 for x in sorted_vals) / float(n) if n > 1 else 0.0
    std_val = math.sqrt(variance)

    def get_pct(p):
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)

    return {
        "count": n,
        "mean": mean_val,
        "std": std_val,
        "p50": get_pct(50),
        "p90": get_pct(90),
        "p95": get_pct(95),
        "p99": get_pct(99),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
    }


def send_streaming_request(
    server_url,
    prompt,
    max_tokens,
    request_id,
    should_cancel=False,
    cancel_after_tokens=4,
    start_barrier=None,
    scheduled_time=0.0,
):
    metric = RequestMetric(request_id=request_id, scheduled_time=scheduled_time)
    url = f"{server_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "stream": True,
        "temperature": 0.7,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    # Arrival synchronization for closed-loop burst testing
    if start_barrier is not None:
        try:
            start_barrier.wait(timeout=10.0)
        except Exception:
            pass

    metric.start_time = time.perf_counter()
    if metric.scheduled_time <= 0:
        metric.scheduled_time = metric.start_time

    last_token_time = 0.0

    try:
        with urllib.request.urlopen(req, timeout=180) as response:
            for line in response:
                line_str = line.decode("utf-8").strip()
                if not line_str or not line_str.startswith("data:"):
                    continue
                data_str = line_str[len("data:"):].strip()
                if data_str == "[DONE]":
                    break

                now = time.perf_counter()
                if metric.completion_tokens == 0:
                    metric.first_token_time = now
                else:
                    metric.inter_token_intervals_ms.append((now - last_token_time) * 1000.0)
                last_token_time = now
                metric.completion_tokens += 1

                if should_cancel and metric.completion_tokens >= cancel_after_tokens:
                    metric.cancelled = True
                    break

            metric.end_time = time.perf_counter()
            metric.success = True
    except Exception as e:
        metric.end_time = time.perf_counter()
        if not metric.cancelled:
            metric.error = str(e)

    return metric


def run_warmup(server_url, warmup_requests=2):
    """Executes discarded warmup requests to prime CUDA context and KV-cache."""
    if warmup_requests <= 0:
        return
    print(f" [Warmup] Sending {warmup_requests} discarded request(s) to prime CUDA context...")
    for i in range(warmup_requests):
        try:
            metric = send_streaming_request(
                server_url=server_url,
                prompt="Hello",
                max_tokens=4,
                request_id=-100 - i,
            )
            if not metric.success:
                print(f"  [!] Warmup warning: request {i+1} returned error: {metric.error}")
        except Exception as e:
            print(f"  [!] Warmup warning: {e}")
    time.sleep(0.5)
    print(" [Warmup] Complete. Starting timed benchmark.\n")


def get_gpu_diagnostics():
    """Captures GPU temperature and power if nvidia-smi is available."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu", "--format=csv,noheader,nounits"],
            stderr=subprocess.DEVNULL,
        ).decode("utf-8").strip()
        parts = [p.strip() for p in out.split(",")]
        return {
            "temp_c": float(parts[0]) if len(parts) > 0 else None,
            "power_w": float(parts[1]) if len(parts) > 1 else None,
            "util_pct": float(parts[2]) if len(parts) > 2 else None,
        }
    except Exception:
        return None


def run_benchmark(
    server_url,
    concurrency,
    total_requests,
    duration,
    arrival_rate,
    max_tokens,
    prompt,
    cancel_rate,
    warmup_requests=2,
    json_out=None,
    seed=None,
):
    is_open_loop = arrival_rate > 0.0
    is_duration_based = duration > 0.0

    mode_str = f"Open-Loop Poisson ({arrival_rate:.1f} req/s)" if is_open_loop else f"Closed-Loop ({concurrency} concurrent clients)"
    load_str = f"Sustained {duration:.1f}s" if is_duration_based else f"{total_requests} Requests"

    print("==================================================================")
    print("           llama.cpp Systems Gold-Standard Benchmark              ")
    print("==================================================================")
    print(f" Target URL     : {server_url}")
    print(f" Workload Mode  : {mode_str}")
    print(f" Duration/Scale : {load_str}")
    print(f" Max Tokens     : {max_tokens}")
    print(f" Cancel Rate    : {cancel_rate * 100:.1f}%")
    print(f" Warmup Requests: {warmup_requests} (discarded)")
    if seed is not None:
        print(f" Random Seed    : {seed} (Common Random Numbers)")
    print("------------------------------------------------------------------")

    # Step 0: Prime server
    run_warmup(server_url, warmup_requests=warmup_requests)

    # Initial hardware telemetry
    gpu_pre = get_gpu_diagnostics()

    results = []
    t_start = time.perf_counter()

    # Deterministic RNG for Common Random Numbers (CRN)
    prng = random.Random(seed) if seed is not None else random.Random()

    if is_open_loop:
        # ----------------------------------------------------------------------
        # Open-Loop Poisson Arrival Mode
        # ----------------------------------------------------------------------
        max_workers = max(concurrency * 4, 256)
        futures = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            req_idx = 0
            cur_scheduled = t_start
            stop_time = t_start + duration if is_duration_based else None

            while True:
                now = time.perf_counter()
                if stop_time and now >= stop_time:
                    break
                if not stop_time and req_idx >= total_requests:
                    break

                # Sample exponential inter-arrival interval
                interval = prng.expovariate(arrival_rate)
                cur_scheduled += interval

                delay = cur_scheduled - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)

                should_cancel = (prng.random() < cancel_rate)
                futures.append(
                    executor.submit(
                        send_streaming_request,
                        server_url,
                        prompt,
                        max_tokens,
                        req_idx,
                        should_cancel,
                        4,
                        None,
                        cur_scheduled,
                    )
                )
                req_idx += 1

            for f in as_completed(futures):
                results.append(f.result())

    elif is_duration_based:
        # ----------------------------------------------------------------------
        # Sustained Duration Closed-Loop Mode
        # ----------------------------------------------------------------------
        stop_event = threading.Event()
        counter_lock = threading.Lock()
        shared_counter = [0]

        def worker_loop(client_id):
            client_prng = random.Random(seed + client_id if seed is not None else None)
            while not stop_event.is_set():
                if time.perf_counter() - t_start >= duration:
                    stop_event.set()
                    break
                with counter_lock:
                    rid = shared_counter[0]
                    shared_counter[0] += 1
                should_cancel = (client_prng.random() < cancel_rate)
                m = send_streaming_request(
                    server_url=server_url,
                    prompt=prompt,
                    max_tokens=max_tokens,
                    request_id=rid,
                    should_cancel=should_cancel,
                    cancel_after_tokens=4,
                    start_barrier=None,
                )
                results.append(m)

        threads = [threading.Thread(target=worker_loop, args=(i,)) for i in range(concurrency)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    else:
        # ----------------------------------------------------------------------
        # Fixed Request Count Closed-Loop Mode with Arrival Barrier
        # ----------------------------------------------------------------------
        barrier_size = min(concurrency, total_requests)
        barrier = threading.Barrier(barrier_size)
        num_to_cancel = int(round(total_requests * cancel_rate))
        if num_to_cancel > 0:
            step = total_requests / float(num_to_cancel)
            cancel_indices = {int(i * step) for i in range(num_to_cancel)}
        else:
            cancel_indices = set()

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = []
            for i in range(total_requests):
                use_barrier = barrier if i < barrier_size else None
                futures.append(
                    executor.submit(
                        send_streaming_request,
                        server_url,
                        prompt,
                        max_tokens,
                        i,
                        i in cancel_indices,
                        4,
                        use_barrier,
                    )
                )
            for f in as_completed(futures):
                results.append(f.result())

    t_end = time.perf_counter()
    total_time_s = t_end - t_start
    total_dispatched = len(results)

    # Post hardware telemetry
    gpu_post = get_gpu_diagnostics()

    successful = [r for r in results if r.success and not r.cancelled]
    cancelled = [r for r in results if r.cancelled]
    failed = [r for r in results if not r.success and not r.cancelled]

    ttft_list = [r.ttft_ms for r in successful if r.ttft_ms > 0]
    total_latency_list = [r.total_latency_ms for r in successful if r.total_latency_ms > 0]
    all_itl = []
    for r in successful:
        all_itl.extend(r.inter_token_intervals_ms)

    total_tokens = sum(r.completion_tokens for r in results)
    tok_per_sec = total_tokens / total_time_s if total_time_s > 0 else 0.0

    ttft_stats = calculate_percentiles(ttft_list)
    itl_stats = calculate_percentiles(all_itl)
    latency_stats = calculate_percentiles(total_latency_list)

    print("-------------------------- SUMMARY RESULTS --------------------------")
    print(f" Total Time Elapsed   : {total_time_s:.2f} s")
    print(f" Requests Succeeded   : {len(successful)} / {total_dispatched}")
    print(f" Requests Cancelled   : {len(cancelled)} / {total_dispatched}")
    print(f" Requests Failed      : {len(failed)} / {total_dispatched}")
    print(f" Total Tokens Emitted : {total_tokens}")
    print(f" Aggregate Throughput : {tok_per_sec:.2f} tokens/sec")
    if gpu_pre and gpu_post:
        print(f" GPU Telemetry        : Start {gpu_pre.get('temp_c')}C ({gpu_pre.get('power_w')}W) -> End {gpu_post.get('temp_c')}C ({gpu_post.get('power_w')}W)")
    print("---------------------------------------------------------------------")

    # Fail loudly if any requests failed
    is_valid_run = (len(failed) == 0 and total_dispatched > 0)
    if not is_valid_run:
        print("\n\033[1m\033[31m[!] LOUD FAILURE: Benchmark run marked INVALID!\033[0m")
        print(f"    {len(failed)} request(s) failed during execution.")
        for r in failed[:5]:
            print(f"    - Request #{r.request_id} error: {r.error}")

    print("\n### 1. TTFT (Time To First Token in ms)")
    print(f"  Mean: {ttft_stats['mean']:.2f} (±{ttft_stats['std']:.2f}) | p50: {ttft_stats['p50']:.2f} | p95: {ttft_stats['p95']:.2f} | p99: {ttft_stats['p99']:.2f} | Max: {ttft_stats['max']:.2f}")

    print("\n### 2. ITL (Inter-Token Latency in ms)")
    print(f"  Mean: {itl_stats['mean']:.2f} (±{itl_stats['std']:.2f}) | p50: {itl_stats['p50']:.2f} | p95: {itl_stats['p95']:.2f} | p99: {itl_stats['p99']:.2f} | Max: {itl_stats['max']:.2f}")

    print("\n### 3. Request Duration (in ms)")
    print(f"  Mean: {latency_stats['mean']:.2f} (±{latency_stats['std']:.2f}) | p50: {latency_stats['p50']:.2f} | p95: {latency_stats['p95']:.2f} | p99: {latency_stats['p99']:.2f}")
    print("=====================================================================\n")

    if json_out:
        data = {
            "valid": is_valid_run,
            "total_time_s": total_time_s,
            "total_requests": total_dispatched,
            "successful": len(successful),
            "cancelled": len(cancelled),
            "failed": len(failed),
            "total_tokens": total_tokens,
            "tok_per_sec": tok_per_sec,
            "ttft": ttft_stats,
            "itl": itl_stats,
            "duration": latency_stats,
            "gpu_telemetry": {
                "pre": gpu_pre,
                "post": gpu_post,
            },
        }
        with open(json_out, "w") as f:
            json.dump(data, f, indent=2)

    if not is_valid_run:
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="llama.cpp server concurrency and latency benchmark")
    parser.add_argument("--url", "--base-url", dest="url", type=str, default="http://localhost:8080", help="Server base URL")
    parser.add_argument("-c", "--concurrency", type=int, default=8, help="Number of concurrent clients (closed loop)")
    parser.add_argument("-n", "--requests", "--num-requests", dest="requests", type=int, default=32, help="Total requests to submit (when duration=0)")
    parser.add_argument("-d", "--duration", type=float, default=0.0, help="Sustained test duration in seconds (0 = fixed request count)")
    parser.add_argument("-r", "--rate", type=float, default=0.0, help="Target request arrival rate (Poisson open-loop lambda in req/s)")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per completion")
    parser.add_argument("--prompt", type=str, default="Explain the theory of relativity in simple terms.", help="Prompt text")
    parser.add_argument("-k", "--cancel", "--cancel-rate", dest="cancel_rate", type=float, default=0.0, help="Probability of client aborting connection mid-generation [0.0 - 1.0]")
    parser.add_argument("--warmup", type=int, default=2, help="Number of discarded warmup requests to prime CUDA context")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for Common Random Numbers (CRN)")
    parser.add_argument("--json-out", type=str, default=None, help="Path to write JSON benchmark metrics")

    args = parser.parse_args()
    run_benchmark(
        server_url=args.url,
        concurrency=args.concurrency,
        total_requests=args.requests,
        duration=args.duration,
        arrival_rate=args.rate,
        max_tokens=args.max_tokens,
        prompt=args.prompt,
        cancel_rate=args.cancel_rate,
        warmup_requests=args.warmup,
        json_out=args.json_out,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
