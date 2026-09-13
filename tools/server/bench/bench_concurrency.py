#!/usr/bin/env python3
"""
Concurrency & Streaming Latency Benchmark for llama-server.
Measures TTFT (Time To First Token), ITL (Inter-Token Latency), Throughput, and Tail Jitter.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class RequestMetric:
    request_id: int
    success: bool = False
    cancelled: bool = False
    error: Optional[str] = None
    start_time: float = 0.0
    first_token_time: float = 0.0
    end_time: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    inter_token_intervals_ms: List[float] = field(default_factory=list)

    @property
    def ttft_ms(self) -> float:
        if self.first_token_time > 0 and self.start_time > 0:
            return (self.first_token_time - self.start_time) * 1000.0
        return 0.0

    @property
    def total_latency_ms(self) -> float:
        if self.end_time > 0 and self.start_time > 0:
            return (self.end_time - self.start_time) * 1000.0
        return 0.0


def calculate_percentiles(values: List[float]) -> dict:
    if not values:
        return {"count": 0, "p50": 0.0, "p95": 0.0, "p99": 0.0, "mean": 0.0, "min": 0.0, "max": 0.0}
    sorted_vals = sorted(values)
    n = len(sorted_vals)

    def get_pct(p: float) -> float:
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_vals[int(k)]
        return sorted_vals[int(f)] * (c - k) + sorted_vals[int(c)] * (k - f)

    return {
        "count": n,
        "mean": sum(sorted_vals) / n,
        "p50": get_pct(50),
        "p95": get_pct(95),
        "p99": get_pct(99),
        "min": sorted_vals[0],
        "max": sorted_vals[-1],
    }


def send_streaming_request(
    server_url: str,
    prompt: str,
    max_tokens: int,
    request_id: int,
    should_cancel: bool = False,
    cancel_after_tokens: int = 4,
) -> RequestMetric:
    metric = RequestMetric(request_id=request_id)
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

    metric.start_time = time.perf_counter()
    last_token_time = 0.0

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
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


def run_benchmark(
    server_url: str,
    concurrency: int,
    total_requests: int,
    max_tokens: int,
    prompt: str,
    cancel_rate: float,
) -> None:
    print("==================================================================")
    print("           llama.cpp Server Concurrency & Latency Benchmark        ")
    print("==================================================================")
    print(f" Target URL     : {server_url}")
    print(f" Concurrency    : {concurrency} clients")
    print(f" Total Requests : {total_requests}")
    print(f" Max Tokens     : {max_tokens}")
    print(f" Cancel Rate    : {cancel_rate * 100:.1f}%")
    print("------------------------------------------------------------------")

    # Determine exact cancellation set deterministically
    num_to_cancel = int(round(total_requests * cancel_rate))
    if num_to_cancel > 0:
        step = total_requests / num_to_cancel
        cancel_indices = {int(i * step) for i in range(num_to_cancel)}
    else:
        cancel_indices = set()

    results: List[RequestMetric] = []
    t_start = time.perf_counter()

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                send_streaming_request,
                server_url,
                prompt,
                max_tokens,
                i,
                i in cancel_indices,
                4,
            )
            for i in range(total_requests)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    t_end = time.perf_counter()
    total_time_s = t_end - t_start

    successful = [r for r in results if r.success and not r.cancelled]
    cancelled = [r for r in results if r.cancelled]
    failed = [r for r in results if not r.success and not r.cancelled]

    ttft_list = [r.ttft_ms for r in successful if r.ttft_ms > 0]
    total_latency_list = [r.total_latency_ms for r in successful if r.total_latency_ms > 0]
    all_itl: List[float] = []
    for r in successful:
        all_itl.extend(r.inter_token_intervals_ms)

    total_tokens = sum(r.completion_tokens for r in results)
    tok_per_sec = total_tokens / total_time_s if total_time_s > 0 else 0.0

    ttft_stats = calculate_percentiles(ttft_list)
    itl_stats = calculate_percentiles(all_itl)
    latency_stats = calculate_percentiles(total_latency_list)

    print("\n-------------------------- SUMMARY RESULTS --------------------------")
    print(f" Total Time Elapsed   : {total_time_s:.2f} s")
    print(f" Requests Succeeded   : {len(successful)} / {total_requests}")
    print(f" Requests Cancelled   : {len(cancelled)} / {total_requests}")
    print(f" Requests Failed      : {len(failed)} / {total_requests}")
    print(f" Total Tokens Emitted : {total_tokens}")
    print(f" Aggregate Throughput : {tok_per_sec:.2f} tokens/sec")
    print("---------------------------------------------------------------------")

    if failed:
        print("\n[!] Sample Failure Error(s):")
        for r in failed[:3]:
            print(f"  Request #{r.request_id}: {r.error}")

    print("\n### 1. TTFT (Time To First Token in ms)")
    print(f"  Mean: {ttft_stats['mean']:.2f} | p50: {ttft_stats['p50']:.2f} | p95: {ttft_stats['p95']:.2f} | p99: {ttft_stats['p99']:.2f} | Max: {ttft_stats['max']:.2f}")

    print("\n### 2. ITL (Inter-Token Latency in ms)")
    print(f"  Mean: {itl_stats['mean']:.2f} | p50: {itl_stats['p50']:.2f} | p95: {itl_stats['p95']:.2f} | p99: {itl_stats['p99']:.2f} | Max: {itl_stats['max']:.2f}")

    print("\n### 3. Request Duration (in ms)")
    print(f"  Mean: {latency_stats['mean']:.2f} | p50: {latency_stats['p50']:.2f} | p95: {latency_stats['p95']:.2f} | p99: {latency_stats['p99']:.2f}")
    print("=====================================================================\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="llama.cpp server concurrency benchmark")
    parser.add_argument("--url", type=str, default="http://localhost:8080", help="Server base URL")
    parser.add_argument("-c", "--concurrency", type=int, default=8, help="Number of concurrent clients")
    parser.add_argument("-n", "--requests", type=int, default=32, help="Total requests to submit")
    parser.add_argument("--max-tokens", type=int, default=64, help="Max tokens per completion")
    parser.add_argument("--prompt", type=str, default="Explain the theory of relativity in simple terms.", help="Prompt text")
    parser.add_argument("-k", "--cancel", "--cancel-rate", dest="cancel_rate", type=float, default=0.0, help="Probability of client aborting connection mid-generation [0.0 - 1.0]")

    args = parser.parse_args()
    run_benchmark(
        server_url=args.url,
        concurrency=args.concurrency,
        total_requests=args.requests,
        max_tokens=args.max_tokens,
        prompt=args.prompt,
        cancel_rate=args.cancel_rate,
    )


if __name__ == "__main__":
    main()
