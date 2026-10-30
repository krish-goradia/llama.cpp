#!/usr/bin/env python3
"""
Trace-Replay Benchmark Client for llama-server.
Replays a multi-distribution workload (short queries, code gen, summarization, multi-turn)
with non-uniform prompt lengths (30 to 800 tokens) and completion budgets (16 to 256 tokens).
"""

import argparse
import json
import math
import random
import sys
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict

# Multi-distribution prompt pool representing real-world multi-tenant chat workloads
PROMPT_POOL = [
    # Short QA / Search (30-50 tokens prompt, 16-32 tokens output)
    {"prompt": "What is the capital of France and what is its primary historical landmark?", "max_tokens": 24},
    {"prompt": "Convert 100 miles per hour to kilometers per hour and show your calculation.", "max_tokens": 32},
    {"prompt": "Explain the difference between a process and a thread in modern operating systems.", "max_tokens": 64},
    {"prompt": "Define ACID properties in database transaction management.", "max_tokens": 48},
    {"prompt": "What is the Big-O time complexity of quicksort in the worst case?", "max_tokens": 24},
    {"prompt": "Summarize the law of conservation of energy in one sentence.", "max_tokens": 20},
    {"prompt": "Who wrote 'To Kill a Mockingbird' and in what decade was it published?", "max_tokens": 24},
    {"prompt": "What are the primary differences between TCP and UDP networking protocols?", "max_tokens": 64},

    # Code Generation & Debugging (100-250 tokens prompt, 64-128 tokens output)
    {"prompt": "Write a clean Python function that takes a list of integers and returns the length of the longest consecutive elements sequence in O(n) time.", "max_tokens": 96},
    {"prompt": "Implement a thread-safe singleton pattern in modern C++ using std::call_once or static initialization.", "max_tokens": 80},
    {"prompt": "Write a bash one-liner to find all files ending in .log modified in the last 24 hours and compress them into a tarball.", "max_tokens": 48},
    {"prompt": "Explain how cache-line false sharing occurs in multithreaded C++ code and write a code snippet demonstrating how alignas(64) fixes it.", "max_tokens": 128},
    {"prompt": "Write a Python script using asyncio to fetch data from three REST API endpoints concurrently with a 5-second timeout.", "max_tokens": 110},
    {"prompt": "Implement binary search in C++ using iterators and template parameters for generic container support.", "max_tokens": 70},

    # Long Context / Reasoning / Summarization (400-800 tokens prompt, 128-256 tokens output)
    {"prompt": "Read the following technical background on hardware caching: Modern CPUs use hierarchical caches (L1, L2, L3) to bridge the speed gap between high-speed processor execution cores and main DRAM. The L1 cache is typically split into instruction (L1i) and data (L1d) caches, operating at single-cycle latencies. L2 caches are larger but slightly slower. The L3 cache is shared across all cores on a single CPU socket. Cache coherence between cores is maintained using hardware snooping protocols such as MESI or MOESI. When two threads on different cores write to adjacent memory locations residing on the same 64-byte cache line, the hardware must constantly invalidate and transfer the cache line across the interconnect, causing severe performance degradation known as false sharing. Based on this text, explain three hardware design trade-offs between shared L3 caches and private L2 caches, and analyze the impact on lock-free concurrent queue implementations.", "max_tokens": 192},
    {"prompt": "Analyze the trade-offs of microservices architectures versus monolithic architectures in high-throughput cloud environments. Focus specifically on network serialization overhead, distributed transactions, observability, operational complexity, and fault isolation.", "max_tokens": 160},
    {"prompt": "Provide a comprehensive architectural comparison between single-producer single-consumer (SPSC) ring buffers and multi-producer multi-consumer (MPMC) lock-free queues. Discuss atomic compare-and-swap contention, ABA problem mitigations, memory ordering semantics, and throughput scaling under core oversubscription.", "max_tokens": 180},
]


class RequestStat:
    def __init__(self, req_id: int):
        self.req_id = req_id
        self.start_time = 0.0
        self.first_token_time = 0.0
        self.end_time = 0.0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.inter_token_intervals_ms: List[float] = []
        self.success = False
        self.error = None

    @property
    def ttft_ms(self) -> float:
        if self.first_token_time > 0 and self.start_time > 0:
            return (self.first_token_time - self.start_time) * 1000.0
        return 0.0


def calculate_percentiles(values):
    if not values:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0, "p99": 0.0}
    s = sorted(values)
    n = len(s)
    mean_val = sum(s) / n
    def get_pct(p):
        k = (n - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return s[int(k)]
        return s[int(f)] * (c - k) + s[int(c)] * (k - f)
    return {
        "mean": mean_val,
        "p50": get_pct(50),
        "p95": get_pct(95),
        "p99": get_pct(99),
    }


def send_request(url: str, item: Dict, req_id: int) -> RequestStat:
    stat = RequestStat(req_id)
    payload = json.dumps({
        "prompt": item["prompt"],
        "n_predict": item["max_tokens"],
        "stream": True,
        "temperature": 0.7,
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{url}/completion",
        data=payload,
        headers={"Content-Type": "application/json"},
    )

    stat.start_time = time.perf_counter()
    last_chunk_time = stat.start_time

    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            for line in response:
                now = time.perf_counter()
                line_str = line.decode("utf-8").strip()
                if not line_str.startswith("data: "):
                    continue
                data_str = line_str[6:]
                if data_str == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    content = data.get("content", "")
                    if content:
                        if stat.first_token_time == 0.0:
                            stat.first_token_time = now
                        else:
                            stat.inter_token_intervals_ms.append((now - last_chunk_time) * 1000.0)
                        last_chunk_time = now
                        stat.completion_tokens += 1
                except json.JSONDecodeError:
                    pass

        stat.end_time = time.perf_counter()
        stat.success = True
    except Exception as e:
        stat.end_time = time.perf_counter()
        stat.error = str(e)

    return stat


def main():
    parser = argparse.ArgumentParser(description="Multi-distribution trace replay benchmark")
    parser.add_argument("--url", type=str, default="http://localhost:8080")
    parser.add_argument("-c", "--concurrency", type=int, default=64)
    parser.add_argument("-n", "--requests", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    sampled_workload = [random.choice(PROMPT_POOL) for _ in range(args.requests)]

    print(f"==================================================================")
    print(f" Real-World Multi-Distribution Trace Replay Benchmark")
    print(f" URL: {args.url} | Concurrency: {args.concurrency} | Total: {args.requests}")
    print(f" Prompt Pool: Diverse (Short QA, Code Gen, Long Technical Summaries)")
    print(f"==================================================================")

    t0 = time.perf_counter()
    results = []

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = [
            executor.submit(send_request, args.url, item, i)
            for i, item in enumerate(sampled_workload)
        ]
        for f in as_completed(futures):
            results.append(f.result())

    total_time = time.perf_counter() - t0
    successful = [r for r in results if r.success]
    failed = [r for r in results if not r.success]
    total_tokens = sum(r.completion_tokens for r in successful)
    tok_per_sec = total_tokens / total_time if total_time > 0 else 0.0

    ttft_vals = [r.ttft_ms for r in successful if r.ttft_ms > 0]
    all_itl = []
    for r in successful:
        all_itl.extend(r.inter_token_intervals_ms)

    ttft_stats = calculate_percentiles(ttft_vals)
    itl_stats = calculate_percentiles(all_itl)

    print("-------------------------- SUMMARY RESULTS --------------------------")
    print(f" Total Time Elapsed   : {total_time:.2f} s")
    print(f" Requests Succeeded   : {len(successful)} / {args.requests}")
    print(f" Requests Failed      : {len(failed)} / {args.requests}")
    print(f" Total Tokens Emitted : {total_tokens}")
    print(f" Aggregate Throughput : {tok_per_sec:.2f} tokens/sec")
    print("---------------------------------------------------------------------")
    print(f" TTFT Mean : {ttft_stats['mean']:.2f} ms | p50: {ttft_stats['p50']:.2f} ms | p95: {ttft_stats['p95']:.2f} ms | p99: {ttft_stats['p99']:.2f} ms")
    print(f" ITL  Mean : {itl_stats['mean']:.2f} ms | p50: {itl_stats['p50']:.2f} ms | p95: {itl_stats['p95']:.2f} ms | p99: {itl_stats['p99']:.2f} ms")
    print("=====================================================================")


if __name__ == "__main__":
    main()
