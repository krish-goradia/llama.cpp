# llama.cpp Architecture & Inference Engine Documentation

This folder contains an in-depth, code-verified technical reference of the `llama.cpp` inference engine, request lifecycle, batching mechanisms, KV cache management, and multi-threaded concurrency architecture.

---

## Documentation Index

| Document | Topic | Thread Perspective & Description |
| :--- | :--- | :--- |
| [**01. Threading & Concurrency Model**](01_threading_and_concurrency.md) | Concurrency & Thread Breakdown | The 2-tier host threading model: HTTP worker pools (`n_threads_http`), the 2-thread server scheduler (`start_loop` + `yield_to_queue`), task queue push/drain lifecycle (`mutex_tasks`), no stop-the-world lock, and GPU worker handoff. |
| [**02. Inference Engine Architecture**](02_inference_engine_architecture.md) | End-to-End Inference Flow | Request lifecycle, server slots, multi-slot batching, ubatching, KV cache cell placement, computation graph build/reuse, and sampling - with thread ownership annotated at every step. |
| [**03. KV Cache Lifecycle & Cleanup**](03_kv_cache_lifecycle_and_cleanup.md) | Memory Ownership, Deallocation & Fragmentation | KV cell ownership (slot vs. ubatch), metadata cleanup (`seq_rm()`), lifecycle trigger points, turnaround to queued tasks, and fragmentation handling (indexed indirect attention & on-demand compaction). |
| [**04. Server Queue Architecture**](04_server_queue_architecture.md) | Forward & Response Pass Queues | The full-duplex dual-queue architecture (`server_queue` vs `server_response`), internal sub-queues (`queue_tasks`, `queue_tasks_deferred`, `queue_tasks_unhandled`, `queue_results`), forward task submission, response streaming, and cooperative yielding. |
| [**05. Future Improvements & Bottlenecks**](05_future_improvements_and_bottlenecks.md) | Optimization Blueprint & Deep Audit | Exhaustive audit of latency bottlenecks across the server: lock-free MPSC ingestion, SPSC response ring buffers, eliminating self-inflicted `NEXT_RESPONSE` lock loops, offloading detokenization/top-k from Main Thread, Radix Tree prefix caching, and template-based SSE serialization. |
| [**Bottleneck 1. Loop & Scheduling Bottlenecks**](BottleNeck1_LoopAndSchedulingBottlenecks.md) | Event Loop & Yield Inefficiencies | Exhaustive breakdown of self-inflicted `NEXT_RESPONSE` lock loops (acquiring `mutex_tasks` 4 times per token) and synchronous thread-bouncing/context-switching in `yield_to_queue`. |
| [**Bottleneck 2. Main Thread Contamination**](BottleNeck2_MainThreadContamination.md) | Critical Path Offloading | Detailed breakdown of Main Thread contamination: synchronous detokenization (`common_token_to_piece`), top-k probability sorting, and dynamic heap allocations on the critical GPU scheduling path. |
| [**Bottleneck 3. Prefix Caching & Slot Selection**](BottleNeck3_PrefixCachingAndSlotSelection.md) | O(S x L) Prefix Scans & Trie Indexing | Analysis of naive token-by-token loops in `get_common_prefix()` across active VRAM slots and RAM prompt caches, and the Radix Tree (Prefix Trie) solution. |
| [**Phase 1. Lock-Free MPSC Ingestion & Cancellation**](Phase1_LockFree_MPSC_Ingestion_Tier.md) | Phase 1 Implementation Reference | Complete design, implementation details, and architecture comparison against the original mutex baseline. |
| [**Phase 1. Benchmark Results & Matrices**](Phase1_Benchmark_Results.md) | Phase 1 Benchmark Matrices | Live benchmarks across 8-client and 16-client loads (0% & 20% cancel), raw C++ stress test outputs (16M-30M ops/s), and latency analysis. |
| [**Phase 2. Lock-Free SPSC Response Tier**](Phase2_LockFree_SPSC_Response_Tier.md) | Phase 2 Implementation Reference | Cacheline-padded (alignas(64)) SPSC ring buffers, dedicated response channels, point-to-point cv.notify_one() signaling, and 0-copy move semantics. |
| [**Phase 2. 64-Client A/B Benchmark Results**](Phase2_AB_Benchmark_Results.md) | High-Concurrency A/B Matrices & Systems Analysis | 64-client and 16-client A/B benchmarks comparing master vs lock-free, raw outputs, +16.2% throughput gain, -24.3% p95 ITL reduction, and wakeup storm elimination. |
| [**Phase 3. Scheduler Loop Streamlining & Queue Modernization**](Phase3_Scheduler_Loop_Streamlining.md) | Phase 3 Implementation Reference | Direct in-memory active slot cycling without dummy NEXT_RESPONSE self-messaging, FIFO std::queue modernization, and 3-tier priority task scheduling. |
| [**Legacy. Mid-Process Cancellation Mechanism**](Legacy_Mid_Process_Cancellation_Mechanism.md) | Legacy Cancellation Architecture Reference | Deep dive into how legacy mutex code handled disconnects: polling timeouts, cleanup_pending_task O(N) queue scanning, SERVER_TASK_TYPE_CANCEL dispatch, and slot eviction. |
| [**Reference. Questions & Answers**](questions_answers.md) | Architectural Q&A Reference | Detailed breakdowns of lock-holding during GPU compute, queue_tasks_unhandled liveloop prevention, and queue_tasks_deferred waiting room semantics. |

---

## High-Level Architecture Overview (by Thread Ownership)

```text
+-----------------------------------------------------------------------------------------------+
| 1. HTTP / INGESTION LAYER                                                                     |
|    Threads: n_threads_http (httplib::ThreadPool)                                              |
|    - Accepts socket connections, parses JSON requests                                         |
|    - Tokenizes prompts using model vocabulary                                                 |
|    - Pushes server_task into server_queue (acquires mutex_tasks for < 1 µs)                   |
+-----------------------------------------------+-----------------------------------------------+
                                                | (Thread-safe queue with mutex_tasks)
                                                v
+-----------------------------------------------------------------------------------------------+
| 2. SCHEDULER & ENGINE CONTROL LAYER (Exactly 2 Threads)                                       |
|                                                                                               |
|    A. Thread 1: Main Inference Loop (server_queue::start_loop)                                |
|       - Drains queue: pops all pending tasks into server_slots                                |
|       - update_slots(): gathers tokens across active slots into llama_batch                   |
|       - llama_decode(): chops batch into ubatches (<= n_ubatch)                               |
|       - KV-Cache manager: finds free cells & updates cell metadata                            |
|       - Graph Builder: builds or reuses computation graph                                     |
|       - Dispatches graph asynchronously via ggml_backend_sched_graph_compute_async()          |
|                                                                                               |
|    B. Thread 2: Queue Auxiliary Worker (server_queue::worker_loop)                            |
|       - Active during yield_to_queue() while Thread 1 waits on decode/GPU                     |
|       - Serves read-only tasks (/metrics, /health, /slots) without blocking                   |
|       - Declines state-modifying tasks (new prompts) until decode finishes                    |
+-----------------------------------------------+-----------------------------------------------+
                                                |
                                                | (Asynchronous Graph Submission)
                                                v
                                    [ HANDOFF TO GPU WORKERS ]
```

---

## Key Source Files Reference

* **Server Scheduling & Queues:**
  * [`../tools/server/server-queue.h`](../tools/server/server-queue.h): Task queue definition, worker thread synchronization.
  * [`../tools/server/server-queue.cpp`](../tools/server/server-queue.cpp): Main server loop, cooperative yield (`yield_to_queue`), task drain.
  * [`../tools/server/server-context.cpp`](../tools/server/server-context.cpp): Slot tracking, multi-request batch rendering, post-decode response handling.
  * [`../tools/server/server-http.cpp`](../tools/server/server-http.cpp): HTTP endpoint routing and worker threadpool setup.
* **Core Engine:**
  * [`../src/llama-context.cpp`](../src/llama-context.cpp): Orchestrates ubatch decoding, graph building, input tensor setup, backend compute dispatch.
  * [`../src/llama-kv-cache.cpp`](../src/llama-kv-cache.cpp): Cell allocation, sequence-to-stream mapping, prefix caching, position tracking.
  * [`../src/llama-batch.cpp`](../src/llama-batch.cpp): Batch memory allocator and micro-batch partitioning.
  * [`../src/llama-sampler.cpp`](../src/llama-sampler.cpp): Top-K, Top-P, Min-P, temperature, repetition penalties, and grammar sampling.
