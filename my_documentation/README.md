# llama.cpp Architecture & Inference Engine Documentation

This folder contains an in-depth, code-verified technical reference of the `llama.cpp` inference engine, request lifecycle, batching mechanisms, KV cache management, and multi-threaded concurrency architecture.

---

## Documentation Index

| Document | Topic | Thread Perspective & Description |
| :--- | :--- | :--- |
| [**01. Inference Engine Architecture**](01_inference_engine_architecture.md) | End-to-End Inference Flow | Request lifecycle, server slots, multi-slot batching, ubatching, KV cache cell placement, computation graph build/reuse, and sampling - with thread ownership annotated at every step. |
| [**02. Threading & Concurrency Model**](02_threading_and_concurrency.md) | Concurrency & Thread Breakdown | The 2-tier host threading model: HTTP worker pools (`n_threads_http`), the 2-thread server scheduler (`start_loop` + `yield_to_queue`), task queue push/drain lifecycle (`mutex_tasks`), no stop-the-world lock, and GPU worker handoff. |

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
