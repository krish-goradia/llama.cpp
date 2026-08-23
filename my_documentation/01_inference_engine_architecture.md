# Inference Engine Architecture & Lifecycle

This document provides a detailed walkthrough of how requests move through the `llama.cpp` inference engine, with every stage annotated with the **specific thread responsible** for execution.

---

## 1. End-to-End Inference Flow (Annotated by Thread)

```text
[THREAD: HTTP Worker Pool (n_threads_http)]
  |-- 1. Client HTTP / SSE request arrives at endpoint (/v1/chat/completions, /completion)
  |-- 2. HTTP thread parses JSON body and tokenizes prompt via model vocabulary
  `-- 3. Wraps data in server_task and pushes into server_queue (protected by mutex_tasks)
                                     |
                                     v
[THREAD 1: Main Server Inference Loop (start_loop)]
  |-- 4. Pops server_task from queue; calls get_available_slot() to assign a server_slot
  |-- 5. pre_decode(): Iterates over all active slots, gathers ready tokens, renders llama_batch
  |-- 6. Calls llama_decode(ctx, batch) -> enters llama_context::decode()
  |-- 7. memory->init_batch(): Chops the logical batch into micro-batches (llama_ubatch <= n_ubatch)
  |-- 8. For each ubatch:
  |      |-- llama_kv_cache::prepare(): Finds & reserves cell indices for (seq_id, pos)
  |      |-- mctx->apply(): Commits cell allocation metadata to KV cache pool
  |      |-- Checks can_reuse(): Reuses existing computation graph or calls model.build_graph()
  |      |-- res->set_inputs(&ubatch): Binds token IDs, positions, and KV cell indices to graph
  |      `-- Calls queue_tasks.yield_to_queue([&] { graph_compute(gf); })
  |                 |
  |                 +---> [THREAD 2: Queue Worker (worker_loop)] (Runs concurrently during decode)
  |                 |       - Wakes up while Thread 1 is blocked on graph_compute
  |                 |       - Handles read-only /metrics, /health, /slots requests immediately
  |                 |       - Declines new prompt tasks, setting them aside in unhandled queue
  |                 |
  |                 v
  `---> Dispatches computation graph via ggml_backend_sched_graph_compute_async(sched, gf)
                                     |
                                     v
[COMPUTE WORKERS: CPU Threadpool or GPU Hardware Compute Units]
  |-- If CPU: ggml_threadpool worker threads (n_threads / n_threads_batch) execute tensor ops
  `-- If GPU: GPU hardware warps/threadgroups execute compute kernels asynchronously
                                     |
                                     v
[THREAD 1: Main Server Inference Loop (start_loop)]
  |-- 9. Graph computation finishes; yield_to_queue() ends (Thread 2 sleeps, declined tasks restored)
  |-- 10. post_decode(): Extracts output logits for tokens with logits[i] == true
  |-- 11. common_sampler_sample(): Runs temperature, top-k, top-p, min-p, penalties, grammar
  |-- 12. Appends sampled token to slot history and pushes token chunk to response stream
                                     |
                                     v
[THREAD: HTTP Worker Pool]
  `-- 13. Wakes up and streams SSE / JSON token chunks back over the socket to the client
```

---

## 2. The Three Primary Architectural Layers

### Layer 1: Server Layer (`tools/server/`)
* **Thread Execution:**
  * **HTTP Worker Pool** (`n_threads_http` in [`../tools/server/server-http.cpp`](../tools/server/server-http.cpp)): Handles socket connections, parsing, and SSE streaming.
  * **Main Server Loop Thread** (1 thread in [`../tools/server/server-queue.cpp`](../tools/server/server-queue.cpp)): Serializes task management, slot bookkeeping, and batch creation.
  * **Queue Auxiliary Worker Thread** (1 thread in [`../tools/server/server-queue.cpp`](../tools/server/server-queue.cpp)): Serves health/metrics requests while inference runs.
* **Key Files:**
  * [`../tools/server/server-queue.h`](../tools/server/server-queue.h): Task queue definition and yield synchronization primitives.
  * [`../tools/server/server-queue.cpp`](../tools/server/server-queue.cpp): Main event loop (`start_loop`) and cooperative yield worker (`worker_loop`).
  * [`../tools/server/server-context.cpp`](../tools/server/server-context.cpp): Slot tracking, multi-request batch rendering, and post-decode response handling.
  * [`../tools/server/server-http.cpp`](../tools/server/server-http.cpp): HTTP endpoint routing and `httplib::ThreadPool` setup.

### Layer 2: Engine Layer (`src/`)
* **Thread Execution:**
  * Driven **strictly by Thread 1 (Main Server Loop Thread)**.
  * Ensures lock-free execution across `llama_context`, `llama_kv_cache`, and `ggml_cgraph`.
* **Key Files:**
  * [`../src/llama-context.cpp`](../src/llama-context.cpp): Decode coordinator, ubatch loop, input tensor setup, and backend scheduler interface.
  * [`../src/llama-kv-cache.cpp`](../src/llama-kv-cache.cpp): KV cache storage, cell metadata, prefix reuse, and context shift.
  * [`../src/llama-memory.cpp`](../src/llama-memory.cpp): Memory status handling and batch-to-memory synchronization.
  * [`../src/llama-graph.cpp`](../src/llama-graph.cpp): Computation graph structures and input bindings.
  * [`../src/llama-sampler.cpp`](../src/llama-sampler.cpp): Sampler implementation.

### Layer 3: Backend Compute Layer (`ggml/`)
* **Thread Execution:**
  * **CPU Backend:** Parallelized across `n_threads` / `n_threads_batch` worker threads via `ggml_threadpool`.
  * **GPU Backends (CUDA / Metal / Vulkan):** GPU hardware warps/wavefronts execute kernels concurrently.
* **Key Files:**
  * [`../ggml/src/ggml-backend.cpp`](../ggml/src/ggml-backend.cpp): Backend scheduler and graph allocator.
  * [`../ggml/src/ggml-cpu/ggml-cpu.c`](../ggml/src/ggml-cpu/ggml-cpu.c): CPU threadpool and CPU tensor kernels.
  * `ggml/src/ggml-cuda/`, `ggml/src/ggml-metal/`, `ggml/src/ggml-vulkan/`: GPU driver bindings and compute kernels.

---

## 3. What is a Slot (`server_slot`)?

A **slot** is host-side bookkeeping for one active request, managed exclusively by **Thread 1 (Main Server Loop Thread)**. It is **not** a batch position, a GPU thread, or a single memory buffer.

```text
+-------------------------------------------------------------------+
|                           server_slot                             |
|                                                                   |
|  - id:                     Slot ID (maps to seq_id in KV cache)   |
|  - state:                  slot_state enum                        |
|  - task:                   std::unique_ptr<const server_task>     |
|  - prompt:                 server_prompt tokens processed         |
|  - generated_tokens:       llama_tokens sampled so far            |
|  - generated_text:         std::string generated text so far      |
|  - n_ctx:                  int32_t context size for this slot     |
|  - smpl:                   common_sampler_ptr (sampler pipeline)  |
|  - spec_ckpt / spec_draft: Speculative decoding checkpoints/draft |
+-------------------------------------------------------------------+
```

### Slot State Machine (Exact `slot_state` enum values)

```text
                        +------------------+
                        | SLOT_STATE_IDLE  | <-------------------------------------+
                        +--------+---------+                                       |
                                 |                                                 |
                                 | Task assigned (get_available_slot)              |
                                 v                                                 |
                        +------------------+                                       |
                        |   SLOT_STARTED   |                                       |
                        +--------+---------+                                       |
                                 |                                                 |
                 +---------------+---------------+                                 |
                 | (If waiting on parent prompt) |                                 |
                 v                               v                                 |
      +----------------------+     +-------------------------------+               |
      | SLOT_STATE_WAIT_OTHER|     |  SLOT_STATE_PROCESSING_PROMPT | <----+        |
      +----------+-----------+     +---------------+---------------+      |        |
                 |                                 |                      | Chunk  |
                 | (Parent finished)               | Prompt fully eval'd  | prompt |
                 +-------------------------------->+                      +--------+
                                                   v                               |
                                      +--------------------------+                 |
                                      |  SLOT_STATE_DONE_PROMPT  |                 |
                                      +------------+-------------+                 |
                                                   |                               |
                 +---------------------------------+-------------------------------+
                 |                                 |
                 | (Embeddings / Rerank: finish)   | (Generation: start next-token loop)
                 v                                 v
          +--------------+             +-------------------------+
          | Release Slot | <-----------+  SLOT_STATE_GENERATING  | <----+
          +--------------+   EOG / max +-------------------------+      | Next token
                                tokens/             |                   | sampled
                                cancel              +-------------------+
```

---

## 4. Multi-Slot Batching vs. Ubatching

There is a clear division of responsibility between **Server-Side Batching** and **Engine-Side Ubatching** across the execution flow:

```text
Slot 0: [Prompt tokens 0..511]
Slot 1: [Token 42] (generating)     ===>  [Thread 1] server_context_impl::pre_decode()
Slot 2: [Prompt tokens 0..255]            Builds single llama_batch (e.g., 769 tokens)
                                                    |
                                                    v
                                          [Thread 1] llama_decode(ctx, batch)
                                                    |
                                                    v
                                          [Thread 1] memory->init_batch(n_ubatch = 512)
                                                    |
               +------------------------------------+------------------------------------+
               |                                                                         |
               v                                                                         v
       llama_ubatch #1 (512 tokens)                                              llama_ubatch #2 (257 tokens)
       - [Thread 1] Prepare KV cells                                             - [Thread 1] Prepare KV cells
       - [Thread 1] Build / reuse graph                                          - [Thread 1] Build / reuse graph
       - [Compute Workers] Execute graph                                         - [Compute Workers] Execute graph
```

### Server Batching (`tools/server/server-context.cpp`)
* Executed by **Thread 1** in [`server_context_impl::pre_decode()`](../tools/server/server-context.cpp#L2838).
* Loops over active slots, gathering ready tokens from generating slots and prompt chunks from new slots.
* Renders a single logical [`llama_batch`](../include/llama.h#L256) containing arrays of token IDs, positions, sequence IDs, and logits flags.

### Engine Ubatching (`src/llama-context.cpp`)
* Executed by **Thread 1** inside [`llama_context::decode()`](../src/llama-context.cpp#L1635).
* Physical compute and memory bounds require splitting the large logical batch into micro-batches ([`llama_ubatch`](../src/llama-batch.h#L15)) bounded by `n_ubatch`.
* Each ubatch is prepared and computed sequentially.

---

## 5. KV Cache Architecture & Location Mapping

The KV cache is a preallocated memory pool containing Key and Value tensor buffers for all context positions across all layers.

```text
+---------------------------------------------------------------------------------------+
| KV Cache Tensor Storage                                                               |
|                                                                                       |
|   Layer 0:  [ Cell 0 ][ Cell 1 ][ Cell 2 ] ... [ Cell N-1 ] (K & V Tensors)           |
|   Layer 1:  [ Cell 0 ][ Cell 1 ][ Cell 2 ] ... [ Cell N-1 ] (K & V Tensors)           |
|   ...                                                                                 |
|   Layer L:  [ Cell 0 ][ Cell 1 ][ Cell 2 ] ... [ Cell N-1 ] (K & V Tensors)           |
+---------------------------------------------------------------------------------------+
| Metadata & Indexing (llama_kv_cache) - Single-Threaded Access (Thread 1)              |
|                                                                                       |
|   - v_cells:         Array of cell metadata (seq_id, pos, references)                 |
|   - v_heads:         Head pointers per stream                                         |
|   - seq_to_stream:   Maps sequence IDs to physical memory streams                     |
+---------------------------------------------------------------------------------------+
```

### Mapping `(seq_id, pos)` -> Cache Cells (Executed by Thread 1)
1. **Find Available Cells:** [`llama_kv_cache::prepare()`](../src/llama-kv-cache.cpp#L747) calls [`llama_kv_cache::find_slot()`](../src/llama-kv-cache.cpp#L894) to locate free or contiguous cell indices in `v_cells`.
2. **Apply Placement:** [`mctx->apply()`](../src/llama-context.cpp#L1326) commits the `(seq_id, pos)` pairs to the cell metadata.
3. **Graph Inputs:** [`res->set_inputs(&ubatch)`](../src/llama-context.cpp#L1380) copies the KV cell indices into graph input tensors.
4. **Recycling:** When a request completes, `slot.mem.seq_rm(slot.id, 0, -1)` releases the cell ownership. The physical tensor memory remains allocated and the cells become reusable for later requests.

---

## 6. Computation Graph Build, Reuse, and Compute

Inside [`llama_context::process_ubatch()`](../src/llama-context.cpp#L1325):

```text
[Thread 1]
+-----------------------------------------------+
| 1. Compute Graph Parameters (graph_params)    |
+-----------------------+-----------------------+
                        |
                        v
          +-------------+-------------+
          | Can reuse previous graph? |
          +------+-------------+------+
             Yes |             | No
                 |             v
                 |      +-----------------------------------------+
                 |      | 2. model.build_graph(gparams)           |
                 |      |    - Allocate graph nodes and tensors   |
                 |      | 3. ggml_backend_sched_alloc_graph()     |
                 |      +--------------------+--------------------+
                 |                           |
                 +-------------+-------------+
                               |
                               v
+-----------------------------------------------+
| 4. res->set_inputs(&ubatch)                   |
|    - Copy token IDs, positions, masks, and    |
|      KV cache cell indices to graph inputs    |
+-----------------------+-----------------------+
                        |
                        v
+-----------------------------------------------+
| 5. queue_tasks.yield_to_queue([&] {           |
|        llama_context::graph_compute();        |
|    })                                         |
+-----------------------+-----------------------+
                        |
                        | (Dispatches to Compute Layer)
                        v
[Compute Workers: CPU Threadpool or GPU Hardware Threads]
  - Execute tensor multiplication, attention, and layer operations in parallel
```

---

## 7. Sampling Pipeline

After graph execution completes:
1. **Thread 1** retrieves logits for tokens marked with `logits[i] == true`.
2. In [`tools/server/server-context.cpp:post_decode()`](../tools/server/server-context.cpp#L3713), logits are evaluated sequentially by [`common_sampler_sample()`](../tools/server/server-context.cpp#L3785) / [`llama_sampler_sample()`](../src/llama-sampler.cpp#L895):
   * Repetition, frequency, and presence penalties
   * Temperature, Top-K, Top-P, Min-P, Typical-P
   * Grammar / JSON schema constraints via PEG parser
3. The selected token is accepted via [`common_sampler_accept()`](../tools/server/server-context.cpp#L3790), appended to the slot history, and dispatched to the client response stream.
4. **HTTP Worker Thread** delivers the chunk over the open network socket to the client.
