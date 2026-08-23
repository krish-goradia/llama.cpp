# Threading & Concurrency Architecture

This document details the multi-threaded execution model of `llama.cpp` up to the point where batches and computation graphs are submitted to the compute/GPU workers.

---

## 1. Thread Hierarchy Overview

```text
+-------------------------------------------------------------------------------+
| TIER 1: HTTP & Ingestion Layer (Concurrent Pool)                              |
|   n_threads_http (~n_parallel + 4 up to CPU cores)                            |
|   - Listens for connections, parses HTTP/JSON, tokenizes inputs               |
|   - Pushes server_task objects to thread-safe server_queue                    |
+---------------------------------------+---------------------------------------+
                                        | (std::mutex mutex_tasks + condition_variable)
                                        v
+-------------------------------------------------------------------------------+
| TIER 2: Server Scheduler & Context Control (Exactly 2 Threads)                |
|                                                                               |
|   1. Main Inference Loop Thread (server_queue::start_loop)                    |
|      - Pops tasks, manages slots, builds batch, orchestrates decode           |
|      - Maps KV-cache locations, builds/reuses computation graph               |
|      - Dispatches graph asynchronously to compute/GPU workers                 |
|                                                                               |
|   2. Queue Auxiliary Worker Thread (server_queue::worker_loop)                |
|      - Active during yield_to_queue() while Main Thread submits / waits       |
|      - Handles /metrics, /health, and cancellations without blocking          |
|      - Declines new prompt tasks until the active submission completes        |
+---------------------------------------+---------------------------------------+
                                        |
                                        | (ggml_backend_sched_graph_compute_async)
                                        v
                            [ HANDOFF TO GPU WORKERS ]
```

---

## 2. Tier 1: Ingestion & HTTP Worker Pool

* **Thread Count:** `n_threads_http` (Configured via `--threads-http`).
* **Source:** [`../tools/server/server-http.cpp`](../tools/server/server-http.cpp#L309-L321).
* **Default Setting:**
  ```cpp
  n_threads_http = std::max(params.n_parallel + 4, static_cast<int32_t>(std::thread::hardware_concurrency() - 1));
  ```

### Lifecycle:
1. Each HTTP request (e.g. `/v1/chat/completions`, `/completion`, `/infill`, `/embeddings`) is handled by a worker thread from `httplib::ThreadPool`.
2. The HTTP worker thread parses the JSON payload, tokenizes the prompt using the model vocabulary, wraps the payload in a [`server_task`](../tools/server/server-queue.h), and posts it to `server_queue`.
3. The HTTP worker thread then blocks on a condition variable or waits on an SSE stream sink until output tokens are generated.

---

## 3. Tier 2: Server Scheduler & Context Control (The 2-Thread Model)

The server scheduling and context preparation path relies on **exactly 2 threads**:

```text
               +-------------------------------------------------+
               |              server_queue Coordinator           |
               +-----------------------+-------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
+------------------------------------+   +------------------------------------+
| Thread 1: Main Inference Loop      |   | Thread 2: Queue Auxiliary Worker   |
| (server_queue::start_loop)         |   | (server_queue::worker_loop)        |
|                                    |   |                                    |
| - Pops tasks from queue            |   | - Wakes up only during             |
| - Updates server_slot bookkeeping  |   |   yield_to_queue()                 |
| - Prepares llama_batch & KV cache  |   | - Handles /metrics, /health,       |
| - Builds / reuses computation graph|   |   and /slots without blocking      |
| - Submits graph to GPU workers     |   | - Declines state-modifying tasks   |
+------------------------------------+   +------------------------------------+
```

### Why Thread #1 (Main Inference Loop) is Serialized
* **Single-Owner Memory Model:** [`llama_context`](../src/llama-context.h), [`llama_kv_cache`](../src/llama-kv-cache.h), and computation graph structures are stateful. Concurrently mutating slots, assigning KV-cache cells, or building graphs across multiple threads would require coarse locking, degrading throughput.
* **Continuous Batching Saturation:** A single thread with a global view across all active slots can inspect ready tokens from every request and combine them into one optimal [`llama_batch`](../include/llama.h).

---

### Why Thread #2 (Queue Auxiliary Worker) Exists: `yield_to_queue()`

A decode submission and synchronization step (e.g. processing a large prompt) can block Thread #1 for tens to hundreds of milliseconds. 

If there were **only 1 thread**, incoming monitoring tasks (like Prometheus scraping `/metrics` or Kubernetes probing `/health`) would be completely blocked until the inference step finished.

To solve this, llama.cpp implements cooperative yielding:

```text
Thread 1 (Main Loop)                            Thread 2 (Queue Worker)
====================                            =======================
yield_to_queue([&] {
    llama_decode(ctx, batch);
})
  |
  |--- 1. Set worker.busy = true, worker.yielding = true
  |--- 2. worker.cv.notify_one() --------------> Wakes up in worker_loop()
  |                                                |
  |--- 3. Executes llama_decode() on Thread 1      |--- 4. Pulls tasks with is_yielding=true
  |       - Chops batch into ubatches              |       - /metrics  ==> Handled immediately!
  |       - Prepares KV cache cell indices         |       - /health   ==> Handled immediately!
  |       - Sets graph inputs                      |       - New prompt==> Declined (set aside)
  |       - Submits graph to GPU workers           |
  |                                                |
  |--- 5. Submission / sync completes              |
  |--- 6. Set worker.yielding = false              |
  |--- 7. Wait for worker to finish task <--------|--- 8. Sets worker.busy = false, sleeps
  |
Restores declined tasks to front of queue
Continues next scheduling iteration
```

### Task Filtering During Yield (`is_yielding`)

In [`tools/server/server-context.cpp:process_single_task()`](../tools/server/server-context.cpp#L2304):

```cpp
bool process_single_task(server_task && task, bool is_yielding) {
    // while yielding, decode is in progress; only read-only server state is safe
    if (is_yielding && task.type != SERVER_TASK_TYPE_METRICS && task.type != SERVER_TASK_TYPE_SLOT_GET) {
        return false; // Decline task; put into queue_tasks_unhandled
    }
    // ...
}
```

* **Read-only tasks** (`SERVER_TASK_TYPE_METRICS`, `SERVER_TASK_TYPE_SLOT_GET`) are executed immediately by Thread #2.
* **State-modifying tasks** (new prompts, generation requests) return `false`, allowing Thread #2 to set them aside in `queue_tasks_unhandled`.
* When Thread #1 finishes, it restores declined tasks to the front of `queue_tasks` in their original order.

---

## 4. Task Queue Synchronization & Lock Mechanics (No Stop-The-World Lock)

There is **no global "stop-the-world" lock** in `llama.cpp`. Network ingestion and task enqueueing remain active and non-blocking at all times.

### Step-by-Step Queue Push & Drain Flow

```text
HTTP Threads (Producers)                          Thread 1: Main Loop (Consumer)
────────────────────────                          ──────────────────────────────
Incoming Request
  │
  ├─► Locks mutex_tasks (<1 µs)
  ├─► queue_tasks.push_back(task)
  ├─► condition_tasks.notify_one()
  └─► Releases mutex_tasks
                                                  1. Calls process_new_tasks(false):
                                                     ┌────────────────────────────────────────┐
                                                     │ Loops until queue is EMPTY:            │
                                                     │  - Locks mutex_tasks (<1 µs)           │
                                                     │  - task = queue_tasks.pop_front()      │
                                                     │  - Unlocks mutex_tasks                 │
                                                     │  - Assigns task to available slot      │
                                                     └────────────────────────────────────────┘
                                                  2. All ready tasks are now in slots!

                                                  3. update_slots() & llama_decode():
New tasks arrive here! ────┐                         - Batches all slots together
  │                        │                         - Submits graph to GPU workers
  ├─► Locks mutex_tasks    │
  ├─► Pushed to queue      │                         (mutex_tasks is free, new tasks
  └─► Releases mutex_tasks │                          safely accumulate in the queue)
                           ▼
                    [ queue_tasks ]
                 (Accumulating in queue)
                                                  4. Decode finishes & tokens sampled

                                                  5. Loop repeats!
                                                     Thread 1 loops back to step 1 and
                                                     drains all newly arrived tasks.
```

### Code Implementation

1. **Pushing into the Queue ([`server_queue::post`](../tools/server/server-queue.cpp#L28)):**
   ```cpp
   int server_queue::post(server_task && task, bool front) {
       std::unique_lock<std::mutex> lock(mutex_tasks); // Acquire lock for < 1 µs
       if (front) {
           queue_tasks.push_front(std::move(task));
       } else {
           queue_tasks.push_back(std::move(task));
       }
       condition_tasks.notify_one();
       return task_id;
   } // Lock is immediately released here
   ```

2. **Draining the Queue ([`server_queue::process_new_tasks`](../tools/server/server-queue.cpp#L138)):**
   ```cpp
   bool server_queue::process_new_tasks(bool is_yielding) {
       while (true) {
           std::unique_lock<std::mutex> lock(mutex_tasks);
           if (queue_tasks.empty()) {
               return false; // Queue is fully drained
           }
           server_task task = std::move(queue_tasks.front());
           queue_tasks.pop_front();
           lock.unlock(); // UNLOCK before doing slot assignment work!

           callback_new_task(std::move(task), is_yielding);
       }
   }
   ```

### Task Accumulation During Inference
While Thread 1 is preparing batches, building graphs, or waiting on GPU computation:
* `mutex_tasks` is **unlocked**.
* HTTP worker threads continue to parse incoming requests and push new `server_task` items into `queue_tasks` with microsecond lock duration.
* These tasks accumulate safely in `queue_tasks` until Thread 1 completes the current iteration and drains them at the start of the next loop.

### Lock Scope Summary

| Lock | Scope | Duration | Impact |
| :--- | :--- | :--- | :--- |
| `mutex_tasks` | Task Queue operations (`post`, `pop`) | Microseconds ($\mu s$) | Protects deque insertion and removal only. |
| `mutex_results` | Response delivery to HTTP threads | Microseconds ($\mu s$) | Protects per-task response queues. |
| **Global Stop-the-World Lock** | **None** | **N/A** | **Does not exist.** |

---

## 5. Handoff to GPU / Compute Workers

Once the ubatch and KV-cache cell locations are assigned, Thread #1 dispatches the computation graph to the backend scheduler:

```cpp
// src/llama-context.cpp: graph_compute()
auto status = ggml_backend_sched_graph_compute_async(sched.get(), gf);
```

At this boundary:
* Thread #1 hands off the graph (`gf`) containing node tensors and assigned KV indices to the backend scheduler.
* Backend drivers (e.g. CUDA / Metal / Vulkan command queues) receive the task asynchronously, concluding the host-side scheduling and preparation phase.

---

## 6. Threading & Batching Parameters Reference

| CLI Flag | Parameter | Default | Purpose |
| :--- | :--- | :--- | :--- |
| `--threads-http` | `params.n_threads_http` | `max(n_parallel + 4, cores - 1)` | Number of worker threads for handling incoming HTTP network requests. |
| `-np`, `--parallel` | `params.n_parallel` | `1` | Number of simultaneous request slots supported by the server. |
| `-b`, `--batch-size` | `cparams.n_batch` | `2048` | Maximum total logical tokens combined across slots in one iteration. |
| `-ub`, `--ubatch-size` | `cparams.n_ubatch` | `512` | Maximum physical micro-batch tokens submitted to the compute backend at once. |
