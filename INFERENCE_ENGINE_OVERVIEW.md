# Inference Engine Overview

## Overall Flow

```text
client request
  -> server task queue
  -> server scheduler
  -> active request slots
  -> shared batch
  -> llama_decode()
  -> KV-cache preparation
  -> graph construction or reuse
  -> CPU/GPU backend
  -> logits
  -> sampling
  -> slot update
  -> next iteration
```

## Main Layers

### `tools/server/`

The server layer accepts requests, queues tasks, manages active request slots, groups work into batches, and streams responses back to clients.

Important files:

- `tools/server/server-queue.cpp`: task queue and main server loop.
- `tools/server/server-queue.h`: queue interface and callbacks.
- `tools/server/server-context.cpp`: request slots, scheduling, batching, and response handling.

### `src/`

The llama.cpp inference-engine layer performs decoding, memory preparation, KV-cache management, graph construction, and sampling.

Important files:

- `src/llama-context.cpp`: decode orchestration.
- `src/llama-memory.cpp`: general memory interface and status handling.
- `src/llama-kv-cache.cpp`: KV-cache storage, cell metadata, placement, and reuse.
- `src/llama-graph.cpp`: computation-graph inputs and graph-related structures.
- `src/llama-sampler.cpp`: token-sampling implementation.

### `ggml/`

The lower-level tensor library executes graph operations on the CPU or GPU.

## Request Queue and Scheduler

A client request enters the queue as a `server_task`. It contains the prompt and request settings. HTTP or request threads can enqueue tasks, but they do not independently run inference for the same model context.

The main scheduling path is:

```text
server_queue::start_loop()
  -> server_queue::process_new_tasks()
  -> server_context_impl::process_single_task()
  -> server_context_impl::update_slots()
```

`server_queue::start_loop()` processes queued tasks and then calls the update-slots callback. `server_context_impl::update_slots()` examines all active slots and creates the work for the next inference step.

## What Is a Slot?

A slot is persistent bookkeeping for one active request. It is not a batch position, a GPU thread, or a KV-cache memory block.

A slot tracks information such as:

```text
request/task state
prompt progress
generated tokens
current token position
sequence ID
KV-cache sequence ownership
sampler state
generation status
```

The intended lifecycle is:

```text
empty
  -> processing prompt
  -> generating
  -> finished
  -> released and reused
```

A useful interpretation is:

> A slot answers: "What is happening with this request, and what does it need next?"

## Batching

The scheduler checks active slots and gathers the work they need. It combines that work into one shared batch.

For example:

```text
slot 0: needs prompt tokens
slot 1: needs one generation token
slot 2: needs prompt tokens

shared batch = slot 0 work + slot 1 work + slot 2 work
```

The slot itself is not sent to the GPU. The scheduler creates batch items containing metadata such as:

```text
token ID
position
sequence ID
output/logits flag
```

The batch may contain work from several slots. Token order remains intact within each individual request.

## KV Cache

The KV cache is a preallocated memory pool containing key and value tensors for previous tokens.

```text
KV-cache pool
  |-- cell 0
  |-- cell 1
  |-- cell 2
  `-- cell N
```

It consists of:

1. Large K/V tensor buffers allocated for the configured cache capacity.
2. Metadata describing cell ownership and token positions.
3. Logic for finding, assigning, removing, and reusing cells.

The cache size depends on factors such as:

```text
number of layers
x cache capacity
x K/V dimensions
x number of streams or sequences
x K/V data types
```

Cells are not individually allocated heap blocks. They are indices into larger K/V tensors.

When a request finishes, its sequence ownership is removed from the cells. The underlying buffers usually remain allocated and the cells become reusable for later requests. There is normally no separate memory thread that frees individual KV blocks.

## Batch Metadata and KV Metadata

The scheduler supplies request information:

```text
sequence ID
current token
position
```

The KV-cache manager maps that information to cache-cell indices:

```text
sequence ID + position
  -> cache-cell indices
  -> K/V tensor locations
```

The batch usually does not contain raw pointers to cached K/V memory. It contains metadata, while the KV-cache layer prepares indices used by the graph.

The current token's K/V values are computed during the current inference step and then written into the cache for future steps.

## Ubatches

A ubatch is a micro-batch: a smaller execution piece of the server's larger batch.

For example:

```text
server batch = 20 tokens
n_ubatch = 8

ubatch 1 = tokens 0-7
ubatch 2 = tokens 8-15
ubatch 3 = tokens 16-19
```

For each ubatch, the inference path generally does this:

```text
prepare KV-cache placement
  -> apply memory context
  -> build or reuse graph
  -> execute graph
  -> advance to next ubatch
```

Ubatches are normally handled sequentially by the inference/control thread. The tensor operations inside each ubatch may execute in parallel on CPU worker threads or the GPU.

## Decode and Graph Execution

The core function path is:

```text
server_context_impl::update_slots()
  -> llama_decode()
  -> llama_context::decode()
  -> memory->init_batch()
  -> llama_kv_cache::prepare()
  -> llama_context::process_ubatch()
  -> memory_context->apply()
  -> model.build_graph()
  -> llama_context::graph_compute()
  -> ggml_backend_sched_graph_compute_async()
```

Responsibilities:

- `update_slots()` selects work from active slots and builds the server batch.
- `llama_context::decode()` accepts the batch and starts engine-side processing.
- `memory->init_batch()` splits the batch into ubatches and prepares memory state.
- `llama_kv_cache::prepare()` finds suitable KV-cache cells and records their indices.
- `memory_context->apply()` commits the prepared memory state for the current ubatch.
- `model.build_graph()` constructs the transformer computation graph.
- `llama_context::process_ubatch()` applies memory state, builds or reuses the graph, sets inputs, and computes it.
- `llama_context::graph_compute()` selects thread settings and submits the graph to ggml.

The graph may be reused when its topology is still valid, so it is not necessarily rebuilt on every decode.

## Sampling and Repetition

After graph execution, the server gets logits for the output associated with each slot.

```text
logits for slot 0 -> slot 0 sampler
logits for slot 1 -> slot 1 sampler
logits for slot 2 -> slot 2 sampler
```

The sampling path commonly includes temperature, top-k, top-p, repetition penalties, and grammar constraints. The main functions are:

- `common_sampler_sample()` in `common/sampling.cpp`.
- `common_sampler_accept()` in `common/sampling.cpp`.
- `llama_sampler_sample()` in `src/llama-sampler.cpp`.

The selected token is added to the request state, converted to text, and streamed to the client. The slot then participates in the next scheduling iteration.

## Thread Model

For one ordinary model context, the control path is generally serialized:

```text
HTTP/request threads
  -> enqueue tasks

one server scheduler/inference loop
  -> process tasks
  -> update slots
  -> build the shared batch
  -> prepare KV-cache metadata
  -> build or reuse graphs
  -> submit backend work
  -> sample results
  -> update slots
```

After backend submission, the expensive numerical work can be parallel:

```text
CPU backend
  -> ggml worker threads execute tensor operations

GPU backend
  -> GPU executes kernels and workgroups
```

Therefore:

```text
serialized:
  request scheduling
  slot bookkeeping
  batch construction
  KV-cache metadata preparation
  graph orchestration
  ubatch sequencing

parallel:
  CPU tensor operations
  GPU tensor execution
```

The scheduler and KV-cache manager are different components, but in the normal path the same inference/control thread calls them sequentially. The KV-cache manager is not normally a separate thread.

## Short Summary

> Multiple request threads submit tasks to a shared queue. One server scheduler consumes those tasks, assigns them to slots, and groups their ready tokens into a batch. `llama_context` prepares KV-cache indices, builds or reuses the graph, and submits it to ggml. The CPU workers or GPU execute the tensor operations. The server samples the results, updates each slot, and repeats until the requests finish.
