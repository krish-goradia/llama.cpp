#include "mpsc-queue.h"

#include <atomic>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <thread>
#include <vector>

// 1. Basic Single-Threaded Verification
static void test_basic_operations() {
    std::cout << "[TEST] Running test_basic_operations...\n";
    mpsc_queue<int> q;

    assert(q.empty());
    int val = -1;
    assert(!q.pop(val));

    q.push(10);
    q.push(20);
    q.push(30);

    assert(!q.empty());

    assert(q.pop(val) && val == 10);
    assert(q.pop(val) && val == 20);
    assert(q.pop(val) && val == 30);
    assert(!q.pop(val));
    assert(q.empty());

    // Test drain
    q.push(100);
    q.push(200);
    q.push(300);

    std::vector<int> drained;
    size_t count = q.drain([&](int v) {
        drained.push_back(v);
    });

    assert(count == 3);
    assert(drained.size() == 3);
    assert(drained[0] == 100);
    assert(drained[1] == 200);
    assert(drained[2] == 300);
    assert(q.empty());

    (void)val;
    (void)count;

    std::cout << "  -> PASS\n";
}

// 2. Lifecycle & Memory Leak Test with Move-Only Types
struct instance_tracker {
    static inline std::atomic<int64_t> alive_instances{0};
    int id = 0;

    instance_tracker(int id_) : id(id_) {
        alive_instances.fetch_add(1, std::memory_order_relaxed);
    }

    ~instance_tracker() {
        if (id != -1) {
            alive_instances.fetch_sub(1, std::memory_order_relaxed);
        }
    }

    instance_tracker(const instance_tracker &) = delete;
    instance_tracker & operator=(const instance_tracker &) = delete;

    instance_tracker(instance_tracker && other) noexcept : id(other.id) {
        other.id = -1;
    }

    instance_tracker & operator=(instance_tracker && other) noexcept {
        if (this != &other) {
            if (id != -1) {
                alive_instances.fetch_sub(1, std::memory_order_relaxed);
            }
            id = other.id;
            other.id = -1;
        }
        return *this;
    }
};

static void test_lifecycle_and_leaks() {
    std::cout << "[TEST] Running test_lifecycle_and_leaks...\n";
    assert(instance_tracker::alive_instances.load() == 0);

    {
        mpsc_queue<instance_tracker> q;
        for (int i = 0; i < 1000; i++) {
            q.push(instance_tracker(i));
        }
        assert(instance_tracker::alive_instances.load() == 1000);

        for (int i = 0; i < 500; i++) {
            instance_tracker item(0);
            bool ok = q.pop(item);
            assert(ok);
            assert(item.id == i);
            (void)ok;
        }
        // 500 items popped & destroyed at loop scope, 500 remaining in queue
        assert(instance_tracker::alive_instances.load() == 500);
    }
    // Queue destructor must clean up remaining 500 items
    assert(instance_tracker::alive_instances.load() == 0);

    std::cout << "  -> PASS\n";
}

// 3. Multi-Threaded Stress Test (MPSC Correctness & Per-Producer FIFO Invariance)
struct test_message {
    int producer_id = 0;
    int seq = 0;
};

static void test_multithreaded_mpsc(int n_producers, int items_per_producer) {
    std::cout << "[TEST] Running test_multithreaded_mpsc (" 
              << n_producers << " producers x " << items_per_producer << " items = "
              << (int64_t)n_producers * items_per_producer << " items)...\n";

    mpsc_queue<test_message> q;
    std::atomic<bool> start_flag{false};
    std::atomic<int> producers_finished{0};

    std::vector<std::thread> producers;
    producers.reserve(n_producers);

    for (int p = 0; p < n_producers; p++) {
        producers.emplace_back([&, p]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int i = 0; i < items_per_producer; i++) {
                q.push({p, i});
            }
            producers_finished.fetch_add(1, std::memory_order_release);
        });
    }

    std::vector<int> expected_seq(n_producers, 0);
    int64_t total_popped = 0;
    const int64_t total_expected = (int64_t)n_producers * items_per_producer;

    auto t_start = std::chrono::high_resolution_clock::now();
    start_flag.store(true, std::memory_order_release);

    test_message msg;
    while (total_popped < total_expected) {
        if (q.pop(msg)) {
            assert(msg.producer_id >= 0 && msg.producer_id < n_producers);
            assert(msg.seq == expected_seq[msg.producer_id]); // Verify strict per-producer FIFO
            expected_seq[msg.producer_id]++;
            total_popped++;
        } else {
            std::this_thread::yield();
        }
    }

    auto t_end = std::chrono::high_resolution_clock::now();
    for (auto & t : producers) {
        t.join();
    }

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(t_end - t_start).count();
    std::cout << "  -> Popped " << total_popped << " items in " << duration_ms << " ms ("
              << (duration_ms > 0 ? (total_popped * 1000 / duration_ms) : 0) << " ops/sec)\n";
    std::cout << "  -> PASS\n";
}

// 4. Atomic Cancellation Stress Test
struct cancellable_task {
    int id = 0;
    int producer_id = 0;
    int seq = 0;
    std::shared_ptr<std::atomic<bool>> cancel_token;

    bool is_cancelled() const {
        return cancel_token && cancel_token->load(std::memory_order_relaxed);
    }
};

static void test_atomic_cancellation_stress(int n_producers, int items_per_producer) {
    std::cout << "[TEST] Running test_atomic_cancellation_stress (" 
              << n_producers << " producers x " << items_per_producer << " items)...\n";

    mpsc_queue<cancellable_task> q;
    std::atomic<bool> start_flag{false};
    std::atomic<int64_t> cancelled_by_producers{0};

    std::vector<std::thread> producers;
    producers.reserve(n_producers);

    for (int p = 0; p < n_producers; p++) {
        producers.emplace_back([&, p]() {
            while (!start_flag.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            for (int i = 0; i < items_per_producer; i++) {
                auto cancel_token = std::make_shared<std::atomic<bool>>(false);
                bool should_cancel = (i % 3 == 0); // Cancel ~33% of requests
                if (should_cancel) {
                    cancel_token->store(true, std::memory_order_relaxed);
                    cancelled_by_producers.fetch_add(1, std::memory_order_relaxed);
                }
                q.push(cancellable_task{p * items_per_producer + i, p, i, cancel_token});
            }
        });
    }

    int64_t total_popped = 0;
    int64_t discarded_count = 0;
    int64_t executed_count = 0;
    const int64_t total_expected = (int64_t)n_producers * items_per_producer;

    start_flag.store(true, std::memory_order_release);

    cancellable_task task;
    while (total_popped < total_expected) {
        if (q.pop(task)) {
            total_popped++;
            if (task.is_cancelled()) {
                discarded_count++;
            } else {
                executed_count++;
            }
        } else {
            std::this_thread::yield();
        }
    }

    for (auto & t : producers) {
        t.join();
    }

    assert(total_popped == total_expected);
    assert(discarded_count == cancelled_by_producers.load());
    assert(executed_count + discarded_count == total_expected);

    std::cout << "  -> Total: " << total_popped << " | Executed: " << executed_count 
              << " | Cancelled & Discarded in O(1): " << discarded_count << "\n";
    std::cout << "  -> PASS\n";
}

int main() {
    std::cout << "==================================================\n";
    std::cout << " Starting Lock-Free MPSC Queue Unit & Stress Tests\n";
    std::cout << "==================================================\n";

    test_basic_operations();
    test_lifecycle_and_leaks();
    test_multithreaded_mpsc(8, 50000);
    test_multithreaded_mpsc(16, 50000);
    test_atomic_cancellation_stress(8, 50000);

    std::cout << "==================================================\n";
    std::cout << " All Lock-Free MPSC Queue Tests PASSED!\n";
    std::cout << "==================================================\n";
    return 0;
}
