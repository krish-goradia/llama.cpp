#pragma once

#include <atomic>
#include <cstddef>
#include <new>
#include <type_traits>
#include <utility>

// Lock-free Single-Producer Single-Consumer (SPSC) ring buffer
// Capacity must be a power of 2
template <typename T, size_t Capacity = 256>
struct spsc_ring_buffer {
    static_assert((Capacity & (Capacity - 1)) == 0, "Capacity must be a power of 2");
    static_assert(Capacity > 0, "Capacity must be positive");

private:
    static constexpr size_t MASK = Capacity - 1;

    // Cacheline isolation to avoid false sharing between producer and consumer
    alignas(64) std::atomic<size_t> tail{0}; // written by producer
    alignas(64) std::atomic<size_t> head{0}; // written by consumer

    // Storage for elements
    alignas(64) typename std::aligned_storage<sizeof(T), alignof(T)>::type storage[Capacity];

public:
    spsc_ring_buffer() = default;

    ~spsc_ring_buffer() {
        // Destroy all remaining active elements directly in-place
        size_t current_head = head.load(std::memory_order_relaxed);
        const size_t current_tail = tail.load(std::memory_order_relaxed);
        while (current_head < current_tail) {
            T * slot = reinterpret_cast<T *>(&storage[current_head & MASK]);
            slot->~T();
            current_head++;
        }
    }

    // Non-copyable, non-movable (pinned ring buffer)
    spsc_ring_buffer(const spsc_ring_buffer &) = delete;
    spsc_ring_buffer & operator=(const spsc_ring_buffer &) = delete;
    spsc_ring_buffer(spsc_ring_buffer &&) = delete;
    spsc_ring_buffer & operator=(spsc_ring_buffer &&) = delete;

    // Push item to the ring buffer (Producer thread only)
    bool try_push(T && item) {
        const size_t current_tail = tail.load(std::memory_order_relaxed);
        const size_t current_head = head.load(std::memory_order_acquire);

        if (current_tail - current_head >= Capacity) {
            return false; // buffer full
        }

        // Placement new in storage slot
        new (&storage[current_tail & MASK]) T(std::move(item));
        tail.store(current_tail + 1, std::memory_order_release);
        return true;
    }

    // Pop item from the ring buffer (Consumer thread only)
    bool try_pop(T & item) {
        const size_t current_head = head.load(std::memory_order_relaxed);
        const size_t current_tail = tail.load(std::memory_order_acquire);

        if (current_head >= current_tail) {
            return false; // buffer empty
        }

        T * slot = reinterpret_cast<T *>(&storage[current_head & MASK]);
        item = std::move(*slot);
        slot->~T();

        head.store(current_head + 1, std::memory_order_release);
        return true;
    }

    size_t size() const noexcept {
        const size_t current_head = head.load(std::memory_order_relaxed);
        const size_t current_tail = tail.load(std::memory_order_relaxed);
        return (current_tail >= current_head) ? (current_tail - current_head) : 0;
    }

    bool empty() const noexcept {
        return head.load(std::memory_order_relaxed) == tail.load(std::memory_order_relaxed);
    }

    static constexpr size_t capacity() noexcept {
        return Capacity;
    }
};
