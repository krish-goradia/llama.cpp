#pragma once

#include <atomic>
#include <cstddef>
#include <utility>

// Lock-free Multi-Producer Single-Consumer (MPSC) queue
// Based on Dmitry Vyukov's non-intrusive MPSC node algorithm
template <typename T>
class mpsc_queue {
public:
    mpsc_queue() {
        node * stub = new node();
        head.store(stub, std::memory_order_relaxed);
        tail = stub;
    }

    ~mpsc_queue() {
        node * tail_node = tail;
        while (true) {
            node * next_node = tail_node->next.load(std::memory_order_relaxed);
            if (next_node == nullptr) {
                break;
            }
            next_node->value.~T();
            delete tail_node;
            tail_node = next_node;
        }
        delete tail_node;
    }

    mpsc_queue(const mpsc_queue &) = delete;
    mpsc_queue & operator=(const mpsc_queue &) = delete;
    mpsc_queue(mpsc_queue &&) = delete;
    mpsc_queue & operator=(mpsc_queue &&) = delete;

    // Multi-producer: push a new item (lock-free / wait-free)
    void push(T value) {
        node * n = new node(std::move(value));
        node * prev = head.exchange(n, std::memory_order_acq_rel);
        prev->next.store(n, std::memory_order_release);
    }

    // Single-consumer: pop the next item (lock-free)
    // Returns true if an item was popped, false if queue is empty or producer linking
    bool pop(T & output) {
        node * tail_node = tail;
        node * next_node = tail_node->next.load(std::memory_order_acquire);
        if (next_node != nullptr) {
            tail = next_node;
            output = std::move(next_node->value);
            next_node->value.~T();
            delete tail_node;
            return true;
        }
        return false;
    }

    // Single-consumer: drain all items currently ready in the queue
    template <typename Callback>
    size_t drain(Callback && cb) {
        size_t count = 0;
        node * tail_node = tail;
        while (true) {
            node * next_node = tail_node->next.load(std::memory_order_acquire);
            if (next_node == nullptr) {
                break;
            }
            tail = next_node;
            cb(std::move(next_node->value));
            next_node->value.~T();
            delete tail_node;
            tail_node = next_node;
            count++;
        }
        return count;
    }

    // Check if queue appears empty to the consumer
    bool empty() const {
        return tail->next.load(std::memory_order_acquire) == nullptr;
    }

private:
    struct node {
        std::atomic<node *> next{nullptr};
        union {
            T value;
        };

        node() {}
        node(T && v) : value(std::move(v)) {}
        ~node() {}
    };

    alignas(64) std::atomic<node *> head;
    alignas(64) node * tail;
};
