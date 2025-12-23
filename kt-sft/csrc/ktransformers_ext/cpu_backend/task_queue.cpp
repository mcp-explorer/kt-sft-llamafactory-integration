/**
 * @Description :
 * @Author    : chenht2022
 * @Date     : 2024-07-17 12:25:51
 * @Version   : 1.0.0
 * @LastEditors : chenht2022
 * @LastEditTime : 2024-10-09 11:08:10
 * @Copyright (c) 2024 by KVCache.AI, All Rights Reserved.
 **/
#include "task_queue.h"
#include <chrono>
#include <thread>
#include <cstdio>
#include <stdexcept>

TaskQueue::TaskQueue() {
    worker = std::thread(&TaskQueue::processTasks, this);
    sync_flag.store(true, std::memory_order_seq_cst);
    exit_flag.store(false, std::memory_order_seq_cst);
}

TaskQueue::~TaskQueue() {
    {
        mutex.lock();
        exit_flag.store(true, std::memory_order_seq_cst);
        mutex.unlock();
    }
    cv.notify_all();
    if (worker.joinable()) {
        worker.join();
    }
}

void TaskQueue::enqueue(std::function<void()> task) {
    {
        mutex.lock();
        tasks.push(task);
        sync_flag.store(false, std::memory_order_seq_cst);
        mutex.unlock();
    }
    cv.notify_one();
}

void TaskQueue::sync() {
    // Use timeout to prevent infinite loops
    auto start = std::chrono::steady_clock::now();
    const auto timeout = std::chrono::seconds(300); // 5 minute timeout
    
    while (!sync_flag.load(std::memory_order_seq_cst)) {
        auto now = std::chrono::steady_clock::now();
        if (now - start > timeout) {
            // Timeout - something went wrong, break to avoid infinite loop
            fprintf(stderr, "[TaskQueue::sync] WARNING: sync() timed out after 5 minutes! Breaking infinite loop.\n");
            fflush(stderr);
            break;
        }
        // Yield to other threads instead of busy-waiting
        std::this_thread::yield();
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }
}

void TaskQueue::processTasks() {
    while (true) {
        std::function<void()> task;
        {
            mutex.lock();
            cv.wait(mutex, [this]() { return !tasks.empty() || exit_flag.load(std::memory_order_seq_cst); });
            if (exit_flag.load(std::memory_order_seq_cst) && tasks.empty()) {
                mutex.unlock();
                return;
            }
            task = tasks.front();
            tasks.pop();
            mutex.unlock();
        }
        
        // Execute task with exception handling to ensure sync_flag is always set
        try {
            task();
        } catch (const std::exception& e) {
            fprintf(stderr, "[TaskQueue::processTasks] Exception in task: %s\n", e.what());
            fflush(stderr);
        } catch (...) {
            fprintf(stderr, "[TaskQueue::processTasks] Unknown exception in task\n");
            fflush(stderr);
        }
        
        {
            mutex.lock();
            if (tasks.empty()) {
                sync_flag.store(true, std::memory_order_seq_cst);
            }
            mutex.unlock();
        }
    }
}