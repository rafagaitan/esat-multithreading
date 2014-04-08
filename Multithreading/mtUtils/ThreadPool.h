/*******************************************************************************
   The MIT License (MIT)

   Copyright (c) 2014 Rafael Gaitan <rafa.gaitan@mirage-tech.com>
                                    http://www.mirage-tech.com

   Permission is hereby granted, free of charge, to any person obtaining a copy
   of this software and associated documentation files (the "Software"), to deal
   in the Software without restriction, including without limitation the rights
   to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
   copies of the Software, and to permit persons to whom the Software is
   furnished to do so, subject to the following conditions:

   The above copyright notice and this permission notice shall be included in
   all copies or substantial portions of the Software.

   THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
   IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
   FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
   AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
   LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
   OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
   THE SOFTWARE.

   -----------------------------------------------------------------------------
   Additional Notes:

   Code for the Multithreading and Parallel Computing Course at ESAT
               -------------------------------
               |     http://www.esat.es      |
               -------------------------------

   more information of the course at:
       -----------------------------------------------------------------
       |  http://www.esat.es/estudios/programacion-multihilo/?pnt=621  |
       -----------------------------------------------------------------
**********************************************************************************/
#pragma once

#include <vector>
#include <memory>
#include <thread>
#include <future>
#include <functional>
#include <stdexcept>
#include <atomic>

#include <mtUtils/ThreadSafeQueue.h>
#include <mtUtils/ScopedThread.h>


class ThreadPool 
{
private:
    void runPendingTask();
public:
    ThreadPool(size_t numThreads = std::thread::hardware_concurrency());
    template<class F, class... Args>
    auto enqueue(F&& f, Args&&... args) -> std::future<typename std::result_of<F(Args...)>::type>;
    ~ThreadPool();

    size_t getNumThreads() const { return workers.size(); }

    static ThreadPool& instance()
    {
        static ThreadPool s_pool;
        return s_pool;
    }
    
private:
    // stop guard
    std::atomic<bool> stop;
    // the task queue
    ThreadSafeQueue< std::function<void()> > tasks;
    // need to keep track of threads so we can join them
    std::vector< ScopedThread > workers;
};

// the constructor just launches some amount of workers
inline ThreadPool::ThreadPool(size_t threads):
    stop(false),
    tasks(),
    workers()
{
    for(size_t i = 0;i<threads;++i)
        workers.emplace_back(std::thread(
            [this]
            {
                while(!stop)
                {
                    runPendingTask();
                }
            }
        ));
}

// add new work item to the pool
template<class F, class... Args>
auto ThreadPool::enqueue(F&& f, Args&&... args)
    -> std::future<typename std::result_of<F(Args...)>::type>
{
    typedef typename std::result_of<F(Args...)>::type return_type;

    // don't allow enqueueing after stopping the pool
    if(stop)
        throw std::runtime_error("enqueue on stopped ThreadPool");

    auto task = std::make_shared< std::packaged_task<return_type()> >(
            std::bind(std::forward<F>(f), std::forward<Args>(args)...)
        );

    std::future<return_type> res = task->get_future();
    tasks.push([task](){ (*task)(); });
    return res;
}


// the destructor joins all threads
inline ThreadPool::~ThreadPool()
{
    stop = true;
    tasks.notify_and_terminate();
    workers.clear();
}

inline void ThreadPool::runPendingTask()
{
    std::function<void()> task;
    if(this->tasks.wait_pop(task))
    {
        task();
    }
}


