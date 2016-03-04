#pragma once


#include <iostream>
#include <string>
#include <atomic>
#include <future>
#include <vector>

#include "ThreadPool.hpp"


#ifndef __APPLE__
#include <chrono>
struct ScopedTimer
{
    typedef std::chrono::duration<float> float_seconds;
    ScopedTimer(const std::string& infoText, unsigned int numTests = 1)
        : _infoText(infoText)
        , _start(std::chrono::high_resolution_clock::now())
        , _numTests(numTests)
    { }
    ~ScopedTimer()
    {
        auto elapsed = std::chrono::duration_cast<float_seconds>(std::chrono::high_resolution_clock::now() - _start);
        std::cout << "Elapsed time for " << _infoText << ": " << (float)elapsed.count() / (float)_numTests << std::endl;
    }
    std::string                                        _infoText;
    std::chrono::time_point<std::chrono::high_resolution_clock> _start;
    unsigned int _numTests;
};
#else
#include <sys/time.h>
#include <stdint.h>
#include <unistd.h>
struct ScopedTimer
{
    typedef uint64_t Timestamp_t;

    std::string  _infoText;
    Timestamp_t  _start;
    unsigned int _numTests;
    double       _secsPerTick;

    #if defined(_POSIX_TIMERS) && ( _POSIX_TIMERS > 0 ) && defined(_POSIX_MONOTONIC_CLOCK)
        #include <time.h>

        Timestamp_t tick()
        {
            struct timespec ts;
            clock_gettime(CLOCK_MONOTONIC, &ts);
            return ((Timestamp_t)ts.tv_sec)*1000000+(Timestamp_t)ts.tv_nsec/1000;
        }
    #else
        #include <sys/time.h>

        Timestamp_t tick()
        {
            struct timeval tv;
            gettimeofday(&tv, NULL);
            return ((Timestamp_t)tv.tv_sec)*1000000+(Timestamp_t)tv.tv_usec;
        }
    #endif
    ScopedTimer(const std::string& infoText, unsigned int numTests = 1) 
        : _infoText(infoText)
        , _start(tick())
        , _numTests(numTests)
        , _secsPerTick(1.0/(double)1000000)
    { }
    ~ScopedTimer()
    {
        double elapsed = (double)(tick() - _start)*_secsPerTick;
        std::cout << "Elapsed time for " << _infoText << ": " << elapsed / (double)_numTests << std::endl;
    }

};

#endif


template<typename Iterator, typename Func>
Func parallel_for_each(Iterator first, Iterator last, Func f)
{
    const size_t length = std::distance(first, last);
    if (!length)
        return std::move(f);

    const size_t minPerThread = 25;
    const size_t maxThreads = (length + minPerThread - 1) / minPerThread;
    const size_t hwThreads = std::thread::hardware_concurrency();
    const size_t numThreads = std::min(hwThreads, maxThreads);
    const size_t blockSize = length / numThreads;

    ThreadPool& pool = ThreadPool::instance();
    std::vector<std::future<Func> > futures(numThreads - 1);
    Iterator blockStart = first;
    for (auto i = 0u; i < (numThreads - 1); ++i)
    {
        Iterator blockEnd = blockStart;
        std::advance(blockEnd, blockSize);
        futures[i] = pool.enqueue(&std::for_each<Iterator, Func>, blockStart, blockEnd, f);
        blockStart = blockEnd;
    }
    std::for_each(blockStart, last, f);
    for (auto i = 0u; i < (numThreads - 1); ++i)
    {
        futures[i].wait();
    }
    return std::move(f);
}

template<typename Iterator, typename T>
struct find_element
{
    void operator()(Iterator begin, Iterator end, const T* value,
        std::promise<Iterator>* result,
        std::atomic<bool>* done)
    {
#if 0
        for (; begin != end && !done->load(); ++begin)
        {
            if (*begin == static_cast<T>(*value))
            {
                result->set_value(begin);
                done->store(true);
                return;
            }
        }
#else
        auto found = std::find(begin, end, *value);
        if (found != end)
        {
            result->set_value(begin);
            done->store(true);
        }
#endif
    }
};


template<typename Iterator, typename T>
Iterator parallel_find_pool(Iterator first, Iterator last, const T& value)
{
    const size_t length = std::distance(first, last);
    if (!length)
        return last;

    const size_t minPerThread = 25;
    const size_t maxThreads = (length + minPerThread - 1) / minPerThread;
    const size_t hwThreads = std::thread::hardware_concurrency();
    const size_t numThreads = std::min(hwThreads, maxThreads);
    const size_t blockSize = length / numThreads;

    ThreadPool& pool = ThreadPool::instance();
    std::vector<std::future<void> > futures(numThreads - 1);
    std::promise<Iterator>          result;
    std::atomic<bool>               done;
    Iterator                        blockStart = first;
    for (auto i = 0u; i < (numThreads - 1) && !done.load(); ++i)
    {
        Iterator blockEnd = blockStart;
        std::advance(blockEnd, blockSize);
        futures[i] = pool.enqueue((find_element<Iterator, T>()), blockStart, blockEnd, &value, &result, &done);
        blockStart = blockEnd;
    }
    find_element<Iterator, T>()(blockStart, last, &value, &result, &done);
    try
    {
        std::for_each(futures.begin(), futures.end(), [](std::future<void>& f) { f.wait(); });
        if (!done.load())
        {
            return last;
        }
        return result.get_future().get();
    }
    catch (...)
    {

    }
    return last;
}

template<typename Iterator, typename T>
Iterator parallel_find_thread(Iterator first, Iterator last, const T& value)
{
    const size_t length = std::distance(first, last);
    if (!length)
        return last;

    const size_t minPerThread = 25;
    const size_t maxThreads = (length + minPerThread - 1) / minPerThread;
    const size_t hwThreads = std::thread::hardware_concurrency();
    const size_t numThreads = std::min(hwThreads, maxThreads);
    const size_t blockSize = length / numThreads;

    std::vector<ScopedThread> threads(numThreads - 1);
    std::promise<Iterator>          result;
    std::atomic<bool>               done;
    Iterator                        blockStart = first;
    for (auto i = 0u; i < (numThreads - 1) && !done.load(); ++i)
    {
        Iterator blockEnd = blockStart;
        std::advance(blockEnd, blockSize);
        threads[i] = std::thread((find_element<Iterator, T>()), blockStart, blockEnd, &value, &result, &done);
        blockStart = blockEnd;
    }
    find_element<Iterator, T>()(blockStart, last, &value, &result, &done);

    threads.clear();

    if (!done.load())
    {
        return last;
    }
    return result.get_future().get();
}

template<typename Iterator, typename T>
Iterator parallel_find_async_impl(Iterator first, Iterator last, const T& value, std::atomic<bool>& done)
{
    try
    {
        const size_t length = std::distance(first, last);
        const size_t minPerThread = 25;
        if (length < (2 * minPerThread))
        {
            for (; (first != last) && !done.load(); ++first)
            {
                if (*first == value)
                {
                    done = true;
                    return first;
                }
            }
            return last;
        }
        else
        {
            const Iterator midPoint = first + (length / 2);
            std::future<Iterator> async_result(std::async(&parallel_find_async_impl<Iterator, T>,
                midPoint, last, std::ref(value), std::ref(done)));
            const Iterator currentResult = parallel_find_async_impl(first, midPoint, value, done);
            return (currentResult == midPoint) ? async_result.get() : currentResult;
        }
    }
    catch (...)
    {
        done = true;
        throw;
    }
}

template<typename Iterator, typename T>
Iterator parallel_find_async(Iterator first, Iterator last, const T& value)
{
    std::atomic<bool> done(false);
    return parallel_find_async_impl(first, last, value, done);
}


