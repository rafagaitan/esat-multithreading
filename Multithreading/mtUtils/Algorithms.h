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

#include <future>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <thread>
#include <string>

#include <mtUtils/ThreadPool.h>
#include <mtUtils/ScopedThread.h>

struct ScopedTimer
{
    typedef std::chrono::duration<float> float_seconds;
    ScopedTimer(const std::string& infoText):
                _infoText(infoText), 
                _start(std::chrono::system_clock::now()) { }
    ~ScopedTimer() 
    {
        auto elapsed = std::chrono::duration_cast<float_seconds>(std::chrono::system_clock::now() - _start);
        std::cout << "Elapsed time for " << _infoText <<": " << elapsed.count() << std::endl;
    }
    std::string                                        _infoText;
    std::chrono::time_point<std::chrono::system_clock> _start;
};

template<typename Iterator, typename Func>
Func parallel_for_each(Iterator first, Iterator last, Func f)
{
    const size_t length = std::distance(first, last);
    if(!length)
        return std::move(f);

    const size_t minPerThread = 25;
    const size_t maxThreads   = (length + minPerThread-1)/minPerThread;
    const size_t hwThreads    = std::thread::hardware_concurrency();
    const size_t numThreads   = std::min(hwThreads,maxThreads);
    const size_t blockSize    = length/numThreads;

    ThreadPool& pool = ThreadPool::instance();
    std::vector<std::future<Func> > futures(numThreads-1);
    Iterator blockStart = first;
    for(auto i = 0u; i < (numThreads-1); ++i)
    {
        Iterator blockEnd = blockStart;
        std::advance(blockEnd,blockSize);
        futures[i] = pool.enqueue(&std::for_each<Iterator,Func>,blockStart, blockEnd, f);
        blockStart = blockEnd;
    }
    std::for_each(blockStart, last, f);
    for(auto i = 0u; i < (numThreads-1); ++i)
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
        for( ; begin!=end && !done->load(); ++begin)
        {
            if( *begin == static_cast<T>(*value) )
            {
                result->set_value(begin);
                done->store(true);
                return;
            }
        }
#else
        auto found = std::find(begin,end,*value);
        if(found != end)
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
    if(!length)
        return last;
    
    const size_t minPerThread = 25;
    const size_t maxThreads   = (length + minPerThread-1)/minPerThread;
    const size_t hwThreads    = std::thread::hardware_concurrency();
    const size_t numThreads   = std::min(hwThreads,maxThreads);
    const size_t blockSize    = length/numThreads;

    ThreadPool& pool = ThreadPool::instance();
    std::vector<std::future<void> > futures(numThreads-1);
    std::promise<Iterator>          result;
    std::atomic<bool>               done;
    Iterator                        blockStart = first;
    for(auto i = 0u; i < (numThreads-1) && !done.load(); ++i)
    {
        Iterator blockEnd = blockStart;
        std::advance(blockEnd,blockSize);
        futures[i] = pool.enqueue((find_element<Iterator,T>()),blockStart, blockEnd, &value, &result, &done);
        blockStart = blockEnd;
    }
    find_element<Iterator,T>()(blockStart, last, &value, &result, &done);
    try
    {
        std::for_each(futures.begin(),futures.end(), [](std::future<void>& f) { f.wait(); });
        if(!done.load())
        {
            return last;
        }
        return result.get_future().get();
    }
    catch(...)
    {
    
    }
    return last;
}

template<typename Iterator, typename T>
Iterator parallel_find_thread(Iterator first, Iterator last, const T& value)
{
    const size_t length = std::distance(first, last);
    if(!length)
        return last;

    const size_t minPerThread = 25;
    const size_t maxThreads   = (length + minPerThread-1)/minPerThread;
    const size_t hwThreads    = std::thread::hardware_concurrency();
    const size_t numThreads   = std::min(hwThreads,maxThreads);
    const size_t blockSize = length/numThreads;
    
    std::vector<ScopedThread> threads(numThreads-1);
    std::promise<Iterator>          result;
    std::atomic<bool>               done;
    Iterator                        blockStart = first;
    for(auto i = 0u; i < (numThreads-1) && !done.load(); ++i)
    {
        Iterator blockEnd = blockStart;
        std::advance(blockEnd,blockSize);
        threads[i] = std::thread((find_element<Iterator,T>()),blockStart, blockEnd, &value, &result, &done);
        blockStart = blockEnd;
    }
    find_element<Iterator,T>()(blockStart, last, &value, &result, &done);

    threads.clear();

    if(!done.load())
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
        if(length < (2*minPerThread))
        {
            for(; (first!=last) && !done.load(); ++first)
            {
                if(*first == value)
                {
                    done = true;
                    return first;
                }
            }
            return last;
        }
        else
        {
            const Iterator midPoint = first + (length/2);
            std::future<Iterator> async_result(std::async(&parallel_find_async_impl<Iterator,T>,
                                                          midPoint,last,std::ref(value), std::ref(done)));
            const Iterator currentResult = parallel_find_async_impl(first, midPoint, value, done);
            return (currentResult==midPoint)?async_result.get():currentResult;
        }
    }
    catch(...)
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



