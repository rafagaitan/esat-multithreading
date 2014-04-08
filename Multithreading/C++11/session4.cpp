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

#include <iostream>
#include <atomic>
#include <numeric>
#include <list>
#include <random>
#include <algorithm>

#include <mtUtils/ScopedThread.h>
#include <mtUtils/ThreadSafeStack.h>
#include <mtUtils/ThreadSafeQueue.h>
#include <mtUtils/ThreadPool.h>
#include <mtUtils/Barrier.h>

template<typename T>
std::list<T> parallel_quick_sort(std::list<T> input)
{
    if(input.empty())
    {
        return input;
    }
    std::list<T> result;
    result.splice(result.begin(), input, input.begin());
    T const& pivot=*(result.begin());
    auto divide_point=std::partition(input.begin(),input.end(), 
        [&] (T const& t) { return t<pivot; });
    std::list<T> lower_part;
    lower_part.splice(lower_part.end(),input,input.begin(), divide_point);

    auto new_lower(std::async(&parallel_quick_sort<T>, std::move(lower_part)));
    auto new_higher(parallel_quick_sort(std::move(input)));

    result.splice(result.end(),new_higher);
    result.splice(result.begin(),new_lower.get());
    return result;
}

int main(int , char**)
{
     if(0)
     {
        std::cout << "ready to have high data contention?" << std::endl;
        std::cin.ignore();

        std::mutex m;
        unsigned long i = 0;
        auto processing_loop = [&]()
        {
            while(true)
            {
                std::lock_guard<std::mutex> lock(m);
                i++;
                if(i > 1000000)
                    return;
                std::this_thread::yield();
            }
        };
        std::vector<ScopedThread> threads;
        for(auto i=0u;i<std::thread::hardware_concurrency();i++)
            threads.emplace_back(std::thread(processing_loop));
    }
    if(0)
    {
        std::cout << "ready to have cache ping-pong?" << std::endl;
        std::cin.ignore();

        std::atomic<unsigned long> counter(0);
        auto processing_loop = [&]()
        {
            while(counter.fetch_add(1) < 100000000)
            {
                std::this_thread::yield();
            }
        };
        std::vector<ScopedThread> threads;
        for(auto i=0u;i<std::thread::hardware_concurrency();i++)
            threads.emplace_back(std::thread(processing_loop));
    }
    if(0)
    {
        std::cout << "ready to have false sharing?" << std::endl;
        std::cin.ignore();

        std::vector<int> data;
        for(unsigned int i=0;i<1000;i++) data.push_back(i);
        std::atomic<unsigned long> counter(0);
        auto processing_loop = [&]()
        {
            while(true)
            {
                auto i = counter++;
                if(i>=data.size())
                    break;
                data[i]*=2;
            }
        };
        std::vector<ScopedThread> threads;
        for(auto i=0u;i<std::thread::hardware_concurrency();i++)
            threads.emplace_back(std::thread(processing_loop));
        for(unsigned int i=0;i<data.size();i++) std::cout << data[i] << ",";
        std::cout << std::endl;
    }
    if(0)
    {
        std::cout << "ready to have oversuscription?" << std::endl;
        std::cin.ignore();

        std::atomic<bool> done(false);

        ThreadSafeQueue<int> ts;
        auto inserter_function = [](ThreadSafeQueue<int>& ts, unsigned int numelements)
        {
            for(unsigned int i=0;i<numelements;i++)
            {
                ts.push(i);
                std::cout << std::this_thread::get_id() << ":pushed value=" << i << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            }
        };

        auto popper_function = [&](ThreadSafeQueue<int>& ts)
        {
            while(!done || !ts.empty())
            {
                int value;
                if(ts.wait_pop(value, std::chrono::milliseconds(30)))
                    std::cout << std::this_thread::get_id() << ":popped value=" << value << std::endl;
            }
            std::cout << std::this_thread::get_id() << ":stack is done" << std::endl;
        };
        std::vector<ScopedThread> producers;
        std::vector<ScopedThread> consumers;
        for(auto i=0;i<4;i++)
            consumers.emplace_back(std::thread(popper_function,std::ref(ts)));

        for(auto i=0;i<200;i++)
            producers.emplace_back(std::thread(inserter_function,std::ref(ts),100));

        producers.clear();
        done = true;
        consumers.clear();
        ts.notify_and_terminate();
    }
    if(1)
    {
        std::cout << "ready to order a list?" << std::endl;
        std::cin.ignore();

        std::vector<int> shuffled_data;
        for(unsigned int i=0;i<1000;i++) 
            shuffled_data.push_back(i);

        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(std::begin(shuffled_data),std::end(shuffled_data),g);

        std::list<int> data(shuffled_data.begin(),shuffled_data.end());
        
        //for(auto i=std::begin(data);i!=std::end(data);++i) std::cout << *i << ",";
        //std::cout << std::endl;

        auto final_data = parallel_quick_sort(data);

        //for(auto i=std::begin(final_data);i!=std::end(final_data);++i) std::cout << *i << ",";
        std::cout <<"first:" << *(final_data.begin()) << ", Last:" << final_data.back() << std::endl;
    }
    if(1)
    {
        std::cout << "ready to crush a queue with a thread pool?" << std::endl;
        std::cin.ignore();

        std::atomic<bool> done(false);
        typedef std::shared_ptr<ThreadSafeQueue<int>> SharedThreadSafeQueue;

        auto ts(std::make_shared<ThreadSafeQueue<int>>());
        
        auto inserter_function = [](SharedThreadSafeQueue ts, unsigned int numelements)
        {
            for(unsigned int i=0;i<numelements;i++)
            {
                ts->push(i);
                //std::cout << std::this_thread::get_id() << ":pushed value=" << i << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        };

        auto popper_function = [&](SharedThreadSafeQueue ts)
        {
            while(!done || !ts->empty())
            {
                int value;
                if(ts->wait_pop(value, std::chrono::milliseconds(30)))
                    std::cout << std::this_thread::get_id() << ":popped value=" << value << std::endl;
            }
            std::cout << std::this_thread::get_id() << ":thread is done" << std::endl;
        };

        ThreadPool pool(40);
        std::vector<std::future<void>> producers;
        std::vector<std::future<void>> consumers;
        for(auto i=0;i<20;i++)
            consumers.emplace_back(pool.enqueue(popper_function,ts));

        for(auto i=0;i<200;i++)
            producers.emplace_back(pool.enqueue(inserter_function,ts,100));

        for(auto i=0u;i<producers.size();i++)
            producers[i].wait();

        done = true;

        for(auto i=0u;i<consumers.size();i++)
            consumers[i].wait();
    }

    if(1)
    {
        std::cout << "ready to sync with a barrier without limit?" << std::endl;
        std::cin.ignore();

        Barrier b;
        std::vector<ScopedThread> threads;
        for(unsigned int i=0;i<4;i++)
        {
            threads.emplace_back(std::thread([&]()
            {
                b.wait();
                std::cout << std::this_thread::get_id() << ":thread synced" << std::endl;
            }));
        }
        std::cout << "Waiting 1 second ..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        b.release();
        threads.clear();

        std::cout << "ready to sync with a barrier with limited number of threads?" << std::endl;
        std::cin.ignore();
        Barrier b2(4);
        for(unsigned int i=0;i<20;i++)
        {
            threads.emplace_back(std::thread([&]()
            {
                b2.wait();
                std::cout << std::this_thread::get_id() << ":thread synced" << std::endl;
            }));
        }
        std::cout << "Waiting 1 second ..." << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(1));
        b2.release();
        threads.clear();
    }


    return EXIT_SUCCESS;
}

