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

#include <vector>
#include <iostream>
#include <thread>
#include <string>
#include <memory>
#include <mutex>
#include <list>
#include <deque>
#include <stack>
#include <atomic>
#include <future>
#include <condition_variable>
#include <numeric>
#include <algorithm>

#include <mtUtils/ScopedThread.h>
#include <mtUtils/ThreadSafeStack.h>
#include <mtUtils/ThreadSafeQueue.h>


template <typename T, typename Container=std::list<T> >
class NaiveThreadSafeDataManager
{
public:
    NaiveThreadSafeDataManager():_data(),_mutex(),_data_cond(),_done(false) {}
    ~NaiveThreadSafeDataManager()
    {
        done();
    }
    void push(T element)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _data.push_back(element);
        _data_cond.notify_one();
    }
    bool find(T element)
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _find(element);
    }
    bool empty()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        return _data.empty();
    }

    bool wait_find(T element, std::chrono::milliseconds t)
    {
        std::unique_lock<std::mutex> lock(_mutex);
        _data_cond.wait_for(lock, t,
                        [&, this]() 
                        { 
                            return (!_data.empty() && _find(element)); 
                        });
        return _find(element);
    }
    void done()
    {
        std::lock_guard<std::mutex> lock(_mutex);
        _done = true;
        _data_cond.notify_one();
    }
protected:
    Container               _data;
    std::mutex              _mutex;
    std::condition_variable _data_cond;
    bool                    _done;
 private:
     bool _find(T element)
     {
         return (std::find(std::begin(_data), 
                 std::end(_data), element) != std::end(_data));
     }

};


int main(int , char**)
{
    {   // wait for it
        std::cout << "ready to fill and find in a list with active waiting?" << std::endl;
        std::cin.ignore();

        NaiveThreadSafeDataManager<int> data_list;
        //bool haveData = false;
        std::mutex mutex;
        /*auto wait_for_data = [&]()
        {
            std::unique_lock<std::mutex> lock(mutex);
            while(!haveData)
            {
                lock.unlock();
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                lock.lock();
            }
        };*/

        std::atomic<bool> haveDataAtomic(false);
        auto wait_for_data_atomic = [&]() 
        {
            while(!haveDataAtomic)
            {
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        };
        auto producer_function = [&](unsigned int numelements)
        {
            for(unsigned int i=0;i<numelements;i++)
            {
                data_list.push(i);
                if(i > numelements/2)
                {
                    haveDataAtomic = true;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
        };
        auto consumer_function = [&](int value_to_find) 
        {
            wait_for_data_atomic(); // wait for it
            auto attempts = 0u;
            bool found = false;
            while(!found && attempts < 100)
            {
                found = data_list.find(value_to_find);
                attempts++;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if(found)
                std::cout << std::this_thread::get_id() << ": Found! after " << attempts << " attempts" << std::endl;
            else
                std::cout << std::this_thread::get_id() << ": not found! :(" << std::endl;
        };
        ScopedThread thread_inserter1(std::thread(producer_function,100));
        ScopedThread thread_inserter2(std::thread(producer_function,100));
        ScopedThread thread_finder1(std::thread(consumer_function, 88));
        ScopedThread thread_finder2(std::thread(consumer_function, 101));
    }

    {   // wait for it
        std::cout << "ready to fill and find in a list waiting with a condition?" << std::endl;
        std::cin.ignore();
        NaiveThreadSafeDataManager<int> data_list;
       
        auto producer_function = [&](unsigned int numelements)
        {
            for(unsigned int i=0;i<numelements;i++)
            {
                data_list.push(i);
            }
        };
        auto consumer_function = [&](unsigned int value_to_find) 
        {
            auto attempts = 0u;
            bool found = false;
            found = data_list.wait_find(value_to_find, std::chrono::milliseconds(10));
            if(found)
            {
                std::cout << std::this_thread::get_id() << 
                ": Found! after " << attempts << " attempts" << std::endl;
            }
            else
            {
                std::cout << std::this_thread::get_id() << 
                ": not found! :(" << std::endl;
            }
        };
        ScopedThread thread_inserter1(std::thread(producer_function,100));
        ScopedThread thread_inserter2(std::thread(producer_function,100));
        ScopedThread thread_finder1(std::thread(consumer_function,88));
        ScopedThread thread_finder2(std::thread(consumer_function,103));
        //data_list.done();
    }


    { // future
        std::cout << "ready to accumulate a vector?" << std::endl;
        std::cin.ignore();

        size_t const length = 10000;
        typedef std::vector<int> Data;
        Data data(length);
        for(size_t i=0;i<length;i++) 
            data[i] = Data::value_type(i);

        size_t const min_per_thread = 25;
        size_t const max_threads = (length+min_per_thread-1)/min_per_thread;
        size_t const hardware_threads = std::thread::hardware_concurrency();
        size_t const num_threads = std::min(hardware_threads!=0?hardware_threads:2,max_threads);
        size_t const block_size = length/num_threads;

        {
            std::vector<std::future<Data::value_type>> futures(num_threads-1);
            auto block_start = std::begin(data);
            for(size_t i=0;i<(num_threads-1);++i)
            {
                auto block_end=block_start;
                std::advance(block_end,block_size);
                futures[i] = std::async(
                    std::accumulate<Data::const_iterator,Data::value_type>, 
                    block_start,block_end,0);
                block_start=block_end;
            }
            Data::value_type result = std::accumulate(block_start,std::end(data),0);
            for(size_t i=0;i<(num_threads-1);++i)
            {
                result+=futures[i].get();
            }
            std::cout << "async total sum (single thread)=" << std::accumulate(std::begin(data),std::end(data),0) << std::endl;
            std::cout << "async total sum (multi thread)=" << result << std::endl;
        }

        {
            std::vector<std::future<Data::value_type>> futures(num_threads-1);
            std::vector<ScopedThread>                  threads(num_threads-1);
            auto block_start = std::begin(data);
            for(size_t i=0;i<(num_threads-1);++i)
            {
                auto block_end=block_start;
                std::advance(block_end,block_size);
                std::packaged_task<Data::value_type()> 
                    task([=]() 
                            { 
                            return std::accumulate(block_start,block_end,0);
                            });
                futures[i] = task.get_future();
                threads[i] = ScopedThread(std::thread(std::move(task)));
                block_start=block_end;
            }
            Data::value_type result = std::accumulate(block_start,std::end(data),0);
            for(size_t i=0;i<(num_threads-1);++i)
            {
                result+=futures[i].get();
            }
            
            std::cout << "packaged_task total sum (single thread)=" << std::accumulate(std::begin(data),std::end(data),0) << std::endl;
            std::cout << "packaged_task total sum (multi thread)=" << result << std::endl;
        }

        {
            std::vector<std::promise<Data::value_type>> promises(num_threads-1);
            std::vector<ScopedThread>                   threads(num_threads-1);
            auto block_start = std::begin(data);
            for(size_t i=0;i<(num_threads-1);++i)
            {
                auto block_end=block_start;
                std::advance(block_end,block_size);
                auto sumfn =[=](std::promise<Data::value_type>* promise) 
                            { 
                                promise->set_value(std::accumulate(block_start,block_end,0));
                            };
                promises[i] = std::promise<Data::value_type>();
                threads[i] = ScopedThread(std::thread(std::move(sumfn),&promises[i]));
                block_start=block_end;
            }
            Data::value_type result = std::accumulate(block_start,std::end(data),0);
            for(size_t i=0;i<(num_threads-1);++i)
            {
                result+=promises[i].get_future().get();
            }
            
            std::cout << "promise total sum (single thread)=" << std::accumulate(std::begin(data),std::end(data),0) << std::endl;
            std::cout << "promise total sum (multi thread)=" << result << std::endl;
        }
    }

    {

        std::cout << "ready to crush a stack?" << std::endl;
        std::cin.ignore();
        std::stack<int> s;
        s.push(42);
        if(!s.empty())
        {
            int const value = s.top();
            s.pop();
            std::cout << value << " from the stack";
        }

        std::atomic<bool> done(false);
        ThreadSafeStack<int> ts;
        auto inserter_function = [&done](ThreadSafeStack<int>& ts, unsigned int numelements)
        {
            for(unsigned int i=0;i<numelements;i++)
            {
                ts.push(i);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            done = true;
        };
        auto popper_function = [&done](ThreadSafeStack<int>& ts)
        {
            while(!done || !ts.empty())
            {
                try
                {
                    if(!ts.empty())
                    {
                        int value;
                        ts.pop(value);
                        std::cout << std::this_thread::get_id() << ":popped value=" << value << std::endl;
                    }
                }
                catch(EmptyStackException& )
                {
                    std::cout << std::this_thread::get_id() << ":(ouch!) stack is empty" << std::endl;
                }
            }
            std::cout << std::this_thread::get_id() << ":stack is done" << std::endl;
        };
        ScopedThread thread_popper1(std::thread(popper_function,std::ref(ts)));
        ScopedThread thread_popper2(std::thread(popper_function,std::ref(ts)));
        ScopedThread thread_popper3(std::thread(popper_function,std::ref(ts)));
        ScopedThread thread_inserter1(std::thread(inserter_function,std::ref(ts),100));
        ScopedThread thread_inserter2(std::thread(inserter_function,std::ref(ts),100));
    }

    {
        std::cout << "ready to crush a stack with conditions?" << std::endl;
        std::cin.ignore();

        std::atomic<bool> done(false);

        ThreadSafeStack<int> ts;
        auto inserter_function = [&done](ThreadSafeStack<int>& ts, unsigned int numelements)
        {
            for(unsigned int i=0;i<numelements;i++)
            {
                ts.push(i);
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            done = true;
        };
        auto popper_function = [&](ThreadSafeStack<int>& ts)
        {
            while(!done || !ts.empty())
            {
                try
                {
                    int value;
                    ts.wait_pop(value);
                    std::cout << std::this_thread::get_id() << ":popped value=" << value << std::endl;
                }
                catch(EmptyStackException& )
                {
                    std::cout << std::this_thread::get_id() << ":(ouch!) stack is empty" << std::endl;
                }
            }
            std::cout << std::this_thread::get_id() << ":stack is done" << std::endl;
        };
        ScopedThread thread_popper1(std::thread(popper_function,std::ref(ts)));
        ScopedThread thread_popper2(std::thread(popper_function,std::ref(ts)));
        ScopedThread thread_popper3(std::thread(popper_function,std::ref(ts)));
        ScopedThread thread_inserter1(std::thread(inserter_function,std::ref(ts),100));
        ScopedThread thread_inserter2(std::thread(inserter_function,std::ref(ts),100));

    }

    {
        std::cout << "ready to crush a queue with conditions?" << std::endl;
        std::cin.ignore();

        std::atomic<bool> done(false);

        ThreadSafeQueue<int> ts;
        auto inserter_function = [](ThreadSafeQueue<int>& ts, unsigned int numelements)
        {
            for(unsigned int i=0;i<numelements;i++)
            {
                ts.push(i);
                std::cout << std::this_thread::get_id() << ":pushed value=" << i << std::endl;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
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
        auto thread_popper1 = std::async(popper_function,std::ref(ts));
        auto thread_popper2 = std::async(popper_function,std::ref(ts));
        auto thread_popper3 = std::async(popper_function,std::ref(ts));
        auto thread_inserter1 = std::async(inserter_function,std::ref(ts),100);
        auto thread_inserter2 = std::async(inserter_function,std::ref(ts),100);

        thread_inserter1.wait();
        thread_inserter2.wait();
        done = true;
        ts.notify_and_terminate();
        thread_popper1.wait();
        thread_popper2.wait();
        thread_popper3.wait();

    }

    return EXIT_SUCCESS;
}

