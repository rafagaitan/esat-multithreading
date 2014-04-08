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
#include <algorithm>
#include <algorithm>

#include <mtUtils/ScopedThread.h>
#include <mtUtils/ThreadSafeStack.h>

void hard_work()
{
    std::cout << "hard work in progress" << std::endl;
    for(unsigned int repeat=0;repeat<10;repeat++)
    {
        const unsigned int size = 100000;
        std::vector<double> data(size);
        for(unsigned int i=0;i<size;i++)
        {
            data[i] = i;
        }
        double addition = 0;
        for(unsigned int i=0;i<size;i++)
        {
            addition += data[i];
        }
        std::cout << "addition=" << addition << std::endl;
    }  
}

void more_hard_work_after_hard_work()
{
    std::cout << "more hard work in progress" << std::endl;
    for(unsigned int repeat=0;repeat<10;repeat++)
    {
        const unsigned int size = 100000;
        std::vector<double> data(size);
        for(unsigned int i=0;i<size;i++)
        {
            data[i] = i;
        }
        double substraction = 0;
        for(unsigned int i=0;i<size;i++)
        {
            substraction -= data[i];
        }
        std::cout << "substraction=" << substraction << std::endl;
    } 
}

class background_task
{
public:
    void operator()() const
    {
        hard_work();
        more_hard_work_after_hard_work();
    }
};

class background_task_with_args
{
public:
    void operator()(int a1, const std::string& a2) const
    {
        std::cout << "t3: a1=" << a1 << " a2=" << a2 << std::endl;
    }
};

class background_task_with_args_in_ctor
{
public:
    background_task_with_args_in_ctor(int a1, const std::string& a2):
        a1(a1), a2(a2) 
    { 
    }
    void operator()() const
    {
        std::cout << "t3: a1=" << a1 << " a2=" << a2 << std::endl;
    }
    int         a1;
    std::string a2;
};

void foo(int a1, const std::string& a2)
{
    std::cout << "t1: a1=" << a1 << " a2=" << a2 << std::endl;
}

void bar(std::vector<unsigned int>& data, unsigned int size)
{
    for(unsigned int i=0;i<size;++i)
    {
        data.push_back(i);
    }
    for(unsigned int i=0;i<size;++i)
    {
        std::cout << "oops?  " << data[i] << ",";
    }
}

void ooops(int param)
{
    std::vector<unsigned int> data;
    std::thread t(bar, std::ref(data), param);
    t.detach();
}

class big_object 
{ 
public:
    void load(const std::string& ) { /** load data **/ } 
    void process() { /** process data **/ } 
    big_object() { }
    ~big_object() { }
};

void process_big_object(std::unique_ptr<big_object> bo) 
{ 
    bo->process(); 
}

std::thread return_thread()
{
    std::thread t(foo, 42, "me transfieren!");
    return t;
}

void join_thread(std::thread input_thread)
{
    input_thread.join();
}

void check_id()
{
    std::cout << "check_id=" << std::this_thread::get_id() << std::endl;
}

std::list<int> some_list;
std::mutex some_mutex;

void add_to_list(int new_value)
{
    std::lock_guard<std::mutex> lock(some_mutex);
    some_list.push_back(new_value);
}

bool contains_in_list(int value_to_find)
{
    std::lock_guard<std::mutex> lock(some_mutex);
    return (std::find(std::begin(some_list), std::end(some_list), value_to_find) != some_list.end());
}


int main(int , char**)
{
    std::cout << "Starting hard_work, ready?";
    std::cin.ignore(); 
    std::thread my_thread(hard_work);
    my_thread.join();

    std::cout << "Starting background_task, ready?";
    std::cin.ignore();
    background_task bt;
    bt();
    std::thread my_thread2(bt);
    //std::thread my_thread2({background_task()});
    my_thread2.join();

    std::cout << "Starting lambda, ready?";
    std::cin.ignore();
    std::thread my_thread3(
        []() 
        {
            hard_work();
            more_hard_work_after_hard_work();
        });
    my_thread3.join();

    std::cout << "ready to crash?";
    std::cin.ignore();
    {
        std::thread my_thread(
        []() 
        {
            hard_work();
            more_hard_work_after_hard_work();
        });
        my_thread.detach();
    }

    {
        std::cout << "ready to pass arguments?" << std::endl;
        std::cin.ignore();
        int a = 42;
        std::thread my_thread(foo, a, "soy un thread!");
        std::string str = "soy otro thread";
        std::thread my_thread2([](int a1, const std::string& a2) { std::cout << "t2: a1=" << a1 << " a2=" << a2 << std::endl; }, a, str);
        std::thread my_thread2_capture([&]() { std::cout << "t2 capturing: a1=" << a << " a2=" << str << std::endl; });
        background_task_with_args bt1;
        std::thread my_thread3(bt1,a, "casi el final...");
        background_task_with_args_in_ctor bt2(a, "el final del todo");
        std::thread my_thread4(bt2);

        my_thread.join();
        my_thread2.join();
        my_thread2_capture.join();
        my_thread3.join();
        my_thread4.join();

    }
    {

        //std::cout << "ready to pass arguments (oops)?" << std::endl;
        //std::cin.ignore();
        //ooops(1000);
        //std::cin.ignore();
    }
    {
        std::unique_ptr<big_object> bo(new big_object);
        bo->load("machodedatos.raw");
        //std::thread process_thread(process_big_object, std::move(bo));
        //process_thread.join();
    }
    {
        std::thread r_thread = return_thread();
        join_thread(std::move(r_thread));
    }
    {
        std::cout << "ready to pass arguments with ScopedThread?" << std::endl;
        std::cin.ignore();
        int a = 42;
        ScopedThread my_thread(std::thread(foo, a, "soy un thread!"));
        std::string str = "soy otro thread";
        ScopedThread my_thread2(std::thread([](int a1, const std::string& a2) { std::cout << "t2: a1=" << a1 << " a2=" << a2 << std::endl; }, a, str));
        ScopedThread my_thread3(std::thread([&]() { std::cout << "t2 capturing: a1=" << a << " a2=" << str << std::endl; }));
        background_task_with_args bt1;
        ScopedThread my_thread4(std::thread(bt1,a, "casi el final..."));
        background_task_with_args_in_ctor bt2(a, "el final del todo");
        ScopedThread my_thread5((std::thread(bt2)));
    }

    {
        std::cout << "ready to count cpus?" << std::endl;
        std::cin.ignore();
        std::cout << "num of cpus:" << std::thread::hardware_concurrency() << std::endl;
    }

    {
        std::cout << "ready to check ids?" << std::endl;
        std::cin.ignore();
        std::cout << "main thread id=" << std::this_thread::get_id() << std::endl;
        std::thread t(check_id);
        std::thread::id id = t.get_id();
        t.join();
        std::cout << "thread id=" << id << std::endl;
    }

    {
        std::cout << "ready to fill and find in a list?" << std::endl;
        std::cin.ignore();
        auto inserter_function = [](unsigned int numelements)
        {
            for(unsigned int i=0;i<numelements;i++)
            {
                add_to_list(i);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        };
        ScopedThread thread_inserter1(std::thread(inserter_function,100));
        ScopedThread thread_inserter2(std::thread(inserter_function,100));
        auto finder_function = [](unsigned int value_to_find) 
        {
            auto attempts = 0u;
            bool found = false;
            while(!found && attempts < 100)
            {
                found = contains_in_list(value_to_find);
                attempts++;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            }
            if(found)
                std::cout << std::this_thread::get_id() << ": Found! after " << attempts << " attempts" << std::endl;
            else
                std::cout << std::this_thread::get_id() << ": not found! :(" << std::endl;
        };
        ScopedThread thread_finder1(std::thread(finder_function,88));
        ScopedThread thread_finder2(std::thread(finder_function,101));
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
            foo(value, "do_something");
        }
        ThreadSafeStack<int> ts;
        auto inserter_function = [](ThreadSafeStack<int>& ts, unsigned int numelements)
        {
            for(unsigned int i=0;i<numelements;i++)
            {
                ts.push(i);
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        };
        auto popper_function = [](ThreadSafeStack<int>& ts)
        {
            while(!ts.empty())
            {
                try
                {
                    int value;
                    ts.pop(value);
                    std::cout << std::this_thread::get_id() << ":popped value=" << value << std::endl;
                    std::this_thread::sleep_for(std::chrono::milliseconds(10));
                }
                catch(EmptyStackException& )
                {
                    break;
                }
            }
            std::cout << std::this_thread::get_id() << ":stack is empty" << std::endl;
        };
        ScopedThread thread_inserter1(std::thread(inserter_function,std::ref(ts),100));
        ScopedThread thread_inserter2(std::thread(inserter_function,std::ref(ts),100));
        ScopedThread thread_popper1(std::thread(popper_function,std::ref(ts)));
        ScopedThread thread_popper2(std::thread(popper_function,std::ref(ts)));
    }

    return EXIT_SUCCESS;
}

