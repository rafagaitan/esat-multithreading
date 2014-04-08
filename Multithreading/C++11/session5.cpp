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
#include <mtUtils/Algorithms.h>

template <typename T>
std::vector<T> fillIntVectorData(size_t numelements = 10, bool sequential = true)
{
    std::vector<T> data;
    data.reserve(numelements);
    for(auto i=0u; i< numelements; i++)
        data.push_back(i);

    if(!sequential)
    {
        std::random_device rd;
        std::mt19937 g(rd());
        std::shuffle(std::begin(data),std::end(data),g);
    }

    return data;
}



struct ITask: public std::enable_shared_from_this<ITask>
{
    virtual ~ITask() { } 
    virtual void execute() = 0;
    virtual void cancel() = 0;
    virtual void update() = 0;
};

struct event_data
{
    enum event_type
    {
        quit,
        start_task,
        stop_task,
        update_task,
        task_complete
    };
    explicit event_data(event_type type): 
                        type(type) { }   
    explicit event_data(event_type type, std::shared_ptr<ITask> task): 
                        type(type), task(task) { } 
    event_type            type;
    std::shared_ptr<ITask> task;
};

ThreadSafeQueue<event_data> eventsQueue;

class Task: public ITask
{
public:
    Task():_taskCancelled(false) { }
    virtual  ~Task() { }
    virtual void execute()
    {
        executeImpl();
        onTaskCompleted();
    }
    virtual void cancel() { _taskCancelled = true; } 
  
protected:
    virtual void onTaskCompletedImpl() = 0;
    virtual void executeImpl() = 0;
    void onTaskCompleted()
    {
        onTaskCompletedImpl();
        eventsQueue.push(event_data(event_data::task_complete, shared_from_this()));
    }
private:
    std::atomic<bool> _taskCancelled;
};

std::shared_ptr<event_data> get_event()
{
    return eventsQueue.wait_pop();
};

void process(ThreadPool& pool, std::shared_ptr<event_data> event)
{
    struct TaskRunner
    {
        void operator() (std::shared_ptr<ITask> task)
        {
            task->execute();
        }
    };
    switch(event->type)
    {
    case event_data::start_task:
        pool.enqueue((TaskRunner()),event->task);
        break;
    case event_data::stop_task:
        event->task->cancel();
        break;
    case event_data::update_task:
        event->task->update();
        break;
    case event_data::task_complete:
        event->task->update();
        break;
    case event_data::quit:
        break;
    }
};

void gui_thread() 
{
    ThreadPool pool(4);
    while(true)
    {
        std::shared_ptr<event_data> event = get_event();
        if(event->type == event_data::quit)
            break;
            
        process(pool, event);
    }
}

class ExecuteTask: public Task
{
public:
    ExecuteTask(const std::string& command): command(command) { }
    virtual ~ExecuteTask() { }
    virtual void executeImpl()
    {
        if(command == "test")
        {
            progress.store(0);
            size_t numElements = 1000;
            for(unsigned int i=1;i<=5;i++)
            {
                std::vector<size_t> data = fillIntVectorData<size_t>(numElements*i);
                std::cout << "Num Elements:" << numElements*i << std::endl;
                {
                    ScopedTimer t("std::for_each");
                    std::for_each(data.begin(), data.end(), [](size_t& v){ v*=2; });
                }
                progress.store(i*100/2.5);
                eventsQueue.push(event_data(event_data::update_task, shared_from_this()));
                {
                    ScopedTimer t("std::parallel_for_each");
                    parallel_for_each(data.begin(), data.end(), [](size_t& v){ v*=2; });
                }
                progress.store(i*100/5);
                eventsQueue.push(event_data(event_data::update_task, shared_from_this()));
                numElements*=10;
            }
        }
        else
        {
            std::cout << "command not found: " << command << std::endl;
        }
    }
    virtual void update()
    {
        std::cout << "Execute task progress:" << progress.load() << std::endl;
    }
protected:
    virtual void onTaskCompletedImpl()
    {
        std::cout << "Execute Task completed" << std::endl;
    }
    std::string command;
    std::atomic<int> progress;
};

class CommandTask: public Task
{
public:
    CommandTask() { }
    virtual ~CommandTask() { }
    virtual void executeImpl()
    {
        while(true)
        {
            char line[256];
            std::cout << "$ >";
            std::cin.getline(line,256,'\n');
            if(std::string(line) == std::string("quit"))
            {
                eventsQueue.push(event_data(event_data::quit));
                break;
            }
            else
                eventsQueue.push(event_data(event_data::start_task, std::make_shared<ExecuteTask>(line)));
        }
    }
    virtual void update() { /** nothing to do **/ }
protected:
    virtual void onTaskCompletedImpl()
    {
        std::cout << "CommandTask finished" << std::endl;
    }
};

int main(int , char**)
{
    ThreadPool::instance();
    bool enableSpawn = true;
    bool enableParallelFind = true;
    bool enableParallelForEach = true;
    bool enableParallelGUIEvents = true;
    if(enableSpawn)
    {
        std::cout << "ready to test spawn threads?" << std::endl;
        std::cin.ignore();
        std::vector<std::future<void>> futures;

        ThreadPool pool;

        for(unsigned int i=0;i<20;i++)
            futures.push_back(pool.enqueue([](int v) { std::cout << "spawned thread: " << v << std::endl; }, i));

        std::for_each(futures.begin(),futures.end(), [](std::future<void>& f) { f.wait(); }); 
    }
    if(enableParallelForEach)
    {
        std::cout << "ready to test parallel for each?" << std::endl;
        std::cin.ignore();
        size_t numElements = 1000;
        for(unsigned int i=1;i<=5;i++)
        {
            std::vector<size_t> data = fillIntVectorData<size_t>(numElements*i);
            std::cout << "Num Elements:" << numElements*i << std::endl;
            {
                ScopedTimer t("std::for_each");
                std::for_each(data.begin(), data.end(), [](size_t& v){ v*=2; });
            }
            {
                ScopedTimer t("std::parallel_for_each");
                parallel_for_each(data.begin(), data.end(), [](size_t& v){ v*=2; });
            }
            numElements*=10;
        }
    }
    if(enableParallelFind)
    {
        std::cout << "ready to test find algorithms?" << std::endl;
        std::cin.ignore();
        const size_t dataSize = 10000000;
        std::vector<size_t> data = fillIntVectorData<size_t>(dataSize, true);
        size_t values_to_find[] = {5647,(dataSize-1),0,dataSize+100};
        for(auto value = std::begin(values_to_find); value != std::end(values_to_find); ++value)
        {
            std::cout << "finding value: " << *value << std::endl;
            {
                ScopedTimer t("parallel_find_thread");
                auto find_elem = parallel_find_thread(data.begin(), data.end(), *value);
                if(find_elem == data.end())
                    std::cout << "not found" << std::endl;
                else
                    std::cout << "found" << std::endl;
            }
            {
                ScopedTimer t("parallel_find_pool");
                auto find_elem = parallel_find_pool(data.begin(), data.end(), *value);
                if(find_elem == data.end())
                    std::cout << "not found" << std::endl;
                else
                    std::cout << "found" << std::endl;
            }
            #ifndef __APPLE__
            {
                ScopedTimer t("parallel_find_async");
                auto find_elem = parallel_find_async(data.begin(), data.end(), *value);
                if(find_elem == data.end())
                    std::cout << "not found" << std::endl;
                else
                    std::cout << "found" << std::endl;
            }
            #endif
            {
                ScopedTimer t("std::find");
                auto find_elem = std::find(data.begin(), data.end(), *value);
                if(find_elem == data.end())
                    std::cout << "not found" << std::endl;
                else
                    std::cout << "found" << std::endl;
            }
        }
    }

    if(enableParallelGUIEvents)
    {
        std::cout << "ready to test parallel GUI events?" << std::endl;
        std::cin.ignore();

        eventsQueue.push(event_data(event_data::start_task, std::make_shared<CommandTask>()));
        gui_thread();
    }
    return EXIT_SUCCESS;
}

