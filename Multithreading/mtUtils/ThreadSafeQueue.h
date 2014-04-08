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

#include <exception>
#include <memory>
#include <queue>
#include <condition_variable>

struct EmptyQueueException: std::exception
{
    const char* what() const throw() { return "queue is empty"; } 
};

template<typename T>
class ThreadSafeQueue
{
private:
    std::queue<T>           _data;
    bool                    _terminate;
    std::condition_variable _data_cond;
    mutable std::mutex      _mtx;
public:
    ThreadSafeQueue():_data(),_terminate(false),_data_cond(),_mtx() { }
    ThreadSafeQueue(const ThreadSafeQueue& other)
    {
        std::lock_guard<std::mutex> lock(other._mtx);
        _data=other._data;
    }
    std::shared_ptr<T> pop()
    {
        std::lock_guard<std::mutex> lock(_mtx);
        return _pop();
    }
    std::shared_ptr<T> wait_pop()
    {
        std::unique_lock<std::mutex> lock(_mtx);
        _data_cond.wait(lock, [this]() { return _terminate || !_data.empty(); });
        return _pop();
    }
    void push(const T& new_value)
    {
        std::lock_guard<std::mutex> lock(_mtx);
        _data.push(new_value);
        _data_cond.notify_one();
    }
    void push(T&& new_value)
    {
        std::lock_guard<std::mutex> lock(_mtx);
        _data.push(std::move(new_value));
        _data_cond.notify_one();
    }
    bool pop(T& value)
    {
        std::lock_guard<std::mutex> lock(_mtx);
        return _pop(value);
    }

    bool wait_pop(T& value)
    {
        std::unique_lock<std::mutex> lock(_mtx);
        _data_cond.wait(lock, [this]() { return _terminate || !_data.empty(); });
        return _pop(value);
    }

    template<typename _Dt>
    bool wait_pop(T& value, _Dt wait_time)
    {
        std::unique_lock<std::mutex> lock(_mtx);
        _data_cond.wait_for(lock, wait_time, [this]() { return _terminate || !_data.empty(); });
        return _pop(value);
    }

    bool empty() const
    {
        std::lock_guard<std::mutex> lock(_mtx);
        return _data.empty();
    }

    void notify()
    {
        std::lock_guard<std::mutex> lock(_mtx);
        _data_cond.notify_all();
    }

    void notify_and_terminate()
    {
        std::lock_guard<std::mutex> lock(_mtx);
        _terminate = true;
        _data_cond.notify_all();
    }
private:
    ThreadSafeQueue& operator=(const ThreadSafeQueue&); // deleted

    bool _pop(T& value)
    {
        if(_data.empty()) return false;
        value = std::move(_data.front());
        _data.pop();
        return true;
    }
    std::shared_ptr<T> _pop()
    {
        if(_data.empty()) throw EmptyQueueException();
        std::shared_ptr<T> const res(std::make_shared<T>(_data.front()));
        _data.pop();
        return res;
    }

};

