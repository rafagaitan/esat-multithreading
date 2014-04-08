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
#include <stack>
#include <condition_variable>

struct EmptyStackException: std::exception
{
    const char* what() const throw() { return "stack is empty"; } 
};
template<typename T>
class ThreadSafeStack
{
private:
    std::stack<T>           _data;
    std::condition_variable _data_cond;
    mutable std::mutex      _mtx;
public:
    ThreadSafeStack():_data(),_data_cond(),_mtx() { }
    ThreadSafeStack(const ThreadSafeStack& other)
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
        std::lock_guard<std::mutex> lock(_mtx);
        _data_cond.wait(lock, [this]() { return !_data.empty(); });
        return _pop;
    }
    void push(T new_value)
    {
        std::lock_guard<std::mutex> lock(_mtx);
        _data.push(new_value);
        _data_cond.notify_one();
    }
    void pop(T& value)
    {
        std::lock_guard<std::mutex> lock(_mtx);
        _pop(value);
    }
    void wait_pop(T& value)
    {
        std::unique_lock<std::mutex> lock(_mtx);
        _data_cond.wait_for(lock, std::chrono::milliseconds(30), [this]() { return !_data.empty(); });
        _pop(value);
    }
    bool empty() const
    {
        std::lock_guard<std::mutex> lock(_mtx);
        return _data.empty();
    }
private:
    ThreadSafeStack& operator=(const ThreadSafeStack&); // deleted

    void _pop(T& value)
    {
        if(_data.empty()) throw EmptyStackException();
        value = _data.top();
        _data.pop();
    }
    std::shared_ptr<T> _pop()
    {
        if(_data.empty()) throw EmptyStackException();
        std::shared_ptr<T> const res(std::make_shared<T>(_data.top()));
        _data.pop();
        return res;
    }

};

