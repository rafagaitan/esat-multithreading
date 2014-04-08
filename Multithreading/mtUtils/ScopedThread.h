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

#include <thread>
#include <stdexcept>

class ScopedThread
{
    std::thread _t;
public:
    ScopedThread(): _t() { }
    explicit ScopedThread(std::thread t):
        _t(std::move(t))
    {
        if(!_t.joinable())
            throw std::logic_error("Thread already joined");
    }
    explicit ScopedThread(ScopedThread&& st):
        _t(std::move(st._t))
    {
    }

    template<class F, class... Args>
    explicit ScopedThread(F&& f, Args&&... args):
        _t(std::bind(std::forward<F>(f), std::forward<Args>(args)...))
    {
    }

    ScopedThread& operator=(ScopedThread&& st)
    {
        _t = std::move(st._t);
        return *this;
    }
    ScopedThread& operator=(std::thread&& t)
    {
        _t = std::move(t);
        return *this;
    }
    ~ScopedThread()
    {
        if(_t.joinable())
            _t.join();
    }
private:
    ScopedThread(const ScopedThread& rhs); // deleted
    ScopedThread& operator=(const ScopedThread& rhs); // deleted
};

