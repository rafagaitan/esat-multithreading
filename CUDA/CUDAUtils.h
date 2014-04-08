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
#include <iostream>
#include <string>
#include <sstream>

#include "cuda_runtime.h"

#include "config.h"

#ifdef CUDAUTILS_USE_EXCEPTIONS
namespace cuda 
{
    struct cuda_exception: public std::exception
    {
        cuda_exception(const std::string& msg):_msg(msg) { }
        virtual const char* what() const override { return _msg.c_str(); }
        std::string _msg;
    };
}
#define CUDAUTIL_THROW true
#else
#define CUDAUTIL_THROW false
#endif
namespace cuda 
{
    template<bool throwException = CUDAUTIL_THROW>
    class Check
    {
    public:
        static bool CUDAError(const cudaError_t& cudaStatus, const std::string& msg)
        {
            if (cudaStatus != cudaSuccess) { 
                if(throwException) throw cuda_exception(msg);
                else return false;
            }
            return true;
        }
        template<typename Arg1>
        static bool CUDAError(const cudaError_t& cudaStatus, const Arg1& arg1) 
        { 
            std::stringstream sstr; sstr << arg1 << ":" << cudaGetErrorString(cudaStatus); 
            return CUDAError(cudaStatus,sstr.str()); 
        }

        template<typename Arg1, typename Arg2>
        static bool CUDAError(const cudaError_t& cudaStatus, const Arg1& arg1, const Arg2& arg2)
        {
            std::stringstream sstr; sstr << arg1 << arg2 << ":" << cudaGetErrorString(cudaStatus); 
            return CUDAError(cudaStatus,sstr.str()); 
        }

        template<typename Arg1, typename Arg2, typename Arg3>
        static bool CUDAError(const cudaError_t& cudaStatus, const Arg1& arg1, const Arg2& arg2, const Arg3& arg3)
        {
            std::stringstream sstr; sstr << arg1 << arg2 << arg3 << ":" << cudaGetErrorString(cudaStatus); 
            return CUDAError(cudaStatus,sstr.str()); 
        }
    };

    class Object
    {
    public:
        Object():_valid(false) { }
        bool valid() { return _valid; }
        virtual ~Object() { }
        template<bool throwException, typename Arg1>
        inline bool validate(const cudaError_t& cudaStatus, const Arg1& arg1)
        {
            _valid =  Check<throwException>::CUDAError(cudaStatus, arg1);
            return _valid;
        }
        template<bool throwException, typename Arg1, typename Arg2>
        inline bool validate(const cudaError_t& cudaStatus, const Arg1& arg1, const Arg2& arg2)
        {
            _valid = Check<throwException>::CUDAError(cudaStatus, arg1, arg2);
            return _valid;
        }
        template<bool throwException, typename Arg1, typename Arg2, typename Arg3>
        inline bool validate(const cudaError_t& cudaStatus, const Arg1& arg1, const Arg2& arg2, const Arg3& arg3)
        {
            _valid =  Check<throwException>::CUDAError(cudaStatus, arg1, arg2, arg3);
            return _valid;
        }
    private:
        bool _valid;
    };

    struct Device: public Object
    {
        Device(unsigned int id): _id(id)
        {
            validate<false>(cudaSetDevice(_id), "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        }
        virtual ~Device()
        {
            // cudaDeviceReset must be called before exiting in order for profiling and
            // tracing tools such as Nsight and Visual Profiler to show complete traces.
             Check<false>::CUDAError(cudaDeviceReset(), "cudaDeviceReset failed!");
        }

        unsigned int _id;
    };

    template<class T>
    struct Buffer: public Object
    {
        Buffer(size_t size):
            _devData(0),
            _size(size)
        {
            validate<false>(cudaMalloc((void**)&_devData, size * sizeof(T)),"cudaMalloc failed!");
        }

        virtual ~Buffer()
        {
            cudaFree(_devData);
        }

        operator T*()
        {
            return _devData;
        }

        operator const T*()
        {
            return _devData;
        }

        __device__ __host__ T& operator[](int i)
        {
            return _devData[i];
        }

        void assign(const T* data)
        {
            validate<CUDAUTIL_THROW>(cudaMemcpy(_devData, data, _size * sizeof(T), cudaMemcpyHostToDevice),"cudaMemcpy to device failed!");
        }

        void retrieve(T* data)
        {
            validate<CUDAUTIL_THROW>(cudaMemcpy(data, _devData, _size * sizeof(T), cudaMemcpyDeviceToHost), "cudaMemcpy from device failed!");
        }

        T*     _devData;
        size_t _size;
    };

}