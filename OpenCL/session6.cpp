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
#include <string>

#include "config.h"

#if defined(__APPLE__) || defined(__MACOSX)
#include <OpenCL/cl.hpp>
#else
#include "cl.hpp"
#endif

#include "OpenCLUtils.h"

namespace test_runtime_api
{
    // Helper function for using OpenCL to add vectors in parallel.
    cl_int addWithOpenCL(int *c, const int *a, const int *b, unsigned int size)
    {
        cl_int err = CL_SUCCESS;
        try
        {
            //std::vector<cl::Platform> platforms;
            //cl::Platform::get(&platforms);
            //if (platforms.size() == 0) {
            //    std::cout << "Platform size 0\n";
            //    return -1;
            //}

            cl::Platform platform;
            cl::Platform::get(&platform);

            cl::Context context(CL_DEVICE_TYPE_GPU); 

            std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();
            std::string sumIntSrc = opencl::loadKernel((std::string(RESOURCES_PATH) + 
                                                        std::string("/kernels.cl")).c_str());
            cl::Program::Sources source(1, std::make_pair(sumIntSrc.c_str(),sumIntSrc.size()));
            cl::Program program_ = cl::Program(context, source);
            program_.build(devices);

            cl::Kernel kernel(program_, "addKernel", &err);

            // Allocate and Copy vectors from host memory to GPU buffers (two input, one output) 
            cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY, size*sizeof(int));
            cl::Buffer buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size*sizeof(int), (void*)a);
            cl::Buffer buffer_b(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size*sizeof(int), (void*)b);

            kernel.setArg(0,buffer_c);
            kernel.setArg(1,buffer_a);
            kernel.setArg(2,buffer_b);

            cl::Event event;
            cl::CommandQueue queue(context, devices[0], 0, &err);
            queue.enqueueNDRangeKernel(
                kernel, 
                cl::NullRange, 
                cl::NDRange(size), // global
                cl::NullRange,     // local
                NULL,
                &event); 

            event.wait();

            queue.enqueueReadBuffer(buffer_c,true,0,size*sizeof(int),c);


 
        }
        catch (cl::Error err) 
        {
            std::cerr 
                << "ERROR: "
                << err.what()
                << "("
                << err.err()
                << ")"
                << std::endl;
        }

        return err;
    }

    void test_opencl()
    {
        const int arraySize = 5;
        const int a[arraySize] = { 1, 2, 3, 4, 5 };
        const int b[arraySize] = { 10, 20, 30, 40, 50 };
        int c[arraySize] = { 0 };

        addWithOpenCL(c, a, b, arraySize);

        std::cout << "{1,2,3,4,5} + {10,20,30,40,50} = {" << 
            c[0] << "," << 
            c[1] << "," <<
            c[2] << "," <<
            c[3] << "," << 
            c[4] << "}" << std::endl;


    }
}


int main()
{
    {
        std::cout << "Test CPP OpenCL API, ready?";
        std::cin.ignore();
        test_runtime_api::test_opencl();
    }
    return 0;
}

