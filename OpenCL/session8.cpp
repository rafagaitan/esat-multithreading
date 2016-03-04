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
#define NOMINMAX
#include <iostream>
#include <string>
#include <cmath>
#include <algorithm>
#include <functional>
#include <memory>

#include <mtUtils/Algorithms.hpp>

#include "Config.hpp"

#include "cl.hpp"

#include "OpenCLUtils.hpp"

namespace test_opencl
{
    template<typename T>
    void printV(std::ostream& out, const T& a, const unsigned int size, unsigned int initElement, unsigned int numElements)
    {
        unsigned int end = std::min(size, initElement + numElements);
        if (initElement == 0)
            out << "{ ";
        else
            out << " ... ";
        for (unsigned int i = initElement; i < end - 1; i++)
        {
            out << a[i] << ",";
        }
        out << a[end - 1];
        if (end == size)
            std::cout << " } ";
        else
            std::cout << " ... ";
    };
    
    class TestKernels
    {
    public:
        cl::Platform platform;
        cl::Context context;
        std::vector<cl::Device> devices;
        cl::Program program;
        cl_int err;
        bool printResults;
        TestKernels(bool print)
        {
            err = CL_SUCCESS;
            cl::Platform::get(&platform);
            
            context = cl::Context(CL_DEVICE_TYPE_GPU);
            
            devices = context.getInfo<CL_CONTEXT_DEVICES>();
            std::string sumIntSrc = opencl::loadKernel("kernels.cl");
            cl::Program::Sources source(1, std::make_pair(sumIntSrc.c_str(),sumIntSrc.size()));
            program = cl::Program(context, source);
            program.build(devices);
            
            printResults = print;
        }
        void multVectorsOpenCL(int arraySize, const int *a, const int *b, int *c)
        {
            _multVectorsOpenCL(c, a, b, arraySize);
        }
        
        void addVectorsCPU(const unsigned int arraySize)
        {
            std::vector<float> a(arraySize);
            std::vector<float> b(arraySize);
            std::vector<float> c(arraySize);
            
            for (unsigned int i = 0; i < arraySize; i++)
            {
                a[i] = static_cast<float>(sin(i)*sin(i));
                b[i] = static_cast<float>(cos(i)*cos(i));
                c[i] = a[i] + b[i];
            }
            
            if (printResults)
            {
                printV(std::cout, a, a.size(),  0, 20); printV(std::cout, a, a.size(), arraySize - 20, 20);
                std::cout << "    +    ";
                printV(std::cout, b, b.size(), 0, 20); printV(std::cout, b, b.size(), arraySize - 20, 20);
                std::cout << "   =   ";
                printV(std::cout, c, c.size(), 0, 20); printV(std::cout, c, c.size(), arraySize - 20, 20);
                std::cout << std::endl;
            }
        }
        void fillThenAddVectorsOpenCL(const unsigned int arraySize)
        {
            std::vector<float> a(arraySize);
            std::vector<float> b(arraySize);
            std::vector<float> c(arraySize);
            
            _fillVectorsOpenCL(a.data(),b.data(), arraySize);
            _addVectorsOpenCL(c.data(),a.data(),b.data(), arraySize);
            
            if (printResults)
            {
                printV(std::cout, a, a.size(), 0, 20); printV(std::cout, a, a.size(), arraySize - 20, 20);
                std::cout << "    +    ";
                printV(std::cout, b, b.size(), 0, 20); printV(std::cout, b, b.size(), arraySize - 20, 20);
                std::cout << "   =   ";
                printV(std::cout, c, c.size(),  0, 20); printV(std::cout, c, c.size(), arraySize - 20, 20);
                std::cout << std::endl;
            }
        }
        
        void fillAndAddVectorsOpenCL(const unsigned int arraySize)
        {
            std::vector<float> a(arraySize);
            std::vector<float> b(arraySize);
            std::vector<float> c(arraySize);
            
            _fillAndAddVectorsOpenCL(c.data(),a.data(),b.data(), arraySize);
            
            if (printResults)
            {
                printV(std::cout, a, a.size(), 0, 20); printV(std::cout, a, a.size(), arraySize - 20, 20);
                std::cout << "    +    ";
                printV(std::cout, b, b.size(), 0, 20); printV(std::cout, b, b.size(), arraySize - 20, 20);
                std::cout << "   =   ";
                printV(std::cout, c, c.size(),  0, 20); printV(std::cout, c, c.size(), arraySize - 20, 20);
                std::cout << std::endl;
            }
        }
    private:
        
        bool executeKernel(std::function<void ()> func)
        {
            try { func(); }
            catch (cl::Error err)
            {
                std::cerr
                << "ERROR: "
                << err.what()
                << "("
                << err.err()
                << ")"
                << std::endl;
                return false;
            }
            return true;
        }
        // Helper function for using OpenCL to add vectors in parallel.
        bool _multVectorsOpenCL(int *c, const int *a, const int *b, unsigned int size)
        {
            return executeKernel([&,this](){
                cl::Kernel kernel(program, "multKernel", &err);
                
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

                queue.enqueueReadBuffer(buffer_c,true,0,size*sizeof(float),c);
            });
        }
        bool _fillVectorsOpenCL(float *a, float *b, unsigned int size)
        {
            return executeKernel([&,this](){
                cl::Kernel kernel(program, "fillKernel", &err);
                
                // Allocate and Copy vectors from host memory to GPU buffers (two input, one output)
                cl::Buffer buffer_a(context, CL_MEM_WRITE_ONLY, size*sizeof(float));
                cl::Buffer buffer_b(context, CL_MEM_WRITE_ONLY, size*sizeof(float));
                
                kernel.setArg(0,buffer_a);
                kernel.setArg(1,buffer_b);
                
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
                
                queue.enqueueReadBuffer(buffer_a,true,0,size*sizeof(float),a);
                queue.enqueueReadBuffer(buffer_b,true,0,size*sizeof(float),b);
            });
        }
        bool _addVectorsOpenCL(float *c, const float *a, const float *b, unsigned int size)
        {
            return executeKernel([&,this](){
                cl::Kernel kernel(program, "addKernel", &err);
                
                // Allocate and Copy vectors from host memory to GPU buffers (two input, one output)
                cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY, size*sizeof(float));
                cl::Buffer buffer_a(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size*sizeof(float), (void*)a);
                cl::Buffer buffer_b(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, size*sizeof(float), (void*)b);
                
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
            });
        }
        bool _fillAndAddVectorsOpenCL(float *c, float *a, float *b, unsigned int size)
        {
            return executeKernel([&,this](){
                cl::Kernel kernel(program, "fillAndAddKernel", &err);
                
                // Allocate and Copy vectors from host memory to GPU buffers (two input, one output)
                cl::Buffer buffer_a(context, CL_MEM_WRITE_ONLY, size*sizeof(float));
                cl::Buffer buffer_b(context, CL_MEM_WRITE_ONLY, size*sizeof(float));
                cl::Buffer buffer_c(context, CL_MEM_WRITE_ONLY, size*sizeof(float));
                
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
                
                queue.enqueueReadBuffer(buffer_a,true,0,size*sizeof(int),a);
                queue.enqueueReadBuffer(buffer_b,true,0,size*sizeof(int),b);
                queue.enqueueReadBuffer(buffer_c,true,0,size*sizeof(int),c);
            });
        }
    };

    void trad_mult(int arraySize, const int *a, const int *b, int *c)
    {
        for (int i=0; i<arraySize; i++)
            c[i] = a[i] * b[i];
    }

    cl_context create_context_c()
    {
        cl_uint numPlatforms = 0;
        cl_uint numDevices = 0;
        //Get the platform ID
        
        
        clGetPlatformIDs(0, NULL, &numPlatforms);
        if (numPlatforms == 0)
            return 0; 

        std::unique_ptr<cl_platform_id[]> platforms(new cl_platform_id[numPlatforms]);
        clGetPlatformIDs(numPlatforms, platforms.get(), NULL);
       
        
        // Get the first GPU device associated with the platform
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, 0, NULL, &numDevices);
        std::unique_ptr<cl_device_id[]> devices(new cl_device_id[numDevices]);
        clGetDeviceIDs(platforms[0], CL_DEVICE_TYPE_GPU, numDevices, devices.get(), NULL);
        //Create an OpenCL context for the GPU device
        cl_context context;
        context = clCreateContext(NULL, 1, &devices[0], NULL, NULL, NULL);

        return context;
    }

    cl::Context create_context_cpp()
    {
        cl::Platform platform;
        cl::Platform::get(&platform);
        return cl::Context(CL_DEVICE_TYPE_GPU);
    }
    
    int num_opencl_plaforms()
    {
        std::vector<cl::Platform> platforms;
        cl::Platform::get(&platforms);
        if (platforms.size() == 0) {
            std::cout << "Platform size 0\n";
            return -1;
        }
        return platforms.size();
    }

    void compile_kernel_c(cl_context context, cl_mem a_buffer, cl_mem b_buffer, cl_mem d_buffer, int N)
    {
        // Build program object and set up kernel arguments
        const char* source = "__kernel void dp_mul(__global const float *a, \n"
                                "                     __global const float *b, \n"
                                "                     __global float *c, \n"
                                "                     int N) \n"
                                "{ \n"
                                " int id = get_global_id(0); \n"
                                " if (id < N) \n"
                                "  c[id] = a[id] * b[id]; \n"
                                "} \n";
        cl_program program = clCreateProgramWithSource(context, 1, &source, NULL, NULL);
        clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        cl_kernel kernel = clCreateKernel(program, "dp_mul", NULL);
        clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&a_buffer);
        clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&b_buffer);
        clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&d_buffer);
        clSetKernelArg(kernel, 3, sizeof(int), (void*)&N);
    }
}


int main()
{
    const int arraySize = 5;
    const int a[arraySize] = { 1, 2, 3, 4, 5 };
    const int b[arraySize] = { 10, 20, 30, 40, 50 };
    const float a_f[arraySize] = { 1, 2, 3, 4, 5 };
    const float b_f[arraySize] = { 10, 20, 30, 40, 50 };
    int c[arraySize]   = { 0 };

    {
        std::cout << "Creating a context in c?" << std::endl;
        cl_context ctx_c = test_opencl::create_context_c();
        clReleaseContext(ctx_c);

        std::cout << "Creating a context in c++?" << std::endl;
        cl::Context ctx_cpp = test_opencl::create_context_cpp();
    }
    {
        std::cout << "Num Plaforms:" << test_opencl::num_opencl_plaforms() << std::endl;
    }
    {
        std::cout << "Compiling a program in c?" << std::endl;
        cl_context ctx_c = test_opencl::create_context_c();
        cl_mem a_buffer = clCreateBuffer(ctx_c, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, arraySize*sizeof(float), (void*)a_f, NULL);
        cl_mem b_buffer = clCreateBuffer(ctx_c, CL_MEM_READ_ONLY| CL_MEM_COPY_HOST_PTR, arraySize*sizeof(float), (void*)b_f, NULL);
        cl_mem d_buffer = clCreateBuffer(ctx_c, CL_MEM_READ_WRITE, arraySize*sizeof(float), NULL, NULL);
        test_opencl::compile_kernel_c(ctx_c, a_buffer, b_buffer, d_buffer, 3);
        
        //// Create a command-queue for a specific device
        //cl_command_queue cmd_queue = clCreateCommandQueue(ctx_c, device_id, 0, NULL);
        
        //// Write to buffer object from host memory
        //clEnqueueWriteBuffer(cmd_queue, a_buffer, CL_FALSE, 0, arraySize*sizeof(float), a_f, 0, NULL, NULL);
        
        //// Set number of work-items in a work-group
        //size_t localWorkSize = 256;
        //int numWorkGroups= (N + localWorkSize - 1) / localWorkSize; // round up
        //size_t globalWorkSize= numWorkGroups* localWorkSize;// must be evenly divisible by localWorkSize
        //clEnqueueNDRangeKernel(cmd_queue, kernel, 1, NULL,
        //                        &globalWorkSize, &localWorkSize, 0, NULL, NULL);
        
        //// Read from buffer object to host memory
        //clEnqueueReadBuffer(cmd_queue, d_buffer, CL_TRUE, 0, arraySize*sizeof(float), c_f, 0, NULL, NULL);
        
        clReleaseMemObject(a_buffer);
        clReleaseMemObject(b_buffer);
        clReleaseMemObject(d_buffer);
        clReleaseContext(ctx_c);
        
    }
    
    {
        std::cout << "Test traditional mult" << std::endl;
        test_opencl::trad_mult(arraySize, a, b, c);
        std::cout << "{1,2,3,4,5} + {10,20,30,40,50} = {" << 
            c[0] << "," << 
            c[1] << "," <<
            c[2] << "," <<
            c[3] << "," << 
            c[4] << "}" << std::endl;
    }
    {
        const unsigned int numTests = 10;
        const unsigned int iniMult = 1;
        const unsigned int maxMult = 10000;
        const bool printResults = false;
        test_opencl::TestKernels testKernels(printResults);
        {
            std::cout << "Test OpenCL mult" << std::endl;
            testKernels.multVectorsOpenCL(arraySize, a, b, c);
            std::cout << "{1,2,3,4,5} + {10,20,30,40,50} = {" << 
                c[0] << "," << 
                c[1] << "," <<
                c[2] << "," <<
                c[3] << "," << 
                c[4] << "}" << std::endl;
        }
        std::cout << "Test AddVectorsCPU, running ..." << std::endl;
        for (unsigned int mult = iniMult; mult <= maxMult; mult *= 10)
        {
            const int arraySize = 1024 * mult;
            std::cout << "Array size: " << arraySize << std::endl;
            ScopedTimer timer("time for AddVectorsCPU", numTests);
            for (unsigned int i = 0; i < numTests; i++)
            {
                testKernels.addVectorsCPU(arraySize);
            }
        }
        std::cout << "Test fillThenAddVectorsOpenCL, running ..." << std::endl;
        for (unsigned int mult = iniMult; mult <= maxMult; mult *= 10)
        {
            const int arraySize = 1024 * mult;
            std::cout << "Array size: " << arraySize << std::endl;
            ScopedTimer timer("time for fillThenAddVectorsOpenCL", numTests);
            for (unsigned int i = 0; i < numTests; i++)
            {
                testKernels.fillThenAddVectorsOpenCL(arraySize);
            }
        }
        std::cout << "Test fillAndAddVectorsOpenCL, running ..." << std::endl;
        for (unsigned int mult = iniMult; mult <= maxMult; mult *= 10)
        {
            const int arraySize = 1024 * mult;
            std::cout << "Array size: " << arraySize << std::endl;
            ScopedTimer timer("time for fillAndAddVectorsOpenCL", numTests);
            for (unsigned int i = 0; i < numTests; i++)
            {
                testKernels.fillAndAddVectorsOpenCL(arraySize);
            }
        }
    }
 

    return 0;
}

