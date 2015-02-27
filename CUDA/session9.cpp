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
#include <fstream>

#include <cuda.h>
#include "cuda_runtime.h"

#include <mtUtils/Algorithms.hpp>

#include "CUDAUtils.hpp"
#include "Matrix.hpp"
#include "MatrixMult.cuh"

namespace test_cuda_apis
{
    void test_driver_api()
    {
        CUdevice cuDevice;
        CUcontext cuContext;
        CUmodule cuModule;
        size_t totalGlobalMem;
        CUfunction matrixMult = 0;
        // cuda driver api intialization
        {
            int major = 0, minor = 0;
            char deviceName[100];

            cuda::Check::CUDAError(cuInit(0), "Error intializing cuda");
            int deviceCount;
            cuda::Check::CUDAError(cuDeviceGetCount(&deviceCount), "Error getting the number of devices");
            if (deviceCount <= 0)
            {
                std::cerr << "No devices found" << std::endl;
                return;
            }

            cuDeviceGet(&cuDevice, 0);

            // get compute capabilities and the devicename
            cuda::Check::CUDAError(cuDeviceComputeCapability(&major, &minor, cuDevice), "Error getting Device compute capability");
            cuda::Check::CUDAError(cuDeviceGetName(deviceName, 256, cuDevice), "Error getting device name");
            std::cout << "> GPU Device has SM " << major << "." << minor << " compute capability" << std::endl;

            cuda::Check::CUDAError(cuDeviceTotalMem(&totalGlobalMem, cuDevice), "Error getting totat global memory");
            std::cout << "  Total amount of global memory:     " << (unsigned long long)totalGlobalMem << " bytes" << std::endl;
            std::string tmp = (totalGlobalMem > (unsigned long long)4 * 1024 * 1024 * 1024L) ? "YES" : "NO";
            std::cout << "  64-bit Memory Address:             " << tmp << std::endl;

            cuda::Check::CUDAError(cuCtxCreate(&cuContext, 0, cuDevice), "Error creating the context");
        }
        // Compile and get the function
        {
            std::string module_path = "MatrixMult.cubin";
            std::cout << "> initCUDA loading module: " << module_path << std::endl;

            cuda::Check::CUDAError(cuModuleLoad(&cuModule, module_path.c_str()), "Error loading module");

            cuda::Check::CUDAError(cuModuleGetFunction(&matrixMult, cuModule, "MatrixMultKernelSimpleDriverAPI"), "Error retrieving the function");
        }
        // Call the kernel
        {
            int WIDTH = BLOCK_SIZE;
            int HEIGHT = BLOCK_SIZE;
            std::stringstream text;
            text << "CUDA Matrix Multiplication (" << WIDTH << "x" << WIDTH << ") Simple method Multiplication time";
            HostMatrix<float> M(WIDTH, HEIGHT); M.fillWithRandomData(); //M.print(std::cout); 
            HostMatrix<float> N(WIDTH, HEIGHT); N.fill_diagonal(2); //N.print(std::cout); 
            HostMatrix<float> C(WIDTH, HEIGHT);
            {
                ScopedTimer t(text.str());

                // allocate device memory
                CUdeviceptr d_M;
                cuda::Check::CUDAError(cuMemAlloc(&d_M, M.sizeInBytes()), "Error allocating memory");
                CUdeviceptr d_N;
                cuda::Check::CUDAError(cuMemAlloc(&d_N, N.sizeInBytes()), "Error allocating memory");

                // copy host memory to device
                cuda::Check::CUDAError(cuMemcpyHtoD(d_M, M, M.sizeInBytes()), "Error uploading memory to device");
                cuda::Check::CUDAError(cuMemcpyHtoD(d_N, N, N.sizeInBytes()), "Error uploading memory to device");

                // allocate device memory for result
                CUdeviceptr d_C;
                cuda::Check::CUDAError(cuMemAlloc(&d_C, C.sizeInBytes()), "Error allocating memory");


                dim3 block(BLOCK_SIZE, BLOCK_SIZE, 1);
                dim3 grid(C.width_ / BLOCK_SIZE, C.height_ / BLOCK_SIZE, 1);
                void *args[6] = { &d_M, &d_N, &d_C, &WIDTH, &WIDTH, &WIDTH};

                // new CUDA 4.0 Driver API Kernel launch call
                cuda::Check::CUDAError(cuLaunchKernel(
                    matrixMult,                                     // Selected kernel function
                    grid.x, grid.y, grid.z,                         // grid config 
                    block.x, block.y, block.z,                      // block config
                    2 * BLOCK_SIZE*BLOCK_SIZE*sizeof(float),        
                    NULL, args, NULL), "Error executing Kernel");

                cuda::Check::CUDAError(cuMemcpyDtoH((void *)C, d_C, C.sizeInBytes()),"Error downloading memory to host");
            }
            C.print(std::cout);
        }

        cuCtxDestroy(cuContext);
    }

    void test_runtime_api()
    {
        try
        {
            cuda::Device device(0);
            for(unsigned int size = 1; size <= 64; size*=2)
            {
                const int WIDTH  = size*BLOCK_SIZE;
                const int HEIGHT = size*BLOCK_SIZE;
                HostMatrix<float> M(WIDTH,HEIGHT); M.fillWithRandomData(); //M.print(std::cout); 
                HostMatrix<float> N(WIDTH,HEIGHT); N.fill_diagonal(2); //N.print(std::cout); 
                HostMatrix<float> P_simple(WIDTH,HEIGHT);
                {
                    std::stringstream text;
                    text << "CUDA Matrix Multiplication (" << WIDTH << "x" << WIDTH << ") Simple method Multiplication time";
                    ScopedTimer t(text.str());
                    MatrixMult(M,N,P_simple, false);
                }
                HostMatrix<float> P_complex(WIDTH,HEIGHT);
                {
                    std::stringstream text;
                    text << "CUDA Matrix Multiplication (" << WIDTH << "x" << WIDTH << ") Complex method Multiplication time";
                    ScopedTimer t(text.str());
                    MatrixMult(M,N,P_complex, true);
                }
                HostMatrix<float> P_cpu(WIDTH,HEIGHT);
                {
                    std::stringstream text;
                    text << "CPU Matrix Multiplication (" << WIDTH << "x" << WIDTH << ")Multiplication time";
                    ScopedTimer t(text.str());
                    P_cpu = M*N;
                }
                std::cout << "------------------------------------------------------" << std::endl;
                //std::cout << "------------------------CUDA SIMPLE -------------------------" << std::endl;
                //P_simple.print(std::cout);
                //std::cout << "----------------------CUDA COMPLEX ---------------------------" << std::endl;
                //P_complex.print(std::cout);
                //std::cout << "----------------------CPU ---------------------------" << std::endl;
                //P_cpu.print(std::cout);
            }
        }
        catch(cuda::cuda_exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }
}


int main()
{
    {
        try
        {
            std::cout << "Test Driver API, ready?";
            std::cin.ignore();
            test_cuda_apis::test_driver_api();
        }
        catch (cuda::cuda_exception& e)
        {
            std::cerr << e.what() << std::endl;
        }
    }
    {
        std::cout << "Test Runtime API, ready?";
        std::cin.ignore();
        test_cuda_apis::test_runtime_api();
    }
    {

    }
    return 0;
}

