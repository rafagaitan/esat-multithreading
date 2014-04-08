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

#include "CUDAUtils.h"
#include "SumIntVectors.cuh"
#include "device_launch_parameters.h"


__global__ void addKernel(int *c, const int *a, const int *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(int *c, const int *a, const int *b, unsigned int size)
{
    cudaError_t cudaStatus;
    // Choose which GPU to run on, change this on a multi-GPU system.
    cuda::Device device(0);
   // Allocate GPU buffers for three vectors (two input, one output)   
    cuda::Buffer<int> buffer_c(size);
    cuda::Buffer<int> buffer_a(size);
    cuda::Buffer<int> buffer_b(size);
    // Copy input vectors from host memory to GPU buffers.
    buffer_a.assign(a);
    buffer_b.assign(b);
    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(buffer_c, buffer_a, buffer_b);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    cuda::Check<CUDAUTIL_THROW>::CUDAError(cudaStatus, "addKernel launch failed",cudaGetErrorString(cudaStatus));   
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda::Check<CUDAUTIL_THROW>::CUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code ", cudaGetErrorString(cudaStatus), " after launching addKernel!");
    // Copy output vector from GPU buffer to host memory.
    buffer_c.retrieve(c);
    return cudaStatus;
}



