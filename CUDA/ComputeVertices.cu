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
#include <cuda.h>
#include <device_launch_parameters.h>

#include "CUDAUtils.hpp"
#include "ComputeVertices.cuh"

__global__ void computeVertices_kernel(float4* pos, unsigned int width, unsigned int height, float time)
{
    unsigned int x = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int y = blockIdx.x * blockDim.x + threadIdx.x;

    // calculate uv coordinates
    float u = x / (float) width;
    float v = y / (float) height;
    u = u*2.0 - 1.0f;
    v = v*2.0 - 1.0f;

    // calculate simple sine wave pattern
    float freq = 4.0f;
    float w = sin(u*freq + time) * cos(v*freq + time) * 0.5f;

    // write output vertex 
    pos[y*width+x] = make_float4(u, w, v, 1.0f);
}

void computeVertices(float4* pos, unsigned int width, unsigned int height, float time)
{
    dim3 dimBlock(32,32);
    dim3 dimGrid(width / dimBlock.x, height / dimBlock.y);
    //dim3 dimGrid(
    //    (width + dimBlock.x - 1) / dimBlock.x,
    //    (height + dimBlock.y - 1) / dimBlock.y
    //    );
    computeVertices_kernel<<<dimGrid, dimBlock>>>(pos, width,height,time);
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    //cuda::Check<CUDAUTIL_THROW>::CUDAError(cudaStatus, "Kernel launch failed: ",cudaGetErrorString(cudaStatus));   
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda::Check::CUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code ", cudaGetErrorString(cudaStatus), " after launching addKernel!");
}
