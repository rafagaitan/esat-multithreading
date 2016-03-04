#define NOMINMAX
#include <algorithm>
#include <iostream>

#include "CUDAUtils.hpp"
#include "SumIntVectors.cuh"
#include "device_launch_parameters.h"


__global__ void addKernel(float *c, const float *a, const float *b)
{
    int i = threadIdx.x;
    c[i] = a[i] + b[i];
}

__global__ void addKernelV2(float *c, const float *a, const float *b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    c[i] = a[i] + b[i];
}

__global__ void fillKernel(float *a)
{
    int i = threadIdx.x;
    a[i] = sin((double)i)*sin((double)i);
}

__global__ void fillKernelV2(float *a, float* b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    a[i] = sin((double)i)*sin((double)i);
    b[i] = cos((double)i)*cos((double)i);
}

__global__ void fillAndAddKernelV2(float* c, float *a, float* b)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    a[i] = sin((double)i)*sin((double)i);
    b[i] = cos((double)i)*cos((double)i);
    c[i] = a[i] + b[i];
}

const unsigned int BLOCK_SIZE = 512;

cudaError_t fillWithCuda(float *a, unsigned int size)
{
    cuda::Buffer<float> buffer_a(size);
    buffer_a.assign(a);

    fillKernel<<<1, size>>>(buffer_a);

    cudaError_t cudaStatus = cudaGetLastError();

    cuda::Check::CUDAError(cudaStatus, "fillKernel launch failed", cudaGetErrorString(cudaStatus));

    cuda::Check::CUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code ", cudaGetErrorString(cudaStatus), " after launching fillKernel!");

    buffer_a.retrieve(a);
    return cudaStatus;
}

cudaError_t fillWithCudaV2(float* a, float* b, unsigned int size)
{
    cuda::Buffer<float> buffer_a(size);
    buffer_a.assign(a);
    cuda::Buffer<float> buffer_b(size);
    buffer_b.assign(b);

    unsigned int totalBlocks = size / BLOCK_SIZE;

    if (size % BLOCK_SIZE != 0)
        totalBlocks++;

    fillKernelV2 << <totalBlocks, BLOCK_SIZE >> >(buffer_a, buffer_b);

    cudaError_t cudaStatus = cudaGetLastError();
    cuda::Check::CUDAError(cudaGetLastError(), "fillKernel launch failed", cudaGetErrorString(cudaStatus));
    cuda::Check::CUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code ", cudaGetErrorString(cudaStatus), " after launching fillKernel!");

    buffer_a.retrieve(a);
    buffer_b.retrieve(b);
    return cudaStatus;
}

// Helper function for using CUDA to add vectors in parallel.
cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size)
{
    cudaError_t cudaStatus;
   // Allocate GPU buffers for three vectors (two input, one output)   
    cuda::Buffer<float> buffer_c(size);
    cuda::Buffer<float> buffer_a(size);
    cuda::Buffer<float> buffer_b(size);
    // Copy input vectors from host memory to GPU buffers.
    buffer_a.assign(a);
    buffer_b.assign(b);
    // Launch a kernel on the GPU with one thread for each element.
    addKernel<<<1, size>>>(buffer_c, buffer_a, buffer_b);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    cuda::Check::CUDAError(cudaStatus, "addKernel launch failed",cudaGetErrorString(cudaStatus));   
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda::Check::CUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code ", cudaGetErrorString(cudaStatus), " after launching addKernel!");
    // Copy output vector from GPU buffer to host memory.
    buffer_c.retrieve(c);
    return cudaStatus;
}

cudaError_t addWithCudaV2(float *c, const float *a, const float *b, unsigned int size)
{

    // Allocate GPU buffers for three vectors (two input, one output)   
    cuda::Buffer<float> buffer_c(size);
    cuda::Buffer<float> buffer_a(size);
    cuda::Buffer<float> buffer_b(size);
    // Copy input vectors from host memory to GPU buffers.
    buffer_a.assign(a);
    buffer_b.assign(b);
    // Launch a kernel on the GPU with size / BLOC_SIZE with BLOCK_SIZE threads per block.
    int totalBlocks = size / BLOCK_SIZE;

    if (size % BLOCK_SIZE != 0)
        totalBlocks++;
    addKernelV2 << < totalBlocks, BLOCK_SIZE >> >(buffer_c, buffer_a, buffer_b);
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    cuda::Check::CUDAError(cudaStatus, "addKernel launch failed", cudaGetErrorString(cudaStatus));
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda::Check::CUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code ", cudaGetErrorString(cudaStatus), " after launching addKernel!");
    // Copy output vector from GPU buffer to host memory.
    buffer_c.retrieve(c);
    return cudaStatus;
}

cudaError_t fillThenAddWithCudaV2(float *c, float *a, float *b, unsigned int size)
{

    // Allocate GPU buffers for three vectors (two input, one output)   
    cuda::Buffer<float> buffer_c(size);
    cuda::Buffer<float> buffer_a(size);
    cuda::Buffer<float> buffer_b(size);
    // Copy input vectors from host memory to GPU buffers.
    buffer_a.assign(a);
    buffer_b.assign(b);
    // Launch a kernel on the GPU with size / BLOC_SIZE with BLOCK_SIZE threads per block.
    int totalBlocks = size / BLOCK_SIZE;

    if (size % BLOCK_SIZE != 0)
        totalBlocks++;

    fillKernelV2 << < totalBlocks, BLOCK_SIZE >> >(buffer_a, buffer_b);

    cudaError_t cudaStatus = cudaGetLastError();
    cuda::Check::CUDAError(cudaStatus, "fillKerneV2 launch failed", cudaGetErrorString(cudaStatus));

    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda::Check::CUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code ", cudaGetErrorString(cudaStatus), " after launching fillKernelV2!");

    addKernelV2 << < totalBlocks, BLOCK_SIZE >> >(buffer_c, buffer_a, buffer_b);
    // Check for any errors launching the kernel
    cudaStatus = cudaGetLastError();
    cuda::Check::CUDAError(cudaStatus, "addKernelV2 launch failed", cudaGetErrorString(cudaStatus));
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda::Check::CUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code ", cudaGetErrorString(cudaStatus), " after launching addKernelV2!");
    // Copy output vector from GPU buffer to host memory.
    buffer_a.retrieve(a);
    buffer_b.retrieve(b);
    buffer_c.retrieve(c);
    return cudaStatus;
}


cudaError_t fillAndAddWithCudaV2(float *c, float *a, float *b, unsigned int size)
{

    // Allocate GPU buffers for three vectors (two input, one output)   
    cuda::Buffer<float> buffer_c(size);
    cuda::Buffer<float> buffer_a(size);
    cuda::Buffer<float> buffer_b(size);
    // Copy input vectors from host memory to GPU buffers.
    buffer_a.assign(a);
    buffer_b.assign(b);
    // Launch a kernel on the GPU with size / BLOC_SIZE with BLOCK_SIZE threads per block.
    int totalBlocks = size / BLOCK_SIZE;

    if (size % BLOCK_SIZE != 0)
        totalBlocks++;

    fillAndAddKernelV2 << < totalBlocks, BLOCK_SIZE >> >(buffer_c, buffer_a, buffer_b);
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    cuda::Check::CUDAError(cudaStatus, "fillAndAddWithCudaV2 launch failed", cudaGetErrorString(cudaStatus));
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda::Check::CUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code ", cudaGetErrorString(cudaStatus), " after launching addKernelV2!");
    // Copy output vector from GPU buffer to host memory.
    buffer_a.retrieve(a);
    buffer_b.retrieve(b);
    buffer_c.retrieve(c);
    return cudaStatus;
}


