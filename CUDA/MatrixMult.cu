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

#include <cuda.h>
#include <device_launch_parameters.h>

#include "CUDAUtils.h"
#include "Matrix.h"
#include "MatrixMult.cuh"



template<class T>
struct Matrix
{
    int             width_;
    int             height_;
    int             stride_;
    T*              elements_;
    __device__ T getData(int row, int col)
    {
        return elements_[row * stride_ + col];
    }

    __device__ void setData(int row, int col, T data)
    {
        elements_[row * stride_ + col] = data;
    }

    __device__ 
        Matrix<T> getSubMatrix(int row, int col, int block_size)
    {
        Matrix<T> sub;
        sub.width_    = block_size;
        sub.height_   = block_size;
        sub.stride_   = stride_;
        sub.elements_ = &elements_[stride_ * block_size * row + block_size * col];
        return sub;
    }
};

typedef Matrix<float> Matrixf;

template<class T>
struct DeviceMatrix 
{
    DeviceMatrix(int width,int height)
        :matrix_()
    {
        matrix_.width_    = width;
        matrix_.height_   = height;
        matrix_.stride_   = width;
        cudaMalloc((void**)&matrix_.elements_, width* height * sizeof(T));
    }

    ~DeviceMatrix()
    {
        cudaFree(matrix_.elements_);
    }

    void setElements(const HostMatrix<T>& hostMatrix)
    {
        cudaMemcpy(matrix_.elements_, hostMatrix.elements_.data(), matrix_.width_* matrix_.height_ * sizeof(T), cudaMemcpyHostToDevice);
    }

    void getElements(HostMatrix<T>& hostMatrix)
    {
        cudaMemcpy(&(hostMatrix.elements_.front()), matrix_.elements_,matrix_.width_* matrix_.height_ * sizeof(T), cudaMemcpyDeviceToHost);
    }

    Matrix<T> matrix_;
};


typedef DeviceMatrix<float> DeviceMatrixf; 


__global__ void MatrixMultKernelComplex(Matrixf M, Matrixf N, Matrixf C)
{
    // Block row and column
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;
    // Each thread block computes one sub-matrix Csub of C
    Matrixf Csub = C.getSubMatrix(blockRow, blockCol, BLOCK_SIZE);
    // Each thread computes one element of Csub
    // by accumulating results into Cvalue
    float Cvalue = 0;
    // Thread row and column within Csub
    int row = threadIdx.y;
    int col = threadIdx.x;
    // Loop over all the sub-matrices of M and N that are
    // required to compute Csub
    // Multiply each pair of sub-matrices together
    // and accumulate the results
    for (int m = 0; m < (M.width_ / BLOCK_SIZE); ++m) {
        // Get sub-matrix Asub of A
        Matrixf Msub = M.getSubMatrix(blockRow, m, BLOCK_SIZE);
        // Get sub-matrix Bsub of B
        Matrixf Nsub = N.getSubMatrix(m, blockCol, BLOCK_SIZE);
        // Shared memory used to store Msub and Nsub respectively
        __shared__ float Ms[BLOCK_SIZE][BLOCK_SIZE];
        __shared__ float Ns[BLOCK_SIZE][BLOCK_SIZE];
        // Load Asub and Bsub from device memory to shared memory
        // Each thread loads one element of each sub-matrix
        Ms[row][col] = Msub.getData(row, col);
        Ns[row][col] = Nsub.getData(row, col);
        // Synchronize to make sure the sub-matrices are loaded
        // before starting the computation
        __syncthreads();
        // Multiply Asub and Bsub together
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += Ms[row][e] * Ns[e][col];
        // Synchronize to make sure that the preceding
        // computation is done before loading two new
        // sub-matrices of A and B in the next iteration
        __syncthreads();
    }
    // Write Csub to device memory
    // Each thread writes one element
    Csub.setData(row, col, Cvalue);
    __syncthreads();
}

__global__ void MatrixMultKernelSimple(Matrixf M, Matrixf N, Matrixf C)
{
    // Each thread computes one element of C
    // by accumulating results into Cvalue
    float Cvalue = 0.0;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if(row > M.height_ || col > N.width_) return;
    for (int e = 0; e < M.width_; ++e)
    {
        //Cvalue += (M.elements_[row * M.width_ + e]) * (N.elements_[e * N.width_ + col]);
        Cvalue += M.getData(row,e) * N.getData(e,col);
    }
    //C.elements_[row * C.width_ + col] = Cvalue;
    C.setData(row,col, Cvalue);
}

void MatrixMult(const HostMatrix<float>& M, const HostMatrix<float>& N, HostMatrix<float>& P, bool complexMult)
{
    if(M.width_%BLOCK_SIZE != 0) throw cuda::cuda_exception("Matrix M width is not multiple of BLOCK_SIZE");
    if(M.height_%BLOCK_SIZE != 0) throw cuda::cuda_exception("Matrix M width is not multiple of BLOCK_SIZE");
    if(N.width_%BLOCK_SIZE != 0) throw cuda::cuda_exception("Matrix N width is not multiple of BLOCK_SIZE");
    if(N.height_%BLOCK_SIZE != 0) throw cuda::cuda_exception("Matrix N width is not multiple of BLOCK_SIZE");

    DeviceMatrix<float> Md(M.width_,M.height_); Md.setElements(M);
    DeviceMatrix<float> Nd(N.width_,N.height_); Nd.setElements(N);
    DeviceMatrix<float> Pd(P.width_,P.height_);

    dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE);
    if(complexMult)
    {
        dim3 dimGrid(N.width_ / dimBlock.x, M.height_ / dimBlock.y);
        MatrixMultKernelComplex<<<dimGrid,dimBlock>>>(Md.matrix_,Nd.matrix_,Pd.matrix_);
    }
    else
    {
        dim3 dimGrid(
            (N.width_ + dimBlock.x - 1) / dimBlock.x,
            (M.height_ + dimBlock.y - 1) / dimBlock.y
            );
        MatrixMultKernelSimple<<<dimGrid,dimBlock>>>(Md.matrix_,Nd.matrix_,Pd.matrix_);
    }
    // Check for any errors launching the kernel
    cudaError_t cudaStatus = cudaGetLastError();
    cuda::Check<CUDAUTIL_THROW>::CUDAError(cudaStatus, "Kernel launch failed",cudaGetErrorString(cudaStatus));   
    // cudaDeviceSynchronize waits for the kernel to finish, and returns
    // any errors encountered during the launch.
    cuda::Check<CUDAUTIL_THROW>::CUDAError(cudaDeviceSynchronize(), "cudaDeviceSynchronize returned error code ", cudaGetErrorString(cudaStatus), " after launching addKernel!");

    Pd.getElements(P);
}



