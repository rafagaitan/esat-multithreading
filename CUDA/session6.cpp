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

#include "cuda_runtime.h"

#include "CUDAUtils.h"
#include "SumIntVectors.cuh"

namespace test_runtime_api
{
    void test_runtime_api()
    {
        const int arraySize = 5;
        const int a[arraySize] = { 1, 2, 3, 4, 5 };
        const int b[arraySize] = { 10, 20, 30, 40, 50 };
        int c[arraySize] = { 0 };

        // Add vectors in parallel.
        cudaError_t cudaStatus = addWithCuda(c, a, b, arraySize);
        cuda::Check<CUDAUTIL_THROW>::CUDAError(cudaStatus, "addWithCuda failed!");

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
        std::cout << "Test Runtime API, ready?";
        std::cin.ignore();
        test_runtime_api::test_runtime_api();
    }
    return 0;
}

