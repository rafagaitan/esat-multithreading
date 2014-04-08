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

#include "cuda_runtime.h"

#include <mtUtils/Algorithms.h>

#include "CUDAUtils.h"
#include "Matrix.h"
#include "MatrixMult.cuh"

namespace test_runtime_api
{
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
        std::cout << "Test Runtime API, ready?";
        std::cin.ignore();
        test_runtime_api::test_runtime_api();
    }
    return 0;
}

