#define NOMINMAX
#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <string>

#include "cuda_runtime.h"

#include <mtUtils/Algorithms.hpp>

#include "CUDAUtils.hpp"
#include "SumIntVectors.cuh"

namespace test_runtime_api
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


    void test_cuda_simple1()
    {
        const int arraySize = 5;
        float a[] = { 1.0f, 2.0f, 3.0f, 4.0f, 5.0f };
        float b[] = { 10, 20, 30, 40, 50 };
        std::vector<float> c(arraySize);

        // Choose which GPU to run on, change this on a multi-GPU system.
        cuda::Device device(0);

        cudaError_t cudaStatus = addWithCuda(c.data(), a, b, arraySize);
        cuda::Check::CUDAError(cudaStatus, "addWithCuda failed!");


        printV(std::cout, a, arraySize,  0, 5);
        std::cout << "    +    ";
        printV(std::cout, b, arraySize, 0, 5);
        std::cout << "   =   ";
        printV(std::cout, c, c.size(), 0, 5);
        std::cout << std::endl;


    }

    void test_cpu_simple2(const unsigned int arraySize, bool printResults = false)
    {
        std::vector<float> a(arraySize);
        std::vector<float> b(arraySize);
        std::vector<float> c(arraySize);

        for (unsigned int i = 0; i < arraySize; i++)
        {
            a[i] = static_cast<float>(sin(i)*sin(i));
            b[i] = static_cast<float>(cos(i)*cos(i));
            c[i] = a[i] + c[i];
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


    void test_cuda_simple2(const int arraySize, bool printResults = false)
    {
        std::vector<float> a(arraySize);
        std::vector<float> b(arraySize);
        std::vector<float> c(arraySize);

        // Choose which GPU to run on, change this on a multi-GPU system.
        cuda::Device device(0);

        cudaError_t cudaStatus;
        float* a_ptr = a.data();
        float* b_ptr = b.data();
        float* c_ptr = c.data();

        cudaStatus = fillWithCudaV2(a_ptr, b_ptr, arraySize);
        cuda::Check::CUDAError(cudaStatus, "fillWithCudaV2 failed!");
        cudaStatus = addWithCudaV2(c_ptr, a_ptr, b_ptr, arraySize);
        cuda::Check::CUDAError(cudaStatus, "addWithCudaV2 failed!");

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

    void test_cuda_simple3(const int arraySize, bool printResults = false)
    {
        std::vector<float> a(arraySize);
        std::vector<float> b(arraySize);
        std::vector<float> c(arraySize);

        // Choose which GPU to run on, change this on a multi-GPU system.
        cuda::Device device(0);

        cudaError_t cudaStatus;
        float* a_ptr = a.data();
        float* b_ptr = b.data();
        float* c_ptr = c.data();

        cudaStatus = fillThenAddWithCudaV2(c_ptr, a_ptr, b_ptr, arraySize);
        cuda::Check::CUDAError(cudaStatus, "fillAndAddWithCuda failed!");

        if (printResults)
        {
            printV(std::cout, a, a.size(), 0, 20); printV(std::cout, a, a.size(), arraySize - 20, 20);
            std::cout << "    +    ";
            printV(std::cout, b, b.size(), 0, 20); printV(std::cout, b, b.size(), arraySize - 20, 20);
            std::cout << "   =   ";
            printV(std::cout, c, c.size(), 0, 20); printV(std::cout, c, c.size(), arraySize - 20, 20);
            std::cout << std::endl;
        }
    }

    void test_cuda_simple4(const int arraySize, bool printResults = false)
    {
        std::vector<float> a(arraySize);
        std::vector<float> b(arraySize);
        std::vector<float> c(arraySize);

        // Choose which GPU to run on, change this on a multi-GPU system.
        cuda::Device device(0);

        cudaError_t cudaStatus;
        float* a_ptr = a.data();
        float* b_ptr = b.data();
        float* c_ptr = c.data();

        cudaStatus = fillAndAddWithCudaV2(c_ptr, a_ptr, b_ptr, arraySize);
        cuda::Check::CUDAError(cudaStatus, "fillAndAddWithCuda failed!");


        if (printResults)
        {
            printV(std::cout, a, a.size(), 0, 20); printV(std::cout, a, a.size(), arraySize - 20, 20);
            std::cout << "    +    ";
            printV(std::cout, b, b.size(), 0, 20); printV(std::cout, b, b.size(), arraySize - 20, 20);
            std::cout << "   =   ";
            printV(std::cout, c, c.size(), 0, 20); printV(std::cout, c, c.size(), arraySize - 20, 20);
            std::cout << std::endl;
        }
    }
}


int main()
{
    try
    {
        std::cout << "Test CUDA simple 1" << std::endl;
        //std::cin.ignore();
        test_runtime_api::test_cuda_simple1();

        const unsigned int numTests = 10;
        const unsigned int iniMult = 1;
        const unsigned int maxMult = 10000;
        const bool printResults = false;
        std::cout << "Test CPU simple 2, running ..." << std::endl;
        for (unsigned int mult = iniMult; mult <= maxMult; mult *= 10)
        {
            const int arraySize = 1024 * mult;
            std::cout << "Array size: " << arraySize << std::endl;
            ScopedTimer timer("time for CPU simple 2", numTests);
            for (unsigned int i = 0; i < numTests; i++)
            {
                test_runtime_api::test_cpu_simple2(arraySize, printResults);
            }
        }

        std::cout << "Test CUDA simple 2, running ..." << std::endl;
        for (unsigned int mult = iniMult; mult <= maxMult; mult *= 10)
        {
            const int arraySize = 1024 * mult;
            std::cout << "Array size: " << arraySize << std::endl;
            ScopedTimer timer("time for CUDA simple 2", numTests);
            for (unsigned int i = 0; i < numTests; i++)
            {
                test_runtime_api::test_cuda_simple2(arraySize, printResults);
            }
        }

        std::cout << "Test CUDA simple 3, running ..." << std::endl;
        for (unsigned int mult = iniMult; mult <= maxMult; mult *= 10)
        {
            const int arraySize = 1024 * mult;
            std::cout << "Array size: " << arraySize << std::endl;
            ScopedTimer timer("time for CUDA simple 3", numTests);
            for (unsigned int i = 0; i < numTests; i++)
            {
                test_runtime_api::test_cuda_simple3(arraySize, printResults);
            }
        }

        std::cout << "Test CUDA simple 4, running ..." << std::endl;
        for (unsigned int mult = iniMult; mult <= maxMult; mult *= 10)
        {
            const int arraySize = 1024 * mult;
            std::cout << "Array size: " << arraySize << std::endl;
            ScopedTimer timer("time for CUDA simple 4", numTests);
            for (unsigned int i = 0; i < numTests; i++)
            {
                test_runtime_api::test_cuda_simple4(arraySize, printResults);
            }
        }
    }
    catch (const cuda::cuda_exception& e)
    {
        std::cerr << "EXCEPTION:" << e.what() << std::endl;
        return -1;
    }
    return 0;
}

