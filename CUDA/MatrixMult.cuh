#pragma once

#include "cuda_runtime.h"
#include "Matrix.hpp"

#define BLOCK_SIZE 32

bool MatrixMult(const HostMatrix<float>& M, const HostMatrix<float>& N, HostMatrix<float>& P, bool complexMult);