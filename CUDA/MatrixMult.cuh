#pragma once

#include "cuda_runtime.h"
#include "Matrix.h"

#define BLOCK_SIZE 32

void MatrixMult(const HostMatrix<float>& M, const HostMatrix<float>& N, HostMatrix<float>& P, bool complexMult);