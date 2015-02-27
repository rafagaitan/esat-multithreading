#include "cuda_runtime.h"

cudaError_t addWithCuda(float *c, const float *a, const float *b, unsigned int size);
cudaError_t addWithCudaV2(float *c, const float *a, const float *b, unsigned int size);
cudaError_t fillThenAddWithCudaV2(float *c, float *a, float *b, unsigned int size);
cudaError_t fillAndAddWithCudaV2(float *c, float *a, float *b, unsigned int size);
cudaError_t fillWithCudaV2(float *a, float *b, unsigned int size);

//__global__ void addKernel(int *c, const int *a, const int *b);
