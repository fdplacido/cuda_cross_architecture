#include <math.h>
#include <iostream>
// #include "cuda_runtime.h"
#include "kernel.h"
#include <stdlib.h>

#include "BackendCommon.h"

using namespace std;

__global__ void vectorMultiplicationKernel(float* A, float* B, float* C, int N) {


	for (unsigned i = threadIdx.x; i < N; i += blockDim.x) {
		C[i] = A[i] * B[i];
	}
}


void vectorMultiplication(float *A, float *B, float *C, int N){

#ifdef TARGET_DEVICE_CPU
    vectorMultiplicationKernel(A, B, C, N);
#elif defined (TARGET_DEVICE_CUDA)
	vectorMultiplicationKernel<<<64,256>>>(A, B, C, N);
#endif
}
