#include <iostream>
#include <vector>
#include <stdlib.h>
#include <time.h>
// #include <cuda_runtime.h>
#include <math.h>
#include <string>

#include "BackendCommon.h"

#include "kernel.h"

void printVector(const std::string& message, float* vec, const int size)
{
	std::cout << message << "\n";
	for (int i=0; i<size; ++i) {
		std::cout << vec[i] << ", ";
	}
	std::cout << "\n\n";
}

int test2()
{
    // Perform matrix multiplication C = A*B
    // where A, B and C are NxN matrices
    int SIZE = 100000;

    // Allocate memory on the host
    std::vector<float> h_A(SIZE);
    std::vector<float> h_B(SIZE);
    std::vector<float> h_C(SIZE);

    // Initialize matrices on the host
    for (int i=0; i<SIZE; i++){
	    h_A[i] = sin(i);
	    h_B[i] = cos(i);
    }

	// printVector("A_orig vec", h_A.data(), SIZE);
	// printVector("B_orig vec", h_B.data(), SIZE);

    // Allocate memory on the device
    float* d_A;
    float* d_B;
    float* d_C;
    Allen::malloc((void**)&d_A, SIZE*sizeof(float));
    Allen::malloc((void**)&d_B, SIZE*sizeof(float));
    Allen::malloc((void**)&d_C, SIZE*sizeof(float));    

	Allen::memcpy(d_A, h_A.data(), SIZE * sizeof(float), Allen::memcpyHostToDevice);
	Allen::memcpy(d_B, h_B.data(), SIZE * sizeof(float), Allen::memcpyHostToDevice);
    // d_A.set(&h_A[0], SIZE);
    // d_B.set(&h_B[0], SIZE);

    vectorMultiplication(d_A, d_B, d_C, SIZE);

	Allen::memcpy(h_C.data(), d_C, SIZE * sizeof(float), Allen::memcpyDeviceToHost);

    float *cpu_C;
    cpu_C = new float[SIZE];

    // Now do the vector multiplication on the CPU
    for (int i=0; i<SIZE; ++i){
    	cpu_C[i] = h_A[i] * h_B[i];
    }

	// printVector("HOST vec", h_C.data(), SIZE);
	// printVector("CPU vec", cpu_C, SIZE);

    double err = 0;
    // Check the result and make sure it is correct
    for (int i=0; i<SIZE; ++i) {
    	err += cpu_C[i] - h_C[i];
    }

    std::cout << "Error: " << err << std::endl;

    return 0;
}
