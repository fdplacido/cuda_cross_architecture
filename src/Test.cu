#include "Test.cuh"

__global__
void saxpy(int n, float a, float *x, float *y)
{
  // TODO convert to strided
  // TODO convert to non-threadIdx dependent code
  int i = blockIdx.x*blockDim.x + threadIdx.x;
  if (i < n) y[i] = a*x[i] + y[i];
}

void test(void)
{
  int N = 1<<20;
  float *x, *y, *d_x, *d_y;
  x = (float*)malloc(N*sizeof(float));
  y = (float*)malloc(N*sizeof(float));

  Allen::malloc((void**)&d_x, N*sizeof(float));
  Allen::malloc((void**)&d_y, N*sizeof(float));

  for (int i = 0; i < N; i++) {
    x[i] = 1.0f;
    y[i] = 2.0f;
  }

  Allen::memcpy(d_x, x, N*sizeof(float), Allen::memcpyHostToDevice);
  Allen::memcpy(d_y, y, N*sizeof(float), Allen::memcpyHostToDevice);


const dim3 grid_dim {(N+255)/256, 1, 1};
const dim3 block_dim {256, 1, 1};


#ifdef TARGET_DEVICE_CPU
  gridDim = {grid_dim.x, grid_dim.y, grid_dim.z};
  for (unsigned int i = 0; i < grid_dim.x; ++i) {
    for (unsigned int j = 0; j < grid_dim.y; ++j) {
      for (unsigned int k = 0; k < grid_dim.z; ++k) {
        blockIdx = {i, j, k};
        // function(std::get<I>(invoke_arguments)...);
        saxpy(N, 2.0f, d_x, d_y);
      }
    }
  }
#elif defined (TARGET_DEVICE_CUDA)

  // Perform SAXPY on 1M elements
  saxpy<<<grid_dim, block_dim>>>(N, 2.0f, d_x, d_y);

#endif

  Allen::memcpy(y, d_y, N*sizeof(float), Allen::memcpyDeviceToHost);

  float maxError = 0.0f;
  for (int i = 0; i < N; i++)
    // maxError = max(maxError, abs(y[i]-4.0f));
    maxError = maxError * abs(y[i]-4.0f);
  printf("Max error: %f\n", maxError);

  Allen::free(d_x);
  Allen::free(d_y);
  free(x);
  free(y);
}