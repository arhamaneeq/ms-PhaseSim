#include <cstdint>
#include <curand_kernel.h>

__global__ void markovSweep(uint8_t* d_input, int w, int h, float T, float mu, curandState* states, int offset);
__global__ void initRNG(curandState* states, unsigned long seed, int w);
__device__ float deltaE(const uint8_t* d_input, int w, int h, int x, int y, int delN, float J, float eps);
__device__ float spinVal(uint8_t v);