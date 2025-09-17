#include <cstdint>
#include <curand_kernel.h>

using Cell = bool;

__global__ void markovSweep(Cell* d_input, int w, int h, float T, float mu, curandState* states, int offset, float J);
__global__ void initRNG(curandState* states, unsigned long seed, int w);
__device__ float deltaE(const Cell* d_input, int w, int h, int x, int y, int delN, float J, float eps);
__device__ float spinVal(uint8_t v);