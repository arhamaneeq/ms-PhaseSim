#include <cstdint>
#include <curand_kernel.h>

__global__ void markovStep(uint8_t* d_input, int w, int h, float T, float mu, curandState* states);
__global__ void initRNG(curandState* states, unsigned long seed);
