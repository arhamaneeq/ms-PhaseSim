#include <cuda_runtime.h>
#include <stdexcept>
#include "kernels.cuh"


void* allocateDeviceMemory(size_t bytes);
void  deallocateDeviceMemory(void* ptr);
void copyMemory(void* dst, const void* src, size_t bytes, int direction);

void markovStep(uint8_t* d_cells, int w, int h, float T, float mu, curandState* states);
curandState* genRands(int w, int h);