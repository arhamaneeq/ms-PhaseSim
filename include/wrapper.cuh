#include <cuda_runtime.h>
#include <stdexcept>

void* allocateDeviceMemory(size_t bytes);
void  deallocateDeviceMemory(void* ptr);
void copyMemory(void* dst, const void* src, size_t bytes, int direction);