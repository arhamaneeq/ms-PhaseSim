#include "wrapper.cuh"

void* allocateDeviceMemory(size_t bytes) {
    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, bytes);

    if (err!=cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }

    return ptr;
}

void deallocateDeviceMemory(void* ptr) {
    cudaError_t err = cudaFree(ptr);
    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}

void copyMemory(void* dst, const void* src, size_t bytes, int direction) {
    cudaError_t err = cudaMemcpy(dst, src, bytes, static_cast<cudaMemcpyKind>(direction));

    if (err != cudaSuccess) {
        throw std::runtime_error(cudaGetErrorString(err));
    }
}