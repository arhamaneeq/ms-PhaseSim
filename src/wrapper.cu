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

void markovStep(uint8_t* d_cells, int w, int h, float T, float mu, curandState* states) {
    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y
    );

    markovSweep<<<grid, block>>>(d_cells, w, h, T, mu, states, 0);
    markovSweep<<<grid, block>>>(d_cells, w, h, T, mu, states, 1);

    cudaDeviceSynchronize();
}

curandState* genRands(int w, int h) {
    curandState* d_states;
    cudaMalloc(&d_states, w * h * sizeof(curandState));

    dim3 block(16, 16);
    dim3 grid((w + block.x - 1) / block.x,
              (h + block.y - 1) / block.y);

    initRNG<<<grid, block>>>(d_states, 42);
    cudaDeviceSynchronize();

    return d_states;
}