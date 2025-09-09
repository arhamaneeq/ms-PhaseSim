#include "kernels.cuh"

__global__ void markovStep(uint8_t* d_input, int w, int h, float T, float mu, curandState* states) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * w + x;

    if (x >= w || y >= h) return;

    curandState localState = states[idx];
    float r1 = curand_uniform(&localState);
    int delN;

    if (d_input[idx] == 0) {
        delN = +1;
    } else if (d_input[idx] == 255) {
        delN = -1;
    } else {
        delN = (r1 < 0.5f) ? -1 : +1;
    }

    float delE = 0;                 // TODO: define hamilltonian
    float k = 1;                    // TODO: remove units, move to dimensionsless everythingggg
    float beta = 1 / (k * T);

    float p_acc = fminf(1.0f , expf(-beta * (delE - mu * delN)));

    float r2 = curand_uniform(&localState);

    if (r2 < p_acc) {
        d_input[idx] += delN;
    }

    states[idx] = localState;

}

__global__ void initRNG(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed, idx, 0, &states[idx]);
}