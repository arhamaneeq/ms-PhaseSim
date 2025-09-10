#include "kernels.cuh"

__global__ void markovStep(uint8_t* d_input, int w, int h, float T, float mu, curandState* states, int offset) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = y * w + x;
    if (x >= w || y >= h) return;
    if ((x + y) % 2 != offset) return;

    curandState localState = states[idx];
    float r1 = curand_uniform(&localState);
    float r2 = curand_uniform(&localState);
    states[idx] = localState;

    int delN;
    if (d_input[idx] == 0) {
        delN = +1;
    } else if (d_input[idx] == 255) {
        delN = -1;
    } else {
        delN = (r1 < 0.5f) ? -1 : +1;
    }

    float J = 1.0;
    float delE = deltaE(d_input, w, h, x, y, delN, J, 0);                 // TODO: define hamilltonian
    // float k = 1;                    // TODO: remove units, move to dimensionsless everythingggg
    // float beta = 1 / (k * T);
    float delPhi = delE - mu * delN;

    float p_acc = fminf(1.0f , expf(-delPhi / T));


    if (r2 < p_acc) {
        d_input[idx] += delN;
    }
}

__global__ void initRNG(curandState* states, unsigned long seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    curand_init(seed, idx, 0, &states[idx]);
}

__device__ float deltaE(const uint8_t* d_input, int w, int h, int x, int y, int delN, float J, float eps) {
    int idx = y * w + x;

    int xL = (x == 0) ? w - 1 : x - 1;
    int xR = (x == w - 1) ? 0 : x + 1;
    int yU = (y == 0) ? h - 1 : y - 1;
    int yD = (y == h - 1) ? 0 : y + 1;
    
    float sumN  = (float) d_input[y * w + xL]
                + (float) d_input[y * w + xR]
                + (float) d_input[yU * w + x]
                + (float) d_input[yD * w + x];

    float deltaE = - J * (float)  delN * sumN + eps * (float) delN;
    return deltaE;

}