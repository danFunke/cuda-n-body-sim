
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include "timer.cuh"
#include <ctime>
#include <cmath>
#define SOFTENING 1e-9f
#define CHECK_CUDA_ERR() { cudaCheckError(__FILE__, __LINE__); }
inline void cudaCheckError(const char* file, int line) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error at %s:%d: %s\n", file, line, cudaGetErrorString(err));
        //exit(EXIT_FAILURE);
    }
}

/*
 * Each body contains x, y, and z coordinate positions,
 * as well as velocities in the x, y, and z directions.
 */

struct Bodies { float* x, * y, * z, * vx, * vy, * vz, * mass; };    //p is now a struct with separate arrays for each component 

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */




__global__ void bodyForce(Bodies p, float dt, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;

    extern __shared__ float s_pos[]; // This shared memory will now only store positions

    if (index >= n) return; // Ensure we don't go out of bounds

    // Separate components of myBody
    float myPosX = p.x[index];
    float myPosY = p.y[index];
    float myPosZ = p.z[index];

    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    // Calculate the number of tiles we need to iterate over
    int numTiles = (n + blockDim.x - 1) / blockDim.x;

    for (int tile = 0; tile < numTiles; ++tile) {
        int idx = tile * blockDim.x + threadIdx.x;

        // Load position data into shared memory
        if (idx < n) {
            s_pos[threadIdx.x] = p.x[idx];
            s_pos[threadIdx.x + blockDim.x] = p.y[idx]; // Assuming blockDim.x is <= half of shared memory size allocated for s_pos
            s_pos[threadIdx.x + 2 * blockDim.x] = p.z[idx]; // "
        }
        __syncthreads();

        // Now we calculate forces using the shared memory
        for (int j = 0; j < blockDim.x; ++j) {
            int sharedIdx = tile * blockDim.x + j;
            // Boundary check for the last tile
            if (sharedIdx >= n) break;

            float dx = s_pos[j] - myPosX;
            float dy = s_pos[j + blockDim.x] - myPosY;
            float dz = s_pos[j + 2 * blockDim.x] - myPosZ;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;

            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    // Update velocities
    p.vx[index] += dt * Fx;
    p.vy[index] += dt * Fy;
    p.vz[index] += dt * Fz;
}



int main(const int argc, const char** argv) {
    int deviceId = 0;
    int numberOfSMs;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, deviceId);
    printf("Device name: %s\n", prop.name);
    printf("Compute capability: %d.%d\n", prop.major, prop.minor);

    cudaGetDevice(&deviceId);
    CHECK_CUDA_ERR();
    printf("Using device ID: %d\n", deviceId);

    srand(time(NULL));

    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    CHECK_CUDA_ERR();
    int nBodies = 65535;

    Bodies h_bodies;
    Bodies d_bodies;

    // Allocate host memory
    h_bodies.x = new float[nBodies];
    h_bodies.y = new float[nBodies];
    h_bodies.z = new float[nBodies];
    h_bodies.vx = new float[nBodies];
    h_bodies.vy = new float[nBodies];
    h_bodies.vz = new float[nBodies];
    h_bodies.mass = new float[nBodies];

    // Allocate device memory
    cudaMalloc(&d_bodies.x, nBodies * sizeof(float));
    cudaMalloc(&d_bodies.y, nBodies * sizeof(float));
    cudaMalloc(&d_bodies.z, nBodies * sizeof(float));
    cudaMalloc(&d_bodies.vx, nBodies * sizeof(float));
    cudaMalloc(&d_bodies.vy, nBodies * sizeof(float));
    cudaMalloc(&d_bodies.vz, nBodies * sizeof(float));
    cudaMalloc(&d_bodies.mass, nBodies * sizeof(float));
    CHECK_CUDA_ERR();
    // Randomly initialize body values
    for (int i = 0; i < nBodies; i++) {
        h_bodies.x[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        h_bodies.y[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        h_bodies.z[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        h_bodies.vx[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        h_bodies.vy[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        h_bodies.vz[i] = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        h_bodies.mass[i] = rand() / (float)RAND_MAX * 10.0f + 0.1f; // Ensure mass is never zero
    }

    // Copy body data from host to device
    cudaMemcpy(d_bodies.x, h_bodies.x, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.y, h_bodies.y, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.z, h_bodies.z, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.vx, h_bodies.vx, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.vy, h_bodies.vy, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.vz, h_bodies.vz, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_bodies.mass, h_bodies.mass, nBodies * sizeof(float), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERR();

    const float dt = 0.01f; // Time step
    const int nIters = 10;  // Simulation iterations


    size_t threadsPerBlock = 128;
    size_t numberOfBlocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;
    double totalTime = 0.0;
    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();
  
        // Integrate positions based on velocities
        for (int i = 0; i < nBodies; i++) { // integrate position
            h_bodies.x[i] += h_bodies.vx[i] * dt;
            h_bodies.y[i] += h_bodies.vy[i] * dt;
            h_bodies.z[i] += h_bodies.vz[i] * dt;
        }

        // Copy updated positions to device
        cudaMemcpy(d_bodies.x, h_bodies.x, nBodies * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bodies.y, h_bodies.y, nBodies * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_bodies.z, h_bodies.z, nBodies * sizeof(float), cudaMemcpyHostToDevice);

        // Launch kernel to compute new velocities
        cudaSetDevice(0);
        size_t sharedMemSize = 3 * threadsPerBlock * sizeof(float);
        bodyForce << <numberOfBlocks, threadsPerBlock, sharedMemSize >> > (d_bodies, dt, nBodies);
        CHECK_CUDA_ERR();
        cudaDeviceSynchronize();

        // Copy updated velocities back to host
        cudaMemcpy(h_bodies.vx, d_bodies.vx, nBodies * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bodies.vy, d_bodies.vy, nBodies * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bodies.vz, d_bodies.vz, nBodies * sizeof(float), cudaMemcpyDeviceToHost);

        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;
      

    }

    double avgTime = totalTime / (double)(nIters);
    float billionsOfOpsPerSecond = 1e-9 * nBodies * (nBodies - 1) / 2 / avgTime;
    printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    // Free device memory
    cudaFree(d_bodies.x);
    cudaFree(d_bodies.y);
    cudaFree(d_bodies.z);
    cudaFree(d_bodies.vx);
    cudaFree(d_bodies.vy);
    cudaFree(d_bodies.vz);
    cudaFree(d_bodies.mass);

    // Free host memory
    free(h_bodies.x);
    free(h_bodies.y);
    free(h_bodies.z);
    free(h_bodies.vx);
    free(h_bodies.vy);
    free(h_bodies.vz);
    free(h_bodies.mass);

    return 0;
}
