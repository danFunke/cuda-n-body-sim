
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
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

typedef struct { float x, y, z, vx, vy, vz, mass; } Body;

/*
 * Calculate the gravitational impact of all bodies in the system
 * on all others.
 */




__global__ void bodyForce(Body* p, float dt, int n) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;


    extern __shared__ Body s_bodies[];
   

    
    Body myBody = p[index];
    float Fx = 0.0f;
    float Fy = 0.0f;
    float Fz = 0.0f;

    for (int tile = 0; tile < gridDim.x; tile++) {
        int offset = tile * blockDim.x + threadIdx.x;
        if (offset < n) {  // Ensure we don't read beyond the number of bodies.
            s_bodies[threadIdx.x] = p[offset];
        }
        __syncthreads();

        // We'll compute forces with all bodies in our current tile.
        // If you're on the last tile and n isn't a multiple of blockDim.x,
        // not all threads might have meaningful data. 
        // This check ensures we don't use data from beyond n.
        int limit = (tile == gridDim.x - 1) ? (n - tile * blockDim.x) : blockDim.x;

       for (int j = 0; j < limit; j++) {
           
           float dx = s_bodies[j].x - myBody.x;
            float dy = s_bodies[j].y - myBody.y;
           float dz = s_bodies[j].z - myBody.z;
           float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
          
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    p[index].vx += dt * Fx;
    p[index].vy += dt * Fy;
    p[index].vz += dt * Fz;
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
    printf("Using device ID: %d\n", deviceId); // Add this line to print the device ID

    srand(time(NULL));

    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);
    CHECK_CUDA_ERR();
    int nBodies = 65535;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);

    const float dt = 0.01f; // Time step

    const int nIters = 10;  // Simulation iterations

    int bytes = nBodies * sizeof(Body);
    Body* p;

    cudaMallocManaged(&p, bytes);
    cudaMemset(p, 0, bytes);
    CHECK_CUDA_ERR();


    //cudaMemPrefetchAsync(p, bytes, deviceId);

    size_t threadsPerBlock = 128;
    size_t numberOfBlocks = ((nBodies + threadsPerBlock - 1) / threadsPerBlock);

    
    // Randomly initialize body values
    for (int i = 0; i < nBodies; i++) {
        p[i].x = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        p[i].y = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        p[i].z = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        p[i].vx = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        p[i].vy = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        p[i].vz = rand() / (float)RAND_MAX * 2.0f - 1.0f;
        p[i].mass = rand() / (float)RAND_MAX * 10.0f;
    }

    double totalTime = 0.0;

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();

        for (int i = 0; i < nBodies; i++) { // integrate position
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
            p[i].vx = 0.0f;
            p[i].vy = 0.0f;
            p[i].vz = 0.0f;
        }


        cudaSetDevice(0);
        bodyForce << <ceil(numberOfBlocks), threadsPerBlock, sizeof(Body)* threadsPerBlock >> > (p, dt, nBodies);
      
        CHECK_CUDA_ERR();
        cudaDeviceSynchronize();
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess)
        {
            printf("CUDA Error: %s\n", cudaGetErrorString(err));
        }
        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;

        double avgTime = totalTime / (double)(nIters);
        float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;

        printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);
    }

    cudaFree(p);

    return 0;
}








