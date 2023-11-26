
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <GL/gl.h>
#include <cuda_gl_interop.h>
#include "timer.cuh"


#define SOFTENING 1e-9f

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
    int stride = blockDim.x * gridDim.x;

    for (int i = index; i < n; i += stride) {
        float Fx = 0.0f; float Fy = 0.0f; float Fz = 0.0f;

        for (int j = 0; j < n; j++) {
            float dx = p[j].x - p[i].x;
            float dy = p[j].y - p[i].y;
            float dz = p[j].z - p[i].z;
            float distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            float invDist = rsqrtf(distSqr);
            float invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3; Fy += dy * invDist3; Fz += dz * invDist3;
        }

        p[i].vx += dt * Fx; p[i].vy += dt * Fy; p[i].vz += dt * Fz;
    }
}


int main(const int argc, const char** argv) {

    int deviceId;
    int numberOfSMs;

    cudaGetDevice(&deviceId);
    cudaDeviceGetAttribute(&numberOfSMs, cudaDevAttrMultiProcessorCount, deviceId);

    const int N = 2 << 24;
    size_t size = N * sizeof(float);






    // The assessment will test against both 2<11 and 2<15.
    // Feel free to pass the command line argument 15 when you generate ./nbody report files

    int nBodies = 65535;
    if (argc > 1) nBodies = 2 << atoi(argv[1]);



    // The assessment will pass hidden initialized values to check for correctness.
    // You should not make changes to these files, or else the assessment will not work.
    const char* initialized_values;
    const char* solution_values;

    if (nBodies == 2 << 11) {
        initialized_values = "09-nbody/files/initialized_4096";
        solution_values = "09-nbody/files/solution_4096";
    }
    else { // nBodies == 2<<15
        initialized_values = "09-nbody/files/initialized_65536";
        solution_values = "09-nbody/files/solution_65536";
    }

    if (argc > 2) initialized_values = argv[2];
    if (argc > 3) solution_values = argv[3];

    const float dt = 0.01f; // Time step
    const int nIters = 10;  // Simulation iterations

    int bytes = nBodies * sizeof(Body);
    float* buf;

    // Initialize GLFW and create an OpenGL window



    cudaMallocManaged(&buf, size);
    cudaMemPrefetchAsync(buf, size, deviceId);

    size_t threadsPerBlock;
   

    threadsPerBlock = 256;
    size_t numberOfBlocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;




    Body* p = (Body*)buf;



    

    /*
     * This simulation will run for 10 cycles of time, calculating gravitational
     * interaction amongst bodies, and adjusting their positions to reflect.
     */

    for (int iter = 0; iter < nIters; iter++) {
        StartTimer();
        double totalTime = 0.0;
        /*
         * You will likely wish to refactor the work being done in `bodyForce`,
         * and potentially the work to integrate the positions.
         */




         /*
          * This position integration cannot occur until this round of `bodyForce` has completed.
          * Also, the next round of `bodyForce` cannot begin until the integration is complete.
          */

        for (int i = 0; i < nBodies; i++) { // integrate position
            p[i].x += p[i].vx * dt;
            p[i].y += p[i].vy * dt;
            p[i].z += p[i].vz * dt;
            p[i].vx = 0.0f;
            p[i].vy = 0.0f;
            p[i].vz = 0.0f;
            p[i].mass = rand() / (float)RAND_MAX * 10.0f;
        }

        bodyForce << <numberOfBlocks, threadsPerBlock >> > (p, dt, nBodies); // compute interbody forces
        cudaDeviceSynchronize();



        // Clear the screen and render particles using OpenGL


        const double tElapsed = GetTimer() / 1000.0;
        totalTime += tElapsed;


        double avgTime = totalTime / (double)(nIters);
        float billionsOfOpsPerSecond = 1e-9 * nBodies * nBodies / avgTime;


        // You will likely enjoy watching this value grow as you accelerate the application,
        // but beware that a failure to correctly synchronize the device might result in
        // unrealistically high values.
        printf("%0.3f Billion Interactions / second\n", billionsOfOpsPerSecond);

    }
    cudaFree(buf);

    // Cleanup resources

    return 0;
}








