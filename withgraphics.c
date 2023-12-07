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
#include <GLFW/glfw3.h>
#include <cmath>
GLuint pbo;
cudaGraphicsResource* cudaPboResource;
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

struct Bodies {
    float* x, * y, * z, * vx, * vy, * vz, * mass;
    float* r, * g, * b;
};    //p is now a struct with separate arrays for each component 

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

/*
GLFWwindow* initOpenGL() {
    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return NULL;
    }

    GLFWwindow* window = glfwCreateWindow(800, 600, "N-body Simulation", NULL, NULL);
    if (!window) {
        fprintf(stderr, "Failed to open GLFW window\n");
        glfwTerminate();
        return NULL;
    }

    glfwMakeContextCurrent(window);
    return window;
}
*/

// Function to draw bodies
void drawBodies(Bodies bodies, int nBodies) {
    glPointSize(2.0f);
    glBegin(GL_POINTS);
    for (int i = 0; i < nBodies; ++i) {
        glColor4f(bodies.r[i], bodies.g[i], bodies.b[i], 0.5f);
        glVertex3f(bodies.x[i], bodies.y[i], bodies.z[i]);
    }
    glEnd();
}

void perspective(GLfloat fov, GLfloat aspect, GLfloat zNear, GLfloat zFar) {
    GLfloat f = 1.0f / tan(fov / 2.0f);
    GLfloat matrix[16] = {
        f / aspect, 0.0f, 0.0f, 0.0f,
        0.0f, f, 0.0f, 0.0f,
        0.0f, 0.0f, (zFar + zNear) / (zNear - zFar), -1.0f,
        0.0f, 0.0f, (2 * zFar * zNear) / (zNear - zFar), 0.0f
    };

    glLoadMatrixf(matrix);
}

// Function to create a look-at matrix similar to gluLookAt
void lookAt(GLfloat eyeX, GLfloat eyeY, GLfloat eyeZ,
    GLfloat centerX, GLfloat centerY, GLfloat centerZ,
    GLfloat upX, GLfloat upY, GLfloat upZ) {
    GLfloat forward[3], side[3], up[3];
    GLfloat matrix[16], resultMatrix[16];

    forward[0] = centerX - eyeX;
    forward[1] = centerY - eyeY;
    forward[2] = centerZ - eyeZ;

    up[0] = upX;
    up[1] = upY;
    up[2] = upZ;

    // Normalize forward
    GLfloat fLength = sqrt(forward[0] * forward[0] + forward[1] * forward[1] + forward[2] * forward[2]);
    for (unsigned int i = 0; i < 3; ++i)
        forward[i] /= fLength;

    // Side = forward x up
    side[0] = forward[1] * up[2] - forward[2] * up[1];
    side[1] = forward[2] * up[0] - forward[0] * up[2];
    side[2] = forward[0] * up[1] - forward[1] * up[0];

    // Normalize side
    GLfloat sLength = sqrt(side[0] * side[0] + side[1] * side[1] + side[2] * side[2]);
    for (unsigned int i = 0; i < 3; ++i)
        side[i] /= sLength;

    // Recompute up as: up = side x forward
    up[0] = side[1] * forward[2] - side[2] * forward[1];
    up[1] = side[2] * forward[0] - side[0] * forward[2];
    up[2] = side[0] * forward[1] - side[1] * forward[0];

    // Fill in the matrix
    matrix[0] = side[0];
    matrix[4] = side[1];
    matrix[8] = side[2];
    matrix[12] = 0.0f;

    matrix[1] = up[0];
    matrix[5] = up[1];
    matrix[9] = up[2];
    matrix[13] = 0.0f;

    matrix[2] = -forward[0];
    matrix[6] = -forward[1];
    matrix[10] = -forward[2];
    matrix[14] = 0.0f;

    matrix[3] = matrix[7] = matrix[11] = 0.0f;
    matrix[15] = 1.0f;

    // Apply translation too
    glLoadIdentity();
    glTranslatef(-eyeX, -eyeY, -eyeZ);
    glGetFloatv(GL_MODELVIEW_MATRIX, resultMatrix);
    glLoadMatrixf(matrix);
    glMultMatrixf(resultMatrix);
}

void framebuffer_size_callback(GLFWwindow* window, int width, int height) {
    // Make sure the viewport matches the new window dimensions
    glViewport(0, 0, width, height);
}

/*

// Define the center point of your galaxy
    float galaxyCenterX = 0.0f; // Adjust as needed
    float galaxyCenterY = 0.0f;
    float galaxyCenterZ = 0.0f;

void updateBodyColors(Bodies bodies, int nBodies, float innerRadius, float outerRadius) {
    for (int i = 0; i < nBodies; ++i) {

        float dx = bodies.x[i] - galaxyCenterX;
        float dy = bodies.y[i] - galaxyCenterY;
        float dz = bodies.z[i] - galaxyCenterZ;
        float distance = sqrtf(dx * dx + dy * dy + dz * dz);

        // Map the distance to a color
        // Closer to center: Red-Yellow-Orange
        // Further away: Blue to Purple
        if (distance < 1.0) {
            // Red to Yellow to Orange
            bodies.r[i] = 1.0f; // Full red
            bodies.g[i] = 1.0f * (distance / innerRadius); // Gradually add green
            bodies.b[i] = 0.0f; // No blue
        }
        else {
            // Blue to Purple
            bodies.r[i] = 0.5f * ((distance - innerRadius) / (outerRadius - innerRadius)); // Gradually add red
            bodies.g[i] = 0.0f; // No green
            bodies.b[i] = 1.0f; // Full blue
        }
    }
}
*/

// Function to update body color based off speed
void updateBodyColorsSpeed(Bodies bodies, int nBodies) {
    for (int i = 0; i < nBodies; ++i) {

        float vx = abs(bodies.vx[i]);
        float vy = abs(bodies.vy[i]);
        float vz = abs(bodies.vz[i]);

        // Map the speed to a color
        // Slow to fast: Green-Yellow-Red
        if (vx < 50.0) {
            // Green
            bodies.r[i] = 0.0f; // No red
            bodies.g[i] = 1.0f; // Full green
            bodies.b[i] = 0.0f; // No blue
        }
        else if (vx < 100.0)
        {
            // Yellow
            bodies.r[i] = 1.0f; // Full red
            bodies.g[i] = 1.0f; // Full green
            bodies.b[i] = 0.0f; // No blue
        }
        else {
            // Red
            bodies.r[i] = 1.0f; // Full red
            bodies.g[i] = 0.0f; // No green
            bodies.b[i] = 0.0f; // No blue
        }
    }
}

float maxVelocity = 0.0f; // Initial value

void updateMaxVelocity(Bodies bodies, int nBodies, float& maxVelocity) {
    for (int i = 0; i < nBodies; ++i) {
        float velocityMagnitude = sqrt(bodies.vx[i] * bodies.vx[i] + bodies.vy[i] * bodies.vy[i] + bodies.vz[i] * bodies.vz[i]);
        if (velocityMagnitude > maxVelocity) {
            maxVelocity = velocityMagnitude;
            printf("Max V: %f\n", maxVelocity);
        }
    }
}

int main(const int argc, const char** argv) {
    int deviceId = 0;
    int numberOfSMs;
    float innerRadius = 1.0f, float outerRadius = 5.0f;
    float maxVelocity = 1.2f;
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

    //colors
    h_bodies.r = new float[nBodies];
    h_bodies.g = new float[nBodies];
    h_bodies.b = new float[nBodies];
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
        h_bodies.r[i] = rand() / (float)RAND_MAX;
        h_bodies.g[i] = rand() / (float)RAND_MAX;
        h_bodies.b[i] = rand() / (float)RAND_MAX;
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

    const float dt = 0.0001f; // Time step
    const int nIters = 10;  // Simulation iterations

    size_t threadsPerBlock = 128;
    size_t numberOfBlocks = (nBodies + threadsPerBlock - 1) / threadsPerBlock;

    if (!glfwInit()) {
        fprintf(stderr, "Failed to initialize GLFW\n");
        return -1;
    }

    // Create a GLFWwindow object
    GLFWwindow* window = glfwCreateWindow(1280, 720, "N-body Simulation", NULL, NULL);
    //Sleep(5000);
    if (window == NULL) {
        fprintf(stderr, "Failed to create GLFW window\n");
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(window);

    // Register the resize callback
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    // Continue with the rest of the initialization...

    // Set up viewport and projection
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    glViewport(0, 0, width, height);
    perspective(45.0f * (float)3.14 / 180.0f, (float)width / (float)height, 0.1f, 100.0f);

    // ... [Rest of your simulation setup]

    // Main loop
    while (!glfwWindowShouldClose(window)) {
        // Clear the screen to black and clear the depth buffer
        glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        // Set up the camera
        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        lookAt(0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f);

        // Simulation update and rendering goes here
        StartTimer();

        // Integrate positions based on velocities
        for (int i = 0; i < nBodies; i++) {
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

        //updateMaxVelocity(h_bodies, nBodies, maxVelocity);

        //updateBodyColors(h_bodies, nBodies, innerRadius, outerRadius);

        updateBodyColorsSpeed(h_bodies, nBodies); // Color based off of speed

        // Draw the bodies
        drawBodies(h_bodies, nBodies); // Make sure to implement this function

        // Swap front and back buffers
        glfwSwapBuffers(window);

        // Poll for and process events
        glfwPollEvents();

        //  const double tElapsed = GetTimer() / 1000.0;
       //   printf("Frame time: %f seconds\n", tElapsed);
    }

    // ... [Cleanup and deallocate resources]

    glfwDestroyWindow(window);

    glfwTerminate();

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
