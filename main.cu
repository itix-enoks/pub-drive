#include <stdio.h>
#include "support.h"

#define BLOCK_SIZE 32

__global__ void squared_matrix_multiply(float* A, float* B, float* C, int size){
    __shared__ float tile_A[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_B[BLOCK_SIZE][BLOCK_SIZE];

    int row = blockIdx.y * BLOCK_SIZE + threadIdx.y;
    int col = blockIdx.x * BLOCK_SIZE + threadIdx.x;

    float value = 0.0f;

    for (int i = 0; i < (size + BLOCK_SIZE - 1) / BLOCK_SIZE; ++i) {
        // Load tiles into shared memory
        if (row < size && i * BLOCK_SIZE + threadIdx.x < size)
            tile_A[threadIdx.y][threadIdx.x] = A[row * size + i * BLOCK_SIZE + threadIdx.x];
        else
            tile_A[threadIdx.y][threadIdx.x] = 0.0f;

        if (col < size && i * BLOCK_SIZE + threadIdx.y < size)
            tile_B[threadIdx.y][threadIdx.x] = B[(i * BLOCK_SIZE + threadIdx.y) * size + col];
        else
            tile_B[threadIdx.y][threadIdx.x] = 0.0f;

        __syncthreads();

        // Multiply the tiles
        for (int j = 0; j < BLOCK_SIZE; ++j)
            value += tile_A[threadIdx.y][j] * tile_B[j][threadIdx.x];

        __syncthreads();
    }

    if (row < size && col < size)
        C[row * size + col] = value;
}

void matMulCPU(float* A, float* B, float* C, int size) {
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            float sum = 0.0;
            for (int k = 0; k < size; ++k) {
                sum += A[i * size + k] * B[k * size + j];
            }
            C[i * size + j] = sum;
        }
    }
}


int main(int argc, char **argv){
    Timer timer;
    cudaError_t cuda_ret;

    // Initialize host variables ----------------------------------------------

    printf("\nSetting up the problem...");
    fflush(stdout);
    startTime(&timer);

    float *A_h, *B_h, *C_h;
    float *A_d, *B_d, *C_d;
    unsigned size;
    unsigned matrix_size;
    dim3 dim_grid, dim_block;

    if (argc == 1){
        size = 1000;
    }
    else if (argc == 2){
        size = atoi(argv[1]);
    }
    else{
        printf("\n    Invalid input parameters!"
               "\n    Usage: ./matmult                # All matrices are 1000 x 1000"
               "\n    Usage: ./matmult <m>            # All matrices are m x m"
               "\n");
        exit(0);
    }

    matrix_size = size * size;

    A_h = (float *)malloc(sizeof(float) * matrix_size);
    for (unsigned int i = 0; i < matrix_size; i++){
        A_h[i] = (rand() % 100) / 100.00;
    }

    B_h = (float *)malloc(sizeof(float) * matrix_size);
    for (unsigned int i = 0; i < matrix_size; i++){
        B_h[i] = (rand() % 100) / 100.00;
    }

    C_h = (float *)malloc(sizeof(float) * matrix_size);

    stopTime(&timer);
    printf("%f ms\n", elapsedTime(timer));
    printf("    Matrix size: %u x %u\n", size, size);

    // Comparison with CPU ----------------------------------------------------

    printf("Baseline CPU matrix multiplication...");
    fflush(stdout);
    startTime(&timer);
    // matMulCPU(A_h, B_h, C_h, size);
    stopTime(&timer);
    printf("%f ms\n", elapsedTime(timer));

    // Allocate device variables ----------------------------------------------

    printf("Allocating device variables...");
    fflush(stdout);
    startTime(&timer);

    /* Add your code here to allocate device memory
        Do not forget to check for errors */
    cuda_ret = cudaMalloc((void**)&A_d, sizeof(float) * matrix_size);
    if (cuda_ret != cudaSuccess) { fprintf(stderr, "cudaMalloc A_d failed\n"); exit(1); }

    cuda_ret = cudaMalloc((void**)&B_d, sizeof(float) * matrix_size);
    if (cuda_ret != cudaSuccess) { fprintf(stderr, "cudaMalloc B_d failed\n"); exit(1); }

    cuda_ret = cudaMalloc((void**)&C_d, sizeof(float) * matrix_size);
    if (cuda_ret != cudaSuccess) { fprintf(stderr, "cudaMalloc C_d failed\n"); exit(1); } 
    

    stopTime(&timer);
    printf("%f ms\n", elapsedTime(timer));

    // Copy host variables to device ------------------------------------------

    printf("Copying data from host to device...");
    fflush(stdout);
    startTime(&timer);

    /* Add your code here to copy host variables to device
        Do not forget to check for errors */
    cuda_ret = cudaMemcpy(A_d, A_h, sizeof(float) * matrix_size, cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) { fprintf(stderr, "cudaMemcpy A_h -> A_d failed\n"); exit(1); }

    cuda_ret = cudaMemcpy(B_d, B_h, sizeof(float) * matrix_size, cudaMemcpyHostToDevice);
    if (cuda_ret != cudaSuccess) { fprintf(stderr, "cudaMemcpy B_h -> B_d failed\n"); exit(1); }
    
    

    stopTime(&timer);
    printf("%f ms\n", elapsedTime(timer));

    // Launch kernel ----------------------------------------------------------

    printf("Launching kernel...");
    fflush(stdout);
    startTime(&timer);

    /* Add your code here to set size and call your kernel
        Do not forget to check for errors */
    dim_block = dim3(BLOCK_SIZE, BLOCK_SIZE);
    dim_grid = dim3((size + BLOCK_SIZE - 1) / BLOCK_SIZE, (size + BLOCK_SIZE - 1) / BLOCK_SIZE);

    squared_matrix_multiply<<<dim_grid, dim_block>>>(A_d, B_d, C_d, size);

    cuda_ret = cudaGetLastError();
    if (cuda_ret != cudaSuccess) { fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cuda_ret)); exit(1); }

    cuda_ret = cudaDeviceSynchronize();
    if (cuda_ret != cudaSuccess) { fprintf(stderr, "cudaDeviceSynchronize failed\n"); exit(1); }
    
    

    stopTime(&timer);
    printf("%f ms\n", elapsedTime(timer));

    // Copy device variables from device to host ------------------------------

    printf("Copying data from device to host...");
    fflush(stdout);
    startTime(&timer);

    /* Add your code here to copy devide variables back to host
        Do not forget to check for errors */
    cuda_ret = cudaMemcpy(C_h, C_d, sizeof(float) * matrix_size, cudaMemcpyDeviceToHost);
    if (cuda_ret != cudaSuccess) { fprintf(stderr, "cudaMemcpy C_d -> C_h failed\n"); exit(1); }

    

    stopTime(&timer);
    printf("%f ms\n", elapsedTime(timer));

    // Free device memory ----------------------------------------------------

    /* Add here your code to free device memory if you did not use CudaMallocManaged */    
    cudaFree(A_d);
    cudaFree(B_d);
    cudaFree(C_d);
    
    

    // Verify correctness -----------------------------------------------------

    /* Uncommend the next lines to check with the CPU if your multiplication is working for small matrices */
    // printf("Verifying results...");
    // fflush(stdout);
    // verify(A_h, B_h, C_h, size, size, size);

    // Free host memory ------------------------------------------------------------

    free(A_h);
    free(B_h);
    free(C_h);

    return 0;
}