#include "kernel.h"
#include <iostream>


#define TILE_WIDTH 32
__global__ void MatrixMulKernel(float *d_A, float *d_B, float *d_C, int M, int K, int N)
{
    // Tiled
    __shared__ float Ads[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Bds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x, by = blockIdx.y;
    int tx = threadIdx.x, ty = threadIdx.y;
    int row = by * TILE_WIDTH + ty;
    int col = bx * TILE_WIDTH + tx;

    float _P_x = 0.0;

    // t for tile index
    for (int t = 0; t < K; t += TILE_WIDTH)
    {
        if (row < M && t + tx < K)
            // memory coalescing: C1 + ty * K + tx
            Ads[ty][tx] = d_A[row * K + t + tx]; // A[row][t + tx];
        else
            Ads[ty][tx] = 0.0;

        if (col < N && t + ty < K)
            // memory coalescing: C2 + ty * N + tx
            Bds[ty][tx] = d_B[(t + ty) * N + col]; // B[t + ty][col];
        else
            Bds[ty][tx] = 0.0;

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            _P_x += Ads[ty][k] * Bds[k][tx]; // avoid shared memory bank conflict

        __syncthreads();
    }
    if (row < M && col < N)
        d_C[row * N + col] = _P_x;
}

void MatrixMuliplication(array2d_t<float> &input1, array2d_t<float> &input2, array2d_t<float> &output)
{
    int M = input1.row_count;
    int K = input1.col_count;
    int N = input2.col_count;
    // Throw an error or handle the dimension mismatch case
    if (input2.row_count != K || output.row_count != M || output.col_count != N)
    {
        std::cerr << "Error: Matrix dimensions are not compatible for multiplication." << std::endl;
        return;
    }

    dim3 dimGrid(ceil(N / (float)TILE_WIDTH), ceil(M / (float)TILE_WIDTH), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
    MatrixMulKernel<<<dimGrid, dimBlock>>>(input1.data_ptr, input2.data_ptr, output.data_ptr, M, K, N);
}