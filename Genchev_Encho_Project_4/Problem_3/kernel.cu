#include <cuda.h>

#define BLUR_SIZE 2
#define TILE_SIZE 16

__global__ void convolutionKernel(int* input, int* output, int* filter, int n_row, int n_col)
{
    __shared__ int tile[TILE_SIZE + 2*BLUR_SIZE][TILE_SIZE + 2*BLUR_SIZE];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row_o = blockIdx.y * TILE_SIZE + ty;
    int col_o = blockIdx.x * TILE_SIZE + tx;

    // Load entire shared memory tile (including halo)
    for (int i = ty; i < TILE_SIZE + 2*BLUR_SIZE; i += blockDim.y)
    {
        for (int j = tx; j < TILE_SIZE + 2*BLUR_SIZE; j += blockDim.x)
        {
            int row_i = blockIdx.y * TILE_SIZE + i - BLUR_SIZE;
            int col_i = blockIdx.x * TILE_SIZE + j - BLUR_SIZE;

            if (row_i >= 0 && row_i < n_row && col_i >= 0 && col_i < n_col)
                tile[i][j] = input[row_i * n_col + col_i];
            else
                tile[i][j] = 0;
        }
    }

    __syncthreads();

    // Compute convolution
    if (row_o < n_row && col_o < n_col)
    {
        int sum_val = 0;

        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                sum_val += tile[ty + i][tx + j] * filter[i*5 + j];
            }
        }

        output[row_o * n_col + col_o] = sum_val;
    }
}