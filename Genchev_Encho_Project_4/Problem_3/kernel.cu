#include <cuda.h>

#define BLUR_SIZE 2
#define TILE_SIZE 16

__global__ void convolutionKernel(int* input, int* output, int* filter, int n_row, int n_col)
{
    //tile including halo for blur
    __shared__ int tile[TILE_SIZE + 2*BLUR_SIZE][TILE_SIZE + 2*BLUR_SIZE];

    //thread coords inside the block
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    //coordinates of the output element
    int row_o = blockIdx.y * TILE_SIZE + ty;
    int col_o = blockIdx.x * TILE_SIZE + tx;

    //load tile 
    for (int i = ty; i < TILE_SIZE + 2*BLUR_SIZE; i += blockDim.y)
    {
        for (int j = tx; j < TILE_SIZE + 2*BLUR_SIZE; j += blockDim.x)
        {
            //map tile index to input coords
            int row_i = blockIdx.y * TILE_SIZE + i - BLUR_SIZE;
            int col_i = blockIdx.x * TILE_SIZE + j - BLUR_SIZE;

            //check bounds
            if (row_i >= 0 && row_i < n_row && col_i >= 0 && col_i < n_col)
                tile[i][j] = input[row_i * n_col + col_i];
            else
                tile[i][j] = 0;
        }
    }

    __syncthreads();

    //compute convolution
    if (row_o < n_row && col_o < n_col)
    {
        int sum_val = 0;

        //5x5 filter on tile
        for (int i = 0; i < 5; i++)
        {
            for (int j = 0; j < 5; j++)
            {
                //multiply
                sum_val += tile[ty + i][tx + j] * filter[i*5 + j];
            }
        }

        //write back to global memory
        output[row_o * n_col + col_o] = sum_val;
    }
}