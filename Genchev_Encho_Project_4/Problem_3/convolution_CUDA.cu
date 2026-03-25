#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <time.h>
#include <cuda.h>

#include "support.h"
#include "kernel.cu"

#define BILLION  1000000000.0
#define MAX_LINE_LENGTH 25000

#define BLUR_SIZE 2
#define TILE_SIZE 16

// Kernel declaration
__global__ void convolutionKernel(int* input, int* output, int* filter, int n_row, int n_col);

int main (int argc, char *argv[])
{
    // Check console errors
    if( argc != 6)
    {
        printf("USE LIKE THIS: convolution_CUDA_parallel n_row n_col mat_input.csv mat_output.csv time.csv\n");
        return EXIT_FAILURE;
    }

    // Get dims
    int n_row = strtol(argv[1], NULL, 10);
    int n_col = strtol(argv[2], NULL, 10);

    // Get files to read/write 
    FILE* inputFile1 = fopen(argv[3], "r");
    if (inputFile1 == NULL){
        printf("Could not open file %s",argv[2]);
        return EXIT_FAILURE;
    }
    FILE* outputFile = fopen(argv[4], "w");
    FILE* timeFile  = fopen(argv[5], "w");

    // Matrices to use
    int* filterMatrix_h = (int*)malloc(5 * 5 * sizeof(int));
    int* inputMatrix_h  = (int*) malloc(n_row * n_col * sizeof(int));
    int* outputMatrix_h = (int*) malloc(n_row * n_col * sizeof(int));

    // read the data from the file
    int row_count = 0;
    char line[MAX_LINE_LENGTH] = {0};
    while (fgets(line, MAX_LINE_LENGTH, inputFile1)) {
        if (line[strlen(line) - 1] != '\n') printf("\n");
        char *token;
        const char s[2] = ",";
        token = strtok(line, s);
        int i_col = 0;
        while (token != NULL) {
            inputMatrix_h[row_count*n_col + i_col] = strtol(token, NULL,10 );
            i_col++;
            token = strtok (NULL, s);
        }
        row_count++;
    }

    // Filling filter
	// 1 0 0 0 1 
	// 0 1 0 1 0 
	// 0 0 1 0 0 
	// 0 1 0 1 0 
	// 1 0 0 0 1 
    for(int i = 0; i< 5; i++)
        for(int j = 0; j< 5; j++)
            filterMatrix_h[i*5+j]=0;

    filterMatrix_h[0*5+0] = 1;
    filterMatrix_h[1*5+1] = 1;
    filterMatrix_h[2*5+2] = 1;
    filterMatrix_h[3*5+3] = 1;
    filterMatrix_h[4*5+4] = 1;
    
    filterMatrix_h[4*5+0] = 1;
    filterMatrix_h[3*5+1] = 1;
    filterMatrix_h[1*5+3] = 1;
    filterMatrix_h[0*5+4] = 1;

    fclose(inputFile1); 

    // Device matrices
    int *inputMatrix_d, *outputMatrix_d, *filterMatrix_d;

    cudaMalloc((void**)&inputMatrix_d, n_row*n_col*sizeof(int));
    cudaMalloc((void**)&outputMatrix_d, n_row*n_col*sizeof(int));
    cudaMalloc((void**)&filterMatrix_d, 5*5*sizeof(int));

    struct timespec start, end;

    // ---------------- Transfer Host to Device ---------------- //
    clock_gettime(CLOCK_REALTIME, &start);

    cudaMemcpy(inputMatrix_d, inputMatrix_h, n_row*n_col*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(filterMatrix_d, filterMatrix_h, 5*5*sizeof(int), cudaMemcpyHostToDevice);

    clock_gettime(CLOCK_REALTIME, &end);
    double time_h2d = (end.tv_sec - start.tv_sec) +
                      (end.tv_nsec - start.tv_nsec) / BILLION;

    // ---------------- Kernel Execution ---------------- //
    dim3 blockSize(TILE_SIZE, TILE_SIZE);
    dim3 gridSize((n_col + TILE_SIZE - 1)/TILE_SIZE, (n_row + TILE_SIZE - 1)/TILE_SIZE);

    clock_gettime(CLOCK_REALTIME, &start);

    convolutionKernel<<<gridSize, blockSize>>>(inputMatrix_d, outputMatrix_d, filterMatrix_d, n_row, n_col);
    cudaDeviceSynchronize();

    clock_gettime(CLOCK_REALTIME, &end);
    double time_kernel = (end.tv_sec - start.tv_sec) +
                         (end.tv_nsec - start.tv_nsec) / BILLION;

    // ---------------- Transfer Device to Host ---------------- //
    clock_gettime(CLOCK_REALTIME, &start);

    cudaMemcpy(outputMatrix_h, outputMatrix_d, n_row*n_col*sizeof(int), cudaMemcpyDeviceToHost);

    clock_gettime(CLOCK_REALTIME, &end);
    double time_d2h = (end.tv_sec - start.tv_sec) +
                      (end.tv_nsec - start.tv_nsec) / BILLION;

    // Save output matrix as csv file
    for (int i = 0; i<n_row; i++)
    {
        for (int j = 0; j<n_col; j++)
        {
            fprintf(outputFile, "%d", outputMatrix_h[i*n_col +j]);
            if (j != n_col -1)
                fprintf(outputFile, ",");
            else if ( i < n_row-1)
                fprintf(outputFile, "\n");
        }
    }

    // Print time (3 rows)
    fprintf(timeFile, "%.20f\n", time_h2d);
    fprintf(timeFile, "%.20f\n", time_kernel);
    fprintf(timeFile, "%.20f\n", time_d2h);

    // Cleanup
    fclose (outputFile);
    fclose (timeFile);

    cudaFree(inputMatrix_d);
    cudaFree(outputMatrix_d);
    cudaFree(filterMatrix_d);

    free(inputMatrix_h);
    free(outputMatrix_h);
    free(filterMatrix_h);

    return 0;
}