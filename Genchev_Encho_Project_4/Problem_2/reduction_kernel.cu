#include <cuda.h>
#include <limits.h>

#define BLOCK_SIZE 1024

__global__ void reduction_kernel(
    unsigned int* hash_array,
    unsigned int* nonce_array,
    unsigned int* out_hash,
    unsigned int* out_nonce,
    unsigned int n
) {
    //shared memory for each block
    __shared__ unsigned int s_hash[BLOCK_SIZE];
    __shared__ unsigned int s_nonce[BLOCK_SIZE];

    //thread id inside block
    unsigned int tid = threadIdx.x;

    //each thread gets two elements
    unsigned int idx = 2 * blockIdx.x * blockDim.x + tid;

    //load first
    unsigned int h1 = UINT_MAX;
    unsigned int n1 = 0;
    if (idx < n) {
        h1 = hash_array[idx];
        n1 = nonce_array[idx];
    }

    //load second
    unsigned int h2 = UINT_MAX;
    unsigned int n2 = 0;
    if (idx + blockDim.x < n) {
        h2 = hash_array[idx + blockDim.x];
        n2 = nonce_array[idx + blockDim.x];
    }

    //take minimum of the two values
    if (h2 < h1) {
        s_hash[tid] = h2;
        s_nonce[tid] = n2;
    } else {
        s_hash[tid] = h1;
        s_nonce[tid] = n1;
    }

    //wait for all threads
    __syncthreads();

    //reduction step
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {

        //half threads in each step
        if (tid < stride) {

            //compare and keep smaller value
            if (s_hash[tid + stride] < s_hash[tid]) {
                s_hash[tid] = s_hash[tid + stride];
                s_nonce[tid] = s_nonce[tid + stride];
            }
        }

        //wait before next step
        __syncthreads();
    }

    //thread 0 writes result for this block
    if (tid == 0) {
        out_hash[blockIdx.x] = s_hash[0];
        out_nonce[blockIdx.x] = s_nonce[0];
    }
}