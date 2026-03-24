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
    __shared__ unsigned int s_hash[BLOCK_SIZE];
    __shared__ unsigned int s_nonce[BLOCK_SIZE];

    unsigned int tid = threadIdx.x;
    unsigned int idx = 2 * blockIdx.x * blockDim.x + tid;

    // Load two elements per thread (like professor pattern)
    unsigned int h1 = (idx < n) ? hash_array[idx] : UINT_MAX;
    unsigned int n1 = (idx < n) ? nonce_array[idx] : 0;

    unsigned int h2 = (idx + blockDim.x < n) ? hash_array[idx + blockDim.x] : UINT_MAX;
    unsigned int n2 = (idx + blockDim.x < n) ? nonce_array[idx + blockDim.x] : 0;

    // Combine immediately (min)
    if (h2 < h1) {
        s_hash[tid] = h2;
        s_nonce[tid] = n2;
    } else {
        s_hash[tid] = h1;
        s_nonce[tid] = n1;
    }

    __syncthreads();

    // Reduction tree
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            if (s_hash[tid + stride] < s_hash[tid]) {
                s_hash[tid] = s_hash[tid + stride];
                s_nonce[tid] = s_nonce[tid + stride];
            }
        }
        __syncthreads();
    }

    // Write result
    if (tid == 0) {
        out_hash[blockIdx.x] = s_hash[0];
        out_nonce[blockIdx.x] = s_nonce[0];
    }
}