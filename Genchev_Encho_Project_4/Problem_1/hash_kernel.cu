#define MAX 123123123
__global__ void hash_kernel(
    unsigned int* nonce_array,
    unsigned int* transactions,
    unsigned int* hash_array,
    unsigned int n_transactions,
    unsigned int trials
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < trials) {
        unsigned int nonce = nonce_array[i];

        unsigned int hash = (nonce + transactions[0] * (i + 1)) % MAX;

        for (int j = 1; j < n_transactions; j++) {
            hash = (hash + transactions[j] * (i + 1)) % MAX;
        }

        hash_array[i] = hash;
    }
}