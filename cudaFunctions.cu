#include "cudaFunctions.cuh"

//Adds to matrices if they are compatible
//Source: https://developer.nvidia.com/blog/even-easier-introduction-cuda/

__global__
void addMatrices(long *a, long *b, long *c, long n) {

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < n; i += stride) {
        c[i] = a[i] + b[i];
    }
}


__global__
void mulMat(long *a, long aColumns, long *b, long bRows, long bColumns, long *c, long cTotal, long cColumns) {

    unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int stride = blockDim.x * gridDim.x;
    for (unsigned int i = index; i < cTotal; i += stride) {
        long sum = 0;//sum of the products
        //multiply the corresponding row & column
        for (long k = 0; k < bRows; ++k) {
            int x = a[((i / cColumns) * aColumns) + k];
            int y = b[(i % bColumns) + (k * bColumns)];
            sum += (x * y);
            //store that value in the correct spot
        }
        c[i] = sum;
    }
}
