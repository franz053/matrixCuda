#ifndef ARRAYCUDA_CUDAFUNCTIONS_CUH
#define ARRAYCUDA_CUDAFUNCTIONS_CUH

__global__
void addMatrices(long *a, long *b, long *c, long n);
__global__
void mulMat(long *a, long aColumns, long *b, long bRows, long bColumns, long *c, long cTotal, long cColumns);


#endif //ARRAYCUDA_CUDAFUNCTIONS_CUH