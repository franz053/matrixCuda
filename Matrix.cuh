#ifndef ARRAYCUDA_MATRIX_CUH
#define ARRAYCUDA_MATRIX_CUH


class Matrix {
private:
    long *values;
    long rows_;
    long columns_;
public:
    Matrix(long rows, long columns);

    ~Matrix();

    void init();

    void print();

    void input();

    void randomFill();

    Matrix add(const Matrix &M);

    Matrix mult(const Matrix &M);
};


#endif //ARRAYCUDA_MATRIX_CUH
