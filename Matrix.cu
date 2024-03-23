#include "Matrix.cuh"
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include "cudaFunctions.cuh"

//Constructor
Matrix::Matrix(long rows, long columns) : rows_(rows), columns_(columns) {
    //Allocate storage for the values in "shared" memory
    cudaMallocManaged(&values, (rows*columns)*sizeof(long));
}

//Destructor
Matrix::~Matrix() {
    cudaFree(values);
}
//Fill the Matrix with 0s
void Matrix::init() {
    for (long i = 0; i < rows_ * columns_; ++i) {
        values[i] = 0;
    }
}

//Prints the Array in the command line
//Limits: Numbers over 6 digits don't display correctly
void Matrix::print() {
    for (long i = 0; i < rows_ * columns_; ++i) {
        if (!(i % columns_)) std::cout  << '\n';
        std::cout << std::setw(7) << values[i];
    }
    std::cout << std::endl;
}

//Prompts the user for input
void Matrix::input() {
    //Show matrix pattern
    for (long i = 0; i < rows_ * columns_; ++i) {
        if (!(i % columns_)) std::cout  << '\n';
        std::cout << '(' << std::setw(2) << (i%columns_)+1 << '/' << std::setw(2) << (i/rows_)+1 << ") ";
    }
    std::cout << std::endl;

    //Ask for values
    for (long i = 0; i < rows_ * columns_; ++i) {
        std::cout << "Enter Value for (" << (i%columns_)+1 << "/" << (i/rows_)+1 << "): ";
        std::cin >> values[i];
    }
    std::cout << std::endl;
}

//Fills the matrix with random values from 0 to 999999
void Matrix::randomFill() {
    //Seed the randomizer
    srand(time(nullptr));
    for (long i = 0; i < rows_ * columns_; ++i) {
        values[i] = static_cast<long>(rand() % 1000000); //not the best rng, but good enough for this
    }
}


/*
//Adds to matrices if they are compatible
Matrix Matrix::add(const Matrix &M) {
    if (this->rows_ != M.rows_ || this->columns_ != M.columns_)
        throw std::invalid_argument("The matrices aren't the same size");

    Matrix c(rows_, columns_); //Stores the result

    for (long i = 0; i < rows_ * columns_; ++i) {
        c.values[i] = M.values[i] + this->values[i];
    }

    return c;
}
*/

//Adds to matrices if they are compatible
Matrix Matrix::add(const Matrix &M) {
    //Error handling
    if (this->rows_ != M.rows_ || this->columns_ != M.columns_)
        throw std::invalid_argument("The matrices aren't the same size");

    Matrix C(rows_, columns_); //Stores the result
    //Size per CUDA Block (Not 100% sure howe to set this properly, but should be a power of 2)
    int blockSize = 1024;
    int numBlocks = (rows_*columns_ + blockSize - 1) / blockSize;
    //Call the cuda function
    addMatrices<<<numBlocks,blockSize>>>(this->values, M.values, C.values, rows_*columns_);
    //Wait for the GPU to finish
    cudaDeviceSynchronize();
    return C;
}

/*
//Multiplies two matrices if they are compatible
// The matrix of which the function is called is on the LHS
Matrix Matrix::mult(const Matrix &M) {
    //Error handling
    if (this->columns_ != M.rows_)
        throw std::invalid_argument("The matrices can't be multiplied");

    Matrix C(this->rows_, M.columns_); //Stores the result

    for (long i = 0; i < C.rows_ * C.columns_; ++i) {
        long sum = 0;//sum of the products
        //multiply the corresponding row & column
        for (long k = 0; k < M.rows_; ++k) {
            int a = this->values[((i/C.columns_)*this->columns_)+k];
            int b = M.values[(i%M.columns_)+(k*M.columns_)];
            sum += (a * b);
            //store that value in the correct spot
        }
        C.values[i] = sum;
    }
    return C;
}
*/

//Multiplies two matrices if they are compatible
// The matrix of which the function is called is on the LHS
Matrix Matrix::mult(const Matrix &M) {
    //Error handling
    if (this->columns_ != M.rows_)
        throw std::invalid_argument("The matrices can't be multiplied");

    Matrix C(this->rows_, M.columns_); //Stores the result

    //Size per CUDA Block (Not 100% sure howe to set this properly, but should be a power of 2)
    int blockSize = 1024;
    int numBlocks = (rows_*columns_ + blockSize - 1) / blockSize;
    //Call the cuda function
    mulMat<<<numBlocks,blockSize>>>(this->values, this->columns_, M.values, M.rows_, M.columns_, C.values, C.columns_*C.rows_, C.columns_);
    //Wait for the GPU to finish
    cudaDeviceSynchronize();
    return C;
}