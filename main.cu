#include "Matrix.cuh"

#include <chrono>
#include <iostream>

#define TIME_MS 60'000
#define SIZE 100000

void timeAdd1() {
    auto duration = std::chrono::milliseconds::zero(); //initialize for later
    long size = SIZE;
    do {
        //Setup matrices
        Matrix m1(size, size);
        Matrix m2(size, size);
        m1.randomFill();
        m2.randomFill();

        //Setup clock
        auto start = std::chrono::high_resolution_clock::now();

        //Operation to time
        m1.add(m2);

        //Stop clock and get time
        auto stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Matrix Size: " << size << "x" << size << " | Duration: " << duration << '\n';
        size++;
    } while (duration.count() < TIME_MS); //Check if TIME_NS has passed
    //print results
    std::cout << "Max size for matrix addition before taking " << TIME_MS << "ms: a=" << size - 1 << " b=" << size - 1
              << std::endl;
}

void timeMult1() {
    auto duration = std::chrono::milliseconds::zero(); //initialize for later
    long size = SIZE;
    do {
        //Setup matrices
        Matrix m1(size, size);
        Matrix m2(size, size);
        m1.randomFill();
        m2.randomFill();

        //Setup clock
        auto start = std::chrono::high_resolution_clock::now();

        //Operation to time
        m1.mult(m2);

        //Stop clock and get time
        auto stop = std::chrono::high_resolution_clock::now();
        duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);
        std::cout << "Matrix Size: " << size << "x" << size << " | Duration: " << duration << '\n';
        size++;
    } while (duration.count() < TIME_MS); //Check if TIME_NS has passed
    //print results
    std::cout << "Max size for matrix multiplication before taking " << TIME_MS << "ms: a=" << size - 1 << " b="
              << size - 1 << std::endl;
}

int main() {
    timeAdd1();
    return 0;
}