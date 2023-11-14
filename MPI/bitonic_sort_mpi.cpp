#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mpi.h>
#include "helper.h"

void compareExchange(float* arr, int i, int j, int dir) {
    if ((arr[i] > arr[j]) == dir) {
        std::swap(arr[i], arr[j]);
    }
}

void bitonicMerge(float* arr, int start, int length, int dir) {
    if (length > 1) {
        int k = length / 2;
        for (int i = start; i < start + k; ++i) {
            compareExchange(arr, i, i + k, dir);
        }
        bitonicMerge(arr, start, k, dir);
        bitonicMerge(arr, start + k, k, dir);
    }
}


void bitonicSort(float* arr, int start, int length, int dir) {
    if (length > 1) {
        int k = length / 2;
        bitonicSort(arr, start, k, 1);
        bitonicSort(arr, start + k, k, 0);
        bitonicMerge(arr, start, length, dir);
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int n = std::atoi(argv[1]);
    int processors = numProcs;
    
    // Broadcast user input to all processes
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&processors, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Ensure the number of processors and number of values is a power of 2
    if (processors != 1 && (processors & (processors - 1)) != 0) {
        if (rank == 0) {
            std::cerr << "Number of processors must be a power of 2." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }
    if (n != 1 && (n & (n - 1)) != 0) {
        if (rank == 0) {
            std::cerr << "Number of values must be a power of 2." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Ensure the number of elements is a multiple of the number of processors
    int elementsPerProc = n / processors;
    if (n % processors != 0) {
        if (rank == 0) {
            std::cerr << "Number of elements must be a multiple of the number of processors." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    float* localArray = new float[elementsPerProc];

    if (rank == 0) {
        // Generate a random array of size n
        std::srand(static_cast<unsigned int>(time(0)));
        float* globalArray = (float*) malloc( n * sizeof(float));
        array_fill(globalArray, n, "Random");

        // Scatter the global array to local arrays
        MPI_Scatter(globalArray, elementsPerProc, MPI_FLOAT, localArray, elementsPerProc, MPI_FLOAT, 0, MPI_COMM_WORLD);

        delete[] globalArray;
    } else {
        // Receive local array from root process
        MPI_Scatter(nullptr, 0, MPI_FLOAT, localArray, elementsPerProc, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    // Sort local array using Bitonic Sort
    bitonicSort(localArray, 0, elementsPerProc, 1);

    // Perform parallel Bitonic Merge
    for (int step = 2; step <= processors; step *= 2) {
        for (int subStep = step / 2; subStep > 0; subStep /= 2) {
            for (int i = 0; i < elementsPerProc; i += subStep) {
                bitonicMerge(localArray, i, subStep, (i / (elementsPerProc / step)) % 2);
            }
        }
    }

    
    if (rank == 0) {
        // Gather the sorted local arrays back to the global array
        float* sortedArray = (float*) malloc( n * sizeof(float));
        MPI_Gather(localArray, elementsPerProc, MPI_FLOAT, sortedArray, elementsPerProc, MPI_FLOAT, 0, MPI_COMM_WORLD);
        
        // Merge the sorted subarrays to get the final sorted array
        bitonicSort(sortedArray, 0, n, 1);

        
        if (correctness_check(sortedArray, n) == 1) {
            std::cout << "Sorted Correctly!" << std::endl;
        } else {
            std::cout << "Did not sort correctly." << std::endl;
        }

        delete[] sortedArray;
    } else {
        // Send local sorted array to the root process
        MPI_Gather(localArray, elementsPerProc, MPI_FLOAT, nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
    }

    delete[] localArray;

    MPI_Finalize();

    return 0;
}