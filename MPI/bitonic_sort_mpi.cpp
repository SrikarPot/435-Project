#include <iostream>
#include <cstdlib>
#include <ctime>
#include <algorithm>
#include <mpi.h>
#include "helper.h"
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

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
    CALI_MARK_BEGIN("main");

    MPI_Init(&argc, &argv);
    int rank, numProcs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

    int n = std::atoi(argv[1]);
    int processors = numProcs;
    std::string input_type = argv[2];
    
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
        CALI_MARK_BEGIN("data_init");
        float* globalArray = (float*) malloc( n * sizeof(float));
        array_fill(globalArray, n, input_type);
        CALI_MARK_END("data_init");

        // Scatter the global array to local arrays
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("MPI_Scatter");
        MPI_Scatter(globalArray, elementsPerProc, MPI_FLOAT, localArray, elementsPerProc, MPI_FLOAT, 0, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Scatter");
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
        delete[] globalArray;
    } else {
        // Receive local array from root process
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("MPI_Scatter");
        MPI_Scatter(nullptr, 0, MPI_FLOAT, localArray, elementsPerProc, MPI_FLOAT, 0, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Scatter");
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
    }

    // Sort local array using Bitonic Sort
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    bitonicSort(localArray, 0, elementsPerProc, 1);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Perform parallel Bitonic Merge
    for (int step = 2; step <= processors; step *= 2) {
        for (int subStep = step / 2; subStep > 0; subStep /= 2) {
            for (int i = 0; i < elementsPerProc; i += subStep) {
                CALI_MARK_BEGIN("comp");
                CALI_MARK_BEGIN("comp_large");
                bitonicMerge(localArray, i, subStep, (i / (elementsPerProc / step)) % 2);
                CALI_MARK_END("comp_large");
                CALI_MARK_END("comp");
            }
        }
    }

    
    if (rank == 0) {
        // Gather the sorted local arrays back to the global array
        float* sortedArray = (float*) malloc( n * sizeof(float));
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("MPI_Gather");
        MPI_Gather(localArray, elementsPerProc, MPI_FLOAT, sortedArray, elementsPerProc, MPI_FLOAT, 0, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Gather");
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
        
        // Merge the sorted subarrays to get the final sorted array
        CALI_MARK_BEGIN("comp");
        CALI_MARK_BEGIN("comp_large");
        bitonicSort(sortedArray, 0, n, 1);
        CALI_MARK_END("comp_large");
        CALI_MARK_END("comp");

        CALI_MARK_BEGIN("correctness_check");
        if (correctness_check(sortedArray, n) == 1) {
            std::cout << "Sorted Correctly!" << std::endl;
        } else {
            std::cout << "Did not sort correctly." << std::endl;
        }
        CALI_MARK_END("correctness_check");

        delete[] sortedArray;
    } else {
        // Send local sorted array to the root process
        CALI_MARK_BEGIN("comm");
        CALI_MARK_BEGIN("comm_large");
        CALI_MARK_BEGIN("MPI_Gather");
        MPI_Gather(localArray, elementsPerProc, MPI_FLOAT, nullptr, 0, MPI_FLOAT, 0, MPI_COMM_WORLD);
        CALI_MARK_END("MPI_Gather");
        CALI_MARK_END("comm_large");
        CALI_MARK_END("comm");
    }

    delete[] localArray;

    CALI_MARK_END("main");

    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "Bitonic"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", input_type.c_str()); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", processors); // The number of processors (MPI ranks)
    // adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 15); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "ONLINE"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    MPI_Finalize();

    return 0;
}