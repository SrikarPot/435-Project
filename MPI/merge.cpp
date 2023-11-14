#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../helper.h"

int rank, size;
int n, loc;

void bitonicMerge(float* values, int l, int r, int step, int dir) {
    if (step > 0) {
        int k = step / 2;

        for (int i = l; i < r - k; ++i) {
            if ((i & step) == 0 && ((values[i] > values[i + k] && dir == 1) || (values[i] < values[i + k] && dir == 0))) {
                float temp = values[i];
                values[i] = values[i + k];
                values[i + k] = temp;
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);

        bitonicMerge(values, l, l + k, k, dir);
        bitonicMerge(values, l + k, r, k, dir);
    }
}

void bitonicSort(float* values, int l, int r, int step, int dir) {
    if (step > 0) {
        int k = step / 2;

        bitonicSort(values, l, l + k, k, 1);  // bitonic sort on ascending order
        bitonicSort(values, l + k, r, k, 0);  // bitonic sort on descending order

        MPI_Barrier(MPI_COMM_WORLD);

        bitonicMerge(values, l, r, step, dir);
    }
}

int main(int argc, char *argv[]) {
    CALI_MARK_BEGIN("main");

    n = atoi(argv[1]); // Change this to your desired array size
    float *arr;
    int local_n;
    std::string input_type = argv[2];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        CALI_MARK_BEGIN("data_init");
        arr = (float*)malloc(n * sizeof(float));
        array_fill(arr, n, input_type);
        CALI_MARK_END("data_init");
    }

    for (int subarr_size = n / size, i = 1; subarr_size <= n; subarr_size <<= 1, i <<= 1) {
        MPI_Comm custom_comm;
        MPI_Comm_split(MPI_COMM_WORLD, (rank % i == 0) ? 1 : MPI_UNDEFINED, rank, &custom_comm);
        if (rank % i == 0) {
            int custom_rank;

            MPI_Comm_rank(custom_comm, &custom_rank);
            float *local_arr = (float*)malloc(subarr_size * sizeof(float));

            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_large");
            CALI_MARK_BEGIN("MPI_Scatter");
            MPI_Scatter(arr, subarr_size, MPI_FLOAT, local_arr, subarr_size, MPI_FLOAT, 0, custom_comm);
            CALI_MARK_END("MPI_Scatter");
            CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");

            CALI_MARK_BEGIN("comp");
            CALI_MARK_BEGIN("comp_large");
            bitonicSort(local_arr, 0, subarr_size, subarr_size, 1); // initial direction is ascending
            CALI_MARK_END("comp_large");
            CALI_MARK_END("comp");

            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_large");
            CALI_MARK_BEGIN("MPI_Gather");
            MPI_Gather(local_arr, subarr_size, MPI_FLOAT, arr, subarr_size, MPI_FLOAT, 0, custom_comm);
            CALI_MARK_END("MPI_Gather");
            CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");

            free(local_arr);
        }
    }

    if (rank == 0) {
        CALI_MARK_BEGIN("correctness_check");
        if (correctness_check(arr, n)) {
            printf("Array correctly sorted!\n");
        } else {
            printf("Array sorting failed\n");
        }
        CALI_MARK_END("correctness_check");
        free(arr);
    }

    CALI_MARK_END("main");
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "BitonicSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", n); // The number of elements in input dataset (1000)
    adiak::value("InputType", (char*)input_type.c_str()); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", size); // The number of processors (MPI ranks)
    // adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 15); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "ONLINE"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    MPI_Finalize();
    return 0;
}
