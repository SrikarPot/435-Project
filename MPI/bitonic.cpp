#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../helper.h"

int rank, size;
int n;

void bitonicSort(float* values, int start, int length, int dir);
void bitonicMerge(float* values, int start, int length, int dir);
void exchange(float &a, float &b);
void bitonicGlobalSort(float* arr, int n, int dir) {
    for (int k = 2; k <= n; k = 2 * k) {
        for (int j = k >> 1; j > 0; j = j >> 1) {
            for (int i = 0; i < n; i++) {
                int ixj = i ^ j;
                if (ixj > i) {
                    if ((i & k) == 0 && arr[i] > arr[ixj]) {
                        exchange(arr[i], arr[ixj]);
                    }
                    if ((i & k) != 0 && arr[i] < arr[ixj]) {
                        exchange(arr[i], arr[ixj]);
                    }
                }
            }
        }
    }
}
int main(int argc, char *argv[]) {
    CALI_MARK_BEGIN("main");

    n = atoi(argv[1]); // Change this to your desired array size
    float *arr ;
    std::string input_type = argv[2];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        CALI_MARK_BEGIN("data_init");
        arr = (float *)malloc(n * sizeof(float));
        array_fill(arr, n, input_type);
        CALI_MARK_END("data_init");
    }

    int chunkSize = n / size; // Assuming n is a multiple of size

    float *local_arr = (float *)malloc(chunkSize * sizeof(float));
    CALI_MARK_BEGIN("MPI_Scatter");
    MPI_Scatter(arr, chunkSize, MPI_FLOAT, local_arr, chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");

    CALI_MARK_BEGIN("comp");
    bitonicSort(local_arr, 0, chunkSize, 1);
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(local_arr, chunkSize, MPI_FLOAT, arr, chunkSize, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    free(local_arr);
    if (rank == 0) {
        // Global sorting of the gathered subarrays
        CALI_MARK_BEGIN("global_sort");
        bitonicGlobalSort(arr, n, 1);
        CALI_MARK_END("global_sort");

        // Correctness check
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
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("Algorithm", "BitonicSort");
    adiak::value("ProgrammingModel", "MPI");
    adiak::value("Datatype", "float");
    adiak::value("SizeOfDatatype", 4);
    adiak::value("InputSize", n);
    adiak::value("InputType", (char*)input_type.c_str());
    adiak::value("num_procs", size);
    adiak::value("group_num", 15);
    adiak::value("implementation_source", "ONLINE");

    MPI_Finalize();
    return 0;
}

void bitonicSort(float* values, int start, int length, int dir) {
    if (length > 1) {
        int k = length / 2;
        bitonicSort(values, start, k, 1);
        bitonicSort(values, start + k, k, 0);
        bitonicMerge(values, start, length, dir);
    }
}

void bitonicMerge(float* values, int start, int length, int dir) {
    if (length > 1) {
        int k = length / 2;
        for (int i = start; i < start + k; i++) {
            if (dir == (values[i] > values[i + k])) {
                exchange(values[i], values[i + k]);
            }
        }
        bitonicMerge(values, start, k, dir);
        bitonicMerge(values, start + k, k, dir);
    }
}

void exchange(float &a, float &b) {
    float temp = a;
    a = b;
    b = temp;
}
