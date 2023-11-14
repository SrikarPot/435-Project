#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../helper.h"

#include <mpi.h>
#include <iostream>
#include <cstdlib>
#include "helper.h"  // Include your helper header

#include <algorithm>


using namespace std;


int world_rank, world_size;

void Merge_low(float local_A[], const float temp_B[], float temp_C[], int local_n) {
    int ai, bi, ci;

    ai = 0;
    bi = 0;
    ci = 0;

    while (ci < local_n) {
        if (ai < local_n && (bi >= local_n || local_A[ai] <= temp_B[bi])) {
            temp_C[ci] = local_A[ai];
            ai++;
        } else {
            temp_C[ci] = temp_B[bi];
            bi++;
        }
        ci++;
    }

    // Copy the sorted elements back into the original array
    for (ai = 0; ai < local_n; ai++)
        local_A[ai] = temp_C[ai];
}

void Merge_high(float local_A[], const float temp_B[], float temp_C[], int local_n) {
    int ai, bi, ci;

    ai = local_n - 1;
    bi = local_n - 1;
    ci = local_n - 1;

    while (ci >= 0) {
        if (ai >= 0 && (bi < 0 || local_A[ai] >= temp_B[bi])) {
            temp_C[ci] = local_A[ai];
            ai--;
        } else {
            temp_C[ci] = temp_B[bi];
            bi--;
        }
        ci--;
    }

    // Copy the sorted elements back into the original array
    for (ai = 0; ai < local_n; ai++)
        local_A[ai] = temp_C[ai];
}





// Comparison function for floats
int Compare_floats(const void* a, const void* b) {
   // Cast pointers to floats and perform comparison
   float fa = *(const float*)a;
   float fb = *(const float*)b;
   return (fa > fb) - (fa < fb);
}



void Odd_even_iter(float local_A[], float temp_B[], float temp_C[],
        int local_n, int phase, int even_partner, int odd_partner,
        int my_rank, int p, MPI_Comm comm) {
   MPI_Status status;

   if (phase % 2 == 0) {
      if (even_partner >= 0) {
         MPI_Sendrecv(local_A, local_n, MPI_FLOAT, even_partner, 0,
            temp_B, local_n, MPI_FLOAT, even_partner, 0, comm, &status);
         if (my_rank % 2 != 0)
            Merge_high(local_A, temp_B, temp_C, local_n);
         else
            Merge_low(local_A, temp_B, temp_C, local_n);
      }
   } else {
      if (odd_partner >= 0) {
         MPI_Sendrecv(local_A, local_n, MPI_FLOAT, odd_partner, 0,
            temp_B, local_n, MPI_FLOAT, odd_partner, 0, comm, &status);
         if (my_rank % 2 != 0)
            Merge_low(local_A, temp_B, temp_C, local_n);
         else
            Merge_high(local_A, temp_B, temp_C, local_n);
      }
   }
}

void Sort(float local_A[], int local_n, int my_rank, int p, MPI_Comm comm) {
   int phase;
   float *temp_B, *temp_C;
   int even_partner, odd_partner;

   temp_B = (float*) malloc(local_n * sizeof(float));
   temp_C = (float*) malloc(local_n * sizeof(float));

   // Determine partners
   if (my_rank % 2 != 0) {
      even_partner = my_rank - 1;
      odd_partner = my_rank + 1;
      if (odd_partner == p) odd_partner = MPI_PROC_NULL;
   } else {
      even_partner = my_rank + 1;
      if (even_partner == p) even_partner = MPI_PROC_NULL;
      odd_partner = my_rank - 1;
   }

   // Sort local list using built-in quick sort for floats
   qsort(local_A, local_n, sizeof(float), Compare_floats);

   for (phase = 0; phase < p; phase++)
      Odd_even_iter(local_A, temp_B, temp_C, local_n, phase, even_partner, odd_partner, my_rank, p, comm);

   free(temp_B);
   free(temp_C);
}

int main(int argc, char** argv) {
    CALI_MARK_BEGIN("main");
    MPI_Init(&argc, &argv);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (argc != 3) {
        if (world_rank == 0) {
            std::cerr << "Usage: mpirun -np <num_processes> ./executable <num_values> <input_type>\n";
        }
        MPI_Finalize();
        return 1;
    }

    int num_values = std::stoi(argv[1]);
    int local_n = num_values / world_size; // Assuming num_values is divisible by world_size
    std::string input_type = argv[2];


    // Allocate memory for the full array only on the root process
    float *arr = nullptr;
    if (world_rank == 0) {
        CALI_MARK_BEGIN("data_init");
        arr = new float[num_values];
        array_fill(arr, num_values, input_type);
        // array_print(arr, local_n);
        CALI_MARK_END("data_init");
    }

    // Allocate memory for the local array on all processes
    float *local_arr = new float[local_n];

    // Scatter the data from the root process to all processes
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Scatter");
    MPI_Scatter(arr, local_n, MPI_FLOAT, local_arr, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Scatter");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Sort the local arrays
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_small");
    std::sort(local_arr, local_arr + local_n);
    CALI_MARK_END("comp_small");
    CALI_MARK_END("comp");

    // Perform the parallel sort
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    Sort(local_arr, local_n, world_rank, world_size, MPI_COMM_WORLD);
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Gather the sorted subarrays into the root array
    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("MPI_Gather");
    MPI_Gather(local_arr, local_n, MPI_FLOAT, arr, local_n, MPI_FLOAT, 0, MPI_COMM_WORLD);
    CALI_MARK_END("MPI_Gather");
    CALI_MARK_END("comm_large");
    CALI_MARK_END("comm");

    // Print the sorted array on the root process
    if (world_rank == 0) 
    {
        CALI_MARK_BEGIN("correctness_check");

        if(correctness_check(arr, local_n)) 
        {
          printf("Array correctly sorted!\n");
        } 

        else 
        {
          printf("Array sorting failed\n");
        }

        CALI_MARK_END("correctness_check");
        delete[] arr;
    }

    delete[] local_arr;

    CALI_MARK_END("main");
    
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "OddEvenTranspositionSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", local_n); // The number of elements in input dataset (1000)
    adiak::value("InputType", (char*)input_type.c_str()); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", world_size); // The number of processors (MPI ranks)
    // adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 15); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "ONLINE"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
    
    MPI_Finalize();
    return 0;
}
