#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../helper.h"

int rank, size;
int n,loc;
void merge(float* values, float* temp, int l, int m, int r)  {

  if(values[m] <= values[m+1]) return; // subarrays already sorted


  int i = l;
  int j = m+1;
  int k = l;

  while(i <= m && j <= r) {
    if(values[i] > values[j])
      temp[k++] = values[j++];
    else
      temp[k++] = values[i++];
  }

  // add left over values from first half
  while(i <= m) {
    temp[k++] = values[i++];
  }

  //add left over values from second half
  while(j <= r) {
    temp[k++] = values[j++];
  }


  // copy over to main array
  for(i = l; i <= r; i++) {
    values[i] = temp[i];
  }
}


void mergeSort(float* values, float* temp, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        mergeSort(values, temp, l, m);
        mergeSort(values, temp, m + 1, r);
        merge(values, temp, l, m, r);
    }
}

int main(int argc, char *argv[]) {
  //  int rank, size;
    n = atoi(argv[1]); // Change this to your desired array size
    float *arr ;
    int local_n;
    std::string input_type = argv[2];
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        arr =(float *)malloc(n * sizeof(int));
        array_fill(arr, n, input_type);
    }

    for(int subarr_size = n / size, i = 1; subarr_size <= n; subarr_size <<=1, i <<=1) {
        MPI_Comm custom_comm;
        MPI_Comm_split(MPI_COMM_WORLD, (rank % i == 0) ? 1 : MPI_UNDEFINED, rank, &custom_comm);
        if(rank % i == 0) {
            int custom_rank;
            MPI_Comm_rank(custom_comm, &custom_rank);
            float *local_arr = (float*)malloc(subarr_size * sizeof(float));
            float *temp = (float*)malloc(subarr_size * sizeof(float));
            MPI_Scatter(arr, subarr_size, MPI_FLOAT, local_arr, subarr_size, MPI_FLOAT, 0, custom_comm);
            mergeSort(local_arr, temp, 0, subarr_size-1);
            MPI_Gather(local_arr, subarr_size, MPI_FLOAT, arr, subarr_size, MPI_FLOAT, 0, custom_comm);
            free(local_arr);
            free(temp);
        }
        // MPI_Comm_free(&custom_comm);
    }


    if (rank == 0) {
        array_print(arr,n);
        free(arr);
    }
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "MergeSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", (char*)input_type.c_str()); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", ); // The number of processors (MPI ranks)
    // adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 15); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "ONLINE"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

    MPI_Finalize();
    return 0;
}