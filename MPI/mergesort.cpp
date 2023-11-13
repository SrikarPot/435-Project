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

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (rank == 0) {
        arr =(float *)malloc(n * sizeof(int));
        array_fill(arr, n);
    }

    for(int subarr_size = n / size; subarr_size <= n; subarr_size <<=1) {
        MPI_Comm custom_comm;
        MPI_Comm_split(MPI_COMM_WORLD, (rank % subarr_size == 0) ? 1 : MPI_UNDEFINED, rank, &custom_comm);
        if(rank % subarr_size == 0) {
            int custom_rank;
            MPI_Comm_rank(custom_comm, &custom_rank);
            float *local_arr = (float*)malloc(subarr_size * sizeof(float));
            float *temp = (float*)malloc(subarr_size * sizeof(float));
            MPI_Scatter(arr, subarr_size, MPI_FLOAT, local_arr, subarr_size, MPI_FLOAT, 0, custom_comm);
            mergeSort(local_arr, temp, 0, subarr_size-1);
            MPI_Gather(arr, subarr_size, MPI_FLOAT, local_arr, subarr_size, MPI_FLOAT, 0, custom_comm);
            free(local_arr);
            free(temp);
        }
    }


    if (rank == 0) {
        array_print(arr,n);
        free(arr);
    }

    MPI_Finalize();
    return 0;
}