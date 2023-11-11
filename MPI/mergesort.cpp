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

  if(rank == 0) {
    printf("%d %d %d\n", l, m, r);
  }

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

      if(rank == 0) {
        //array_print(temp, n);
        array_print(values, n);
     }
}


void mergeSort(float* values, float* temp, int l, int r) {
    if (l < r) {
        int m = l + (r - l) / 2;

        mergeSort(values, temp, l, m);
        mergeSort(values, temp, m + 1, r);
        merge(values, temp, l, m, r);
        printf("--------------\n");
        array_print(values, n);
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
 //       array_print(arr,n);
    }

    local_n = n / size;
    loc = local_n;
    float *local_arr = (float *)malloc(local_n * sizeof(int));
    float *temp = (float*)malloc(local_n*sizeof(int));

    MPI_Scatter(arr, local_n, MPI_INT, local_arr, local_n, MPI_INT, 0, MPI_COMM_WORLD);
  //  array_print(local_arr, local_n);
      // if(rank == 1)
      //   array_print(local_arr, local_n);
    mergeSort(local_arr, temp, 0, local_n - 1);
    // if(rank == 1)
    //    array_print(local_arr, local_n);

    float *sorted = NULL;
    if (rank == 0) {
        sorted = (float *)malloc(n * sizeof(float));
    }

    MPI_Gather(local_arr, local_n, MPI_INT, sorted, local_n, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        float* glob_temp = (float *)malloc(n * sizeof(float));
        array_print(sorted, n);
  //      printf("---------------\n");
        mergeSort(sorted, glob_temp, 0, n - 1);

    //    printf("final:\n");
        array_print(sorted,n);

        free(sorted);
        free(arr);
        free(glob_temp);
    }

    free(local_arr);
    free(temp);
    MPI_Finalize();
    return 0;
}