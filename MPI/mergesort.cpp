#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int L[n1], R[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;
    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        } else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void merge_sort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        merge_sort(arr, left, mid);
        merge_sort(arr, mid + 1, right);

        merge(arr, left, mid, right);
    }
}

int main(int argc, char *argv[]) {
    int rank, size;
    int array_size;

    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }

    array_size = atoi(argv[1]);
    int array[array_size];

    for (int i = 0; i < array_size; i++)
        array[i] = rand() % 100; // Filling array with random values for demonstration

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_size = array_size / size;
    int local_array[local_size];

    MPI_Scatter(array, local_size, MPI_INT, local_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    merge_sort(local_array, 0, local_size - 1);

    int sorted_array[array_size];

    MPI_Gather(local_array, local_size, MPI_INT, sorted_array, local_size, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        merge_sort(sorted_array, 0, array_size - 1);

        printf("Original Array: ");
        for (int i = 0; i < array_size; i++)
            printf("%d ", array[i]);
        printf("\n");

        printf("Sorted Array: ");
        for (int i = 0; i < array_size; i++)
            printf("%d ", sorted_array[i]);
        printf("\n");
    }

    MPI_Finalize();
    return 0;
}
