#ifndef HELPER_H  // Header guards specific to "helper.h"
#define HELPER_H

// Include necessary libraries if needed
#include <iostream>
#include <string>

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}

void array_print(float *arr, int length) 
{
  int i;
  for (i = 0; i < length; ++i) {
    printf("%1.3f ",  arr[i]);
  }
  printf("\n");
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_float();
  }
}

int correctness_check(float *h_a, int n)
{
    for (int i = 1; i < n; i++)
    {
        if (h_a[i - 1] > h_a[i])
        {
            return 0; // Array is not sorted correctly
        }
    }
    return 1; // Array is sorted correctly
}

#endif  // End of header guard HELPER_H