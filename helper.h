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
//   return (float)rand()/(float)1000000;
  return (float)rand()/(float)FLT_MAX;
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

void array_fill(float *arr, int length, const std::string& input_type)
{
    srand(static_cast<unsigned>(time(nullptr)));
    // printf("input_type == %s\n\n", input_type);

    if (input_type == "Sorted") {
        // Fill the array with sorted values
        for (int i = 0; i < length; ++i) {
            arr[i] = static_cast<float>(i) / static_cast<float>(length);
        }
    } else if (input_type == "ReverseSorted") {
        // Fill the array with reverse sorted values
        for (int i = 0; i < length; ++i) {
            arr[i] = static_cast<float>(length - i - 1) / static_cast<float>(length);
        }
    } else if (input_type == "Random") {
        // Fill the array with random values
        for (int i = 0; i < length; ++i) {
            arr[i] = random_float();
        }
    } else if (input_type == "1%%perturbed") {
        // Fill the array with values slightly perturbed from a sorted sequence
        for (int i = 0; i < length; ++i) {
            arr[i] = static_cast<float>(i) / static_cast<float>(length);
        }
        int num_swaps = length / 100;  // 1% of the array size
        for (int i = 0; i < num_swaps; ++i) {
            int index1 = rand() % length;
            int index2 = rand() % length;
            std::swap(arr[index1], arr[index2]);
        }
    } else {
        // Handle unknown input_type or provide a default behavior
        // For example, you could fill the array with random values as a default
        printf("did not recognize input type. Filling with random values\n\n");
        for (int i = 0; i < length; ++i) {
            arr[i] = random_float();
        }
    }
}

int correctness_check(float *h_a, int n)
{
    for (int i = 1; i < n; i++)
    {
        if (h_a[i - 1] > h_a[i])
        {
            printf("%d %f %f\n", i, h_a[i-1], h_a[i]);
            array_print(h_a, 20);
            return 0; // Array is not sorted correctly
        }
    }
    return 1; // Array is sorted correctly
}

#endif  // End of header guard HELPER_H
