/*
 * Parallel bitonic sort using CUDA.
 * Compile with
 * nvcc bitonic_sort.cu
 * Based on http://www.tools-of-computing.com/tc/CS/Sorts/bitonic_sort.htm
 * License: BSD 3
 */

#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
#include <cuda.h>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>
#include "../helper.h"

int THREADS;
int BLOCKS;
int NUM_VALS;
int kernelCalls = 0;

const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

// Store results in these variables.
float effective_bandwidth_gb_s;
float bitonic_sort_step_time;
float cudaMemcpy_host_to_device_time;
float cudaMemcpy_device_to_host_time;




__device__ void merge(float* values, float* temp, int l, int m, int r)  {
  int i = l;
  int j = m;
  int k = l;

  while(i < m && j < r) {
    if(values[i] > values[j])
      temp[k++] = values[j++];
    else
      temp[k++] = values[i++];
  }

  // add left over values from first half
  while(i < m) {
    temp[k++] = values[i++];
  }

  //add left over values from second half
  while(j < r) {
    temp[k++] = values[j++];
  }

  // copy over to main array
  for(i = l; i < r; i++) {
    values[i] = temp[i];
  }

}


__global__ void merge_sort(float* values, float* temp, int num_vals, int window) {
    int id = threadIdx.x + blockDim.x * blockIdx.x;
    int l = id*window;
    int r = l+window;

    if(r > num_vals) { // final window might be smaller
      r = num_vals;
    }

    int m = l + (r-l)/2;  

    if(l < num_vals) { // check if thread is neccesary
      merge(values, temp, l, m, r);
    }
} 


/**
 * Inplace merge sort using CUDA.
 */
void merge_sort_caller(float *values)
{
  float *dev_values, *temp;
  int size = NUM_VALS * sizeof(float);

  cudaMalloc((void**)&dev_values, size);
  cudaMalloc((void**)&temp, size);
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);


  
  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */

  for(int window = 2; window <= size; window <<=1) {
    merge_sort<<<blocks, threads>>>(dev_values, temp, NUM_VALS, window);
  }

  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  cudaFree(dev_values);
  cudaFree(temp);
  
}

int main(int argc, char *argv[])
{
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);
  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  clock_t start, stop;

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  array_fill(values, NUM_VALS);

  start = clock();
  merge_sort_caller(values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);
  array_print(values, NUM_VALS);


  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}