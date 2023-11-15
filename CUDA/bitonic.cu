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
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>
#include "../helper.h"

int THREADS;
int BLOCKS;
int NUM_VALS;

cudaEvent_t start_sort, end_sort, start_host_device, end_host_device, start_device_host, end_device_host;

// Store results in these variables.
float effective_bandwidth_gb_s = 0;
float bitonic_sort_step_time = 0;
float cudaMemcpy_host_to_device_time = 0;
float cudaMemcpy_device_to_host_time = 0;
__global__ void bitonic_sort_step(float *dev_values, int j, int k)
{
  unsigned int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) {
    if ((i&k)==0) {
      /* Sort ascending */
      if (dev_values[i]>dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
    if ((i&k)!=0) {
      /* Sort descending */
      if (dev_values[i]<dev_values[ixj]) {
        /* exchange(i,ixj); */
        float temp = dev_values[i];
        dev_values[i] = dev_values[ixj];
        dev_values[ixj] = temp;
      }
    }
  }
}

/**
 * Inplace bitonic sort using CUDA.
 */
void bitonic_sort(float *values)
{
  float *dev_values;
  size_t size = NUM_VALS * sizeof(float);

  cudaMalloc((void**) &dev_values, size);
  
  //MEM COPY FROM HOST TO DEVICE
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  cudaEventRecord(start_host_device);
  CALI_MARK_BEGIN("cudaMemcpyHostToDevice");
  cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice);
  CALI_MARK_END("cudaMemcpyHostToDevice");
  cudaEventRecord(end_host_device);
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");

  cudaEventSynchronize(end_host_device);
  cudaEventElapsedTime(&cudaMemcpy_host_to_device_time, start_host_device, end_host_device);

  dim3 blocks(BLOCKS,1);    /* Number of blocks   */
  dim3 threads(THREADS,1);  /* Number of threads  */
  
  int j, k;
  int kernel_call = 1;
  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");

  cudaEventRecord(start_sort);
  /* Major step */
  for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
      bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
      kernel_call++;
    }
  }
  cudaEventRecord(end_sort);
  cudaDeviceSynchronize();
  CALI_MARK_END("comp_large");
  CALI_MARK_END("comp");
  cudaEventElapsedTime(&bitonic_sort_step_time, start_sort, end_sort);

  effective_bandwidth_gb_s = ((kernel_call * 6 * NUM_VALS * sizeof(float)) / 1e9) / (bitonic_sort_step_time / 1000);
  
  
  //MEM COPY FROM DEVICE TO HOST
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  cudaEventRecord(start_device_host);
  CALI_MARK_BEGIN("cudaMemcpyDeviceToHost");
  cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost);
  CALI_MARK_END("cudaMemcpyDeviceToHost");
  cudaEventRecord(end_device_host);
  

  cudaEventSynchronize(end_device_host);
  cudaEventElapsedTime(&cudaMemcpy_device_to_host_time, start_device_host, end_device_host);
  
  CALI_MARK_BEGIN("cudaFree");
  cudaFree(dev_values);
  CALI_MARK_END("cudaFree");
  
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");
}

int main(int argc, char *argv[])
{
  cudaEventCreate(&start_sort);
  cudaEventCreate(&end_sort);
  cudaEventCreate(&start_host_device);
  cudaEventCreate(&end_host_device);
  cudaEventCreate(&start_device_host);
  cudaEventCreate(&end_device_host);
  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;
  std::string input_type = argv[3];

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();

  clock_t start, stop;

  float *values = (float*) malloc( NUM_VALS * sizeof(float));
  CALI_MARK_BEGIN("data_init");
  array_fill(values, NUM_VALS, input_type);
  CALI_MARK_END("data_init");

  start = clock();
  bitonic_sort(values); /* Inplace */
  stop = clock();

  print_elapsed(start, stop);

  std::cout << "bitonic sort step time: " << bitonic_sort_step_time << std::endl;
  std::cout << "host to device time: " << cudaMemcpy_host_to_device_time << std::endl;
  std::cout << "device to host time: " << cudaMemcpy_device_to_host_time << std::endl;
  std::cout << "effective bandwidth: " << effective_bandwidth_gb_s << std::endl;

  CALI_MARK_BEGIN("correctness_check");
  bool correct = correctness_check(values, NUM_VALS);
  CALI_MARK_END("correctness_check");
  if(correct) printf("Array correctly sorted\n");
  
 adiak::init(NULL);
  adiak::launchdate();    // launch date of the job
  adiak::libraries();     // Libraries used
  adiak::cmdline();       // Command line used to launch the job
  adiak::clustername();   // Name of the cluster
  adiak::value("Algorithm", "BitonicSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
  adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
  adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
  adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
  adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
  adiak::value("InputType", "Sorted"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
  // adiak::value("num_procs", ); // The number of processors (MPI ranks)
  adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
  adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
  adiak::value("group_num", 15); // The number of your group (integer, e.g., 1, 10)
  adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}