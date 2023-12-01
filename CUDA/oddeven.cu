#include <stdio.h>
#include <cuda_runtime.h>
// #include <caliper/cali.h>
#include <stdlib.h>
#include <time.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>


#include "../helper.h"

int THREADS;
int BLOCKS;
int NUM_VALS;

// Define the array size
#define N 8


// CUDA kernel function for odd-even transposition sort
__global__ void oddEvenSortKernel(float *d_a, int n, int phase)
{
  int index = threadIdx.x + blockDim.x * blockIdx.x;
  int idx1 = index;
  int idx2 = index + 1;
  // Check whether we are in an odd or even phase
  if ((phase % 2 == 0) && (idx2 < n) && (idx1 % 2 == 0))
  { // Even phase
    if (d_a[idx1] > d_a[idx2])
    {
      // Swap elements
      float temp = d_a[idx1];
      d_a[idx1] = d_a[idx2];
      d_a[idx2] = temp;
    }
  }
  else if ((phase % 2 == 1) && (idx2 < n) && (idx1 % 2 == 1))
  { // Odd phase
    if (d_a[idx1] > d_a[idx2])
    {
      // Swap elements
      float temp = d_a[idx1];
      d_a[idx1] = d_a[idx2];
      d_a[idx2] = temp;
    }
  }
}

// Host function to run the odd-even transposition sort
void cudaOddEvenSort(float *h_a, int n)
{
  float *d_a;
  
  // Allocate memory on the device
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  CALI_MARK_BEGIN("cudaMalloc");
  cudaMalloc(&d_a, n * sizeof(int));
  CALI_MARK_END("cudaMalloc");
  // Copy data from host to device
  CALI_MARK_BEGIN("cudaMemcpyHostToDevice");
  cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
  CALI_MARK_END("cudaMemcpyHostToDevice");
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");

  // Setup block and grid dimensions
  dim3 blocks(BLOCKS, 1);   /* Number of blocks   */
  dim3 threads(THREADS, 1); /* Number of threads  */

  // Caliper instrumentation for computation region
  // CALI_MARK_BEGIN("comp");
  // CALI_MARK_BEGIN("comp_large");

  // Launch the kernel multiple times
  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");
  for (int i = 0; i < n; ++i)
  {
    oddEvenSortKernel<<<blocks, threads>>>(d_a, n, i);
    cudaDeviceSynchronize();

  }
  CALI_MARK_END("comp_large");
  CALI_MARK_END("comp");

  // CALI_MARK_END("comp_large");
  // CALI_MARK_END("comp");

  // Copy the sorted array back to the host
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  CALI_MARK_BEGIN("cudaMemcpyDeviceToHost");
  cudaMemcpy(h_a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
  CALI_MARK_END("cudaMemcpyDeviceToHost");
  
  // Free device memory
  CALI_MARK_BEGIN("cudaFree");
  cudaFree(d_a);
  CALI_MARK_END("cudaFree");
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");
}





int main(int argc, char *argv[])
{
  CALI_MARK_BEGIN("main");

  THREADS = atoi(argv[1]);  // Number of threads
  NUM_VALS = atoi(argv[2]); // Number of values in the array
  std::string input_type = argv[3];
  BLOCKS = NUM_VALS / THREADS;
  printf(input_type.c_str());
  float *values = (float *)malloc(NUM_VALS * sizeof(float));

  // CALI_MARK_BEGIN("main");

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Initialize data in the host array
  // CALI_MARK_BEGIN("data_init");
  // data_init(h_a, N);
  array_fill(values, NUM_VALS, input_type);
  array_print(values, NUM_VALS);

  cali::ConfigManager mgr;
  mgr.start();

  // CALI_MARK_END("data_init");

  // Caliper annotation for communication region, for example with MPI (not present in the code)
  // CALI_MARK_BEGIN("comm");

  // CALI_MARK_END("comm");

  // Caliper instrumentation for computation region
  CALI_MARK_BEGIN("comp");
  CALI_MARK_BEGIN("comp_large");
  
  // Perform sorting on the GPU
  cudaOddEvenSort(values, NUM_VALS);
  cudaDeviceSynchronize();

  CALI_MARK_END("comp_large");
  CALI_MARK_END("comp");

  // Caliper annotation for checking the correctness of the sorting operation
  CALI_MARK_BEGIN("correctness_check");
  int is_correct = correctness_check(values, NUM_VALS);
  if (is_correct)
  {
    printf("The array is sorted correctly.\n");
  }
  else
  {
    printf("The array is NOT sorted correctly.\n");
  }
  CALI_MARK_END("correctness_check");


    array_print(values, NUM_VALS);
    CALI_MARK_END("main");
    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "OddEvenTranspositionSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", (char*)input_type.c_str()); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    // adiak::value("num_procs", ); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 15); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").

  // Thicket.tree();
  mgr.stop();
  mgr.flush();

  return 0;
}
