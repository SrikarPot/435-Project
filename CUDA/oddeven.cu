#include <stdio.h>
#include <cuda_runtime.h>
// #include <caliper/cali.h>
#include <stdlib.h>
#include <time.h>
#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

// Define the array size
#define N 8

__device__ void device_array_print(float *arr, int length)
{
  int i;
  for (i = 0; i < length; ++i)
  {
    printf("%1.3f ", arr[i]);
  }
}

void array_print(float *arr, int length)
{
  int i;
  for (i = 0; i < length; ++i)
  {
    printf("%1.3f ", arr[i]);
  }
  printf("\n");
}

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
  cudaMalloc(&d_a, n * sizeof(int));
  // Copy data from host to device
  CALI_MARK_BEGIN("comm");
  CALI_MARK_BEGIN("comm_large");
  cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
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
  cudaMemcpy(h_a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
  CALI_MARK_END("comm_large");
  CALI_MARK_END("comm");
  // Free device memory
  cudaFree(d_a);
}

// Function to initialize data in the array
void data_init(int *h_a, int n)
{
  int init_data[N] = {7, 3, 5, 8, 2, 9, 4, 1};
  for (int i = 0; i < n; i++)
  {
    h_a[i] = init_data[i];
  }
}

// Function to check the correctness of the sort
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

float random_float()
{
  return (float)rand() / (float)RAND_MAX;
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i)
  {
    arr[i] = random_float();
  }
}

int main(int argc, char *argv[])
{
  CALI_MARK_BEGIN("main");

  THREADS = atoi(argv[1]);  // Number of threads
  NUM_VALS = atoi(argv[2]); // Number of values in the array
  BLOCKS = NUM_VALS / THREADS;
  float *values = (float *)malloc(NUM_VALS * sizeof(float));

  // CALI_MARK_BEGIN("main");

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Initialize data in the host array
  // CALI_MARK_BEGIN("data_init");
  // data_init(h_a, N);
  array_fill(values, NUM_VALS);
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
  CALI_MARK_END("correctness_check");

  if (is_correct)
  {
    printf("The array is sorted correctly.\n");
  }
  else
  {
    printf("The array is NOT sorted correctly.\n");
  }
    array_print(values, NUM_VALS);
   CALI_MARK_END("main");
adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "ODDEVENSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Sorted"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
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