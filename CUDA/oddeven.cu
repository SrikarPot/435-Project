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

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

int THREADS;
int BLOCKS;
int NUM_VALS;

const char* bitonic_sort_step_region = "bitonic_sort_step";
const char* cudaMemcpy_host_to_device = "cudaMemcpy_host_to_device";
const char* cudaMemcpy_device_to_host = "cudaMemcpy_device_to_host";

/* Define Caliper region names */
const char* host_to_device = "host_to_device";
const char* device_to_host = "device_to_host";
const char* bitonic_step = "bitonic_step";

int bitonic_counter = 0;

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

__global__ void enumerationSort(int *array, int *rank, int n, int THREADS) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = k; i < n; i += THREADS){
        
        if (i < n) {
            rank[i] = 0;
            for (int j = 0; j < n; j++) {
                if (array[j] < array[i] || (array[j] == array[i] && j < i)) {
                    rank[i]++;
                }
            }
        }
    }
}

// Helper function to swap two integers
__device__ void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

// CUDA kernel for sorting the array based on ranks
__global__ void sortArray(int *array, int *rank, int n, int THREADS) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    for(int i = k; i < n; i += THREADS){
        if (i < n) {
            for (int j = 0; j < n; j++) {
                if (rank[j] == i) {
                    swap(array[j], array[i]);
                    swap(rank[j], rank[i]);
                    break;
                }
            }
        }
    }
}


int main(int argc, char *argv[])
{
  

  THREADS = atoi(argv[1]);
  NUM_VALS = atoi(argv[2]);
  BLOCKS = NUM_VALS / THREADS;

  printf("Number of threads: %d\n", THREADS);
  printf("Number of values: %d\n", NUM_VALS);
  printf("Number of blocks: %d\n", BLOCKS);

  // Create caliper ConfigManager object
  cali::ConfigManager mgr;
  mgr.start();







    const int n = NUM_VALS; // Size of the array
    int *h_array = new int[n];
    int *h_rank = new int[n];
    int *sorted_array = new int[n];

    // Initialize the array with random values
    srand(static_cast<unsigned int>(time(nullptr)));
    for (int i = 0; i < n; i++) {
        h_array[i] = rand() % 100;
    }

    // Print the og array
    std::cout << "Original Array: ";
    for (int i = 0; i < n; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // Device arrays
    int *d_array, *d_rank;
    cudaMalloc((void**)&d_array, sizeof(int) * n);
    cudaMalloc((void**)&d_rank, sizeof(int) * n);

    // Copy data from host to device
    cudaMemcpy(d_array, h_array, sizeof(int) * n, cudaMemcpyHostToDevice);

    // Launch the enumeration sort kernel
    enumerationSort<<<BLOCKS, THREADS>>>(d_array, d_rank, n, THREADS);
    cudaDeviceSynchronize();

    // Launch the sorting kernel to rearrange the array
    // sortArray<<<BLOCKS, THREADS>>>(d_array, d_rank, n, THREADS);
    // cudaDeviceSynchronize();

    // Copy the sorted array and ranks back to the host
    cudaMemcpy(h_array, d_array, sizeof(int) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_rank, d_rank, sizeof(int) * n, cudaMemcpyDeviceToHost);

    for (int i = 0; i < NUM_VALS; i++){
        sorted_array[h_rank[i]] = h_array[i];
    }

    // Print the sorted array
    std::cout << "Sorted Array: ";
    for (int i = 0; i < n; i++) {
        std::cout << sorted_array[i] << " ";
    }
    std::cout << std::endl;
 
    // Clean up
    delete[] h_array;
    delete[] h_rank;
    delete[] sorted_array;
    cudaFree(d_array);
    cudaFree(d_rank);



    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "EnumerationSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "int"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", "sizeOfDatatype"); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", "inputSize"); // The number of elements in input dataset (1000)
    adiak::value("InputType", "inputType"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", "num_procs"); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", "group_number"); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Online"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").







//   print_elapsed(start, stop);

  // Store results in these variables.

  
//   printf("\neffective_bandwidth_gb_s: %.7f\n", effective_bandwidth_gb_s);
//   printf("bitonic_sort_step_time: %.7f\n", bitonic_sort_step_time);
//   printf("cudaMemcpy_host_to_device_time: %.7f\n", cudaMemcpy_host_to_device_time);
//   printf("cudaMemcpy_device_to_host_time: %.7f\n", cudaMemcpy_device_to_host_time);
//   printf("\ntotal time: %.7f\n", (ms_bitonic_step + ms_host_to_device + ms_device_to_host));

//   adiak::init(NULL);
//   adiak::user();
//   adiak::launchdate();
//   adiak::libraries();
//   adiak::cmdline();
//   adiak::clustername();
//   adiak::value("num_threads", THREADS);
//   adiak::value("num_blocks", BLOCKS);
//   adiak::value("num_vals", NUM_VALS);
//   adiak::value("program_name", "cuda_bitonic_sort");
//   adiak::value("datatype_size", sizeof(float));
//   adiak::value("effective_bandwidth (GB/s)", effective_bandwidth_gb_s);
//   adiak::value("bitonic_sort_step_time", bitonic_sort_step_time);
//   adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
//   adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);

  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}