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
const char* comm = "comm";
const char* comm_large = "comm_large";
const char* comp = "comp";
const char* comp_large = "comp_large";

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

__global__ void enumerationSort(float *array, int *rank, int n, int THREADS) {
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
__global__ void sortArray(float *array, float *sorted_array, int *rank, int n, int THREADS) {
    for (int i = 0; i < n; i += THREADS){
    int k = blockIdx.x * blockDim.x + threadIdx.x;

    // for(int i = k; i < n; i += THREADS){
    //     if (i < n) {
    //         for (int j = 0; j < n; j++) {
    //             if (rank[j] == i) {
    //                 swap(array[j], array[i]);
    //                 swap(rank[j], rank[i]);
    //                 break;
    //             }
    //         }
    //     }
    // }
    for (int i = k; i < n; i += THREADS){
        sorted_array[rank[i]] = array[i];
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
    float *h_array = new float[n];
    int *h_rank = new int[n];
    float *sorted_array = new float[n];

    // Initialize the array with random values
    array_fill(h_array, n);
    

    // Print the og array
    std::cout << "Original Array: ";
    for (int i = 0; i < n; i++) {
        std::cout << h_array[i] << " ";
    }
    std::cout << std::endl;

    // Device arrays
    float *d_array, *sorted_array_device;
    int* d_rank;
    cudaMalloc((void**)&d_array, sizeof(float) * n);
    cudaMalloc((void**)&d_rank, sizeof(int) * n);
    cudaMalloc((void**)&sorted_array_device, sizeof(float) * n);

    CALI_MARK_BEGIN("comm");
    CALI_MARK_BEGIN("comm_large");

    // Copy data from host to device
    cudaMemcpy(d_array, h_array, sizeof(float) * n, cudaMemcpyHostToDevice);

    CALI_MARK_BEGIN("comm_large");
    CALI_MARK_BEGIN("comm");

    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");

    // Launch the enumeration sort kernel
    enumerationSort<<<BLOCKS, THREADS>>>(d_array, d_rank, n, THREADS);
    cudaDeviceSynchronize();
    CALI_MARK_END("comp_large");

    CALI_MARK_BEGIN("comp_large");
    // Launch the sorting kernel to rearrange the array
    sortArray<<<BLOCKS, THREADS>>>(d_array, sorted_array_device, d_rank, n, THREADS);
    cudaDeviceSynchronize();

    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    CALI_MARK_BEGIN("comm");
    
    // Copy the sorted array and ranks back to the host
    // CALI_MARK_BEGIN("comm_large");
    // cudaMemcpy(h_array, d_array, sizeof(float) * n, cudaMemcpyDeviceToHost);
    // CALI_MARK_END("comm_large");

    // CALI_MARK_BEGIN("comm_large");
    // cudaMemcpy(h_rank, d_rank, sizeof(int) * n, cudaMemcpyDeviceToHost);
    // CALI_MARK_END("comm_large");

    CALI_MARK_BEGIN("comm_large");
    cudaMemcpy(sorted_array, sorted_array_device, sizeof(float) * n, cudaMemcpyDeviceToHost);
    CALI_MARK_END("comm_large");

    CALI_MARK_END("comm");

    // for (int i = 0; i < NUM_VALS; i++){
    //     sorted_array[rank[i]] = h_array[i];
    // }

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
    cudaFree(sorted_array_device);



    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "EnumerationSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "CUDA"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", float); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", "Sorted"); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    // adiak::value("num_procs", ); // The number of processors (MPI ranks)
    adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 15); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").



//   print_elapsed(start, stop);


  // Flush Caliper output before finalizing MPI
  mgr.stop();
  mgr.flush();
}