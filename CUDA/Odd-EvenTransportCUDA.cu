#include <stdio.h>
#include <cuda_runtime.h>
#include <caliper/cali.h>
#include <time.h>
#include <iostream>

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#include <cuda_runtime.h>
#include <cuda.h>

int THREADS;
int BLOCKS;
int NUM_VALS;

// Define the array size
#define N 8

cudaEvent_t start_sort, end_sort, start_host_device, end_host_device, start_device_host, end_device_host;

void print_elapsed(clock_t start, clock_t stop)
{
    double elapsed = ((double)(stop - start)) / CLOCKS_PER_SEC;
    printf("Elapsed time: %.3fs\n", elapsed);
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
            int temp = d_a[idx1];
            d_a[idx1] = d_a[idx2];
            d_a[idx2] = temp;
        }
    }
    else if ((phase % 2 == 1) && (idx2 < n) && (idx1 % 2 == 1))
    { // Odd phase
        if (d_a[idx1] > d_a[idx2])
        {
            // Swap elements
            int temp = d_a[idx1];
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
    cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);

    // Setup block and grid dimensions
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;

    // Caliper instrumentation for computation region
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");

    // Launch the kernel multiple times
    for (int i = 0; i < n; ++i)
    {
        oddEvenSortKernel<<<blocksPerGrid, threadsPerBlock>>>(d_a, n, i);
        cudaDeviceSynchronize();
    }

    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    // Copy the sorted array back to the host
    cudaMemcpy(h_a, d_a, n * sizeof(int), cudaMemcpyDeviceToHost);
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

void array_print(float *arr, int length)
{
    int i;
    for (i = 0; i < length; ++i)
    {
        printf("%1.3f ", arr[i]);
    }
    printf("\n");
}

int main()
{

    int h_a[N];
    float *values = (float *)malloc(5 * sizeof(float));

    CALI_MARK_BEGIN("main");

    // Initialize data in the host array
    CALI_MARK_BEGIN("data_init");
    // data_init(h_a, N);
    array_fill(values, 5);
    array_print(values, 5);

    CALI_MARK_END("data_init");

    // Caliper annotation for communication region, for example with MPI (not present in the code)
    CALI_MARK_BEGIN("comm");

    CALI_MARK_END("comm");

    cali::ConfigManager mgr;
    mgr.start();

    clock_t start, stop;

    // Caliper instrumentation for computation region
    CALI_MARK_BEGIN("comp");
    CALI_MARK_BEGIN("comp_large");
    start = clock();
    // Perform sorting on the GPU
    cudaOddEvenSort(values, 5);
    stop = clock();
    CALI_MARK_END("comp_large");
    CALI_MARK_END("comp");

    print_elapsed(start, stop);

    // Caliper annotation for checking the correctness of the sorting operation
    CALI_MARK_BEGIN("correctness_check");
    int is_correct = correctness_check(values, 5);
    CALI_MARK_END("correctness_check");

    if (is_correct)
    {
        printf("The array is sorted correctly.\n");
    }
    else
    {
        printf("The array is NOT sorted correctly.\n");
    }

    CALI_MARK_END("main");

    array_print(values, 5);

    adiak::init(NULL);
    adiak::user();
    adiak::launchdate();
    adiak::libraries();
    adiak::cmdline();
    adiak::clustername();
    adiak::value("num_threads", THREADS);
    adiak::value("num_blocks", BLOCKS);
    adiak::value("num_vals", NUM_VALS);
    adiak::value("program_name", "cuda_odd-even_transposition");
    adiak::value("datatype_size", sizeof(float));
    adiak::value("effective_bandwidth (GB/s)", effective_bandwidth_gb_s);
    adiak::value("odd_even_transposition_sort_step_time", odd_even_transposition_sort_step_time);
    adiak::value("cudaMemcpy_host_to_device_time", cudaMemcpy_host_to_device_time);
    adiak::value("cudaMemcpy_device_to_host_time", cudaMemcpy_device_to_host_time);

    mgr.stop();
    mgr.flush();

    return 0;
}