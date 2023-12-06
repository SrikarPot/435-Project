# CSCE 435 Group project

## 1. Group members:
1. Aurko Routh
2. Srikar Potlapalli
3. Andrew Chian
4. Soohwan Kim

---

## 2. Project topic
Performance comparison of sorting and graph algorithms with CUDA and MPI implementations
## 3. Project description
We will be implementing the following algorithms in MPI and CUDA:
- Bitonic Sort
- Odd-Even Transport Sort
- Merge sort
- bubble sort
- quick sort
- radix sort
- cascade merge sort
- parallel external sort

## 3. Project implementation
Implement your proposed algorithms, and test them starting on a small scale.
Instrument your code, and turn in at least one Caliper file per algorithm;
if you have implemented an MPI and a CUDA version of your algorithm,
turn in a Caliper file for each.

### 3a. Caliper instrumentation
Please use the caliper build `/scratch/group/csce435-f23/Caliper/caliper/share/cmake/caliper` 
(same as lab1 build.sh) to collect caliper files for each experiment you run.

Your Caliper regions should resemble the following calltree
(use `Thicket.tree()` to see the calltree collected on your runs):
```
main
|_ data_init
|_ comm
|    |_ MPI_Barrier
|    |_ comm_small  // When you broadcast just a few elements, such as splitters in Sample sort
|    |   |_ MPI_Bcast
|    |   |_ MPI_Send
|    |   |_ cudaMemcpy
|    |_ comm_large  // When you send all of the data the process has
|        |_ MPI_Send
|        |_ MPI_Bcast
|        |_ cudaMemcpy
|_ comp
|    |_ comp_small  // When you perform the computation on a small number of elements, such as sorting the splitters in Sample sort
|    |_ comp_large  // When you perform the computation on all of the data the process has, such as sorting all local elements
|_ correctness_check
```
### Psuedo code for Odd-Even Transposition Sort(CUDA)
```
// Define the array size
let N = size of the array

// CUDA kernel function for odd-even transposition sort
__global__ void oddEvenSortKernel(float *d_a, int n, int phase)
{
  // Calculate global thread index
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

// Host function to run the parallelized odd-even transposition sort
void cudaOddEvenSort(float *h_a, int n)
{
  float *d_a;

  // Allocate memory on the device
  cudaMalloc(&d_a, n * sizeof(float));

  // Copy data from host to device
  cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice);

  // Setup block and grid dimensions
  dim3 blocks(BLOCKS, 1);   // Number of blocks
  dim3 threads(THREADS, 1); // Number of threads per block

  // Launch the kernel for each phase
  for (int i = 0; i < n; ++i)
  {
    oddEvenSortKernel<<<blocks, threads>>>(d_a, n, i);
    cudaDeviceSynchronize();
  }

  // Copy the sorted array back to the host
  cudaMemcpy(h_a, d_a, n * sizeof(float), cudaMemcpyDeviceToHost);

  // Free device memory
  cudaFree(d_a);
}

// Main program
int main()
{
  // Initialize array and other variables
  float *array = initializeArray(N);
  
  // Perform sorting on the GPU
  cudaOddEvenSort(array, N);

  // Print the sorted array or perform further operations

  // Cleanup and exit
  return 0;
}

```


Required code regions:
- `main` - top-level main function.
    - `data_init` - the function where input data is generated or read in from file.
    - `correctness_check` - function for checking the correctness of the algorithm output (e.g., checking if the resulting data is sorted).
    - `comm` - All communication-related functions in your algorithm should be nested under the `comm` region.
      - Inside the `comm` region, you should create regions to indicate how much data you are communicating (i.e., `comm_small` if you are sending or broadcasting a few values, `comm_large` if you are sending all of your local values).
      - Notice that auxillary functions like MPI_init are not under here.
    - `comp` - All computation functions within your algorithm should be nested under the `comp` region.
      - Inside the `comp` region, you should create regions to indicate how much data you are computing on (i.e., `comp_small` if you are sorting a few values like the splitters, `comp_large` if you are sorting values in the array).
      - Notice that auxillary functions like data_init are not under here.

All functions will be called from `main` and most will be grouped under either `comm` or `comp` regions, representing communication and computation, respectively. You should be timing as many significant functions in your code as possible. **Do not** time print statements or other insignificant operations that may skew the performance measurements.

**Nesting Code Regions** - all computation code regions should be nested in the "comp" parent code region as following:
```
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
mergesort();
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Looped GPU kernels** - to time GPU kernels in a loop:
```
### Bitonic sort example.
int count = 1;
CALI_MARK_BEGIN("comp");
CALI_MARK_BEGIN("comp_large");
int j, k;
/* Major step */
for (k = 2; k <= NUM_VALS; k <<= 1) {
    /* Minor step */
    for (j=k>>1; j>0; j=j>>1) {
        bitonic_sort_step<<<blocks, threads>>>(dev_values, j, k);
        count++;
    }
}
CALI_MARK_END("comp_large");
CALI_MARK_END("comp");
```

**Calltree Examples**:

```
# Bitonic sort tree - CUDA looped kernel
1.000 main
├─ 1.000 comm
│  └─ 1.000 comm_large
│     └─ 1.000 cudaMemcpy
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Matrix multiplication example - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  ├─ 1.000 comm_large
│  │  ├─ 1.000 MPI_Recv
│  │  └─ 1.000 MPI_Send
│  └─ 1.000 comm_small
│     ├─ 1.000 MPI_Recv
│     └─ 1.000 MPI_Send
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

```
# Mergesort - MPI
1.000 main
├─ 1.000 comm
│  ├─ 1.000 MPI_Barrier
│  └─ 1.000 comm_large
│     ├─ 1.000 MPI_Gather
│     └─ 1.000 MPI_Scatter
├─ 1.000 comp
│  └─ 1.000 comp_large
└─ 1.000 data_init
```

#### 3b. Collect Metadata

Have the following `adiak` code in your programs to collect metadata:
```
adiak::init(NULL);
adiak::launchdate();    // launch date of the job
adiak::libraries();     // Libraries used
adiak::cmdline();       // Command line used to launch the job
adiak::clustername();   // Name of the cluster
adiak::value("Algorithm", algorithm); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
adiak::value("ProgrammingModel", programmingModel); // e.g., "MPI", "CUDA", "MPIwithCUDA"
adiak::value("Datatype", datatype); // The datatype of input elements (e.g., double, int, float)
adiak::value("SizeOfDatatype", sizeOfDatatype); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
adiak::value("InputSize", inputSize); // The number of elements in input dataset (1000)
adiak::value("InputType", inputType); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
adiak::value("num_procs", num_procs); // The number of processors (MPI ranks)
adiak::value("num_threads", num_threads); // The number of CUDA or OpenMP threads
adiak::value("num_blocks", num_blocks); // The number of CUDA blocks 
adiak::value("group_num", group_number); // The number of your group (integer, e.g., 1, 10)
adiak::value("implementation_source", implementation_source) // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").
```

They will show up in the `Thicket.metadata` if the caliper file is read into Thicket.

**See the `Builds/` directory to find the correct Caliper configurations to get the above metrics for CUDA, MPI, or OpenMP programs.** They will show up in the `Thicket.dataframe` when the Caliper file is read into Thicket.
