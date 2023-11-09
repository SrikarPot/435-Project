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
- enumeration sort

### Psuedo code for Bitonic Sort(CUDA)
```
global function bitonic_sort(values):
    allocate device memory for dev_values
    size = NUM_VALS * size_of(float)

    cudaMalloc((void**) &dev_values, size)

    // Memory copy from host to device
    cudaMemcpy(dev_values, values, size, cudaMemcpyHostToDevice)

    dim3 blocks(BLOCKS, 1)     // Number of blocks
    dim3 threads(THREADS, 1)   // Number of threads

    create CUDA events for timing

    /* Major step */
    for k from 2 to NUM_VALS (doubling each time):
        /* Minor step */
        for j from k / 2 to 1 (halving each time):
            launch bitonic_sort_step CUDA kernel with arguments (dev_values, j, k) using blocks and threads
            // Timing events for this iteration

    // Memory copy from device to host
    cudaMemcpy(values, dev_values, size, cudaMemcpyDeviceToHost)

    // Free device memory
    cudaFree(dev_values)

    global CUDA kernel function bitonic_sort_step(dev_values, j, k):
        i = threadIdx.x + blockDim.x * blockIdx.x
        ixj = i ^ j // Sorting partners

        if ixj > i:
            if (ixj) > i and (i & k) == 0:
                // Sort ascending
                if dev_values[i] > dev_values[ixj]:
                    swap dev_values[i] with dev_values[ixj]
    
            if (ixj) > i and (i & k) != 0:
                // Sort descending
                if dev_values[i] < dev_values[ixj]:
                    swap dev_values[i] with dev_values[ixj]

```

### Psuedo Code for Odd-Even Transposition Sort (CUDA)
```
Define oddEvenSortKernel with input array d_a, size n, and phase:
  Calculate index based on thread and block indices
  Define idx1 as index
  Define idx2 as index + 1

  // Odd and even phases
  If phase is even and idx2 is less than n and idx1 is even:
    If d_a[idx1] is greater than d_a[idx2]:
      Swap d_a[idx1] with d_a[idx2]
  
  Else if phase is odd and idx2 is less than n and idx1 is odd:
    If d_a[idx1] is greater than d_a[idx2]:
      Swap d_a[idx1] with d_a[idx2]

Define cudaOddEvenSort with input array h_a and size n:
  Allocate memory for array d_a on the device

  Copy array h_a to device memory (d_a)

  Define number of blocks and threads per block

  For i from 0 to n-1:
    Call oddEvenSortKernel with d_a, n, and i
    Synchronize device to ensure kernel completion
    Optionally, copy d_a to h_a to check intermediate sorting state
    Print array h_a to display sorting progress (optional step)

  Copy the sorted array d_a back to host array h_a

  Free the device memory allocated for d_a
```
### Psuedo code for Merge Sort(CUDA)
```
global function merge_sort_caller(values):
    allocate device memory for dev_values and temp
    copy values from host to device (dev_values)
    
    blocks = number_of_blocks
    threads = number_of_threads

    for window from 2 to size of values (doubling each time):
        launch merge_sort CUDA kernel with arguments (dev_values, temp, size of values, window) using blocks and threads

    copy values from device to host (dev_values to values)
    free device memory (dev_values, temp)

global CUDA kernel function merge_sort(values, temp, num_vals, window):
    id = threadIdx.x + blockDim.x * blockIdx.x
    l = id * window
    r = l + window

    if r > num_vals:
        r = num_vals

    m = l + (r - l) / 2

    if l < num_vals:
        call merge function on the device (values, temp, l, m, r)
```
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
