#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <time.h>

// Function to parse command line arguments and initialize global_n and local_n
void Get_args(int argc, char *argv[], int *global_n, int *local_n, int my_rank, int p, MPI_Comm comm)
{
    if (my_rank == 0)
    {
        // Assume that the global list size and local list size are passed as command line arguments
        if (argc != 3)
        {
            fprintf(stderr, "Usage: %s <global list size> <local list size>\n", argv[0]);
            MPI_Abort(comm, 1); // Exit if the arguments are not valid
        }
        *global_n = atoi(argv[1]);
        *local_n = atoi(argv[2]);
    }
    // Broadcast global_n and local_n to all processes
    MPI_Bcast(global_n, 1, MPI_INT, 0, comm);
    MPI_Bcast(local_n, 1, MPI_INT, 0, comm);
}

// Function to generate a list of random floats for each process
void Generate_list(float local_A[], int local_n, int my_rank)
{
    srand(time(NULL) + my_rank); // Seed the random number generator
    for (int i = 0; i < local_n; i++)
    {
        local_A[i] = (float)rand() / (float)(RAND_MAX / 100); // Generate random floats between 0 and 100
    }
}

// Function to print the local lists of each process
void Print_local_lists(const float local_A[], int local_n, int my_rank, int p, MPI_Comm comm)
{
    MPI_Barrier(comm); // Synchronize before starting to print
    for (int i = 0; i < p; i++)
    {
        if (my_rank == i)
        {
            printf("Process %d: ", my_rank);
            for (int j = 0; j < local_n; j++)
            {
                printf("%.2f ", local_A[j]);
            }
            printf("\n");
        }
        MPI_Barrier(comm); // Synchronize after each process prints
    }
}

void Merge_low(float local_A[], const float temp_B[], float temp_C[], int local_n) {
    int ai, bi, ci;
    ai = bi = ci = 0;

    while (ci < local_n) {
        if (ai < local_n && (bi >= local_n || local_A[ai] <= temp_B[bi])) {
            temp_C[ci] = local_A[ai];
            ai++;
        } else {
            temp_C[ci] = temp_B[bi];
            bi++;
        }
        ci++;
    }

    // Copy elements from temp_C back to local_A
    for (int i = 0; i < local_n; i++) {
        local_A[i] = temp_C[i];
    }
}

void Merge_high(float local_A[], const float temp_B[], float temp_C[], int local_n) {
    int ai, bi, ci;
    ai = bi = ci = local_n - 1;

    while (ci >= 0) {
        if (ai >= 0 && (bi >= 0 && local_A[ai] >= temp_B[bi])) {
            temp_C[ci] = local_A[ai];
            ai--;
        } else {
            temp_C[ci] = temp_B[bi];
            bi--;
        }
        ci--;
    }

    // Copy elements from temp_C back to local_A
    for (int i = 0; i < local_n; i++) {
        local_A[i] = temp_C[i];
    }
}


// // Function to sort the local list of each process
// void Sort(float local_A[], int local_n, int my_rank, int p, MPI_Comm comm)
// {
//     // A simple bubble sort for demonstration purposes
//     float temp;
//     for (int i = 0; i < local_n - 1; i++)
//     {
//         for (int j = 0; j < local_n - i - 1; j++)
//         {
//             if (local_A[j] > local_A[j + 1])
//             {
//                 temp = local_A[j];
//                 local_A[j] = local_A[j + 1];
//                 local_A[j + 1] = temp;
//             }
//         }
//     }
//     // In a real application, this might involve more complex operations
//     // like a parallel sorting algorithm or a sorting network
// }

// Function to print the global list after all local lists have been gathered and sorted
void Print_global_list(const float local_A[], int local_n, int my_rank, int p, MPI_Comm comm)
{
    float *global_A = NULL;
    if (my_rank == 0)
    {
        global_A = (float *)malloc(p * local_n * sizeof(float));
    }
    MPI_Gather(local_A, local_n, MPI_FLOAT, global_A, local_n, MPI_FLOAT, 0, comm);
    if (my_rank == 0)
    {
        printf("Global list: ");
        for (int i = 0; i < p * local_n; i++)
        {
            printf("%.2f ", global_A[i]);
        }
        printf("\n");
        free(global_A);
    }
}

void Odd_even_iter(float local_A[], float temp_B[], float temp_C[],
                   int local_n, int phase, int even_partner, int odd_partner,
                   int my_rank, int p, MPI_Comm comm)
{
    MPI_Status status;

    if (phase % 2 == 0)
    { /* even phase */
        if (even_partner >= 0)
        { /* check for even partner */
            MPI_Sendrecv(local_A, local_n, MPI_FLOAT, even_partner, 0,
                         temp_B, local_n, MPI_FLOAT, even_partner, 0, comm,
                         &status);
            if (my_rank % 2 != 0) /* odd rank */
                // local_A have largest local_n floats from local_A and even_partner
                Merge_high(local_A, temp_B, temp_C, local_n);
            else /* even rank */
                // local_A have smallest local_n floats from local_A and even_partner
                Merge_low(local_A, temp_B, temp_C, local_n);
        }
    }
    else
    { /* odd phase */
        if (odd_partner >= 0)
        { /* check for odd partner */
            MPI_Sendrecv(local_A, local_n, MPI_FLOAT, odd_partner, 0,
                         temp_B, local_n, MPI_FLOAT, odd_partner, 0, comm,
                         &status);
            if (my_rank % 2 != 0) /* odd rank */
                Merge_low(local_A, temp_B, temp_C, local_n);
            else /* even rank */
                Merge_high(local_A, temp_B, temp_C, local_n);
        }
    }
} /* Odd_even_iter */

// Comparison function for floats to be used with qsort
int Compare_floats(const void *a, const void *b)
{
    float arg1 = *(const float *)a;
    float arg2 = *(const float *)b;
    if (arg1 < arg2)
        return -1;
    if (arg1 > arg2)
        return 1;
    return 0;
}

// Adjusted Sort function to work with float arrays
void Sort(float local_A[], int local_n, int my_rank, int p, MPI_Comm comm)
{
    int phase;
    float *temp_B, *temp_C;
    int even_partner; /* phase is even or left-looking */
    int odd_partner;  /* phase is odd or right-looking */

    /* Allocate temporary storage for merging */
    temp_B = (float *)malloc(local_n * sizeof(float));
    temp_C = (float *)malloc(local_n * sizeof(float));

    /* Determine partners for the odd and even phases */
    if (my_rank % 2 != 0)
    {
        even_partner = my_rank - 1;
        odd_partner = my_rank + 1;
        if (odd_partner == p)
            odd_partner = MPI_PROC_NULL; // No partner, idle during odd phase
    }
    else
    {
        even_partner = my_rank + 1;
        if (even_partner == p)
            even_partner = MPI_PROC_NULL; // No partner, idle during even phase
        odd_partner = my_rank - 1;
    }

    /* Sort local list using built-in quick sort */
    qsort(local_A, local_n, sizeof(float), Compare_floats);

    for (phase = 0; phase < p; phase++)
        Odd_even_iter(local_A, temp_B, temp_C, local_n, phase,
                      even_partner, odd_partner, my_rank, p, comm);

    // Deallocate memory
    free(temp_B);
    free(temp_C);
} /* Sort */

int main(int argc, char *argv[])
{
    int my_rank, p; // rank, number processes
    char g_i;       // holds either g or i depending on user input
    float *local_A;   // local list: size of local number of elements * size of int
    int global_n;   // number of elements in global list
    int local_n;    // number of elements in local list (process list)
    MPI_Comm comm;
    double start, finish, loc_elapsed, elapsed;

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    Get_args(argc, argv, &global_n, &local_n, my_rank, p, comm);
    local_A = (float *)malloc(local_n * sizeof(float));

    // generate random list based on user input
    Generate_list(local_A, local_n, my_rank);
    Print_local_lists(local_A, local_n, my_rank, p, comm);

    MPI_Barrier(comm);
    start = MPI_Wtime();
    Sort(local_A, local_n, my_rank, p, comm);
    finish = MPI_Wtime();
    loc_elapsed = finish - start;
    MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

    Print_global_list(local_A, local_n, my_rank, p, comm);

    free(local_A); // deallocate memory

    if (my_rank == 0)
        printf("Sorting took %f milliseconds \n", loc_elapsed * 1000);

    MPI_Finalize();

    return 0;
} /* main */