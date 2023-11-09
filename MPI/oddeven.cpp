#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// Helper function to get command line arguments
void Get_args(int argc, char *argv[], int *global_n, int *local_n, char *g_i, int my_rank, int p, MPI_Comm comm)
{
    if (my_rank == 0)
    {
        if (argc != 3)
        {
            fprintf(stderr, "Usage: mpiexec -np <p> %s <g|i> <global_n>\n", argv[0]);
            *global_n = -1; // Use an invalid value to indicate error
        }
        else
        {
            *g_i = argv[1][0]; // Expecting 'g' or 'i'
            *global_n = atoi(argv[2]);

            // Check for a valid 'g_i' and positive 'global_n'
            if ((*g_i != 'g' && *g_i != 'i') || *global_n <= 0)
            {
                fprintf(stderr, "Error: invalid arguments\n");
                *global_n = -1; // Use an invalid value to indicate error
            }
        }
    }

    // Broadcast global_n and g_i to all processes
    MPI_Bcast(global_n, 1, MPI_INT, 0, comm);
    MPI_Bcast(g_i, 1, MPI_CHAR, 0, comm);

    if (*global_n <= 0)
    {
        MPI_Finalize();
        exit(-1);
    }

    // Compute the local number of elements
    *local_n = *global_n / p;

    // Handle the case where global_n is not divisible by p
    if (*global_n % p != 0)
    {
        if (my_rank == 0)
        {
            fprintf(stderr, "Error: The number of elements must be divisible by the number of processes.\n");
        }
        MPI_Finalize();
        exit(-1);
    }
}

// Helper function to generate a list of integers
void Generate_list(int *local_A, int local_n, int my_rank)
{
    // Seed the random number generator to get different results each run
    srand(time(NULL) + my_rank);
    for (int i = 0; i < local_n; i++)
    {
        local_A[i] = rand() % 100; // Fill the array with random numbers from 0 to 99
    }
}

// Helper function to read in a user-defined list
void Read_list(int *local_A, int local_n, int my_rank, int p, MPI_Comm comm)
{
    // You should implement this to read the user's list from command line or file.
    // This is a placeholder implementation
    for (int i = 0; i < local_n; i++)
    {
        local_A[i] = my_rank + i; // Simple pattern for demonstration
    }
}

// Helper function to print a process's local list
void Print_local_lists(const int *local_A, int local_n, int my_rank, int p, MPI_Comm comm)
{
    for (int i = 0; i < p; i++)
    {
        if (my_rank == i)
        {
            printf("Process %d's local list: ", my_rank);
            for (int j = 0; j < local_n; j++)
            {
                printf("%d ", local_A[j]);
            }
            printf("\n");
        }
        MPI_Barrier(comm); // Synchronize before the next process prints
    }
}

// Helper function to sort local list
void Sort(int *local_A, int local_n, int my_rank, int p, MPI_Comm comm)
{
    // Implement your preferred sorting algorithm here
    // Placeholder: Use qsort for demonstration
    qsort(local_A, local_n, sizeof(int), compare);
}

// Comparison function for qsort
int compare(const void *a, const void *b)
{
    return (*(int *)a - *(int *)b);
}

// Helper function to print the global list
void Print_global_list(int *local_A, int local_n, int my_rank, int p, MPI_Comm comm)
{
    // This will require gathering the sorted local lists to the root process
    // and then printing the entire sorted list
    int *global_A = NULL;
    if (my_rank == 0)
    {
        global_A = (int *)malloc(p * local_n * sizeof(int));
    }
    MPI_Gather(local_A, local_n, MPI_INT, global_A, local_n, MPI_INT, 0, comm);

    if (my_rank == 0)
    {
        printf("Global list: ");
        for (int i = 0; i < p * local_n; i++)
        {
            printf("%d ", global_A[i]);
        }
        printf("\n");
        free(global_A);
    }
}

void Sort(int local_A[], int local_n, int my_rank,
          int p, MPI_Comm comm)
{
    int phase;
    int *temp_B, *temp_C;
    int even_partner; /* phase is even or left-looking */
    int odd_partner;  /* phase is odd or right-looking */

    /* Temporary storage used in merge-split */
    temp_B = (int *)malloc(local_n * sizeof(int));
    temp_C = (int *)malloc(local_n * sizeof(int));

    /* Find partners:  negative rank => do nothing during phase */
    if (my_rank % 2 != 0)
    { /* odd rank */
        even_partner = my_rank - 1;
        odd_partner = my_rank + 1;
        if (odd_partner == p)
            odd_partner = MPI_PROC_NULL; // Idle during odd phase
    }
    else
    { /* even rank */
        even_partner = my_rank + 1;
        if (even_partner == p)
            even_partner = MPI_PROC_NULL; // Idle during even phase
        odd_partner = my_rank - 1;
    }

    /* Sort local list using built-in quick sort */
    qsort(local_A, local_n, sizeof(int), Compare);

#ifdef DEBUG
    printf("Proc %d > before loop in sort\n", my_rank);
    fflush(stdout);
#endif

    for (phase = 0; phase < p; phase++)
        Odd_even_iter(local_A, temp_B, temp_C, local_n, phase,
                      even_partner, odd_partner, my_rank, p, comm);

    // deallocate memory
    free(temp_B);
    free(temp_C);
} /* Sort */

void Odd_even_iter(int local_A[], int temp_B[], int temp_C[], int local_n, int phase, int even_partner, int odd_partner, int my_rank, int p, MPI_Comm comm)
{
    MPI_Status status;

    if (phase % 2 == 0)
    { /* even phase */
        if (even_partner >= 0)
        { /* check for even partner */
            MPI_Sendrecv(local_A, local_n, MPI_INT, even_partner, 0,
                         temp_B, local_n, MPI_INT, even_partner, 0, comm,
                         &status);
            if (my_rank % 2 != 0) /* odd rank */
                // local_A have largest local_n ints from local_A and even_partner
                Merge_high(local_A, temp_B, temp_C, local_n);
            else /* even rank */
                // local_A have smallest local_n ints from local_A and even_partner
                Merge_low(local_A, temp_B, temp_C, local_n);
        }
    }
    else
    { /* odd phase */
        if (odd_partner >= 0)
        { /* check for odd partner */
            MPI_Sendrecv(local_A, local_n, MPI_INT, odd_partner, 0,
                         temp_B, local_n, MPI_INT, odd_partner, 0, comm,
                         &status);
            if (my_rank % 2 != 0) /* odd rank */
                Merge_low(local_A, temp_B, temp_C, local_n);
            else /* even rank */
                Merge_high(local_A, temp_B, temp_C, local_n);
        }
    }
} /* Odd_even_iter */

float random_float()
{
    return (int)rand() / (int)RAND_MAX;
}

void array_fill(int *arr, int length)
{
    srand(time(NULL));
    int i;
    for (i = 0; i < length; ++i)
    {
        arr[i] = random_float();
    }
}

void Print_local_lists(float local_A[], int local_n, int my_rank, int p, MPI_Comm comm)
{
    printf("Process %d's local list:\n", my_rank);
    for (int i = 0; i < local_n; i++)
    {
        printf("%f ", local_A[i]);
    }
    printf("\n");
    fflush(stdout); // Ensure the output is printed immediately
}

int main(int argc, char *argv[])
{
    int my_rank, p; // rank, number processes
    char g_i;       // holds either g or i depending on user input
    int *local_A;   // local list: size of local number of elements * size of int
    int global_n;   // number of elements in global list
    int local_n;    // number of elements in local list (process list)
    MPI_Comm comm;
    double start, finish, loc_elapsed, elapsed;

    MPI_Init(&argc, &argv);
    comm = MPI_COMM_WORLD;
    MPI_Comm_size(comm, &p);
    MPI_Comm_rank(comm, &my_rank);

    Get_args(argc, argv, &global_n, &local_n, &g_i, my_rank, p, comm);
    local_A = (int *)malloc(local_n * sizeof(int));

    // generate random list based on user input
    if (g_i == 'g')
    {
        array_fill(local_A, local_n, my_rank);
        Print_local_lists(local_A, local_n, my_rank, p, comm);
    }

#ifdef DEBUG
    printf("Proc %d > Before Sort\n", my_rank);
    fflush(stdout);
#endif

    MPI_Barrier(comm);
    start = MPI_Wtime();
    Sort(local_A, local_n, my_rank, p, comm);
    finish = MPI_Wtime();
    loc_elapsed = finish - start;
    MPI_Reduce(&loc_elapsed, &elapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);

#ifdef DEBUG
    Print_local_lists(local_A, local_n, my_rank, p, comm);
    fflush(stdout);
#endif

    Print_global_list(local_A, local_n, my_rank, p, comm);

    free(local_A); // deallocate memory

    if (my_rank == 0)
        printf("Sorting took %f milliseconds \n", loc_elapsed * 1000);

    MPI_Finalize();

    return 0;
} /* main */