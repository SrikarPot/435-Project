/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:  
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks-1 worker tasks.
*   NOTE:  C and Fortran versions of this code differ because of the way
*   arrays are stored/passed.  C arrays are row-major order but Fortran
*   arrays are column-major order.
* AUTHOR: Blaise Barney. Adapted from Ros Leibensperger, Cornell Theory
*   Center. Converted to MPI: George L. Gusciora, MHPCC (1/95)
* LAST REVISED: 09/29/2021
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <limits.h>
#include <iostream>
#include <ostream>
#include "../helper.h"

#include <caliper/cali.h>
#include <caliper/cali-manager.h>
#include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int THREADS;
int NUM_VALS;


int main (int argc, char *argv[])
{
CALI_CXX_MARK_FUNCTION;
    
NUM_VALS = atoi(argv[1]);
std::string input_type = argv[2];



int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers_inc_master,            /* number of worker tasks */
	source,                /* task id of message source */
	dest,                  /* task id of message destination */
	mtype,                 /* message type */
	i, j, k, rc;           /* misc */

MPI_Status status;

// double worker_receive_time,       /* Buffer for worker recieve times */
//    worker_calculation_time,      /* Buffer for worker calculation times */
//    worker_send_time = 0;         /* Buffer for worker send times */
// double whole_computation_time,    /* Buffer for whole computation time */
//    master_initialization_time,   /* Buffer for master initialization time */
//    master_send_receive_time = 0; /* Buffer for master send and receive time */
/* Define Caliper region names */
    // const char* whole_computation = "whole_computation";
    // const char* master_initialization = "master_initialization";
    // const char* master_send_recieve = "master_send_recieve";
    // const char* worker_recieve = "worker_recieve";
    // const char* worker_calculation = "worker_calculation";
    // const char* worker_send = "worker_send";


MPI_Init(&argc,&argv);
MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
if (numtasks < 2 ) {
  printf("Need at least two MPI tasks. Quitting...\n");
  MPI_Abort(MPI_COMM_WORLD, rc);
  exit(1);
  }
numworkers_inc_master = numtasks;

int calculations_per_worker = NUM_VALS / numworkers_inc_master;
int rank[calculations_per_worker];
int rank_idx[calculations_per_worker];
float received_data[NUM_VALS];
float sorted_array[NUM_VALS];


// WHOLE PROGRAM COMPUTATION PART STARTS HERE
double total_time_start = MPI_Wtime();
// CALI_MARK_BEGIN(whole_computation);

// Create caliper ConfigManager object
cali::ConfigManager mgr;
mgr.start();

/**************************** master task ************************************/
   if (taskid == MASTER)
   {
   
        // INITIALIZATION PART FOR THE MASTER PROCESS STARTS HERE
            printf("mpi_mm has started with %d tasks.\n",numtasks);
            printf("Initializing arrays...\n");

            const int n = NUM_VALS;
            float *h_array;

            CALI_MARK_BEGIN("data_init");
            h_array = (float*)malloc(n * sizeof(float));
            array_fill(h_array, n, input_type);
            CALI_MARK_END("data_init");
        //INITIALIZATION PART FOR THE MASTER PROCESS ENDS HERE

        std::cout << "Original Array: ";
        for (int i = 0; i < n; i++) {
            std::cout << h_array[i] << " ";
        }
        std::cout << std::endl;
      
      
        //SEND ARRAY DATA PART FOR THE MASTER PROCESS STARTS HERE
            /* Send matrix data to the worker tasks */
            mtype = FROM_MASTER;
            printf("Sending array to tasks");

            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_large");
            CALI_MARK_BEGIN("MPI_Bcast");
            MPI_Bcast(h_array, NUM_VALS, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
            CALI_MARK_END("MPI_Bcast");
            CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");

            printf("Sent array to all tasks");

        //SEND ARRAY DATA PART FOR THE MASTER PROCESS ENDS HERE

        
        /* Do master thread calculations */
            int count = 0;
            CALI_MARK_BEGIN("comp");
            CALI_MARK_BEGIN("comp_large");
            for(int i = taskid; i < NUM_VALS; i += numworkers_inc_master){
                
                if (i < NUM_VALS) {
                    rank[count] = 0;
                    rank_idx[count] = i;
                    for (int j = 0; j < NUM_VALS; j++) {
                        if (received_data[j] < received_data[i] || (received_data[j] == received_data[i] && j < i)) {
                            rank[count]++;
                        }
                    }
                }
                
                count++;
            }
            CALI_MARK_END("comp_large");
            CALI_MARK_END("comp");

            CALI_MARK_BEGIN("comp");
            CALI_MARK_BEGIN("comp_small");
            for (int i = 0; i < calculations_per_worker; i++){
                std::cout << "master rank_idx: " << rank_idx[i] << std::endl;
                std::cout << "master rank: " << rank[i] << std::endl;
                sorted_array[rank[i]] = h_array[rank_idx[i]];
            }
            CALI_MARK_END("comp_small");
            CALI_MARK_END("comp");


        /* Receive results from worker tasks */
            mtype = FROM_WORKER;
            CALI_MARK_BEGIN("comm");
            CALI_MARK_BEGIN("comm_large");
            CALI_MARK_BEGIN("MPI_Recv");
            for (source=1; source<numworkers_inc_master; source++)
            {
                MPI_Recv(&rank, calculations_per_worker, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
                MPI_Recv(&rank_idx, calculations_per_worker, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);

                for (int i = 0; i < calculations_per_worker; i++){
                    sorted_array[rank[i]] = h_array[rank_idx[i]];
                }

                printf("Received results from task %d\n",source);
                std::cout << std::endl;
            }
            CALI_MARK_END("MPI_Recv");
            CALI_MARK_END("comm_large");
            CALI_MARK_END("comm");
      
        std::cout << "Sorted Array: ";
        for (int i = 0; i < n; i++) {
            std::cout << sorted_array[i] << " ";
        }
        std::cout << std::endl;
      
        CALI_MARK_BEGIN("correctness_check");
        if (correctness_check(h_array, n)) {
            printf("Array correctly sorted!\n");
        } else {
            printf("Array sorting failed\n");
        }
        CALI_MARK_END("correctness_check");
        free(h_array);
      
      
   }


/**************************** worker task ************************************/
   if (taskid > MASTER)
   {
      //RECEIVING PART FOR WORKER PROCESS STARTS HERE
        MPI_Bcast(received_data, NUM_VALS, MPI_FLOAT, MASTER, MPI_COMM_WORLD);
        printf("worker recieved array");
        
      //RECEIVING PART FOR WORKER PROCESS ENDS HERE
      

      //CALCULATION PART FOR WORKER PROCESS STARTS HERE

        int count = 0;
        for(int i = taskid; i < NUM_VALS; i += numworkers_inc_master){
            
            if (i < NUM_VALS) {
                rank[count] = 0;
                rank_idx[count] = i;
                for (int j = 0; j < NUM_VALS; j++) {
                    if (received_data[j] < received_data[i] || (received_data[j] == received_data[i] && j < i)) {
                        rank[count]++;
                    }
                }
            }
            count++;
        }
         
      //CALCULATION PART FOR WORKER PROCESS ENDS HERE
      
      
      //SENDING PART FOR WORKER PROCESS STARTS HERE

        mtype = FROM_WORKER;
        MPI_Send(&rank, calculations_per_worker, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
        MPI_Send(&rank_idx, calculations_per_worker, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);

      //SENDING PART FOR WORKER PROCESS ENDS HERE

   }

   // WHOLE PROGRAM COMPUTATION PART ENDS HERE


    adiak::init(NULL);
    adiak::launchdate();    // launch date of the job
    adiak::libraries();     // Libraries used
    adiak::cmdline();       // Command line used to launch the job
    adiak::clustername();   // Name of the cluster
    adiak::value("Algorithm", "EnumerationSort"); // The name of the algorithm you are using (e.g., "MergeSort", "BitonicSort")
    adiak::value("ProgrammingModel", "MPI"); // e.g., "MPI", "CUDA", "MPIwithCUDA"
    adiak::value("Datatype", "float"); // The datatype of input elements (e.g., double, int, float)
    adiak::value("SizeOfDatatype", 4); // sizeof(datatype) of input elements in bytes (e.g., 1, 2, 4)
    adiak::value("InputSize", NUM_VALS); // The number of elements in input dataset (1000)
    adiak::value("InputType", (char*)input_type.c_str()); // For sorting, this would be "Sorted", "ReverseSorted", "Random", "1%perturbed"
    adiak::value("num_procs", numtasks); // The number of processors (MPI ranks)
    // adiak::value("num_threads", THREADS); // The number of CUDA or OpenMP threads
    // adiak::value("num_blocks", BLOCKS); // The number of CUDA blocks 
    adiak::value("group_num", 15); // The number of your group (integer, e.g., 1, 10)
    adiak::value("implementation_source", "Handwritten"); // Where you got the source code of your algorithm; choices: ("Online", "AI", "Handwritten").


   // Flush Caliper output before finalizing MPI
   mgr.stop();
   mgr.flush();

   MPI_Finalize();
}