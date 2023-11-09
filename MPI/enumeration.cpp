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

// #include <caliper/cali.h>
// #include <caliper/cali-manager.h>
// #include <adiak.hpp>

#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

int THREADS;
int NUM_VALS;
float random_float()
{
  return (float)rand()/(float)RAND_MAX;
}
int main (int argc, char *argv[])
{
// CALI_CXX_MARK_FUNCTION;
    
NUM_VALS = atoi(argv[1]);



int	numtasks,              /* number of tasks in partition */
	taskid,                /* a task identifier */
	numworkers,            /* number of worker tasks */
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
numworkers = numtasks-1;

int calculations_per_worker = NUM_VALS / numworkers;
int rank[calculations_per_worker];
int rank_idx[calculations_per_worker];
float received_data[NUM_VALS];
float sorted_array[NUM_VALS];

int is_master =(taskid == 0) ? 1 : 0;

MPI_Comm worker_comm;
MPI_Comm_split(MPI_COMM_WORLD, is_master, taskid, &worker_comm);

// WHOLE PROGRAM COMPUTATION PART STARTS HERE
double total_time_start = MPI_Wtime();
// CALI_MARK_BEGIN(whole_computation);

// Create caliper ConfigManager object
// cali::ConfigManager mgr;
// mgr.start();

/**************************** master task ************************************/
   if (taskid == MASTER)
   {
   
      // INITIALIZATION PART FOR THE MASTER PROCESS STARTS HERE

        printf("mpi_mm has started with %d tasks.\n",numtasks);
        printf("Initializing arrays...\n");


        // int rank, size;
        // MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        // MPI_Comm_size(MPI_COMM_WORLD, &size);

            const int n = NUM_VALS; // Size of the array
            float *h_array = new float[n];
            // float *h_rank = new float[n];
            

            srand(time(NULL));
            int i;
            for (i = 0; i < NUM_VALS; ++i) {
                h_array[i] = random_float();
            }

            // Print the og array
            std::cout << "Original Array: ";
            for (int i = 0; i < n; i++) {
                std::cout << h_array[i] << " ";
            }
            std::cout << std::endl;

      
            
      //INITIALIZATION PART FOR THE MASTER PROCESS ENDS HERE
      
      
      //SEND-RECEIVE PART FOR THE MASTER PROCESS STARTS HERE
        
        /* Send matrix data to the worker tasks */
        
        mtype = FROM_MASTER;
        numworkers = THREADS;
        for (dest=1; dest<=numworkers; dest++)
        {
            MPI_Send(h_array, NUM_VALS, MPI_INT, dest, mtype, MPI_COMM_WORLD);
            printf("Sent array to tasks%d\n",dest);
        }

        /* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (source=1; source<=numworkers; source++)
        {
            MPI_Recv(&rank, calculations_per_worker, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
            MPI_Recv(&rank_idx, calculations_per_worker, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);

            for (int i = 0; i < calculations_per_worker; i++){
                // sorted_array[rank[i]] = h_array[rank_idx];
            }

            printf("Received results from task %d\n",source);
        }
      
      //SEND-RECEIVE PART FOR THE MASTER PROCESS ENDS HERE
        std::cout << "Sorted Array: ";
        for (int i = 0; i < n; i++) {
            std::cout << sorted_array[i] << " ";
        }
        std::cout << std::endl;
      
        delete[] h_array;
      
      
   }


/**************************** worker task ************************************/
   if (taskid > MASTER)
   {
      //RECEIVING PART FOR WORKER PROCESS STARTS HERE
        mtype = FROM_MASTER;
        MPI_Recv(received_data, NUM_VALS, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
        
      //RECEIVING PART FOR WORKER PROCESS ENDS HERE
      

      //CALCULATION PART FOR WORKER PROCESS STARTS HERE

        int count = 0;
        for(int i = (taskid - 1); i < NUM_VALS; i += numworkers){
            
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

   double worker_receive_time_max,
      worker_receive_time_min,
      worker_receive_time_sum,
      worker_recieve_time_average,
      worker_calculation_time_max,
      worker_calculation_time_min,
      worker_calculation_time_sum,
      worker_calculation_time_average,
      worker_send_time_max,
      worker_send_time_min,
      worker_send_time_sum,
      worker_send_time_average = 0; // Worker statistic values.

   /* USE MPI_Reduce here to calculate the minimum, maximum and the average times for the worker processes.
   MPI_Reduce (&sendbuf,&recvbuf,count,datatype,op,root,comm). https://hpc-tutorials.llnl.gov/mpi/collective_communication_routines/ */
                            //    MPI_Reduce(&worker_receive_time,&worker_receive_time_max,1,MPI_DOUBLE,MPI_MAX,0,worker_comm);
                            //    MPI_Reduce(&worker_receive_time,&worker_receive_time_min,1,MPI_DOUBLE,MPI_MIN,0,worker_comm);
                            //    MPI_Reduce(&worker_receive_time,&worker_receive_time_sum,1,MPI_DOUBLE,MPI_SUM,0,worker_comm);

                            //    MPI_Reduce(&worker_calculation_time,&worker_calculation_time_max,1,MPI_DOUBLE,MPI_MAX,0,worker_comm);
                            //    MPI_Reduce(&worker_calculation_time,&worker_calculation_time_min,1,MPI_DOUBLE,MPI_MIN,0,worker_comm);
                            //    MPI_Reduce(&worker_calculation_time,&worker_calculation_time_sum,1,MPI_DOUBLE,MPI_SUM,0,worker_comm);

                            //    MPI_Reduce(&worker_send_time,&worker_send_time_max,1,MPI_DOUBLE,MPI_MAX,0,worker_comm);
                            //    MPI_Reduce(&worker_send_time,&worker_send_time_min,1,MPI_DOUBLE,MPI_MIN,0,worker_comm);
                            //    MPI_Reduce(&worker_send_time,&worker_send_time_sum,1,MPI_DOUBLE,MPI_SUM,0,worker_comm);


    //timing stuff
        //    if (taskid == 0)
        //    {
        //       // Master Times
        //       printf("******************************************************\n");
        //       printf("Master Times:\n");
        //       printf("Whole Computation Time: %f \n", whole_computation_time);
        //       printf("Master Initialization Time: %f \n", master_initialization_time);
        //       printf("Master Send and Receive Time: %f \n", master_send_receive_time);
        //       printf("\n******************************************************\n");

        //       // Add values to Adiak
        //       adiak::value("MPI_Reduce-whole_computation_time", whole_computation_time);
        //       adiak::value("MPI_Reduce-master_initialization_time", master_initialization_time);
        //       adiak::value("MPI_Reduce-master_send_receive_time", master_send_receive_time);

        //       // Must move values to master for adiak
        //       mtype = FROM_WORKER;
        //       MPI_Recv(&worker_receive_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        //       MPI_Recv(&worker_receive_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        //       MPI_Recv(&worker_recieve_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        //       MPI_Recv(&worker_calculation_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        //       MPI_Recv(&worker_calculation_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        //       MPI_Recv(&worker_calculation_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        //       MPI_Recv(&worker_send_time_max, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        //       MPI_Recv(&worker_send_time_min, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);
        //       MPI_Recv(&worker_send_time_average, 1, MPI_DOUBLE, 1, mtype, MPI_COMM_WORLD, &status);

        //       adiak::value("MPI_Reduce-worker_receive_time_max", worker_receive_time_max);
        //       adiak::value("MPI_Reduce-worker_receive_time_min", worker_receive_time_min);
        //       adiak::value("MPI_Reduce-worker_recieve_time_average", worker_recieve_time_average);
        //       adiak::value("MPI_Reduce-worker_calculation_time_max", worker_calculation_time_max);
        //       adiak::value("MPI_Reduce-worker_calculation_time_min", worker_calculation_time_min);
        //       adiak::value("MPI_Reduce-worker_calculation_time_average", worker_calculation_time_average);
        //       adiak::value("MPI_Reduce-worker_send_time_max", worker_send_time_max);
        //       adiak::value("MPI_Reduce-worker_send_time_min", worker_send_time_min);
        //       adiak::value("MPI_Reduce-worker_send_time_average", worker_send_time_average);
        //    }
        //    else if (taskid == 1)
        //    { // Print only from the first worker.
        //       // Print out worker time results.
            
        //       // Compute averages after MPI_Reduce
        //       worker_recieve_time_average = worker_receive_time_sum / (double)numworkers;
        //       worker_calculation_time_average = worker_calculation_time_sum / (double)numworkers;
        //       worker_send_time_average = worker_send_time_sum / (double)numworkers;

        //       printf("******************************************************\n");
        //       printf("Worker Times:\n");
        //       printf("Worker Receive Time Max: %f \n", worker_receive_time_max);
        //       printf("Worker Receive Time Min: %f \n", worker_receive_time_min);
        //       printf("Worker Receive Time Average: %f \n", worker_recieve_time_average);
        //       printf("Worker Calculation Time Max: %f \n", worker_calculation_time_max);
        //       printf("Worker Calculation Time Min: %f \n", worker_calculation_time_min);
        //       printf("Worker Calculation Time Average: %f \n", worker_calculation_time_average);
        //       printf("Worker Send Time Max: %f \n", worker_send_time_max);
        //       printf("Worker Send Time Min: %f \n", worker_send_time_min);
        //       printf("Worker Send Time Average: %f \n", worker_send_time_average);
        //       printf("\n******************************************************\n");

        //       mtype = FROM_WORKER;
        //       MPI_Send(&worker_receive_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        //       MPI_Send(&worker_receive_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        //       MPI_Send(&worker_recieve_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        //       MPI_Send(&worker_calculation_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        //       MPI_Send(&worker_calculation_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        //       MPI_Send(&worker_calculation_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        //       MPI_Send(&worker_send_time_max, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        //       MPI_Send(&worker_send_time_min, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        //       MPI_Send(&worker_send_time_average, 1, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
        //    }




   // Flush Caliper output before finalizing MPI
//    mgr.stop();
//    mgr.flush();

   MPI_Finalize();
}