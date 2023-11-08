#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

void exchange(std::vector<int>& data, int n, int m, int rank, int size, MPI_Comm comm) {
    int partnerRank = rank ^ (1 << m);
    if (rank < partnerRank) {
        std::vector<int> recvData(n);
        MPI_Recv(recvData.data(), n, MPI_INT, partnerRank, 0, comm, MPI_STATUS_IGNORE);

        if (rank & (1 << m)) {
            std::vector<int> mergedData(n);
            int i = 0, j = 0;
            for (int k = 0; k < n; k++) {
                if (i < n / 2 && (j == n / 2 || recvData[i] < data[j])) {
                    mergedData[k] = recvData[i];
                    i++;
                } else {
                    mergedData[k] = data[j];
                    j++;
                }
            }
            data = mergedData;
        } else {
            std::vector<int> mergedData(n);
            int i = n - 1, j = n - 1;
            for (int k = n - 1; k >= 0; k--) {
                if (i >= n / 2 && (j < n / 2 || data[i] > recvData[j])) {
                    mergedData[k] = data[i];
                    i--;
                } else {
                    mergedData[k] = recvData[j];
                    j--;
                }
            }
            data = mergedData;
        }
    } else {
        MPI_Send(data.data(), n, MPI_INT, partnerRank, 0, comm);
    }
}

void bitonicSortParallel(std::vector<int>& data, int rank, int size, MPI_Comm comm) {
    int n = data.size();
    for (int m = 1; m <= size; m <<= 1) {
        for (int j = m >> 1; j > 0; j >>= 1) {
            for (int i = 0; i < n; i++) {
                int bit = (i & j) >> (m - 2);
                if ((i & (m - 1)) == 0 && (i | (j - 1)) < n) {
                    if (bit == rank) {
                        exchange(data, n, m, rank, size, comm);
                    }
                }
            }
        }
    }
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int size, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (argc != 3) {
        if (rank == 0) {
            std::cerr << "Usage: " << argv[0] << " <num_elements> <num_processors>" << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    int numElements = std::atoi(argv[1]);
    int numProcessors = std::atoi(argv[2]);

    if (numProcessors != size) {
        if (rank == 0) {
            std::cerr << "Number of processors must match MPI communicator size." << std::endl;
        }
        MPI_Finalize();
        return 1;
    }

    // Generate random data on rank 0 and distribute it to all processors
    std::vector<int> data;
    if (rank == 0) {
        data.resize(numElements);
        for (int i = 0; i < numElements; i++) {
            data[i] = rand() % 1000;
        }
    }
    MPI_Bcast(data.data(), numElements, MPI_INT, 0, MPI_COMM_WORLD);

    // Perform parallel bitonic sort
    bitonicSortParallel(data, rank, size, MPI_COMM_WORLD);

    // Gather sorted data on rank 0
    std::vector<int> sortedData(numElements);
    MPI_Gather(data.data(), numElements, MPI_INT, sortedData.data(), numElements, MPI_INT, 0, MPI_COMM_WORLD);

    // Output sorted data on rank 0
    if (rank == 0) {
        std::cout << "Sorted Data: ";
        for (int i = 0; i < numElements; i++) {
            std::cout << sortedData[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Finalize();

    return 0;
}
