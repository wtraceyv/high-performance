// Copyright 2019 by Alan M. Ferrenberg
#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#define MPI_WTIME_IS_GLOBAL 1

// Create a random number seed by reading bits from /dev/urandom
unsigned int getRanSeed(void) {
  unsigned int myRandomNumber = 0;
  size_t size = sizeof(myRandomNumber);
  std::ifstream urandom("/dev/urandom", std::ios::in|std::ios::binary);

  if (urandom) {
    urandom.read(reinterpret_cast<char*>(&myRandomNumber), size);
  }
  urandom.close();
  return(myRandomNumber);
}

// Print out the proper usage of the program
void usageError() {
  std::cerr << std::endl << "Usage:" << std::endl;
  std::cerr << std::endl << "pingpong array_size numLoops" << std::endl;
}

// Check the number of arguments, returning 1 if there's a problem
int processArgv(int argc, char *argv[], int *N, int *numLoops) {
  int status = 0;

  // Should be 2 arguments (N and numLoops) so argc should be 3
  if (argc == 3) {
    *N = std::stoi(argv[1]);
    *numLoops = std::stoi(argv[2]);
  } else {
    // There's a problem so print out an error message and return 1
    usageError();
    status = 1;
  }
  return status;
}

void doMPIStuff(int argc, char *argv[], int *numTasks, int *rank) {
  int value;

  // Initialize MPI for this program
  if ((value = MPI_Init(&argc, &argv)) != 0) {
    std::cerr << "Problem with MPI_Init" << std::endl;
    exit(value);
  }

  // Get the number of processes, exit if there aren't 2
  if ((value = MPI_Comm_size(MPI_COMM_WORLD, numTasks)) != 0) {
    std::cerr << "Problem with MPI_Comm_size" << std::endl;
    exit(value);
  }
  if (*numTasks != 2) {
    std::cerr << "Program is designed to run with 2 tasks" << std::endl;
    exit(3);
  }

  // Get my rank
  if ((value = MPI_Comm_rank(MPI_COMM_WORLD, rank)) != 0) {
    std::cerr << "Problem with MPI_Comm_rank" << std::endl;
    exit(value);
  }
}

void manager(int N, int numLoops, double *eTime, double *cTime, int array[] ) {
  double CPU, totalCPU, eStart, eEnd;
  clock_t t1;

  // Initialize the random number generator
  unsigned int seed = getRanSeed();
  srand(seed);

  // Initialize the array with random integers
  for (int i = 0; i < N; i++) {
    array[i] = rand_r(&seed);
  }

  // Initialize the clocks for elapsed and CPU time
  eStart = MPI_Wtime();
  t1 = clock();

  // Perform the requested number of loops, first sending data to the
  // other process then receiving it back.
  for (int i = 0; i < numLoops; i++) {
    MPI_Send(array, N, MPI_INT, 1, 1, MPI_COMM_WORLD);
    MPI_Recv(array, N, MPI_INT, 1, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  }

  // Get the end CPU time and use MPI_Reduce to accumulate it
  t1 = clock() - t1;
  CPU = static_cast<double>(t1)/static_cast<double>(CLOCKS_PER_SEC);
  MPI_Reduce(&CPU, &totalCPU, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

  // Get the end elapsed time and determine the transfer rate
  eEnd = MPI_Wtime();
  *eTime = (eEnd - eStart);
  *cTime = totalCPU;
}

void worker(int N, int numLoops, int array[]) {
  double CPU, totalCPU;
  clock_t t1;

  // Initialize the clock for CPU time
  t1 = clock();

  // Perform the requested number of loops, first receiving data from the
  // other process then sending it back.
  for (int i = 0; i < numLoops; i++) {
    MPI_Recv(array, N, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Send(array, N, MPI_INT, 0, 1, MPI_COMM_WORLD);
  }

  // Get the end CPU time and use MPI_Reduce to accumulate it
  t1 = clock() - t1;
  CPU = static_cast<double>(t1)/static_cast<double>(CLOCKS_PER_SEC);
  MPI_Reduce(&CPU, &totalCPU, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
  int N, numTasks, numLoops, rank, *array;

  // Process the command line arguments, exit if status is 1
  if (processArgv(argc, argv, &N, &numLoops) == 1) {
    exit(1);
  }

  // Do the MPI initialization and get my rank and the number of tasks
  doMPIStuff(argc, argv, &numTasks, &rank);

  // Declare an integer array of length N.  The manager will initialize it.
  array = new int[N];

  // Do differentiated work based on the rank.  First for the manager task
  if (rank == 0) {
    double elapsedTime, cpuTime;
    manager(N, numLoops, &elapsedTime, &cpuTime, array);
    std::cout << "elapsed time = " << elapsedTime;
    std::cout << " CPU time = " << cpuTime << std::endl;
  } else {
    // Now tasks for the other process.
    worker(N, numLoops, array);
  }

  // Wrap up MPI and clear out memory
  MPI_Finalize();
  delete [] array;

  // Exit the program
  exit(0);
}
