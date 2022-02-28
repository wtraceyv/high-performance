// Copyright (C) 2020 Wall-e traceywd@miamioh.edu
// with great reference to code by Alan Ferrenberg

// #ifdef OUTPUT
//   for (int n = 0; n < N; n++) {
//     cout << n << " " << oR[n] << " " << oI[n] << endl;
//   }
//   exit(0);
// #endif

// mpic++ -O3 -std=c++14 -Wall to compile
// mpirun -np [numTasks] ./Executable

#include <mpi.h>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <cmath>

// ease of namespaces
using std::cerr;
using std::cout;
using std::endl;
using std::stoi;

// macros
// - - - macro for conditional compilation/debug?
#define MPI_WTIME_IS_GLOBAL 1
#define ZERO 0.0000000000000000000000000000000000

// Init origin function (oR, oI) and fR, fI for computations 
void initialize(int N, double oR[], double oI[], double fR[], double fI[]) {
  double a = 2.0;
  oR[0] = a;
  oI[0] = ZERO;
  fR[0] = ZERO;
  fI[0] = ZERO;

  for (int n = 1; n < N; n++) {
    oR[n] = sin(a*n)/n;
    oI[n] = ZERO;
    fR[n] = ZERO;
    fI[n] = ZERO;
  }
}  // END initialize

void mpiPrepare(int argc, char *argv[], int *numTasks, int *rank) {
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
  // Get my rank
  if ((value = MPI_Comm_rank(MPI_COMM_WORLD, rank)) != 0) {
    std::cerr << "Problem with MPI_Comm_rank" << std::endl;
    exit(value);
  }
}


// COMPUTE DFT, only one Xk
// return real and imaginary components for given k in *fR and *fI
void computeFT(int N, int k, double oR[], double oI[], double *fR, double *fI) {
  double tempR = ZERO, tempI = ZERO;
  for (int n = 0; n < N; n++) {
    double arg = 2.0*M_PI*k*n/N;
    double cosArg = cos(arg);
    double sinArg = sin(arg);
    tempR += oR[n]*cosArg + oI[n]*sinArg;
    tempI += oI[n]*cosArg - oR[n]*sinArg;
  }
  *fR = tempR;
  *fI = tempI;
}

void printFT(double reals[], double imag[], int length) {
  for (int k = 0; k < length; k++) {
    cout << "k = " << k << ", " << reals[k] << ", " << imag[k] << endl;
  }
}

void manager(int N, int numTasks, double oR[], double oI[], double fR[], double fI[]) {
  double CPU, totalCPU, eStart, eEnd;
  clock_t t1;
  fR = new double[N];  fI = new double[N];  oR = new double[N];
  oI = new double[N];
  initialize(N, oR, oI, fR, fI);
  eStart = MPI_Wtime();
  t1 = clock();
  // Bcast origin arrays
  MPI_Bcast(oR, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Bcast(oI, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  // send workers initial k values to calculate (0-numTasks)
  int k = 0;
  double tempR, tempI;
  for (int i = 0; i < numTasks; i++) {
    MPI_Send(&i, 1, MPI_INT, i + 1, 0, MPI_COMM_WORLD);
  }
  while (k < N) {
    MPI_Status status;  // expecting k value calculated within status
    MPI_Recv(&tempR, 1, MPI_DOUBLE, k + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    MPI_Recv(&tempI, 1, MPI_DOUBLE, k + 1, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    int rank = status.MPI_SOURCE, tag = status.MPI_TAG;  // make sure works??
    fR[tag] = tempR;  fI[tag] = tempI;  // tag = k value calculated
    k++;
    if (k < N) {
      MPI_Send(&k, 1, MPI_INT, rank, 0, MPI_COMM_WORLD);  // tag not matter
    }
  }
  // presumably we are done..tell everybody
  k = -1;
  MPI_Bcast(&k, 1, MPI_INT, 0, MPI_COMM_WORLD);

  // finish timing save
  t1 = clock() - t1;
  CPU = static_cast<double>(t1)/static_cast<double>(CLOCKS_PER_SEC);
  MPI_Reduce(&CPU, &totalCPU, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
  eEnd = MPI_Wtime();
  double elapsed = (eEnd - eStart);
  cerr << "e: " << elapsed << ", CPU: " << totalCPU << endl;
  printFT(fR, fI, N);
  delete [] fR;
  delete [] fI;
}  // END manager's routine

void worker(int N) {
  double CPU, totalCPU, tempR, tempI;
  double oR[N], oI[N];
  int k = 1;
  MPI_Status status;
  // init timing tools
  clock_t t1;
  t1 = clock();

  MPI_Recv(oR, N, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  MPI_Recv(oI, N, MPI_DOUBLE, 0, MPI_ANY_TAG, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
  // recieve Bcasts for arrays
  while (k >= 0) {
    MPI_Recv(&k, 1, MPI_INT, 0, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    k = status.MPI_TAG;
    if (k < 0)
      break;
    computeFT(N, k, oR, oI, &tempR, &tempI);
    MPI_Send(&tempR, 1, MPI_DOUBLE, 0, k, MPI_COMM_WORLD);
    MPI_Send(&tempI, 1, MPI_DOUBLE, 0, k, MPI_COMM_WORLD);
  }

  // finish my CPU timing save
  t1 = clock() - t1;
  CPU = static_cast<double>(t1)/static_cast<double>(CLOCKS_PER_SEC);
  MPI_Reduce(&CPU, &totalCPU, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
}  // END workers' routine

int processArgv(int argc, char *argv[], int *N) {
  if (argc == 2) {
    *N = stoi(argv[1]);
  } else {
    cerr << endl << "Usage:   ./mpi_fourier N" << endl;  // mpirun -np 2-8
    return 1;
  }
  return 0;
}

int main(int argc, char *argv[]) {
  int N = 0, numTasks, rank;
  double *oR, *oI, *fR, *fI;

  // Check whether N has been provided
  if (processArgv(argc, argv, &N) != 0){
    exit(1);
  }
  oR = new double[N];
  oI = new double[N];
  fR = new double[N];
  fI = new double[N];

  // init MPI
  mpiPrepare(argc, argv, &numTasks, &rank);
  // divvy manager, workers
  if (rank == 0) {
    manager(N, numTasks, oR, oI, fR, fI);
  } else {
    worker(N);
  }
  // Write out the real and imaginary components of the Fourier transform
  for (int k = 0; k < N; k++) {
    cout << k << " " << fR[k] << " " << fI[k] << endl;
  }
  cout << "..now finalizing MPI" << endl;

  MPI_Finalize();
  delete [] oR;   delete [] oI;   delete [] fR;   delete [] fI;
  exit(0);
}
