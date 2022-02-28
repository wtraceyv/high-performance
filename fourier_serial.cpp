#include <cmath>
#include <iostream>

// Copyright 2020 Alan M. Ferrenberg

using std::cerr;
using std::cout;
using std::endl;
using std::stoi;

#define ZERO 0.0000000000000000000000000000000000

// Compute the discrete Fourier transform of the function provided
void computeFT(int N, double oR[], double oI[], double fR[], double fI[]) {
  int k, n;
  double tempR, tempI;

  // Loop over the N frequency values
  for (k = 0; k < N; k++) {
    tempR = ZERO;
    tempI = ZERO;
    // Loop over the N spatial/temporal values
    for (n = 0; n < N; n++) {
      double arg = 2.0*M_PI*k*n/N;
      double cosArg = cos(arg);
      double sinArg = sin(arg);
      
      // Accumulate the real and imaginary components of the Fourier transform
      // for frequency k in temporary variables
      tempR += oR[n]*cosArg + oI[n]*sinArg;
      tempI += oI[n]*cosArg - oR[n]*sinArg;
    }
    // Update the values of for the real and imaginary components of the
    // Fourier transform
    fR[k] = tempR;
    fI[k] = tempI;
  }
}

// Initialize the real and imaginary components of the original function and
// the Fourier transform.  The function is sinc(x) = sin(ax)/x
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
}

// Main method for this program
int main(int argc, char *argv[]) {
  // Declare pointers to the Function and fourier transform arrays
  double *oR, *oI, *fR, *fI;

  // Check whether the number of samples has been provided
  if (argc < 2) {
    cerr << endl << "Usage:  fourier_serial number_of_samples." << endl;
    exit(1);
  }

  // Number of samples is the command line argument
  int N = stoi(argv[1]);

  // Allocate arrays for the function and its Fourier transform.  This puts
  // them in the heap, not the stack.
  oR = new double[N];
  oI = new double[N];
  fR = new double[N];
  fI = new double[N];

  // Initialize the original function and its Fourier transform
  initialize(N, oR, oI, fR, fI);

  // If OUTPUT is defined, print out the original function and exit
#ifdef OUTPUT
  for (int n = 0; n < N; n++) {
    cout << n << " " << oR[n] << " " << oI[n] << endl;
  }
  exit(0);
#endif

  // Compute the Fourier transform of the function
  computeFT(N, oR, oI, fR, fI);

  // Write out the real and imaginary components of the Fourier transform
  for (int k = 0; k < N; k++) {
    cout << k << " " << fR[k] << " " << fI[k] << endl;
  }

  // Free up the memory on the heap
  delete [] oR;   delete [] oI;   delete [] fR;   delete [] fI;

  // Exit!
  exit(0);
}
