#ifndef CONST_H
#define CONST_H

#define N 480                     // Number of particles
#define G 1
#define ALPHA 0.005

//#define MASS                     // Enable for using Salpeter mass function 
#define POTENTIAL                  // Disable it for evaluating the Galactic potential 
                                   // by CPUs. 

//#define EXT_DATA                 // Enable EXT_DATA if you want use the input data file 
                                   // ExtData.txt. ExtData.txt contains the initial conditions
                                   // of all part.s. The format is the following: x,y and z for
                                   // the position; u,v and w for the velocity and the mass, m.
                                   // EX.: 
                                   //   x y z u v w m

#define FRAME 10                   // number of output files 

#define S_M 2.35                   // It represents the exponent of the Salpeter function
#define CLASS_N 5                  // Class number 

#define THREADS_PER_BLOCK 16       // Number of threads per block on GPU device
#define NUM_GPUS 1				   // Number of GPUs
#define NUM_THREADS 8              // Number of Threads 

#endif
