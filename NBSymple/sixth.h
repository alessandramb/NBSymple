#ifndef SIXTH_H
#define SIXTH_H

#include "header.h"
#include "double4.h"

using namespace std;

class Sixth: 
 public Particles
{
  ofstream outES;
  ofstream outPos;
    
  Vel *vel0;
  
  double *C;
  double *D;
    
  double4 *pos0;
  double4 *pos1;

  #ifdef DS_SIXTH
     DS4 *pos1_temp;

     float4 *acc1;
  #else
     double4 *acc1;
  #endif
 public:
  Sixth();
  ~Sixth();

  void integration();
};
#endif


