#ifndef LEAPFROG_H
#define LEAPFROG_H

#include "header.h"

using namespace std;

class Leapfrog: 
 public Particles
{
  ofstream outEL;
  ofstream outPos;

  float4 *acc1;
 public:
  Leapfrog();
  ~Leapfrog();

  void integration();
};
#endif
