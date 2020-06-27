/****************************************************
 ############ LAST UPDATE 09/10/2009 ################
 
 In Particles Class is enclosed all the elements 
 that characterize each particle.
 Particle:
 - Position, pos:{pos.x, pos.y, pos.z, pos.w (MASS) }
 - Velocity, vel:{vel.u, vel.v, vel.w}
 - Mutual distances , **dist 
 - Total mass of the system, M
 - Average distance between each particle, mDist
 - Epsilon, eps
 
****************************************************/
#ifndef PARTICLE_H
#define PARTICLE_H

#include "const.h"
#include "double4.h"


using namespace std;

class Particles
{
  ofstream out;
 public: 
  int n;

  double E0, E, L0, L; 
  double mDisk, mHalo, mBulge;
  double bB, aD, bD, aH;
  double eps;
  double h, beta;

  double rCl, M, M_BH;

  #ifdef LEAPFROG_METHOD
     float4 *pos;
  #else
     double4 *pos;
  #endif
  
  Vel *vel;
 public:
   
  Particles();
  ~Particles();
  
  void massFunction();
  void equalMasses();
  void genInitialConditions();
  void normalization();
    
  double kinEnergy(int n = N); 
 
  template<class T> 
  double potEnergy(T *pos, double M)
  {
    double radius;
    double pE =.0;
    double bulgeE,diskE, haloE;


    for (int i = 0; i < N; i++){
      for(int j = 0; j <= i-1; j++)
		  pE -= pos[i].w*pos[j].w / (sqrt((pos[i].x-pos[j].x) * (pos[i].x-pos[j].x) + (pos[i].y-pos[j].y) * 
										  (pos[i].y-pos[j].y) + (pos[i].z-pos[j].z)*(pos[i].z-pos[j].z)+ eps*eps));  
    } 

  
    mBulge =  1.40592e10 / M;
    mDisk  =  8.56080e10 / M;
    mHalo  = 10.70680e10 / M;

    bB =   387.3 / rCl;
    aD =  5317.8 / rCl;
    bD =   250.0 / rCl;
    aH = 12000.0 / rCl;
  
    bulgeE = .0;
    diskE  = .0;
    haloE  = .0;
  
    for (int i = 0; i < N ; i++){
      radius  = sqrt(pos[i].x*pos[i].x + pos[i].y*pos[i].y + pos[i].z*pos[i].z);
      bulgeE -= pos[i].w / sqrt(radius*radius + bB*bB);
      diskE  -= pos[i].w / sqrt( pos[i].x*pos[i].x + pos[i].y*pos[i].y + ( aD + sqrt( pos[i].z*pos[i].z +
                bD*bD ) )*( aD + sqrt(pos[i].z*pos[i].z + bD*bD) ) );
      haloE  += pos[i].w * ( log( 1 + pow( radius / aH , 1.02) ) - 3.186322775);
    }
  
    bulgeE *= mBulge;
    diskE  *= mDisk;
    haloE  *= mHalo/(1.02*aH);  
  
    pE += bulgeE + diskE + haloE;
  
    return pE;
  }

  /********* This function evaluates the sys. total angular momentum **************/
  template<class T> 
  double angMomentum(T *pos, Vel vel[])
 {
    double angMx =.0;
    double angMy =.0;
    double angMz =.0;
 
    for(int i = 0; i < N; i++){
      angMx += pos[i].w*(pos[i].y * vel[i].w - pos[i].z*vel[i].v);
      angMy -= pos[i].w*(pos[i].x * vel[i].w - pos[i].z*vel[i].u);
      angMz += pos[i].w*(pos[i].x * vel[i].v - pos[i].y*vel[i].u);
    }
    return sqrt(angMx*angMx + angMy*angMy + angMz*angMz);
  }

  inline void closeFile(ofstream &stream){ stream.close();}
  void openFile(ofstream &stream, const char *str);
  void readExtData(const char *str);
  
  template<class T> 
  void print(const char *str, T *p, Vel v[])
  {
    openFile( out, str );
 
   out << setiosflags( ios_base::scientific | ios_base::showpos | ios_base::right);
   out << setw( 10 );
   out << setprecision(8);

    for (int i = 0; i < N; i++) { 
      out << p[i].x << " " << p[i].y << " " << p[i].z << " ";
      out << v[i].u << " " << v[i].v << " " << v[i].w << " ";
      out << p[i].w << endl;
    }
    closeFile(out);
  }
     
  template<class T> 
  void print2(const char *str, T *p, Vel v[],const int i)
  {
    char s[40];
    char t[10];
     
    strcpy(s,str);
    
    sprintf(t, "%d", i);
    strcat(t, ".txt");
    strcat(s,t);

    print(s, p, v);

  } 

  void readInput(const char *str);
};

#endif
