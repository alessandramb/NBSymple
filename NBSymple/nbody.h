#include "const.h"
#include "double4.h"
/**************************  DOUBLE SINGLE  **************************************/
/* See article Gaburov, Harfst, Portegies Zwart: "SAPPORO: A way to turn your
 * graphics cards into a GRAPE-6", New Astronomy 14 (2009) 630-637.
 */
/*********************************************************************************/
#ifdef DS_SIXTH                                           
// This function computes c = a + b.
__device__ DS dsAdd(DS a, DS b)
{
// Compute dsa + dsb using Knuth's trick.
  float t1 = a.x + b.x;
  float e = t1 - a.x;
  float t2 = ((b.x - e) + (a.x - (t1 - e))) + a.y + b.y;
                                                                                                                             
// The result t1 + t2, after normalization.
  DS c;
  c.x = e = t1 + t2;
  c.y = t2 - (e - t1);
  return c;
}

// This function computes c = a + b.
__device__ DS dsAdd(DS a, float b)
{
// Compute dsa + b using Knuth's trick.
  float t1 = a.x + b;
  float e = t1 - a.x;
  float t2 = ((b - e) + (a.x - (t1 - e))) + a.y;
                                                                                                                             
// The result t1 + t2, after normalization.
  DS c;
  c.x = e = t1 + t2;
  c.y = t2 - (e - t1);
  return c;
}
#endif

/*********************************************************************************/
/* See article Nyland et al: "Fast N-Body Simulation with CUDA"                  */
/*********************************************************************************/
template<class T1, class T2> __device__ T1 
bodyBodyInteraction(T2 bi, T2 bj, T1 ai, float eps) 
{ 
   float3 r; 
	
  #ifdef DS_SIXTH
     r.x = (bj.x.x - bi.x.x) + (bj.x.y - bi.x.y);
     r.y = (bj.y.x - bi.y.x) + (bj.y.y - bi.y.y);
     r.z = (bj.z.x - bi.z.x) + (bj.z.y - bi.z.y);
  #else
     r.x = bj.x - bi.x;
     r.y = bj.y - bi.y;
     r.z = bj.z - bi.z;
  #endif

  float distSqr= r.x*r.x + r.y*r.y + r.z*r.z + eps*eps;

  #ifdef SIXTH_METHOD 
    float distSixth = distSqr*distSqr*distSqr;
    float invDistCube = 1.0/sqrt(distSixth);
  #else
    float distSixth = distSqr*distSqr*distSqr;
    float invDistCube = 1.0f/sqrtf(distSixth);
  #endif

  #ifdef DS_SIXTH
    float s = bj.w.x * invDistCube;
  #else
    float s = bj.w * invDistCube;
  #endif 

  ai.x += r.x*s;
  ai.y += r.y*s;
  ai.z += r.z*s;

  return ai;
}

template<class T1, class T2> __device__ T1 
tile_calculation(T2 myPosition, T1 accel, float eps) 
{ 
  int i; 
  extern __shared__ T2 shPosition[];  

  int block = blockDim.x;

  for(i = 0; i < block; i++)
     accel = bodyBodyInteraction(myPosition, shPosition[i], accel, eps);
  return accel;
}

template<class T1, class T2> __device__ T1 
galactic_accelerations(float M,float rCluster, T2 myPosition)
{
  T1 accel;

  #ifdef SIXTH_METHOD
    double mBulge, mHalo, mDisk, aD, aH, bD, bB;
    double R2, r2, r;
    double axG_t, ayG_t, azG_t;
  #else
    float mBulge,mHalo, mDisk, aD, aH, bD, bB;
    float R2, r2, r;
    float axG_t, ayG_t, azG_t;
  #endif

  mBulge =   1.40592e10 / M;
  bB     = 387.3 / rCluster;
  
  mDisk  =    8.5608e10 / M;
  aD     = 5317.8 / rCluster;
  bD     =  250.0 / rCluster;
	
  mHalo  =     10.7068e10 / M;
  aH     =  12000.0 / rCluster;

  R2 = myPosition.x*myPosition.x + myPosition.y*myPosition.y;
  r2 = myPosition.x*myPosition.x + myPosition.y*myPosition.y + myPosition.z*myPosition.z;
  r  = sqrtf(r2);

  axG_t = -mBulge * myPosition.x / powf((r2 + bB*bB),1.5);
  ayG_t = -mBulge * myPosition.y / powf((r2 + bB*bB),1.5);
  azG_t = -mBulge * myPosition.z / powf((r2 + bB*bB),1.5);

  axG_t -= mDisk * myPosition.x / powf(R2 + (aD + sqrt(bD*bD + myPosition.z*myPosition.z))*
                                      (aD+ sqrtf(bD*bD + myPosition.z*myPosition.z)),1.5);
  ayG_t -= mDisk * myPosition.y / powf(R2 + (aD + sqrt(bD*bD + myPosition.z*myPosition.z))*
                                      (aD+ sqrtf(bD*bD + myPosition.z*myPosition.z)),1.5);
  azG_t -= mDisk * myPosition.z *(aD+ sqrtf(bD*bD + myPosition.z*myPosition.z))/
           (sqrtf(bD*bD + myPosition.z*myPosition.z) * powf(R2 + (aD+ sqrtf(bD*bD + 
            myPosition.z*myPosition.z))*(aD+ sqrtf(bD*bD + myPosition.z*myPosition.z)),1.5));

  /***************  see  Espresate,[2002] *******************/ 
  axG_t -= mHalo * myPosition.x * powf(r/aH,0.02)/((aH*aH)*r* (1. + powf(r/aH,1.02)));
  ayG_t -= mHalo * myPosition.y * powf(r/aH,0.02)/((aH*aH)*r* (1. + powf(r/aH,1.02)));
  azG_t -= mHalo * myPosition.z * powf(r/aH,0.02)/((aH*aH)*r* (1. + powf(r/aH,1.02)));
     
  accel.x = axG_t;
  accel.y = ayG_t;   
  accel.z = azG_t;

  return accel;	
} 	

/***** See article "Fast N-Body Simulation with CUDA", Nyland at al *******/
__global__  
void calculate_forces(void *devX, void *devA, float rCluster, float M, unsigned int istart)
{
  #ifdef DS_SIXTH  
    extern __shared__ DS4 shPosition[];
 
    DS4 *globalX = (DS4 *)devX;
    float4 *globalA = (float4 *)devA; 
    DS4 myPosition;
  #elif LEAPFROG_METHOD 
    extern __shared__ float4 shPosition[];
  
    float4 *globalX = (float4 *)devX;
    float4 *globalA = (float4 *)devA;
    float4 myPosition;
  #elif SIXTH_METHOD
    extern __shared__ double4 shPosition[];
   
    double4 *globalX = (double4 *)devX;
    double4 *globalA = (double4 *)devA;
    double4 myPosition;
  #endif

  int i, tile;

  #ifdef SIXTH_METHOD
   double3 acc = {0.0, 0.0, 0.0};
  #else 
   float3 acc  = {0.0f, 0.0f, 0.0f};
  #endif

  int gtid = blockIdx.x*blockDim.x + threadIdx.x + istart;
  int p = blockDim.x;
  
  myPosition.x = globalX[gtid].x;
  myPosition.y = globalX[gtid].y;
  myPosition.z = globalX[gtid].z;
  myPosition.w = globalX[gtid].w;
	
  /***************************Softening*******************************/
  float densNUM;
  float dmedia;
  float eps;
  densNUM  = 3 * N/ (4*M_PI);
  dmedia   = 2 * powf(densNUM,-1./3.);
  eps      = ALPHA * dmedia;
  /*******************************************************************/
	
  #ifdef POTENTIAL
   #ifdef DS_SIXTH
     acc = galactic_accelerations<float3, float4>(M, rCluster, (float4){ myPosition.x.x + myPosition.x.y, 
                                                                         myPosition.y.x + myPosition.y.y, 
                                                                         myPosition.z.x + myPosition.z.y,
                                                                         myPosition.w.x + myPosition.w.y } );
   #elif LEAPFROG_METHOD
     acc = galactic_accelerations<float3,float4>( M , rCluster, myPosition );
   #else
     acc = galactic_accelerations<double3,double4>( M, rCluster, myPosition);
   #endif
  #endif // POTENTIAL

  for(i = 0, tile = 0; i < N; i += p, tile++) {
     int idx = tile * blockDim.x + threadIdx.x ;
     
     shPosition[threadIdx.x].x = globalX[idx].x;
     shPosition[threadIdx.x].y = globalX[idx].y;
     shPosition[threadIdx.x].z = globalX[idx].z;
     shPosition[threadIdx.x].w = globalX[idx].w;
     
     __syncthreads();
     #ifdef LEAPFROG_METHOD
       acc = tile_calculation<float3, float4>( myPosition, acc, eps );
     #elif DS_SIXTH
       acc = tile_calculation<float3, DS4>( myPosition, acc, eps );	
     #else
       acc = tile_calculation<double3, double4>( myPosition, acc, eps );
     #endif
     __syncthreads();	
  }
 
  #ifdef SIXTH_METHOD
    double4 acc4 = {acc.x, acc.y, acc.z, 0.0}; 
  #else
    float4  acc4 = {acc.x, acc.y, acc.z, 0.0f};
  #endif

  globalA[gtid-istart].x = acc4.x;
  globalA[gtid-istart].y = acc4.y;
  globalA[gtid-istart].z = acc4.z;
}   
