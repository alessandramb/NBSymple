#include "leapfrog.h"
#include "nbody.h"
#include "cutil.h"

Leapfrog::Leapfrog()
{  
  #ifndef EXT_DATA 
    genInitialConditions();
   #ifdef MASS
    massFunction();  
   #else
    cout << "Equal masses Leapfrog \n";
    equalMasses();  
   #endif // MASS
   normalization();
  
   /******* Print initial positions ****************/
   print( "./Data/leapfrog0.txt", pos, vel );
   /************************************************/  
  #else
   readExtData( "./Data/ExtData.txt" );
  #endif // EXT_DATA

  acc1 = new float4 [N];

  E0 = kinEnergy() + potEnergy<float4>(pos,M);
  L0 = angMomentum<float4>(pos,vel);
}

Leapfrog::~Leapfrog()
{
  delete [] acc1;
}

/**************************************************************/
/*                 LEAPFROG INTEGRATION METHOD                */
/**************************************************************/
void Leapfrog::integration()
{
  /******** Variables for the evalutation of the loop time ****/
  clock_t start, end;
  struct timeval tv;
  int secStart, secEnd; 
  int microSecStart, microSecEnd;
  double time;
  /************************************************************/
  int t_i = ((int) n/FRAME); 
  int q = 1;
  
  int i;
  int j;
  
  #ifdef _OMP
   omp_set_dynamic( NUM_THREADS );
   omp_set_num_threads( NUM_THREADS );
  #endif
  
  #ifndef POTENTIAL
  double R2, r2, r;
  double a, b, c;
  double axG_t, ayG_t, azG_t;

  mBulge =  1.40592e10 / M;
  mDisk  =  8.56080e10 / M;
  mHalo  = 10.70680e10 / M;
  
  bB =   387.3 / rCl;
  aD =  5317.8 / rCl;
  bD =   250.0 / rCl;
  aH = 12000.0 / rCl;
  #endif
  /****************** START CLOCK *****************************/
  start = clock();
  gettimeofday(&tv, NULL);
  secStart = tv.tv_sec;
  microSecStart = tv.tv_usec;
  /************************ START LOOP ************************/
  for (int k = 1; k <= n; k++){
   
    #pragma omp parallel private(i)
    {
     #pragma omp for 
     for (i = 0; i < N; i++){
       pos[i].x = pos[i].x + vel[i].u * h * .5;
       pos[i].y = pos[i].y + vel[i].v * h * .5;
       pos[i].z = pos[i].z + vel[i].w * h * .5;
     }
    } 
    #ifdef _OMP
      omp_set_num_threads( NUM_GPUS );
    #endif
           
    #pragma omp parallel default( shared )
    {      
      /********************************************************************/
      unsigned int cpu_thread_id = omp_get_thread_num();
      unsigned num_cpu_threads = omp_get_num_threads();
      int gpu_id = -1;

      CUDA_SAFE_CALL( cudaSetDevice((cpu_thread_id % num_cpu_threads) + 1 ) );
      CUDA_SAFE_CALL( cudaGetDevice(&gpu_id) );
      /********************************************************************/  
 
      float4 *accD;
      CUDA_SAFE_CALL( cudaMalloc((void**)&accD, N*sizeof(float4)) );

      float4 *posD; 
      CUDA_SAFE_CALL( cudaMalloc((void**)&posD, N*sizeof(float4)) );
      /********************************************************************/

      unsigned int istart =  cpu_thread_id * N / num_cpu_threads;
      float4 *sub_a = acc1 + cpu_thread_id * N / num_cpu_threads;

      /********************************************************************/
      dim3 gpu_threads( THREADS_PER_BLOCK );
      dim3 gpu_blocks( N/ (gpu_threads.x * num_cpu_threads) );
      int sharedMem = sizeof(float4)*gpu_threads.x; 

      /********************************************************************/
      CUDA_SAFE_CALL( cudaMemcpy(posD, pos, N*sizeof(float4), cudaMemcpyHostToDevice) ); 
 
      calculate_forces<<< gpu_blocks, gpu_threads, sharedMem >>>( posD, accD, rCl, M, istart );

      CUDA_SAFE_CALL( cudaMemcpy(sub_a, accD, (N / num_cpu_threads) * sizeof(float4), 
                                 cudaMemcpyDeviceToHost) );

      CUDA_SAFE_CALL( cudaFree( accD ) );
      CUDA_SAFE_CALL( cudaFree( posD ) );
    } 

    #ifdef _OMP
     omp_set_num_threads( NUM_THREADS);
    #endif

    #pragma omp parallel private( i ) default( shared )
    {
    /****  if the potential is not evaluted with CUDA **********/
    #ifndef POTENTIAL
    #pragma omp for private(R2,r2,r,a,b,c,axG_t,ayG_t,azG_t)
    for(i = 0; i < N; i++){
       R2 = pos[i].x*pos[i].x + pos[i].y*pos[i].y;
       r2 = pos[i].x*pos[i].x + pos[i].y*pos[i].y + pos[i].z*pos[i].z;
       r  = sqrt(r2);
     
       a = pow((r2 + bB*bB),1.5);
       axG_t = -mBulge * pos[i].x / a;
       ayG_t = -mBulge * pos[i].y / a;
       azG_t = -mBulge * pos[i].z / a;
 
       b = pow(R2 + (aD + sqrt(bD*bD + pos[i].z*pos[i].z))*(aD+ sqrt(bD*bD + pos[i].z*pos[i].z)),1.5);
       axG_t -= mDisk * pos[i].x / b;
       ayG_t -= mDisk * pos[i].y / b;
       azG_t -= mDisk * pos[i].z *(aD+ sqrt(bD*bD + pos[i].z*pos[i].z))/(sqrt(bD*bD + pos[i].z*pos[i].z) 
       	   * pow(R2+ (aD+ sqrt(bD*bD + pos[i].z*pos[i].z))*(aD+ sqrt(bD*bD + pos[i].z*pos[i].z)),1.5));

       /***************  see  Espresate,[2002] *******************/ 
       c = pow(r/aH,0.02)/((aH*aH)*r* (1. + pow(r/aH,1.02)));
       axG_t -= mHalo * pos[i].x * c;
       ayG_t -= mHalo * pos[i].y * c;
       azG_t -= mHalo * pos[i].z * c;
     
       acc1[i].x += axG_t;
       acc1[i].y += ayG_t;   
       acc1[i].z += azG_t;
    }    
    #endif
    #pragma omp for
    for(i = 0; i < N; i++){  
       vel[i].u = vel[i].u + h * acc1[i].x;
       pos[i].x = pos[i].x + vel[i].u * h * .5;
	 
       vel[i].v = vel[i].v + h * acc1[i].y;    
       pos[i].y = pos[i].y + vel[i].v * h * .5;
	 
       vel[i].w = vel[i].w + h * acc1[i].z;
       pos[i].z = pos[i].z + vel[i].w * h * .5;
	 
    }
    }

   /***********************************************************/
   if( k == q*t_i && q < FRAME+1 ) {      
       print2("./Data/leapfrog", pos, vel, q);
       q++;
    }
   /***********************************************************/      
  }  
  end = clock();
  gettimeofday(&tv, NULL);
  secEnd = tv.tv_sec;
  microSecEnd = tv.tv_usec;    
  /********************* END LOOP *****************************/

  time = (((double)(end-start))/((double)CLOCKS_PER_SEC));

  cout << setprecision(10);
  cout << "Time clock: " << time << " Get time of day: ";
  cout << secEnd + microSecEnd * 0.000001 - secStart - microSecStart*0.000001 << endl; 
     
  E = kinEnergy() + potEnergy(pos,M);
  L = angMomentum(pos,vel);
  cout << "E0 = " << E0 << "; E = " << E << "; DE =(E-E0)/E0 = " << (E-E0)/fabs(E0) << endl;
  cout << "L0 = " << L0 << "; L = " << L << "; DL =(L-L0)/L0 = " << (L-L0)/fabs(L0) << endl;

  /******** Print on the file *************************/
  openFile(outEL, "./Data/energiaL.txt");
  outEL << "####################################\n";
  outEL << "#          LEAPFROG METHOD:        #\n";
  outEL << "####################################\n";
  outEL << "Total Number of steps: " << n << ".\n";
  outEL << "Integration step: " << h << ".\n";
  outEL << "Cluster radius :" << rCl << ".\n";
  outEL << "Beta: " << beta << ".\n\n"; 
  outEL << "E0 = " << E0 << "; E = " << E << "; DE =(E-E0)/E0 = " << (E-E0)/fabs(E0) << endl;
  outEL << "L0 = " << L0 << "; L = " << L << "; DL =(L-L0)/L0 = " << (L-L0)/fabs(L0) << endl;
  outEL << endl;
  outEL << "Time clock: " << time << " Get time of day: ";
  outEL << secEnd + microSecEnd * 0.000001 - secStart - microSecStart*0.000001 << endl;
  closeFile(outEL);
  
  /****************************************************/
}

