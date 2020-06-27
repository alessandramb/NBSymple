#include "sixth.h" 
#include "nbody.h"
#include "cutil.h"

Sixth::Sixth()
{  
  #ifndef EXT_DATA
   genInitialConditions();
   #ifdef MASS
    massFunction();  
   #else
     cout << "Sixth equal masses \n";
     equalMasses();  
   #endif // MASS  
   normalization();
   
   /******* Print initial positions ****************/
   print( "./Data/sixth0.txt", pos, vel );
   /************************************************/  
  #else
   readExtData( "./Data/ExtData.txt" );
  #endif // EXT_DATA
    
  pos1 = new double4 [N];

  #ifdef DS_SIXTH
    pos1_temp = new DS4 [N];
    
    acc1 = new float4 [N];
  #else
    acc1 = new double4 [N];
  #endif


  /**************** Sixth order coefficients *******/
  C = new double [8];
  C[0] = 0.392256805238780;
  C[1] = 0.510043411918458;
  C[2] =-0.471053385409758;
  C[3] = 0.0687531682525198;
  C[4] = 0.0687531682525198;
  C[5] =-0.471053385409758;
  C[6] = 0.510043411918458;
  C[7] = 0.392256805238780;

  D = new double [8];
  D[0] = 0.784513610477560;
  D[1] = 0.235573213359357;
  D[2] =-1.17767998417887;
  D[3] = 1.31518632068391;
  D[4] =-1.17767998417887;
  D[5] = 0.235573213359357;
  D[6] = 0.784513610477560;
  D[7] = 0.;
  /*************************************************/

  for (int i = 0; i < N; i++){
     pos1[i].w = pos[i].w;
  }   

  E0 = kinEnergy() + potEnergy( pos, M );
  L0 = angMomentum( pos, vel );

}

Sixth::~Sixth()
{
  delete [] pos1;

  #ifdef DS_SIXTH
     delete [] pos1_temp;
  #endif
   
  delete [] C;
  delete [] D;
  
  delete [] acc1;
}

/**************************************************************/
/*              SIXTH ORDER INTEGRATION METHOD                */
/**************************************************************/
void Sixth::integration()
{
  /******** Variables for the evalutation of the loop time ****/
  clock_t start, end;
  struct timeval tv;
  int secStart, secEnd;
  int microSecStart, microSecEnd;
  double time;
  /************************************************************/
  int t_i = ((int) n/FRAME);
  int q   = 1;
  
  int i; 
  int j; 

  #ifdef _OMP
   omp_set_dynamic( NUM_THREADS );
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
   for(int l = 0; l < 8; l++){    
    #pragma omp parallel default(shared) private(i)
    {
    #pragma omp for 
    for (i = 0; i < N; i++){
       pos1[i].x = pos[i].x + C[l] * vel[i].u * h;
       pos1[i].y = pos[i].y + C[l] * vel[i].v * h;
       pos1[i].z = pos[i].z + C[l] * vel[i].w * h;
       #ifdef DS_SIXTH
         pos1_temp[i] = (DS4){ to_DS( pos1[i].x ), to_DS( pos1[i].y ),
                               to_DS( pos1[i].z ), to_DS( pos1[i].w ) }; 
       #endif
      }
    }
    if(l != 7){
    #ifdef _OMP
    omp_set_num_threads( NUM_GPUS );
    #endif
    #pragma omp  parallel default( shared )
    { 
      /*********************************************************************/
      unsigned int cpu_thread_id = omp_get_thread_num();
      unsigned int num_cpu_threads = omp_get_num_threads();
      int gpu_id = -1;

      /*
       * The device of the GPUs are set from cudaSetDevice( 1 ) to
       * cudaSetDevice( ngpus ) where ngpus in the number of the n-th
       * GPU device allowed.
       */
      CUDA_SAFE_CALL( cudaSetDevice(cpu_thread_id % num_cpu_threads + 1 ) );
      CUDA_SAFE_CALL( cudaGetDevice(&gpu_id) );
      /*********************************************************************/
      #ifdef SIXTH_METHOD     
        double4 *accD;
        CUDA_SAFE_CALL( cudaMalloc( ( void** )&accD, N * sizeof( double4 ) ) );
     
        double4 *posD;
	CUDA_SAFE_CALL( cudaMalloc( ( void** )&posD, N * sizeof( double4 ) ) );
      #else
        float4 *accD;
        CUDA_SAFE_CALL( cudaMalloc( ( void** )&accD, N * sizeof( float4 ) ) );

        DS4 *posD;
        CUDA_SAFE_CALL( cudaMalloc( ( void** )&posD, N * sizeof( DS4 ) ) );
      #endif
      /*********************************************************************/
      int istart = cpu_thread_id * N/num_cpu_threads;
      #ifdef SIXTH_METHOD
        double4 *sub_a = acc1 + cpu_thread_id * N/ num_cpu_threads;
      #else
        float4 *sub_a = acc1 + cpu_thread_id * N/ num_cpu_threads;
      #endif
      /********************************************************************/
      dim3 gpu_threads( THREADS_PER_BLOCK );
      dim3 gpu_blocks( N / ( gpu_threads.x * num_cpu_threads ) );
      
      #ifdef SIXTH_METHOD
        int sharedMem = sizeof( double4 ) * gpu_threads.x;
        CUDA_SAFE_CALL( cudaMemcpy( posD, 
	                            pos1, 
	                            N * sizeof( double4 ),
				    cudaMemcpyHostToDevice ) );
      #else
        int sharedMem = sizeof( DS4 ) * gpu_threads.x;
        CUDA_SAFE_CALL( cudaMemcpy( posD, 
	                            pos1_temp, 
				    N * sizeof( DS4 ),
				    cudaMemcpyHostToDevice ) );
      #endif
      /*******************************************************************/
      calculate_forces <<< gpu_blocks, gpu_threads, sharedMem >>> ( posD, 
                                                                    accD,
								    rCl, 
								    M, 
								    istart );
      #ifdef SIXTH_METHOD
        CUDA_SAFE_CALL( cudaMemcpy( sub_a, accD, 
	                            ( N / num_cpu_threads ) * sizeof( double4 ), 
                                    cudaMemcpyDeviceToHost ) );
      #else
        CUDA_SAFE_CALL( cudaMemcpy( sub_a, accD,
                                    ( N / num_cpu_threads ) * sizeof( float4 ),
                                    cudaMemcpyDeviceToHost) );
      #endif
      CUDA_SAFE_CALL( cudaFree( accD ));    
      CUDA_SAFE_CALL( cudaFree( posD ));
    } 
  
        
    #ifdef _OMP
      omp_set_num_threads( NUM_THREADS );
    #endif
    }
    
    #pragma omp parallel default( shared ) private( i )
    { 
    #ifndef POTENTIAL        
    #pragma omp for private( R2, r2, r, a, b, c, axG_t, ayG_t, azG_t )
    for( i = 0; i < N; i++ ){
       R2 = pos1[i].x*pos1[i].x + pos1[i].y*pos1[i].y;
       r2 = pos1[i].x*pos1[i].x + pos1[i].y*pos1[i].y + pos1[i].z*pos1[i].z;
       r  = sqrt(r2);
     
       a = pow((r2 + bB*bB),1.5);
       axG_t = -mBulge * pos1[i].x / a;
       ayG_t = -mBulge * pos1[i].y / a;
       azG_t = -mBulge * pos1[i].z / a;
 
       b = pow(R2 + (aD + sqrt(bD*bD + pos1[i].z*pos1[i].z))*(aD+ sqrt(bD*bD + pos1[i].z*pos1[i].z)),1.5);
       axG_t -= mDisk * pos1[i].x / b;
       ayG_t -= mDisk * pos1[i].y / b;
       azG_t -= mDisk * pos1[i].z *(aD+ sqrt(bD*bD + pos1[i].z*pos1[i].z))/(sqrt(bD*bD + pos1[i].z*pos1[i].z) 
       	        * pow(R2+ (aD+ sqrt(bD*bD + pos1[i].z*pos1[i].z))*(aD+ sqrt(bD*bD + pos1[i].z*pos1[i].z)),1.5));

       /***************  see  Espresate,[2002] *******************/ 
       c = pow(r/aH,0.02)/((aH*aH)*r* (1. + pow(r/aH,1.02)));
       axG_t -= mHalo * pos1[i].x * c;
       ayG_t -= mHalo * pos1[i].y * c;
       azG_t -= mHalo * pos1[i].z * c;
     
       acc1[i].x += axG_t;
       acc1[i].y += ayG_t;   
       acc1[i].z += azG_t;
    }    
    #endif

    #pragma omp for
    for(i = 0; i < N; i++){  
       vel[i].u = vel[i].u + D[l] * h * acc1[i].x;
       pos[i].x = pos1[i].x;
	 
       vel[i].v = vel[i].v + D[l] * h * acc1[i].y;
       pos[i].y = pos1[i].y;    
       
       vel[i].w = vel[i].w + D[l] * h * acc1[i].z;
       pos[i].z = pos1[i].z;
     }
    }
   }
   /***********************************************************/
   if( k == q*t_i && q < FRAME+1 ) {
       print2("./Data/sixth", pos1, vel, q);
       q++;
    }
   /***********************************************************/
            
  }
  /************************* END LOOP *************************/
  end = clock();
  gettimeofday(&tv, NULL);
  secEnd = tv.tv_sec;
  microSecEnd = tv.tv_usec;
  
  /************ Evaluation of the distances *******************/  
 
  time = (((double)(end-start))/((double)CLOCKS_PER_SEC));

  cout << setprecision(10);
  cout << "Time clock: " << time << " Get time of day: ";
  cout << secEnd + microSecEnd * 0.000001 - secStart - microSecStart*0.000001 << endl;

  E = kinEnergy() + potEnergy( pos, M );
  L = angMomentum( pos, vel );
  
  cout << "E0 = " << E0 << "; E = " << E << "; DE =(E-E0)/E0 = " << (E-E0)/fabs(E0) << endl;
  cout << "L0 = " << L0 << "; L = " << L << "; DL =(L-L0)/L0 = " << (L-L0)/fabs(L0) << endl;

  /************** Print on the file *************************/
  openFile(outES, "./Data/energiaS.txt");
  outES << "####################################\n";
  outES << "#       SIXTH ORDER METHOD:        #\n";
  outES << "####################################\n";
  outES << "Total Number of steps: " << n << ".\n";
  outES << "Integration step: " << h << ".\n";
  outES << "Cluster radius :" << rCl << ".\n";
  outES << "Beta: " << beta << ".\n\n";
  outES << "E0 = " << E0 << "; E = " << E << "; DE =(E-E0)/E0 = " << (E-E0)/fabs(E0) << endl;
  outES << "L0 = " << L0 << "; L = " << L << "; DL =(L-L0)/L0 = " << (L-L0)/fabs(L0) << endl;
  outES << endl;
  outES << "Time clock: " << time << " Get time of day: ";
  outES << secEnd + microSecEnd * 0.000001 - secStart - microSecStart*0.000001 << endl;
  
  closeFile(outES);
  /*********************************************************/
}

