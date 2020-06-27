/********************************************************
 For more informations about the variables open the file 
 header (particle.h)
********************************************************/

#include "header.h"

using namespace std;

Particles::Particles()
{
  readInput("input.txt");
 
  #ifdef LEAPFROG_METHOD 
      pos = new float4 [N];
  #else
      pos = new double4 [N];
  #endif
  vel = new Vel [N];
 
}

Particles::~Particles()
{
  delete [] vel;  
  delete [] pos;
 
}


void Particles::openFile(ofstream &stream, const char *str)
{
  stream.open(str, ios::binary);
  if (!stream.is_open()){
    cout << str << " does not exist.\n";
    exit(1);
  }
}



//***** This function evaluates the kinetic energy. *****//
double Particles::kinEnergy(int n)
{
  double k = .0;
  for (int i = 0; i < n ; i++)
    k += pos[i].w * (vel[i].u*vel[i].u + vel[i].v*vel[i].v + vel[i].w*vel[i].w);
  
  return .5 * k;
}

//***** This function generates the initial condition. *****//
void Particles::genInitialConditions()
{
  double x_t, y_t, z_t;
  double u_t, v_t, w_t;
  
  srand(time(0));
  
  for (int i = 0; i < N-1; i++){
    x_t = (double)rand()/((double)RAND_MAX) - .5;
    y_t = (double)rand()/((double)RAND_MAX) - .5;
    z_t = (double)rand()/((double)RAND_MAX) - .5;
    if ((x_t*x_t + y_t*y_t + z_t*z_t) < .25){
       pos[i].x = x_t;
       pos[i].y = y_t;
       pos[i].z = z_t;
    } else i--;
  }

  for(int i = 0; i < N-1; i++){
    u_t = (double)rand()/((double)RAND_MAX) - .5;
    v_t = (double)rand()/((double)RAND_MAX) - .5;
    w_t = (double)rand()/((double)RAND_MAX) - .5;
    if((u_t*u_t + v_t*v_t + w_t*w_t) < .25){
       vel[i].u = u_t;
       vel[i].v = v_t;
       vel[i].w = w_t;
    } else i--;
  }


}

/**** Sampling the Salpeter's function massFunction() assigning the  *****/
/**** the mass to each body.                                         *****/
void Particles::massFunction()
{
   int n_tot = 0;
   int diff, randomIdx;
   int nBodyC[CLASS_N];  // Numero di corpi contenuti in ogni classe 

   double mMin, mMax;       // massa minima e massima 
   double mBin[CLASS_N+1];    // Estremo inferiore del bin 
   double deltaLogM;

   /************* Black Hole Mass ***********************/
   pos[N-1].w = M_BH;
   /*****************************************************/
   
   mMin = .1;
   mMax = 15.;
   mBin[0] = mMin;
   nBodyC[0] = 0;

   deltaLogM = (log10(mMax)-log10(mMin))/(double)CLASS_N;

   for (int i = 1; i < CLASS_N+1; i++) {
     mBin[i]   = pow(10, log10(mBin[i-1]) + deltaLogM);
     nBodyC[i] = (int)(N * (pow(mBin[i], 1 - S_M) - pow(mBin[i-1],1 - S_M))/
     	         (pow(mBin[CLASS_N], 1 - S_M) - pow(mBin[0], 1 - S_M)));
     n_tot     = nBodyC[i];
   }

   srand(time(0));
   diff = N - n_tot;
   
 
   for (int i = 0; i < diff ; i++){
     randomIdx              = (CLASS_N-1)*(int)fabs( ((double)rand())/((double)RAND_MAX) );
     nBodyC[randomIdx + 1] += 1;
   }
  
   n_tot = 0;
   
   for(int j = 1; j < CLASS_N+1; j++){
     for(int i = n_tot; i < n_tot+nBodyC[j]; i++)
       pos[i].w = mBin[j]+(mBin[j-1]-mBin[j])*fabs((double)rand()/((double)(RAND_MAX)));
       n_tot   += nBodyC[j];
   }
}

/*********** This fuctions gives for each body the same mass ************/
void Particles::equalMasses()
{
   M = .0;
   for (int i = 0; i < N-1; i++){
     pos[i].w = 1.;
     M       += pos[i].w;
   }  
   
   /********** Black Hole Mass *******************/
   pos[N-1].w = M_BH;
   /**********************************************/

   M += pos[N-1].w;
}

/********** Serve per redere le variabili adimensionali ***********/
void Particles::normalization()
{
   double r2Max, r2In, rMax;
   double xCM, yCM, zCM;
   double velUnit, vEq;
   double densNum, mDist;
   double uCM, vCM, wCM, uBH, escVel;
   double eGrav = .0, k;
   double virialRatio, virialRatioNew, scale;

   M = .0;
   xCM = yCM = zCM = .0;
   
   for (int i = 0; i < N; i++)  M += pos[i].w;
   for (int i = 0; i < N-1; i++){
      xCM += pos[i].x*pos[i].w/(M-pos[N-1].w);
      yCM += pos[i].y*pos[i].w/(M-pos[N-1].w);
      zCM += pos[i].z*pos[i].w/(M-pos[N-1].w);
   }
  
   pos[N-1].x = xCM;
   pos[N-1].y = yCM;
   pos[N-1].z = zCM;
   
   r2Max = .0;

   for(int i = 0; i<N; i++){
     pos[i].x -= xCM;  
     pos[i].y -= yCM;
     pos[i].z -= zCM;  
   
     r2In = pos[i].x*pos[i].x + pos[i].y*pos[i].y + pos[i].z*pos[i].z;
     if (r2In > r2Max) r2Max = r2In ;
   }
   
   rMax = sqrt(r2Max);
   
   for (int i = 0; i < N; i++){
      pos[i].x /= rMax;
      pos[i].y /= rMax;
      pos[i].z /= rMax;
      pos[i].w /= M;
   }
   
   velUnit = sqrt(6.67259e-11 * M * 1.989e30/(rCl * 3.086e16));	
   vEq = 220000./velUnit;  

   /***************** SOFTENING ************************************/
   densNum = 3.*N/(4.*M_PI);
   mDist = 2. * pow(densNum, -1./3.);
   eps = ALPHA * mDist;   // mDist = Average distance
   
   /*************** ESCAPE VELOCITY ********************************/
   uBH = .0; //Black hole potential
   
   for(int i = 0; i < N-1; i++){
        uBH += pos[i].w / sqrt((pos[i].x-pos[ N-1].x) * (pos[i].x-pos[ N-1].x) + (pos[i].y-pos[N-1].y) * 
                          (pos[i].y-pos[ N-1].y) + (pos[i].z-pos[ N-1].z)*(pos[i].z-pos[ N-1].z)+ eps*eps);
//	uBH += pos[i].w / dist;
   }
   escVel = sqrt(2*uBH);
   
   /******************* ANGLE **************************************/
   double theta, phi;
   
   theta = M_PI*.5;
   phi = M_PI*.5;
  
   /******************** BH INITIAL VELOCITY ***********************/
   vel[N-1].u = beta * escVel * sin(theta) * cos(phi);
   vel[N-1].v = beta * escVel * sin(theta) * sin(phi);
   vel[N-1].w = beta * escVel * cos(theta);

   /******************* VELOCITY IN THE CM REFERENCE ***************/
   uCM = vCM = wCM = .0; 

   for(int i = 0; i < N-1; i++){
	uCM += vel[i].u*pos[i].w;   
	vCM += vel[i].v*pos[i].w;       
	wCM += vel[i].w*pos[i].w;     
   } 
				 
   uCM /= (1.-pos[N-1].w);   
   vCM /= (1.-pos[N-1].w);       
   wCM /= (1.-pos[N-1].w);
				 
   uCM += vel[N-1].u*pos[N-1].w/(1-pos[N-1].w);   
   vCM += vel[N-1].v*pos[N-1].w/(1-pos[N-1].w);       
   wCM += vel[N-1].w*pos[N-1].w/(1-pos[N-1].w); 
				 
   for(int i = 0; i < N-1; i++){
      vel[i].u -= uCM;  
      vel[i].v -= vCM;
      vel[i].w -= wCM;   
   }		

   /**************** Virial ratio without BH ***********************/
   
   for(int i = 1; i < N-1; i++){
      for(int j = 0; j <= i-1; j++){
        eGrav -= pos[i].w*pos[j].w /( sqrt((pos[i].x-pos[j].x) * (pos[i].x-pos[j].x) + (pos[i].y-pos[j].y) * 
                          (pos[i].y-pos[j].y) + (pos[i].z-pos[j].z)*(pos[i].z-pos[j].z)+ eps*eps));   
      }            
   }
 
 
 //  eGrav = -3./5.*(1-pos[N-1].w)*(1-pos[N-1].w);  
   k = kinEnergy(N-1);
   
   virialRatio = 2.*k/fabs(eGrav);
   virialRatioNew = .05;
   
   if (virialRatio > .05){
     scale = sqrt(virialRatioNew/virialRatio);
     for(int i = 0; i < N-1; i++){
	 vel[i].u *= scale;  
	 vel[i].v *= scale;
	 vel[i].w *= scale;             
     }
   
     k = kinEnergy(N-1);

     virialRatio = 2. *k/fabs(eGrav);
   }
   
   /************** Velocity in the c.m. reference system *************/ 
   uCM = vCM = wCM = .0;
   
   for(int i = 0; i < N-1; i++){
	uCM += vel[i].u*pos[i].w;   
	vCM += vel[i].v*pos[i].w;       
	wCM += vel[i].w*pos[i].w;     
   } 
				 
   uCM /= (1.-pos[N-1].w);   
   vCM /= (1.-pos[N-1].w);       
   wCM /= (1.-pos[N-1].w);
				 
   uCM += vel[N-1].u*pos[N-1].w/(1.-pos[N-1].w);   
   vCM += vel[N-1].v*pos[N-1].w/(1.-pos[N-1].w);       
   wCM += vel[N-1].w*pos[N-1].w/(1.-pos[N-1].w); 
   			 
   for(int i = 0; i < N-1; i++){
      vel[i].u -= uCM;  
      vel[i].v -= vCM;
      vel[i].w -= wCM;   
   }
   			
   /********** Evaluation of the virial ratio with the BH ***********/
//  eGrav = -3./5.;
  
   eGrav = 0;
   for (int i = 1; i < N; i++){
     for (int j = 0; j < i; j++){
         eGrav -= pos[i].w*pos[j].w/(sqrt((pos[i].x-pos[j].x) * (pos[i].x-pos[j].x) + (pos[i].y-pos[j].y) * 
                          (pos[i].y-pos[j].y) + (pos[i].z-pos[j].z)*(pos[i].z-pos[j].z)+ eps*eps));  
     }
   }


   k = kinEnergy(N);
   virialRatio = 2.*k/fabs(eGrav);

   /* In the Galaxy reference sys. with the cluster on x-axis and on the xy-plane */
   for(int i = 0; i < N; i++){
      pos[i].x += (8500./rCl);
      vel[i].v += vEq;
   }	 
}

// It reads the data input from an external file
void Particles::readExtData(const char *str)
{
   double densNum, mDist;
   ifstream in;
   int i;
   
   in.open(str, ios::binary);   
   cout << "Ext. Data\n";  
   if (!in.is_open()){
      cout << str <<" does not exist!\n";
      exit(1);
   }
   
   for (i = 0; i < N; i++) 
        in >> pos[i].x >> pos[i].y >> pos[i].z >> vel[i].u >> vel[i].v >>
	    vel[i].w >> pos[i].w; 
   
   in.close();

   /***************** SOFTENING *************************************/
   densNum = 3.*N/(4.*M_PI);
   mDist = 2. * pow(densNum, -1./3.);
   eps = ALPHA * mDist;   // mDist = Average distance
}

void Particles::readInput(const char *str)
{
   ifstream in;
   
   in.open(str);
   if (!in.is_open()){
     cout << str << " does not exist!\n";
     exit(1);
   }

   in.ignore(30, ':');
   in >> n;
   in.ignore(30, ':');
   in >> h;
   in.ignore(30, ':');
   in >> rCl;
   in.ignore(30, ':');
   in >> beta;
   in.ignore(50, ':');
   in >> M;
   in.ignore(50, ':');
   in >> M_BH;

   in.close();
}
