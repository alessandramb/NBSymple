#include <iostream>
#include <fstream>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <ctime>
#include <new>

#include "stdlib.h"
#include "sys/time.h"
#include "omp.h"

#include "const.h"
#include "particle.h"

#ifdef LEAPFROG_METHOD
 #include "leapfrog.h"
#else
 #include "sixth.h"
#endif
