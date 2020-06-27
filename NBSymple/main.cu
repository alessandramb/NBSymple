/****************************************
 *                NBSymple
 *                (main.cu)
 ****************************************/

#include "header.h"

using namespace std;

int main(int argc, char **argv)
{
  #ifdef LEAPFROG_METHOD
    cout << "LEAPFROG METHOD " << endl;
    Leapfrog A;
    A.integration();
  #else
    cout << "SIXTH ORDER METHOD " << endl;
    Sixth B;
    B.integration();
  #endif
   
  return 0;
}
