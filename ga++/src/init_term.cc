
#include "ga++.h"

void
GA::Initialize(int argc, char *argv[], size_t limit) {
 
  int val;
  
#ifndef MPIPP
  PBEGIN_(argc, argv);
#else
  if((val=MPI_Init(&argc, &argv)) < 0) 
    fprintf(stderr, "MPI_Init() failed\n");
#endif

  // GA Initialization
  if(limit == 0) 
    GA_Initialize();
  else 
    GA_Initialize_ltd(limit);
}

void 
GA::Initialize(int argc, char *argv[], unsigned long heapSize, 
	   unsigned long stackSize, int type, size_t limit) {
  
  int val;
  
  // Initialize MPI/TCGMSG  
#ifndef MPIPP
  PBEGIN_(argc, argv);
#else
  if((val=MPI_Init(&argc, &argv)) < 0) 
    fprintf(stderr, "MPI_Init() failed\n");
#endif
  
  
  // GA Initialization
  if(limit == 0) 
    GA_Initialize();
  else 
    GA_Initialize_ltd(limit);

  
  //if(GA_Uses_ma()) {
  
  int nProcs = GA_Nnodes();
  
  // Initialize memory allocator
  heapSize /= ((unsigned long) nProcs);
  stackSize /= ((unsigned long) nProcs);
  
  if(!MA_init(type, stackSize, heapSize)) 
    GA_Error((char *)"MA_init failed",stackSize+heapSize);
  // }
}

void 
GA::Terminate()
{
  
  /* Terminate GA */
  GA_Terminate();    
  
#ifndef MPIPP
  PEND_();
#else
  MPI_Finalize();
#endif    
}

GA::GAServices GA::SERVICES = GA::GAServices();
