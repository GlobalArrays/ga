#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if MSG_COMMS_TCGMSG || MSG_COMMS_TCGMSG5 || MSG_COMMS_TCGMSGMPI
#   include "tcgmsg.h"
#elif MSG_COMMS_MPI
#   include "mpi.h"
#endif

#include "ga++.h"

void
GA::Initialize(int argc, char *argv[], size_t limit) {
#if MSG_COMMS_TCGMSG || MSG_COMMS_TCGMSG5 || MSG_COMMS_TCGMSGMPI
  tcg_pbegin(argc, argv);
#elif MSG_COMMS_MPI
#   ifdef DCMF
  int status;
  int desired = MPI_THREAD_MULTIPLE;
  int provided;
  printf("using MPI_Init_thread\n");
  status = MPI_Init_thread(&argc, &argv, desired, &provided);
  if ( provided != MPI_THREAD_MULTIPLE ) {
      printf("provided != MPI_THREAD_MULTIPLE\n");
  } else if ( provided == MPI_THREAD_SERIALIZED ) {
      printf("provided = MPI_THREAD_SERIALIZED\n"); \
  } else if ( provided == MPI_THREAD_FUNNELED ) {
      printf("provided = MPI_THREAD_FUNNELED\n"); \
  } else if ( provided == MPI_THREAD_SINGLE ) {
      printf("provided = MPI_THREAD_SINGLE\n");
  }
#   else
  int val;
  if((val=MPI_Init(&argc, &argv)) < 0) 
    fprintf(stderr, "MPI_Init() failed\n");
#   endif
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
  // Initialize MPI/TCGMSG  
#if MSG_COMMS_TCGMSG || MSG_COMMS_TCGMSG5 || MSG_COMMS_TCGMSGMPI
  tcg_pbegin(argc, argv);
#elif MSG_COMMS_MPI
#   ifdef DCMF
  int status;
  int desired = MPI_THREAD_MULTIPLE;
  int provided;
  printf("using MPI_Init_thread\n");
  status = MPI_Init_thread(&argc, &argv, desired, &provided);
  if ( provided != MPI_THREAD_MULTIPLE ) {
      printf("provided != MPI_THREAD_MULTIPLE\n");
  } else if ( provided == MPI_THREAD_SERIALIZED ) {
      printf("provided = MPI_THREAD_SERIALIZED\n"); \
  } else if ( provided == MPI_THREAD_FUNNELED ) {
      printf("provided = MPI_THREAD_FUNNELED\n"); \
  } else if ( provided == MPI_THREAD_SINGLE ) {
      printf("provided = MPI_THREAD_SINGLE\n");
  }
#   else
  int val;
  if((val=MPI_Init(&argc, &argv)) < 0) 
    fprintf(stderr, "MPI_Init() failed\n");
#   endif
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
  
#if MSG_COMMS_TCGMSG || MSG_COMMS_TCGMSG5 || MSG_COMMS_TCGMSGMPI
  tcg_pend();
#elif MSG_COMMS_MPI
  MPI_Finalize();
#endif    
}
