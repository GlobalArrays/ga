/* Define PBEGIN_C so that global variables in tcgmsgP.h are defined here
   and declared extern everywhere else ... SGI linker is a whiner */
#define PBEGIN_C
#include "tcgmsgP.h"

extern void TrapSigint(void);
extern void TrapSigchld(void);
extern int WaitAll(long);

#if !(defined(KSR) || defined(CRAY))
extern void bzero(char *, int);
#endif
extern pid_t getpid(void), fork(void);

void PBEGIN_(int argc, char **argv)
/*
  Simple shared-memory only version of TCGMSG
*/  
{
  long arg, node, i, max_n_msg;

  TCGMSG_nodeid = 0;
  TCGMSG_nnodes = 1;		/* By default just sequential */

#ifdef CRAY_T3D
  TCGMSG_nnodes = (long)_num_pes(); 
#else
  for (arg=1; arg<(argc-1); arg++)
    if (strcmp(argv[arg],"-p") == 0) {
      TCGMSG_nnodes = atol(argv[arg+1]);
      break;
    }
#endif

  if (TCGMSG_nnodes > MAX_PROC){
     if(NODEID_()){
        sleep(1);
        return;
     }
     fprintf(stderr,"\nTCGMSG has been configured for up to %d processes\n",
                    MAX_PROC);
     fprintf(stderr,"Please change MAX_PROC in `tcgmsgP.h` and recompile\n\n");
     sleep(1);
     Error("aborting ... ",0);
  }
  if (TCGMSG_nnodes == 1) return;

  /* Set up handler for SIGINT and SIGCHLD */

#ifndef CRAY_T3D
  TrapSigint();
  TrapSigchld();
#endif

  /* Allocate the process info structures */

  if (!(TCGMSG_proc_info = (ProcInfo *)
	malloc((size_t) (TCGMSG_nnodes*sizeof(ProcInfo)))))
    Error("pbegin: failed to malloc procinfo",
	  (long) (TCGMSG_nnodes*sizeof(ProcInfo)));
  bzero((char *) TCGMSG_proc_info, (int) (TCGMSG_nnodes*sizeof(ProcInfo)));

  /* Allocate a ring of message q entries to avoid having a malloc/free
     pair for every message sent */

  max_n_msg = 2*TCGMSG_nnodes;
  if (max_n_msg < 64) max_n_msg = 64;

  if (!(TCGMSG_sendq_ring = (SendQEntry *)
	malloc((size_t) (max_n_msg*sizeof(SendQEntry)))))
    Error("pegin: failed to malloc entries for send q", 0L);
  
  for (i=0; i<max_n_msg; i++) {
    TCGMSG_sendq_ring[i].active = 0;
    TCGMSG_sendq_ring[i].next_in_ring = TCGMSG_sendq_ring + ((i+1)%max_n_msg);
  }

  /* Create the shared memory and fill with zeroes */
  
#ifdef CRAY_T3D
  TCGMSG_shmem_size = (long)(TCGMSG_nnodes * sizeof(ShmemBuf));

  TCGMSG_shmem = (char *) TCGMSG_receive_buffer;
#else
  TCGMSG_shmem_size = (long) (TCGMSG_nnodes * TCGMSG_nnodes * sizeof(ShmemBuf));
  
  TCGMSG_shmem = CreateSharedRegion(&TCGMSG_shmem_id, &TCGMSG_shmem_size);
#endif
  
  bzero(TCGMSG_shmem, (int) TCGMSG_shmem_size);
  
  /* Fork the child processes */
  
  TCGMSG_proc_info[0].pid = getpid();
  
#ifdef CRAY_T3D

  TCGMSG_nodeid = (long) _my_pe();     /* get my unique id */

#else

  for (node=1; node<TCGMSG_nnodes; node++) {
    pid_t pid = fork();

    if (pid == 0) {
      TCGMSG_nodeid = node;	/* Generate my unique id */
      break;
    }      
    else {
      TCGMSG_proc_info[node].pid = pid;
    }
  }
#endif

  /* Now everyone initializes the pointers to the shared-
     memory buffers.

     Each process has TCGMSG_nnodes buffers, one for each
     other process that it can receive from via shared
     memory (plus one extra!). */

  /*shmem_set_cache_inv();*/
  for (node=0; node<TCGMSG_nnodes; node++) {
    long me = TCGMSG_nodeid;
    if (me != node) {

#     ifdef CRAY_T3D
         TCGMSG_proc_info[node].sendbuf = ((ShmemBuf *) TCGMSG_shmem) + me;
         TCGMSG_proc_info[node].recvbuf = ((ShmemBuf *) TCGMSG_shmem) + node;
#     else
         TCGMSG_proc_info[node].sendbuf = ((ShmemBuf *) TCGMSG_shmem) +
	   (node*TCGMSG_nnodes + me);
         TCGMSG_proc_info[node].recvbuf = ((ShmemBuf *) TCGMSG_shmem) +
	   (me*TCGMSG_nnodes + node);
#     endif
    }
  }

#ifdef CRAY_T3D
  /* initialize T3D gops/brodcast work array */
  t3d_gops_init();
#endif


  /* At this point communication is possible. 

     Synchronize and continue. */

  {
    Integer type = 1;
    
    SYNCH_(&type);
  }

}

void PEND_(void)
{
  Integer type = 999;

#ifndef CRAY_T3D
  (void) signal(SIGCHLD, SIG_DFL); /* Death of children now OK */
#endif

  SYNCH_(&type);
  
#ifdef SYSV 
  if (TCGMSG_nodeid == 0 && TCGMSG_nnodes > 1) {
    int status;
    status = WaitAll(TCGMSG_nnodes-1);       /* Wait for demise of children */
    DeleteSharedRegion(TCGMSG_shmem_id);
    if (status) exit(1);
  }
#endif
}
