#include <mpi.h>
#include "tcgmsgP.h"

#ifndef CRAY_YMP
#define USE_MPI_ABORT   
#endif

char      tcgmsg_err_string[ERR_STR_LEN];
MPI_Comm  TCGMSG_Comm;
int       _tcg_initialized=0;
long       DEBUG_;
int       SR_parallel; 
int       SR_single_cluster =1;

static int SR_initialized=0;
long TCGREADY_()
{
     return (long)SR_initialized;
}

/*\ number of processes
\*/
long FATR NNODES_()
{
int numprocs;

   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#  ifdef NXTVAL_SERVER
     if(SR_parallel) return((long)numprocs-1);
#  endif
   return((long)numprocs);
}


/*\ Get calling process id
\*/
long FATR NODEID_()
{
int myid;

    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    return((long)myid);
}

void Error(string, code)
     char *string;
     long  code;
{
    fprintf(stdout,"%3ld: %s %ld (%#lx).\n", (long)NODEID_(), string,
           (long)code,(long)code);
    fflush(stdout);
    fprintf(stderr,"%3ld: %s %ld (%#lx).\n", (long)NODEID_(), string,
           (long)code,(long)code);

    finalize_nxtval(); /* clean nxtval resources */
#ifdef USE_MPI_ABORT
    MPI_Abort(MPI_COMM_WORLD,(int)code);
#else
    exit(1);
#endif
}



void make_tcgmsg_comm()
{
/*  this is based on the MPI Forum decision that MPI_COMM_WORLD 
 *  is a C constant 
 */
extern int single_cluster();

# ifdef NXTVAL_SERVER
    if( SR_parallel ){   
          /* data server for a single process */
          int server;
          MPI_Group MPI_GROUP_WORLD, tcgmsg_grp;

          MPI_Comm_size(MPI_COMM_WORLD, &server);
          server --; /* the highest numbered process will be excluded */
          MPI_Comm_group(MPI_COMM_WORLD, &MPI_GROUP_WORLD);
          MPI_Group_excl(MPI_GROUP_WORLD, 1, &server, &tcgmsg_grp); 
          MPI_Comm_create(MPI_COMM_WORLD, tcgmsg_grp, &TCGMSG_Comm); 
      }else
#   endif
          TCGMSG_Comm = MPI_COMM_WORLD; 

#if !defined(NXTVAL_SERVER) && !defined(ARMCI)
#ifdef SGI
       SR_single_cluster = single_cluster();
       if(!SR_single_cluster)
         Error("native nxtval not supported multiple hosts",0); 
#endif
#endif

}
        

#ifdef CRAY_YMP
#define BROKEN_MPI_INITIALIZED
#endif

/*\ Alternative initialization for C programs
 *  used to address argv/argc manipulation in MPI
\*/
void ALT_PBEGIN_(int *argc, char **argv[])
{
int numprocs, myid;
int init=0;

   if(SR_initialized)Error("TCGMSG initialized already???",-1);
   else SR_initialized=1;

   /* check if another library initialized MPI already */
   MPI_Initialized(&init);

#ifdef BROKEN_MPI_INITIALIZED
   /* we really do not have any choice but call MPI_Init possibly again */
   if(init) init = 0;
#endif

   if(!init){ 
      /* nope */
      MPI_Init(argc, argv);
      MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
   }

   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   SR_parallel = numprocs > 1 ? 1 : 0;

   make_tcgmsg_comm();
   MPI_Barrier(MPI_COMM_WORLD);
   /* printf("%d:ready to go\n",NODEID_()); */
   install_nxtval();
}

/*\ Initialization for C programs
\*/
void PBEGIN_(int argc, char* argv[])
{
   ALT_PBEGIN_(&argc, &argv);
}



/*\ shut down message-passing library
\*/ 
void FATR PEND_()
{
#   ifdef NXTVAL_SERVER
       long zero=0;
       if( SR_parallel )  (void) NXTVAL_(&zero);
       MPI_Barrier(MPI_COMM_WORLD);
#   endif
    finalize_nxtval();
    MPI_Finalize();
    exit(0);
}



double FATR TCGTIME_()
{
  static int first_call = 1;
  static double first_time, last_time, cur_time;
  double diff;

  if (first_call) {
    first_time = MPI_Wtime();
    first_call = 0;
    last_time  = -1e-9; 
  }

  cur_time = MPI_Wtime();
  diff = cur_time - first_time;

  /* address crappy MPI_Wtime: consectutive calls must be at least 1ns apart  */
  if(diff - last_time < 1e-9) diff +=1e-9;
  last_time = diff;

  return diff;                  /* Add logic here for clock wrap */
}



long FATR MTIME_()
{
  return (long) (TCGTIME_()*100.0); /* time in centiseconds */
}



/*\ longerface from Fortran to C error routine
\*/
void FATR PARERR_(code)
   long *code;
{
  Error("User detected error in FORTRAN", *code);
}


void FATR SETDBG_(onoff)
     long *onoff;
{
     DEBUG_ = *onoff;
}

void FATR STATS_()
{
  printf("STATS not implemented\n");
} 
