#include "mpi.h"
#include "tcgmsgP.h"

char      tcgmsg_err_string[ERR_STR_LEN];
MPI_Comm  TCGMSG_Comm;
Int       DEBUG_;
int       SR_parallel; 



/*\ number of processes
\*/
Int NNODES_()
{
int numprocs;

   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
#  ifdef NXTVAL_SERVER
     if(SR_parallel) return((Int)numprocs-1);
#  endif
   return((Int)numprocs);
}


/*\ Get calling process id
\*/
Int NODEID_()
{
int myid;

    MPI_Comm_rank(MPI_COMM_WORLD,&myid);
    return((Int)myid);
}

void Error(string, code)
     char *string;
     Int  code;
{
    fprintf(stdout,"%3ld: %s %ld (%#lx).\n", (long)NODEID_(), string,
           (long)code,(long)code);
    fflush(stdout);
    fprintf(stderr,"%3ld: %s %ld (%#lx).\n", (long)NODEID_(), string,
           (long)code,(long)code);
    MPI_Abort(MPI_COMM_WORLD,(int)code);
}



void make_tcgmsg_comm()
{
/*  this is based on the MPI Forum decision that MPI_COMM_WORLD 
 *  is a C constant 
 */
#   ifdef NXTVAL_SERVER
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
}
        


/*\ Initialization for C programs
\*/
void PBEGIN_(argc, argv)
int argc;
char *argv[];
{
int numprocs, myid;


   MPI_Init(&argc, &argv);

   MPI_Errhandler_set(MPI_COMM_WORLD, MPI_ERRORS_RETURN);
   MPI_Comm_size(MPI_COMM_WORLD, &numprocs);
   MPI_Comm_rank(MPI_COMM_WORLD, &myid);
   SR_parallel = numprocs > 1 ? 1 : 0;

   make_tcgmsg_comm();
   MPI_Barrier(MPI_COMM_WORLD);
/*   printf("ready to go\n");*/
   install_nxtval();
}



/*\ shut down message-passing library
\*/ 
void PEND_()
{
#   ifdef NXTVAL_SERVER
       Int zero=0;
       if( SR_parallel )  (void) NXTVAL_(&zero);
       MPI_Barrier(MPI_COMM_WORLD);
#   endif
    MPI_Finalize();
    exit(0);
}



Double TCGTIME_()
{
  static int first_call = 1;
  static double first_time;
  double diff;

  if (first_call) {
    first_time = MPI_Wtime();
    first_call = 0;
  }

  diff = MPI_Wtime() - first_time;

  return (Double)diff;                  /* Add logic here for clock wrap */
}



Int  MTIME_()
{
  return (Int) (TCGTIME_()*100.0); /* time in centiseconds */
}



/*\ Interface from Fortran to C error routine
\*/
void PARERR_(code)
   Int *code;
{
  Error("User detected error in FORTRAN", *code);
}


void SETDBG_(onoff)
     Int *onoff;
{
     DEBUG_ = *onoff;
}

void STATS_()
{
  printf("STATS not implemented\n");
} 
