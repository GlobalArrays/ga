/*$id$*/
#include <stdio.h>
#include <stdlib.h>
#include "armci.h"
#include <mpi.h>
int me,nprocs;
int LOOP=10;
int main(int argc, char **argv)
{
int i;
double **myptrs;
double t0,t1,tget=0,tnbget=0,tput=0,tnbput=0,tnbwait=0,t2=0;
    MPI_Init(&argc,&argv);
    MPI_Comm_rank(MPI_COMM_WORLD,&me);
    MPI_Comm_size(MPI_COMM_WORLD,&nprocs);
    myptrs = (double **)malloc(sizeof(double *)*nprocs);
    ARMCI_Init();
    ARMCI_Malloc((void **)myptrs,LOOP*sizeof(double)); 
    MPI_Barrier(MPI_COMM_WORLD);
    if(me==0){
       for(i=0;i<10;i++){
         ARMCI_Get(myptrs[me]+i,myptrs[me+1]+i,sizeof(double),me+1);
       }
       t0 = MPI_Wtime(); 
       for(i=0;i<LOOP;i++){
         ARMCI_Get(myptrs[me]+i,myptrs[me+1]+i,sizeof(double),me+1);
       }
       t1 = MPI_Wtime(); 
       printf("\nGet Latency=%lf\n",1e6*(t1-t0)/LOOP);fflush(stdout);
       t1=t0=0;
       for(i=0;i<LOOP;i++){
         armci_hdl_t nbh;
         ARMCI_INIT_HANDLE(&nbh);
         t0 = MPI_Wtime(); 
         ARMCI_NbGet(myptrs[me]+i,myptrs[me+1]+i,sizeof(double),me+1,&nbh);
         t1 = MPI_Wtime(); 
         ARMCI_Wait(&nbh);
         t2 = MPI_Wtime();
         tnbget+=(t1-t0);
         tnbwait+=(t2-t1);
       }
       printf("\nNb Get Latency=%lf Nb Wait=%lf\n",1e6*tnbget/LOOP,1e6*tnbwait/LOOP);fflush(stdout);
    }
    else
      sleep(1);
    MPI_Barrier(MPI_COMM_WORLD);
    ARMCI_Finalize();
    MPI_Finalize();
    
}
