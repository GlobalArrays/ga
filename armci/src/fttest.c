#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#   include <mpi.h>
#   define MP_BARRIER()      MPI_Barrier(MPI_COMM_WORLD)
#   define MP_FINALIZE()     MPI_Finalize()
#   define MP_INIT(arc,argv) MPI_Init(&(argc),&(argv))
#   define MP_MYID(pid)      MPI_Comm_rank(MPI_COMM_WORLD, (pid))
#   define MP_PROCS(pproc)   MPI_Comm_size(MPI_COMM_WORLD, (pproc));

#include "armci.h"
int me,nproc;
double **ptr_arr;
long size;
void do_work(int sz)
{
    int i;
    static int d=1;
    for(i=0;i<sz;i++){
       *((double *)(ptr_arr[me])+i)=i+1.12*d++;
    }
}
int main(int argc, char* argv[])
{
    int ndim;
    int rc,i,rid;
    armci_ckpt_ds_t ckptds;
    ARMCI_Group *grp;
    MP_INIT(argc, argv);
    MP_PROCS(&nproc);
    MP_MYID(&me);

    if(me==0){
       printf("ARMCI test program (%d processes)\n",nproc); 
       fflush(stdout);
       sleep(1);
    }
    
    ARMCI_Init();
    grp = ARMCI_Get_world_group();
    size = 131072;
    rc=ARMCI_Malloc((void **)ptr_arr,size*8);
    (void)ARMCI_Ckpt_create_ds(&ckptds,1); 
    /*ckptds.ptr_arr[0]=&me;
    ckptds.ptr_arr[1]=&nproc;
    ckptds.ptr_arr[2]=&size;*/
    ckptds.ptr_arr[0]=ptr_arr[me];
    /*ckptds.sz[0]=sizeof(int);
    ckptds.sz[1]=sizeof(int);
    ckptds.sz[2]=sizeof(long);*/
    for(size=1024;size<2048;size*=2){
       ckptds.sz[0]=size*8;
       rid=ARMCI_Ckpt_init(NULL,grp,0,0,&ckptds);
       for(i=0;i<5;i++){
         rc = ARMCI_Ckpt(rid);
         do_work(size);
       }
       MP_BARRIER();
       ARMCI_Ckpt_finalize(rid);
       printf("\n%d:done for size %d",me,size);fflush(stdout);
    }

    MP_BARRIER();
    ARMCI_Finalize();
    MP_FINALIZE();
    return(0);
}
