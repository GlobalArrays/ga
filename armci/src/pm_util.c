#include <mpi.h> 
#include <assert.h>
#include "armcip.h"

void pm_abort(int code)
{   
    MPI_Abort(MPI_COMM_WORLD, code);
    assert(0);
}               


void check_server_context()
{
    if (SERVER_CONTEXT) {
        printf("Server Context should not make MPI Calls\n");
        pm_abort(1);
    }
}

void pm_init(int *argc, char *(*argv[])) 
{
    
  //MPI_Init(argc, argv);
}

void pm_finalize() 
{
  //MPI_Finalize();
}

double pm_time(void) 
{
   return MPI_Wtime(); 
}

/**
 * @param src src data ptr
 * @param sbytes/rbytes @bytes to be sent/recvd per destination proc
 */
void pm_alltoall(void *src, int sbytes, void *dst, int rbytes) 
{
    check_server_context();
    assert(sbytes == rbytes);
    MPI_Alltoall(src,sbytes,MPI_CHAR,dst,sbytes,MPI_CHAR, MPI_COMM_WORLD);
}

void pm_barrier(void) 
{
   
    check_server_context();
    MPI_Barrier(MPI_COMM_WORLD);
}

int pm_rank() 
{
    int ret;

    static int counter = 0;
    if (counter == 0 || !SERVER_CONTEXT) {
        counter++;
        MPI_Comm_rank(MPI_COMM_WORLD, &ret);
    } else
        ret = armci_me;
    return ret;
}

int pm_nproc() 
{
    int ret;
    static int counter = 0;
    if (counter == 0 || !SERVER_CONTEXT) {
        counter++;
        MPI_Comm_size(MPI_COMM_WORLD, &ret);
    } else
        ret = armci_nproc;
    return ret;
}

int PM_UNDEFINED() 
{
    check_server_context();
    assert(0);
    return 0;
}

int PM_ERR_GROUP() 
{ 
    check_server_context();
    assert(0);
    return 0;
}

int PM_SUCCESS() 
{ 
    check_server_context();
    assert(0);
    return 0;
}

void pm_bcast(void *buffer, int len, int root) 
{
    check_server_context();
    MPI_Bcast(buffer, len, MPI_CHAR, root, MPI_COMM_WORLD);
}

void pm_send(void *buffer, int len, int to, int tag) 
{
    check_server_context();
    MPI_Send(buffer, len, MPI_CHAR, to, tag, MPI_COMM_WORLD);
}

int pm_recv(void *buffer, int buflen, int *from, int tag) 
{
    int msglen;
    MPI_Status status;
    int proc = (*from == -1)? MPI_ANY_SOURCE: *from;
    MPI_Recv(buffer, buflen, MPI_CHAR, proc, tag, MPI_COMM_WORLD, &status);
    MPI_Get_count(&status, MPI_CHAR, &msglen);
    if(*from == -1) 
        *from = (int)status.MPI_SOURCE;
    return msglen;
}


