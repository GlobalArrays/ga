#include "mpi.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define NLEN    (1024 * 1024)
#define MPI_TAG 12383

int main(int argc, char **argv)
{
    int  me, nprocs;
    int *sbuf    = NULL;
    int *gpuBuf  = NULL;
    int *ipcBuf  = NULL;
    void *vbuf   = NULL;
    int  nghbr, sndr;
    int  lowest, my_master, myGPU, ranks_on_node;
    int *masters  = NULL;
    int  ngpu;
    cudaIpcMemHandle_t *handles = NULL;
    cudaIpcMemHandle_t  handle;
    MPI_Request req   = MPI_REQUEST_NULL;
    MPI_Status  status;
    int t_ok, ok, i;

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nprocs);

    char  my_name[MPI_MAX_PROCESSOR_NAME];
    char *all_names = (char*)malloc(MPI_MAX_PROCESSOR_NAME * nprocs);
    int   name_len;

    MPI_Get_processor_name(my_name, &name_len);
    MPI_Allgather(my_name,   MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  all_names, MPI_MAX_PROCESSOR_NAME, MPI_CHAR,
                  MPI_COMM_WORLD);

    lowest        = nprocs;
    ranks_on_node = 0;
    for (i = 0; i < nprocs; i++) {
        if (strcmp(&all_names[i * MPI_MAX_PROCESSOR_NAME], my_name) == 0) {
            ranks_on_node++;
            if (i < lowest) lowest = i;
        }
    }

    my_master = lowest;
    masters   = (int*)malloc(sizeof(int) * nprocs);
    MPI_Allgather(&my_master, 1, MPI_INT,
                  masters,    1, MPI_INT, MPI_COMM_WORLD);

    printf("p[%d] name=%s  lowest=%d  my_master=%d  ranks_on_node=%d\n",
           me, my_name, lowest, my_master, ranks_on_node);
    fflush(stdout);

    cudaGetDeviceCount(&ngpu);
    int gpus_needed = ranks_on_node - 1;
    t_ok = (ngpu >= gpus_needed) ? 1 : 0;
    MPI_Allreduce(&t_ok, &ok, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
    if (!ok) {
        if (me == 0)
            printf("Not enough GPUs: need %d, have %d\n", gpus_needed, ngpu);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    myGPU  = -1;
    gpuBuf = NULL;

    if (me != my_master) {
        myGPU = (me - lowest) - 1;
        cudaSetDevice(myGPU);
        cudaMalloc((void**)&gpuBuf, sizeof(int) * NLEN);
        cudaMemset(gpuBuf, 0, sizeof(int) * NLEN);
        printf("p[%d] worker → GPU %d  gpuBuf=%p\n", me, myGPU, (void*)gpuBuf);
        fflush(stdout);
    } else {
        printf("p[%d] master (no own GPU buffer)\n", me);
        fflush(stdout);
    }

    handles = (cudaIpcMemHandle_t*)malloc(sizeof(cudaIpcMemHandle_t) * nprocs);
    memset(&handle, 0, sizeof(handle));
    if (me != my_master)
        cudaIpcGetMemHandle(&handle, gpuBuf);

    MPI_Allgather(&handle,  sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                  handles,  sizeof(cudaIpcMemHandle_t), MPI_BYTE,
                  MPI_COMM_WORLD);

    sbuf  = (int*)malloc(sizeof(int) * NLEN);
    nghbr = -1;

    if (me != my_master) {
        int workers     = ranks_on_node - 1;
        int my_offset   = (me - lowest) - 1;
        int next_offset = (my_offset + 1) % workers;
        nghbr = lowest + 1 + next_offset;

        if (workers == 1)
            nghbr = (my_master + ranks_on_node) % nprocs;

        for (i = 0; i < NLEN; i++)
            sbuf[i] = nghbr * NLEN + i;
    }

    printf("p[%d] nghbr=%d\n", me, nghbr); fflush(stdout);

    ipcBuf = NULL;

    if (me == my_master && nprocs > ranks_on_node) {
        int prev_master  = (lowest - ranks_on_node + nprocs) % nprocs;
        int prev_workers = ranks_on_node - 1;
        sndr = prev_master + prev_workers;

        int dst_worker = lowest + 1;
        cudaSetDevice(0);
        cudaIpcOpenMemHandle(&vbuf, handles[dst_worker],
                             cudaIpcMemLazyEnablePeerAccess);
        ipcBuf = (int*)vbuf;
        cudaDeviceSynchronize();

        printf("p[%d] (master) opened IPC buf of rank %d → ipcBuf=%p\n",
               me, dst_worker, (void*)ipcBuf);
        printf("p[%d] (master) MPI_Irecv(ipcBuf) from rank %d\n", me, sndr);
        fflush(stdout);

        MPI_Irecv(ipcBuf, NLEN, MPI_INT, sndr, MPI_TAG,
                  MPI_COMM_WORLD, &req);
    }

    if (nghbr != -1) {
        int same_node = (strcmp(&all_names[nghbr * MPI_MAX_PROCESSOR_NAME],
                                my_name) == 0);
        if (same_node) {
            int peer_gpu = (nghbr - lowest) - 1;
            printf("p[%d] IPC → rank %d (GPU %d)\n", me, nghbr, peer_gpu);
            fflush(stdout);

            cudaSetDevice(peer_gpu);
            cudaIpcOpenMemHandle(&vbuf, handles[nghbr],
                                 cudaIpcMemLazyEnablePeerAccess);
            int *peerBuf = (int*)vbuf;
            cudaDeviceSynchronize();
            cudaMemcpy(peerBuf, sbuf, sizeof(int) * NLEN,
                       cudaMemcpyHostToDevice);
            cudaIpcCloseMemHandle(vbuf);
            cudaDeviceSynchronize();

        } else {
            cudaSetDevice(myGPU);
            cudaMemcpy(gpuBuf, sbuf, sizeof(int) * NLEN,
                       cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            printf("p[%d] MPI_Send(gpuBuf) → rank %d (cross-node master)\n",
                   me, masters[nghbr]);
            fflush(stdout);

            MPI_Send(gpuBuf, NLEN, MPI_INT, masters[nghbr], MPI_TAG,
                     MPI_COMM_WORLD);
        }
    }

    printf("p[%d] done sending\n", me); fflush(stdout);

    if (me == my_master && ipcBuf != NULL) {
        MPI_Wait(&req, &status);
        printf("p[%d] (master) MPI_Wait done\n", me); fflush(stdout);
        cudaIpcCloseMemHandle(vbuf);
        cudaDeviceSynchronize();
    }

    MPI_Barrier(MPI_COMM_WORLD);
    printf("p[%d] past barrier\n", me); fflush(stdout);

    t_ok = 1;
    if (me != my_master) {
        int *check = (int*)malloc(sizeof(int) * NLEN);
        cudaSetDevice(myGPU);
        cudaMemcpy(check, gpuBuf, sizeof(int) * NLEN, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        for (i = 0; i < NLEN; i++) {
            if (check[i] != me * NLEN + i) {
                printf("p[%d] MISMATCH at [%d]: got %d  expected %d\n",
                       me, i, check[i], me * NLEN + i);
                t_ok = 0;
                break;
            }
        }
        free(check);
    }

    MPI_Allreduce(&t_ok, &ok, 1, MPI_INT, MPI_PROD, MPI_COMM_WORLD);
    if (me == 0) {
        if (ok) printf("\n*** PASSED ***\n\n");
        else    printf("\n*** FAILED ***\n\n");
        fflush(stdout);
    }

    if (gpuBuf)    cudaFree(gpuBuf);
    if (sbuf)      free(sbuf);
    if (handles)   free(handles);
    if (masters)   free(masters);
    if (all_names) free(all_names);

    MPI_Finalize();
    return ok ? 0 : 1;
}