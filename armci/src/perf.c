/* Copyright (c)  1999 Pacific Northwest National Laboratory
 * All rights reserved.
 *
 *    Author: Jialin Ju, PNNL
 */

/***
   NAME
     test.c
   PURPOSE
     compare the performance of MPI2 GET and ARMCI GET
   NOTES

   HISTORY
     jju - Mar 15, 1999: Created.
***/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <mpi.h>
#include "armci.h"
#include "armcip.h"

#define SIZE 550
#define MAXPROC 10

#define CHUNK_NUM 28
int chunk[CHUNK_NUM] = {1,3,4,6,9,12,16,20,24,30,40,48,52,64,78,91,104,
                        128,142,171,210,256,300,353,400,440,476,512};

void fill_array(double *arr, int count, int which);
void check_result(double *src_buf, double *dst_buf,
                 int *stride, int *count, int stride_levels, int proc);

double time_get(double *src_buf, double *dst_buf,
                int chunk, int loop, int proc, int levels)
{
    int i, bal = 0;
    
    int stride[2];
    int count[2];
    int stride_levels = levels;

    double start_time, stop_time, total_time = 0;

    stride[0] = SIZE * sizeof(double);
    count[0] = chunk * sizeof(double); count[1] = chunk;
    
    for(i=0; i<loop; i++) {
        start_time = MPI_Wtime();

        ARMCI_GetS(src_buf, stride, dst_buf, stride,
                   count, stride_levels, proc);

        stop_time = MPI_Wtime();
        total_time += (stop_time - start_time);

        /* test result: only once */
        if(i == 0) {
            double *buf;
            buf = (double *)malloc(SIZE * SIZE * sizeof(double));
            assert(buf != NULL);
            fill_array(buf, SIZE*SIZE, proc);

            ARMCI_PutS(buf, stride, src_buf, stride, count,
                       stride_levels, proc);
            ARMCI_GetS(src_buf, stride, dst_buf, stride, count,
                       stride_levels, proc);
            
            check_result(buf, dst_buf, stride, count, stride_levels, proc);
            free(buf);
        }
        
        /* prepare next src and dst ptrs: avoid cache locality */
        if(bal == 0) {
            src_buf += chunk * (loop - i - 1);
            dst_buf += chunk * (loop - i - 1);
            bal = 1;
        } else {
            src_buf -= chunk * (loop - i - 1);
            dst_buf -= chunk * (loop - i - 1);
            bal = 0;
        }
    }
    
    return(total_time/loop);
}

double time_put(double *src_buf, double *dst_buf,
                int chunk, int loop, int proc, int levels)
{
    int i, bal = 0;

    int stride[2];
    int count[2];
    int stride_levels = levels;

    double start_time, stop_time, total_time = 0;

    stride[0] = SIZE * sizeof(double);
    count[0] = chunk * sizeof(double); count[1] = chunk;

    for(i=0; i<loop; i++) {
        start_time = MPI_Wtime();

        ARMCI_PutS(src_buf, stride, dst_buf, stride,
                   count, stride_levels, proc);

        stop_time = MPI_Wtime();
        total_time += (stop_time - start_time);
        
        /* prepare next src and dst ptrs: avoid cache locality */
        if(bal == 0) {
            src_buf += chunk * (loop - i - 1);
            dst_buf += chunk * (loop - i - 1);
            bal = 1;
        } else {
            src_buf -= chunk * (loop - i - 1);
            dst_buf -= chunk * (loop - i - 1);
            bal = 0;
        }
    }
    
    return(total_time/loop);
}

double time_acc(double *src_buf, double *dst_buf,
                int chunk, int loop, int proc, int levels)
{
    int i, bal = 0;

    int stride[2];
    int count[2];
    int stride_levels = levels;

    double start_time, stop_time, total_time = 0;

    stride[0] = SIZE * sizeof(double);
    count[0] = chunk * sizeof(double); count[1] = chunk;

    for(i=0; i<loop; i++) {
        double scale = (double)i;
        
        start_time = MPI_Wtime();

        ARMCI_AccS(ARMCI_ACC_DBL, &scale, src_buf, stride, dst_buf, stride,
                   count, stride_levels, proc);

        stop_time = MPI_Wtime();
        total_time += (stop_time - start_time);

        /* prepare next src and dst ptrs: avoid cache locality */
        if(bal == 0) {
            src_buf += chunk * (loop - i - 1);
            dst_buf += chunk * (loop - i - 1);
            bal = 1;
        } else {
            src_buf -= chunk * (loop - i - 1);
            dst_buf -= chunk * (loop - i - 1);
            bal = 0;
        }
    }
    
    return(total_time/loop);
}

void test_1D()
{
    int i, j;
    int src, dst;
    int ierr;
    double *buf;
    void *ptr[MAXPROC];

    /* find who I am and the dst process */
    src = armci_me;
    
    /* memory allocation */
    if(armci_me == 0) {
        buf = (double *)malloc(SIZE * SIZE * sizeof(double));
        assert(buf != NULL);
        
        fill_array(buf, SIZE*SIZE, armci_me*10);
    }
    
    ierr = ARMCI_Malloc(ptr, (SIZE * SIZE * sizeof(double)));
    assert(ierr == 0); assert(ptr[armci_me]);

    /* ARMCI - initialize the data window */
    fill_array(ptr[armci_me], SIZE*SIZE, armci_me);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* only the proc 0 doest the work */
    if(armci_me == 0) {
        printf("\n\t\t\tRemote 1-D Array Section\n");
        printf("  section               get                 put");
        printf("                 acc\n");
        printf("bytes   loop       sec      MB/s       sec      MB/s");
        printf("       sec      MB/s\n");
        printf("------- ------  --------  --------  --------  --------");
        printf("  --------  --------\n");
        fflush(stdout);
        
        for(i=0; i<CHUNK_NUM; i++) {
            int loop;
            int bytes = chunk[i] * chunk[i] * sizeof(double);
            
            double t_get = 0, t_put = 0, t_acc = 0;
            double latency_get, latency_put, latency_acc;
            double bandwidth_get, bandwidth_put, bandwidth_acc;
            
            loop = (SIZE * SIZE) / (chunk[i] * chunk[i]);
            loop = (int)sqrt((double)loop);
            
            for(dst=1; dst<armci_nclus; dst++) {
                /* strided get */
                t_get += time_get((double *)(ptr[dst]), (double *)buf,
                                  chunk[i]*chunk[i], loop, dst, 0);
                
                /* strided put */
                t_put += time_put((double *)buf, (double *)(ptr[dst]),
                                  chunk[i]*chunk[i], loop, dst, 0);
                
                /* strided acc */
                t_acc += time_acc((double *)buf, (double *)(ptr[dst]),
                                  chunk[i]*chunk[i], loop, dst, 0);
            }
            
            latency_get = t_get/(armci_nclus - 1);
            latency_put = t_put/(armci_nclus - 1);
            latency_acc = t_acc/(armci_nclus - 1);
            
            bandwidth_get = (bytes * (armci_nclus - 1) * 1e-6)/t_get;
            bandwidth_put = (bytes * (armci_nclus - 1) * 1e-6)/t_put;
            bandwidth_acc = (bytes * (armci_nclus - 1) * 1e-6)/t_acc;

            /* print */
            printf("%d\t%d\t%.2e  %.2e  %.2e  %.2e  %.2e  %.2e\n",
                   bytes, loop, latency_get, bandwidth_get,
                   latency_put, bandwidth_put, latency_acc, bandwidth_acc);
        }
    }
    else sleep(4);
    
    MPI_Barrier(MPI_COMM_WORLD);
    
    /* cleanup */
    ARMCI_Free(ptr[armci_me]);
    if(armci_me == 0) free(buf);
}

void test_2D()
{
    int i, j;
    int src, dst;
    int ierr;
    double *buf;
    void *ptr[MAXPROC];

    /* find who I am and the dst process */
    src = armci_me;
    
    /* memory allocation */
    if(armci_me == 0) {
        buf = (double *)malloc(SIZE * SIZE * sizeof(double));
        assert(buf != NULL);
        
        fill_array(buf, SIZE*SIZE, armci_me*10);
    }
    
    ierr = ARMCI_Malloc(ptr, (SIZE * SIZE * sizeof(double)));
    assert(ierr == 0); assert(ptr[armci_me]);
    
    /* ARMCI - initialize the data window */
    fill_array(ptr[armci_me], SIZE*SIZE, armci_me);

    MPI_Barrier(MPI_COMM_WORLD);
    
    /* only the proc 0 doest the work */
    /* print the title */
    if(armci_me == 0) {
        printf("\n\t\t\tRemote 2-D Array Section\n");
        printf("  section               get                 put");
        printf("                 acc\n");
        printf("bytes   loop       sec      MB/s       sec      MB/s");
        printf("       sec      MB/s\n");
        printf("------- ------  --------  --------  --------  --------");
        printf("  --------  --------\n");
        fflush(stdout);
        
        for(i=0; i<CHUNK_NUM; i++) {
            int loop;
            int bytes = chunk[i] * chunk[i] * sizeof(double);

            double t_get = 0, t_put = 0, t_acc = 0;
            double latency_get, latency_put, latency_acc;
            double bandwidth_get, bandwidth_put, bandwidth_acc;
            
            loop = SIZE / chunk[i];

            for(dst=1; dst<armci_nclus; dst++) {
                /* strided get */
                t_get += time_get((double *)(ptr[dst]), (double *)buf,
                                 chunk[i], loop, dst, 1);
 
                /* strided put */
                t_put += time_put((double *)buf, (double *)(ptr[dst]),
                                 chunk[i], loop, dst, 1);
                
                /* strided acc */
                t_acc += time_acc((double *)buf, (double *)(ptr[dst]),
                                 chunk[i], loop, dst, 1);
            }
            
            latency_get = t_get/(armci_nclus - 1);
            latency_put = t_put/(armci_nclus - 1);
            latency_acc = t_acc/(armci_nclus - 1);
            
            bandwidth_get = (bytes * (armci_nclus - 1) * 1e-6)/t_get;
            bandwidth_put = (bytes * (armci_nclus - 1) * 1e-6)/t_put;
            bandwidth_acc = (bytes * (armci_nclus - 1) * 1e-6)/t_acc;

            /* print */
            if(armci_me == 0)
                printf("%d\t%d\t%.2e  %.2e  %.2e  %.2e  %.2e  %.2e\n",
                       bytes, loop, latency_get, bandwidth_get,
                       latency_put, bandwidth_put, latency_acc, bandwidth_acc);
        }
    }
    else sleep(4);
    
    /* cleanup */
    ARMCI_Free(ptr[armci_me]);
    free(buf);
}

    
main(int argc, char **argv)
{
    int nproc, me;
    
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &me);
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);

    if(nproc < 2) {
        if(me == 0)
            fprintf(stderr,
                    "USAGE: 2 <= processes < %d\n", MAXPROC);
        MPI_Barrier(MPI_COMM_WORLD);
        MPI_Finalize();
        exit(0);
    }
    
    /* initialize ARMCI */
    ARMCI_Init();

    if(nproc != armci_nclus) {
        if(me == 0)
            fprintf(stderr, "USAGE: Please run one process on each node.\n");
        MPI_Barrier(MPI_COMM_WORLD);
        ARMCI_Finalize();
        MPI_Finalize();
        exit(0);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    
    /* test 1 dimension array */
    test_1D();
    
    /* test 2 dimension array */
    test_2D();

    MPI_Barrier(MPI_COMM_WORLD);
    /* done */
    ARMCI_Finalize();
    MPI_Finalize();
}    

void fill_array(double *arr, int count, int which)
{
    int i;

    for(i=0; i<count; i++) arr[i] = i * 8.23 + which * 2.89;
}

void check_result(double *src_buf, double *dst_buf,
                int *stride, int *count, int stride_levels, int proc)
{
    int i, j, size;
    long idx;
    int n1dim;  /* number of 1 dim block */
    int bvalue[MAX_STRIDE_LEVEL], bunit[MAX_STRIDE_LEVEL];

    /* number of n-element of the first dimension */
    n1dim = 1;
    for(i=1; i<=stride_levels; i++)
        n1dim *= count[i];

    /* calculate the destination indices */
    bvalue[0] = 0; bvalue[1] = 0; bunit[0] = 1; bunit[1] = 1;
    for(i=2; i<=stride_levels; i++) {
        bvalue[i] = 0;
        bunit[i] = bunit[i-1] * count[i-1];
    }

    for(i=0; i<n1dim; i++) {
        idx = 0;
        for(j=1; j<=stride_levels; j++) {
            idx += bvalue[j] * stride[j-1];
            if((i+1) % bunit[j] == 0) bvalue[j]++;
            if(bvalue[j] > (count[j]-1)) bvalue[j] = 0;
        }

        size = count[0] / sizeof(double);
        for(j=0; j<size; j++)
            if(((double *)((char *)src_buf+idx))[j] !=
               ((double *)((char *)dst_buf+idx))[j])
                fprintf(stdout, "Error: comparison failed: (%d) (%f : %f)\n",
                        j, ((double *)((char *)src_buf+idx))[j],
                        ((double *)((char *)dst_buf+idx))[j]);
    }
}
