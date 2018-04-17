#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <mpi.h>

#include "armci.h"
#include "message.h"

static int me;
static int nproc;

#define PUTS  0
#define GETS  1
#define ACCS  2

#define MAX_MESSAGE_SIZE 1024*1024
#define MEDIUM_MESSAGE_SIZE 8192
#define ITER_SMALL 100
#define ITER_LARGE 10

#define WARMUP 2
static void fill_array(double *arr, int count, int which);
static void strided_test(size_t buffer_size, int op);

double dclock()
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return(tv.tv_sec * 1.0e6 + (double)tv.tv_usec);
}

int main(int argc, char **argv)
{
    armci_msg_init(&argc,&argv);
    ARMCI_Init_args(&argc, &argv);
    me = armci_msg_me();
    nproc = armci_msg_nproc();

    /* This test only works for two processes */

    assert(nproc == 2);

    if (0 == me) {
        printf("\n\n");
        printf("msg size (bytes)     avg time (us)    avg b/w (MB/sec)\n");
    }

    if (0 == me) {
        printf("\n\n");
        printf("#PNNL armci Put Strided Test\n");
    }
    strided_test(MAX_MESSAGE_SIZE, PUTS);


    if (0 == me) {
        printf("\n\n");
        printf("#PNNL armci Get Strided Test\n");
    }
    strided_test(MAX_MESSAGE_SIZE, GETS);
    
   
    if (0 == me) {
        printf("\n\n");
        printf("#PNNL armci Accumulate Strided Test\n");
    }
    strided_test(MAX_MESSAGE_SIZE, ACCS);
    
    
    ARMCI_Finalize();
    armci_msg_finalize();

    return 0;
}


static void fill_array(double *arr, int count, int which)
{
    int i;

    for (i = 0; i < count; i++) {
        arr[i] = i;
    }
}


static void strided_test(size_t buffer_size, int op)
{
    void **dst_ptr;
    void **put_buf;
    void **get_buf;
    double *times;
    size_t msg_size;
    int dst = 1;
    double scale = 1;
    /* Information for strided data transfer */
    int levels = 1;
    int count[2];
    int stride[1];
    size_t xdim, ydim;

    dst_ptr = (void*)malloc(nproc * sizeof(void*));
    put_buf = (void*)malloc(nproc * sizeof(void*));
    get_buf = (void*)malloc(nproc * sizeof(void*));
    times = (double*)malloc(nproc * sizeof(double));
    ARMCI_Malloc(dst_ptr, buffer_size);
    ARMCI_Malloc(put_buf, buffer_size);
    ARMCI_Malloc(get_buf, buffer_size);

    /* initialize what we're putting */
    fill_array((double*)put_buf[me], buffer_size/sizeof(double), me);

    for (msg_size = 16; msg_size <= buffer_size; msg_size *= 2) {


        int j;
        int iter = msg_size > MEDIUM_MESSAGE_SIZE ? ITER_LARGE : ITER_SMALL;

        for (xdim = 8; xdim <= msg_size; xdim *=2 ) {
            double t_start, t_end;
            ydim = msg_size / xdim;
            count[0] = xdim;
            count[1] = ydim;
            stride[0] = xdim;

            if (0 == me) {
                for (j= 0; j < iter + WARMUP; ++j) {

                    if (WARMUP == j) {
                        t_start = dclock();
                    }

                    switch (op) {
                        case PUTS:
                            ARMCI_PutS(put_buf[me], stride, dst_ptr[dst], stride, 
                                    count, levels, dst);
                            break;
                        case GETS:
                            ARMCI_GetS(dst_ptr[dst], stride, get_buf[me], stride, 
                                    count, levels, dst);
                            break;
                        case ACCS:
                            ARMCI_AccS(ARMCI_ACC_DBL, (void *)&scale, 
                                    put_buf[me], stride, dst_ptr[dst], stride,
                                    count, levels, dst);
                            break;
                        default:
                            ARMCI_Error("oops", 1);
                    }

                }
            }
            ARMCI_Barrier();
            /* calculate total time and average time */
            t_end = dclock();


            if (0 == me) {
                printf("%8lu\t\t%6.2f\t\t%6.2f\t\t%lu\t\t%lu\n",
                        (unsigned long)msg_size,
                        ((t_end  - t_start))/iter,
                        msg_size*(nproc-1)*iter/((t_end - t_start)),
                        (unsigned long)xdim,
                        (unsigned long)ydim);
            }
        }
    }
    ARMCI_Free(dst_ptr[me]);
    ARMCI_Free(put_buf[me]);
    ARMCI_Free(get_buf[me]);
    free(dst_ptr);
    free(put_buf);
    free(get_buf);
    free(times);
}
