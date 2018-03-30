#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <mpi.h>

#include "comex.h"
#include "stats.h"

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
    comex_init_args(&argc, &argv);
    comex_group_rank(COMEX_GROUP_WORLD, &me);
    comex_group_size(COMEX_GROUP_WORLD, &nproc);

    /* This test only works for two processes */

    assert(nproc == 2);

    if (0 == me) {
        printf("msg size (bytes)     avg time (us)    avg b/w (MB/sec)\n");
    }

    if (0 == me) {
        printf("\n\n");
        printf("#PNNL ComEx Put Strided Test\n");
    }
    strided_test(MAX_MESSAGE_SIZE, PUTS);

    if (0 == me) {
        printf("\n\n");
        printf("#PNNL ComEx Get Strided Test\n");
    }
    strided_test(MAX_MESSAGE_SIZE, GETS);
   
    if (0 == me) {
        printf("\n\n");
        printf("#PNNL ComEx Accumulate Strided Test\n");
    }
    strided_test(MAX_MESSAGE_SIZE, ACCS);
    
    comex_finalize();
    MPI_Finalize();

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
    stats_t stats_latency;
    stats_t stats_bandwidth;

    stats_clear(&stats_latency);
    stats_clear(&stats_bandwidth);
    dst_ptr = (void*)malloc(nproc * sizeof(void*));
    put_buf = (void*)malloc(nproc * sizeof(void*));
    get_buf = (void*)malloc(nproc * sizeof(void*));
    times = (double*)malloc(nproc * sizeof(double));
    comex_malloc(dst_ptr, buffer_size, COMEX_GROUP_WORLD);
    comex_malloc(put_buf, buffer_size, COMEX_GROUP_WORLD);
    comex_malloc(get_buf, buffer_size, COMEX_GROUP_WORLD);

    /* initialize what we're putting */
    fill_array((double*)put_buf[me], buffer_size/sizeof(double), me);

    size_t msg_size;

    int dst = 1;
    double scale = 1;

    /* Information for strided data transfer */

    int levels = 1;
    int count[2];
    int stride[1];

    size_t xdim, ydim;
    for (msg_size = 16; msg_size <= buffer_size; msg_size *= 2) {


        int j;
        int iter = msg_size > MEDIUM_MESSAGE_SIZE ? ITER_LARGE : ITER_SMALL;

        for (xdim = 8; xdim <= msg_size; xdim *=2 ) {
            ydim = msg_size / xdim;
            count[0] = xdim;
            count[1] = ydim;
            stride[0] = xdim;

            double t_start, t_end;
            if (0 == me) {
                for (j= 0; j < iter + WARMUP; ++j) {

                    if (WARMUP == j) {
                        t_start = dclock();
                    }

                    switch (op) {
                        case PUTS:
                            comex_puts(put_buf[me], stride, dst_ptr[dst], stride, 
                                    count, levels, dst, COMEX_GROUP_WORLD);
                            break;
                        case GETS:
                            comex_gets(dst_ptr[dst], stride, get_buf[me], stride, 
                                    count, levels, dst, COMEX_GROUP_WORLD);
                            break;
                        case ACCS:
                            comex_accs(COMEX_ACC_DBL, (void *)&scale, 
                                    put_buf[me], stride, dst_ptr[dst], stride,
                                    count, levels, dst, COMEX_GROUP_WORLD);
                            break;
                        default:
                            comex_error("oops", 1);
                    }

                }
            }
            comex_barrier(COMEX_GROUP_WORLD);
            /* calculate total time and average time */
            t_end = dclock();


            if (0 == me) {
                double latency = (t_end-t_start)/iter;
                double bandwidth = msg_size*(nproc-1)*iter/(t_end-t_start);
                printf("%5zu\t\t%6.2f\t\t%6.2f\t\t%zu\t\t%zu\n",
                        msg_size, latency, bandwidth, xdim, ydim);
                stats_sample_value(&stats_latency, latency);
                stats_sample_value(&stats_bandwidth, bandwidth);
            }
        }
    }
    if (0 == me) {
        printf("Latency avg %6.2f +- %6.2f\n",
                stats_latency._mean, stats_stddev(&stats_latency));
        printf("Latency min %6.2f\n", stats_latency._min);
        printf("Latency max %6.2f\n", stats_latency._max);
        printf("Bandwidth avg %6.2f +- %6.2f\n",
                stats_bandwidth._mean, stats_stddev(&stats_bandwidth));
        printf("Bandwidth min %6.2f\n", stats_bandwidth._min);
        printf("Bandwidth max %6.2f\n", stats_bandwidth._max);
    }
    comex_free(dst_ptr[me], COMEX_GROUP_WORLD);
    comex_free(put_buf[me], COMEX_GROUP_WORLD);
    comex_free(get_buf[me], COMEX_GROUP_WORLD);
    free(dst_ptr);
    free(put_buf);
    free(get_buf);
    free(times);
}
