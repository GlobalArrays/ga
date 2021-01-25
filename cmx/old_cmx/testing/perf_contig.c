#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <mpi.h>

#include "cmx.h"

static int me;
static int nproc;

#define PUT  0
#define GET 1
#define ACC 2

#define MAX_MESSAGE_SIZE 1024*1024
#define MEDIUM_MESSAGE_SIZE 8192
#define ITER_SMALL 1000
#define ITER_LARGE 100

#define WARMUP 20
static void fill_array(double *arr, int count, int which);
static void contig_test(cmxInt buffer_size, int op);

double dclock()
{
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return(tv.tv_sec * 1.0e6 + (double)tv.tv_usec);
}

int main(int argc, char **argv)
{
  cmx_init_args(&argc, &argv);
  cmx_group_rank(CMX_GROUP_WORLD, &me);
  cmx_group_size(CMX_GROUP_WORLD, &nproc);

  if (0 == me) {
    printf("msg size (bytes)     avg time (us)    avg b/w (MB/sec)\n");
    printf("\n\n");
  }

  if (0 == me) {
    printf("CMX Put Test\n");
  }
  contig_test(MAX_MESSAGE_SIZE, PUT);

  if (0 == me) {
    printf("\n\n");
    printf("CMX Get Test\n");
  }
  contig_test(MAX_MESSAGE_SIZE, GET);

  if (0 == me) {
    printf("\n\n");
    printf("CMX Accumulate Test\n");
  }
  contig_test(MAX_MESSAGE_SIZE, ACC);

  cmx_finalize();
  MPI_Finalize();

  return 0;
}


static void fill_array(double *arr, int count, int which)
{
  int i;

  for (i = 0; i < count; i++) {
    arr[i] = i * 8.23 + which * 2.89;
  }
}


static void contig_test(cmxInt buffer_size, int op)
{
  cmx_handle_t rem_hdl;
  void *put_buf;
  void *get_buf;
  double *times;
  int participate = 1;

  put_buf = (void*)malloc(buffer_size);
  get_buf = (void*)malloc(buffer_size);
  times = (double*)malloc(nproc * sizeof(double));
  cmx_malloc(&rem_hdl, buffer_size, CMX_GROUP_WORLD);

  /* initialize what we're putting */
  fill_array((double*)put_buf, buffer_size/sizeof(double), me);

  cmxInt msg_size;

  int dst = me + 1;
  if (nproc%2 != 0 && nproc-1 == me) participate = 0;
  double scale = 1.0;
  for (msg_size = 16; msg_size <= buffer_size; msg_size *= 2) {

    int j;
    int iter = msg_size > MEDIUM_MESSAGE_SIZE ? ITER_LARGE : ITER_SMALL;

    double t_start, t_end;
    if (me%2 == 0 && participate) {
      for (j= 0; j < iter + WARMUP; ++j) {

        if (WARMUP == j) {
          t_start = dclock();
        }

        switch (op) {
          case PUT:
            cmx_put(put_buf, 0, msg_size, dst, rem_hdl);
            break;
          case GET:
            cmx_get(get_buf, 0, msg_size, dst, rem_hdl);
            break;
          case ACC:
            cmx_acc(CMX_ACC_DBL, &scale, 
                put_buf, 0, msg_size, dst, rem_hdl);
            break;
          default:
            cmx_error("oops", 1);
        }

      }
    }
    cmx_barrier(CMX_GROUP_WORLD);
    /* calculate total time and average time */
    t_end = dclock();


    if (0 == me) {
      printf("%8zu\t\t%8.2f\t\t%10.2f\n",
          msg_size,
          ((t_end  - t_start))/iter,
          msg_size*(nproc-1)*iter/((t_end - t_start)));
    }
  }
  cmx_free(rem_hdl);
  free(put_buf);
  free(get_buf);
  free(times);
}
