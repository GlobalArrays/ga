#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <mpi.h>

#include "cmx.h"

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
static void strided_test(cmxInt buffer_size, int op);

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
    printf("msg size (bytes)  avg time (us)  avg b/w (MB/sec)  xdim    ydim\n");
  }

  if (0 == me) {
    printf("\n\n");
    printf("CMX Put Strided Test\n");
  }
  strided_test(MAX_MESSAGE_SIZE, PUTS);


  if (0 == me) {
    printf("\n\n");
    printf("CMX Get Strided Test\n");
  }
  strided_test(MAX_MESSAGE_SIZE, GETS);


  if (0 == me) {
    printf("\n\n");
    printf("CMX Accumulate Strided Test\n");
  }
  strided_test(MAX_MESSAGE_SIZE, ACCS);


  cmx_finalize();
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


static void strided_test(cmxInt buffer_size, int op)
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
  if (nproc%2 != 0 && me == nproc-1) participate = 0;
  double scale = 1;

  /* Information for strided data transfer */

  int levels = 1;
  cmxInt count[2];
  cmxInt stride[1];

  cmxInt xdim, ydim;
  for (msg_size = 16; msg_size <= buffer_size; msg_size *= 2) {


    int j;
    int iter = msg_size > MEDIUM_MESSAGE_SIZE ? ITER_LARGE : ITER_SMALL;

    for (xdim = 8; xdim <= msg_size; xdim *=2 ) {
      ydim = msg_size / xdim;
      count[0] = xdim;
      count[1] = ydim;
      stride[0] = xdim;

      double t_start, t_end;
      if (me%2 == 0 && participate) {
        for (j= 0; j < iter + WARMUP; ++j) {

          if (WARMUP == j) {
            t_start = dclock();
          }

          switch (op) {
            case PUTS:
              cmx_puts(put_buf, stride, 0, stride, 
                  count, levels, dst, rem_hdl);
              break;
            case GETS:
              cmx_gets(get_buf, stride, 0, stride, 
                  count, levels, dst, rem_hdl);
              break;
            case ACCS:
              cmx_accs(CMX_ACC_DBL, (void *)&scale, 
                  put_buf, stride, 0, stride,
                  count, levels, dst, rem_hdl);
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
        printf("%5zu\t\t%6.2f\t\t%6.2f\t\t%zu\t\t%zu\n",
            msg_size,
            ((t_end  - t_start))/iter,
            msg_size*(nproc-1)*iter/((t_end - t_start)), xdim, ydim);
      }
    }
  }
  cmx_free(rem_hdl);
  free(put_buf);
  free(get_buf);
  free(times);
}
