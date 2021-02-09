/* Test Rmw Performance
 * The number of processes are increases from 2 to the number of 
 * processes present in the job */

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>

#include <mpi.h>

#include "cmx.h"

static int me;
static int nproc;

#define FETCH_AND_ADD  0
#define FETCH_AND_ADD_LONG  1
#define SWAP 2
#define SWAP_LONG 3

#define MAX_MESSAGE_SIZE 1024
#define MEDIUM_MESSAGE_SIZE 8192
#define ITER_SMALL 10000
#define ITER_LARGE 10000

#define WARMUP 20
static void fill_array(double *arr, int count, int which);
static void rmw_test(size_t buffer_size, int op);

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

  /* This test only works for two processes */

  if (0 == me) {
    printf("#Processes     avg time (us)\n");
    printf("\n\n");
  }

  if (0 == me) {
    printf("CMX Rmw-Fetch and Add Long Test\n");
    printf("\n\n");
  }
  rmw_test(MAX_MESSAGE_SIZE, FETCH_AND_ADD_LONG);

  if (0 == me)
    printf("\n\n");


  if (0 == me) {
    printf("CMX Rmw-Fetch and Add Test\n");
  }
  rmw_test(MAX_MESSAGE_SIZE, FETCH_AND_ADD);
  if (0 == me)
    printf("\n\n");


  if (0 == me) {
    printf("CMX Rmw-Swap Long Test\n");
  }
  rmw_test(MAX_MESSAGE_SIZE, SWAP_LONG);

  if (0 == me)
    printf("\n\n");


  if (0 == me) {
    printf("CMX Rmw-Swap Test\n");
  }
  rmw_test(MAX_MESSAGE_SIZE, SWAP);

  if (0 == me)
    printf("\n\n");
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


static void rmw_test(size_t buffer_size, int op)
{
  void *put_buf;
  cmx_handle_t dst_hdl;
  double *times;
  int dst = 0;
  double t_start, t_end;
  int j;
  int iter = ITER_LARGE;
  int part_proc;


  times = (double*)malloc(nproc * sizeof(double));
  put_buf = malloc(buffer_size);
  cmx_malloc(&dst_hdl, (cmxInt)buffer_size, CMX_GROUP_WORLD);

  /* initialize what we're putting */
  fill_array((double*)put_buf, buffer_size/sizeof(double), me);

  /* All processes perform Rmw on process 0*/
  for (part_proc = 2; part_proc <= nproc; part_proc *= 2) {
    if (me < part_proc) {
      for (j= 0; j < iter + WARMUP; ++j) {

        if (WARMUP == j) {
          t_start = dclock();
        }

        switch (op) {
          case FETCH_AND_ADD:
            cmx_rmw(CMX_FETCH_AND_ADD,
                put_buf, 0, 1, dst, dst_hdl);
            break;
          case FETCH_AND_ADD_LONG:
            cmx_rmw(CMX_FETCH_AND_ADD_LONG,
                put_buf, 0, 1, dst, dst_hdl);
            break;
          case SWAP:
            cmx_rmw(CMX_SWAP,
                put_buf, 0, 1, dst, dst_hdl);
            break;
          case SWAP_LONG:
            cmx_rmw(CMX_SWAP_LONG,
                put_buf, 0, 1, dst, dst_hdl);
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
      printf("%5d\t\t%6.2f\n",
          part_proc,
          ((t_end  - t_start))/iter);
    }
  }

  cmx_free(dst_hdl);
  free(times);
  free(put_buf);
}
