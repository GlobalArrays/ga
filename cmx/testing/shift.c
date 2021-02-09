#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "cmx.h"

static int me;
static int nproc;
static int size[] = {2,4,8,16,32,64,128,256,512,1024,0}; /* 0 is sentinal */

#define PUT_FORWARD  0
#define PUT_BACKWARD 1
#define GET_FORWARD  2
#define GET_BACKWARD 3

static void fill_array(double *arr, int count, int which);
static void shift(cmxInt buffer_size, int op);


int main(int argc, char **argv)
{
  int i;

  cmx_init_args(&argc, &argv);
  cmx_group_rank(CMX_GROUP_WORLD, &me);
  cmx_group_size(CMX_GROUP_WORLD, &nproc);

  if (0 == me) {
    printf("msg size (bytes)     avg time (milliseconds)    avg b/w (bytes/sec)\n");
  }

  if (0 == me) {
    printf("shifting put forward\n");
  }
  for (i=0; size[i]!=0; ++i) {
    shift(size[i], PUT_FORWARD);
  }

  if (0 == me) {
    printf("shifting put backward\n");
  }
  for (i=0; size[i]!=0; ++i) {
    shift(size[i], PUT_BACKWARD);
  }

  if (0 == me) {
    printf("shifting get forward\n");
  }
  for (i=0; size[i]!=0; ++i) {
    shift(size[i], GET_FORWARD);
  }

  if (0 == me) {
    printf("shifting get backward\n");
  }
  for (i=0; size[i]!=0; ++i) {
    shift(size[i], GET_BACKWARD);
  }

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


static void shift(cmxInt buffer_size, int op)
{
  cmx_handle_t rem_hdl;
  void *put_buf;
  void *get_buf;
  int i=0;
  double *times;
  double *result;
  double total_time=0;
  MPI_Comm comm = MPI_COMM_NULL;

  cmx_group_comm(CMX_GROUP_WORLD, &comm);
  put_buf = (void*)malloc(buffer_size);
  get_buf = (void*)malloc(buffer_size);
  times = (double*)malloc(nproc * sizeof(double));
  result = (double*)malloc(nproc * sizeof(double));
  cmx_malloc(&rem_hdl, buffer_size, CMX_GROUP_WORLD);

  /* initialize what we're putting */
  fill_array((double*)put_buf, buffer_size/sizeof(double), me);

  /* initialize time keepers */
  (void)memset(times, 0, nproc*sizeof(double));
  (void)memset(result, 0, nproc*sizeof(double));
  times[me] = MPI_Wtime()*1.0e6;

  /* the shift */
  switch (op) {
    case PUT_FORWARD:
      for (i=1; i<nproc; ++i) {
        int dst = (me+i)%nproc;
        cmx_put(put_buf, 0, buffer_size, dst, rem_hdl);
        cmx_barrier(CMX_GROUP_WORLD);
      }
      break;
    case PUT_BACKWARD:
      for (i=1; i<nproc; ++i) {
        int dst = me<i ? me-i+nproc : me-i;
        cmx_put(put_buf, 0, buffer_size, dst, rem_hdl);
        cmx_barrier(CMX_GROUP_WORLD);
      }
      break;
    case GET_FORWARD:
      for (i=1; i<nproc; ++i) {
        int dst = (me+i)%nproc;
        cmx_get(get_buf, 0, buffer_size, dst, rem_hdl);
        cmx_barrier(CMX_GROUP_WORLD);
      }
      break;
    case GET_BACKWARD:
      for (i=1; i<nproc; ++i) {
        int dst = me<i ? me-i+nproc : me-i;
        cmx_get(get_buf, 0, buffer_size, dst, rem_hdl);
        cmx_barrier(CMX_GROUP_WORLD);
      }
      break;
    default:
      cmx_error("oops", 1);
  }

  /* calculate total time and average time */
  times[me] = MPI_Wtime()*1.0e6 - times[me];
  MPI_Allreduce(times, result, nproc, MPI_DOUBLE, MPI_SUM, comm);
  for (i=0; i<nproc; ++i) {
    total_time += times[i];
  }
  if (0 == me) {
    printf("%5zu                %6.2f                   %10.2f\n",
        buffer_size,
        total_time/nproc*1000,
        buffer_size*(nproc-1)/total_time);
  }

  cmx_free(rem_hdl);
  free(put_buf);
  free(get_buf);
  free(times);
}
