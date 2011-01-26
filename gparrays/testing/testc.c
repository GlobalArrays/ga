/* Test algorithm parameters */

#include "ga.h"
#include "macdecls.h"
#include "mp3.h"

#include <stdlib.h>

#define N_I  4
#define N_J  4

#define Q_I 2
#define Q_J 2

/* get random patch for GP */
void get_range( int ndim, int dims[], int lo[], int hi[])
{
  int dim, nproc, i, itmp;
  nproc = GA_Nodeid();
  /* Mix up values on different processors */
  for (i=0; i<nproc; i++) itmp = rand();

  for(dim=0; dim<ndim; dim++){
    int toss1, toss2;
    toss1 = rand()%dims[dim];
    toss2 = rand()%dims[dim];
    if (toss1<toss2) {
      lo[dim]=toss1;
      hi[dim]=toss2;
    } else {
      hi[dim]=toss1;
      lo[dim]=toss2;
    }
  }
}

void do_work()
{
  int g_p, me, i, ii, j, jj, l, k;
  int m_k_ij, m_l_ij, idx;
  int dims[2], lo[2], hi[2], ndim;
  int nelems, nsize;
  int idim, jdim, subscript[2], size;
  int ld[2], ld_sz[2];
  int *ptr;
  void **buf_ptr;
  void *buf;
  int *buf_size;


  /* Create Global Pointer array */
  dims[0] = N_I;
  dims[1] = N_J;
  ndim = 2;
  me = NGA_Nodeid();

  g_p = GP_Create_handle();
  GP_Set_dimensions(g_p, ndim, dims);
  GP_Allocate(g_p);

  /* Find locally owned elements in Global Pointer array.
     Only these elements can be assigned to data using the
     GP_Assign_local_element routine. */
  GP_Distribution(g_p, me, lo, hi);
  idim = hi[0] - lo[0] + 1;
  jdim = hi[1] - lo[1] + 1;
  for (i=0; i<idim; i++) {
    ii = i + lo[0];
    for (j=0; j<jdim; j++) {
      jj = j + lo[1];
      idx = jj*N_I + ii;
      m_k_ij = ii%Q_I + 1;
      m_l_ij = jj%Q_J + 1;
      /* Allocate local memory for object and assign it values */
      size = sizeof(int)*(m_k_ij*m_l_ij+2);
      ptr = (int*)malloc(size);
      ptr[0] = m_k_ij;
      ptr[1] = m_l_ij;
      for (k=0; k<m_k_ij; k++) {
        for (l=0; l<m_l_ij; l++) {
          ptr[l*m_k_ij+k+2] = l*m_k_ij+k+idx;
        }
      }
      subscript[0] = ii;
      subscript[1] = jj;
      printf("p[%d] size is: %d location is [%d:%d] ptr: %p\n",me,size,
          subscript[0],subscript[1], ptr);
      GP_Assign_local_element(g_p, subscript, (void*)ptr, size);
    }
  }
  
  /* Guarantee data consistency */
  NGA_Sync();
  GP_Debug(g_p);

  /* Generate bounding coordinates to an arbitrary patch in GP array */
  get_range(ndim, dims, lo, hi);
  printf("p[%d] Getting patch [%d:%d] [%d:%d]\n",me,lo[0],hi[0],lo[1],hi[1]);

  /* Find the total amount of data contained in the patch */
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  printf("p[%d] Total size of patch: %d\n",me,nsize);
  GP_Get_size(g_p, lo, hi, &size);
  printf("p[%d] Total size of patch data: %d\n",me,size);

  /* Allocate buffers and retrieve data */
  buf = (void*)malloc(size);
  buf_ptr = (void**)malloc(nsize*sizeof(void*));
  buf_size = (int*) malloc(nsize*sizeof(int*));
  ld[0] = hi[0]-lo[0]+1;
  ld[1] = hi[1]-lo[1]+1;
  ld_sz[0] = hi[0]-lo[0]+1;
  ld_sz[1] = hi[1]-lo[1]+1;
  GA_Set_debug(1);
  GP_Get(g_p, lo, hi, buf, buf_ptr, ld, buf_size, ld_sz, &size);
  GA_Set_debug(0);
  printf("p[%d] Returned from GP_Get size: %d\n",me,size);
  
#if 0
  /* Check contents of buffers to see if data is as expected */
  for (i=lo[0]; i<=hi[0]; i++) {
    ii = i - lo[0];
    for (j=lo[1]; j<=hi[1]; j++) {
      jj = j - lo[1];
      idx = j*N_I + i;
      ptr = (int*)buf_ptr[jj*ld[0]+ii];
      if (ptr[0] != i%Q_I + 1) {
        NGA_Error("Element dimension i does not match",ptr[0]);
      }
      if (ptr[1] != j%Q_J + 1) {
        NGA_Error("Element dimension j does not match",ptr[1]);
      }
      m_k_ij = i%Q_I + 1;
      m_l_ij = j%Q_J + 1;
      for (k=0; k<ptr[0]; k++) {
        for (l=0; l<ptr[1]; l++) {
          if (ptr[l*ptr[0]+k+2] != l*m_k_ij+l+idx) {
            NGA_Error("Element ij does not match",ptr[l*ptr[0]+k+2]);
          }
        }
      }
    }
  }

  /* Clean up buffers and Global Pointer array */
  free(buf);
  free(buf_ptr);
  free(buf_size);
  GP_Distribution(g_p, me, lo, hi);
  for (i=lo[0]; i<hi[0]; i++) {
    subscript[0] = i;
    for (j=lo[1]; j<hi[1]; j++) {
      subscript[1] = j;
      GP_Free_local_element(g_p, subscript);
    }
  }

  /* destroy Global Pointer array */
  GP_Destroy(g_p);
#endif
}

int main(int argc, char **argv)
{
  int heap=20000, stack=20000;
  int me, nproc;

  MP_INIT(argc, argv);

  NGA_Initialize();
  GP_Initialize();
  me = NGA_Nodeid();
  nproc = NGA_Nnodes();

  if (me==0) {
    if (GA_Uses_fapi()) NGA_Error("Program runs with C array API only",0);
    printf("Using %ld processes\n", (long)nproc);
    fflush(stdout);
  }

  heap /= nproc;
  stack /= nproc;

  if (!MA_init(MT_F_DBL, stack, heap))
    NGA_Error("MA_init failed",stack+heap);

  do_work();

  if (me==0) printf("Terminating ..\n");
  GP_Terminate();
  NGA_Terminate();

  MP_FINALIZE();

  return 0;
}
