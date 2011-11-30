/* Test algorithm parameters */
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "ga.h"
#include "gp.h"
#include "macdecls.h"
#include "mp3.h"

#include <stdlib.h>

/*
#define N_I  4
#define N_J  4
*/
#define N_I  32
#define N_J  32
/*
#define Q_I 2
#define Q_J 2
*/
#define Q_I 8
#define Q_J 8

/* get random patch for GP */
void get_range( int ndim, int dims[], int lo[], int hi[], int g_p)
{
  int dim, nproc, i, itmp;
  nproc = NGA_Nodeid();
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
  /*
  nproc = (nproc+8)%NGA_Nnodes();
  GP_Distribution(g_p, nproc, lo, hi);
  */
}

void do_work()
{
  int g_p, me, i, ii, j, jj, l, k;
  int m_k_ij, m_l_ij, idx;
  int dims[2], lo[2], hi[2], ndim;
  int lo_t[2], hi_t[2];
  int nelems, nsize;
  int idim, jdim, subscript[2], size;
  int ld[2], ld_sz[2];
  int *ptr;
  void **buf_ptr;
  void *buf;
  int *buf_size;
  void *elem_buf;


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
  /*bjp
  printf("p[%d] GP_Dist lo[0]: %d hi[0]: %d lo[1]: %d hi[1]: %d\n",me,lo[0],hi[0],lo[1],hi[1]);
  */
  idim = hi[0] - lo[0] + 1;
  jdim = hi[1] - lo[1] + 1;
  for (ii=0; ii<idim; ii++) {
    i = ii + lo[0];
    for (jj=0; jj<jdim; jj++) {
      j = jj + lo[1];
      /*bjp
      printf("p[%d] jj: %d\n",me,jj);
      */
      idx = j*N_I + i;
      m_k_ij = i%Q_I + 1;
      m_l_ij = j%Q_J + 1;
      /*
      m_k_ij = 2;
      m_l_ij = 2;
      */
      /* Allocate local memory for object and assign it values */
      size = sizeof(int)*(m_k_ij*m_l_ij+2);
      /*bjp
      printf("p[%d] allocating data of size: %d (ptr=%p)\n",GA_Nodeid(),size,ptr);
      */
      /*bjp
      printf("p[%d] Original i: %d j: %d idx: %d\n",me,i,j,idx);
      */
      ptr = (int*)GP_Malloc(size);
      /*bjp
      printf("p[%d] src_ptr: %p\n",me,ptr);
      */
      /*bjp
      printf("p[%d] finished allocating data of size: %d (ptr=%p)\n",GA_Nodeid(),size, ptr);
      */
      ptr[0] = m_k_ij;
      ptr[1] = m_l_ij;
      for (k=0; k<m_k_ij; k++) {
        for (l=0; l<m_l_ij; l++) {
          ptr[l*m_k_ij+k+2] = l*m_k_ij+k+idx;
        }
      }
      subscript[0] = i;
      /*bjp
      printf("p[%d] ii: %d jj: %d\n",me,ii,jj);
      */
      subscript[1] = j;
      /*bjp
      printf("p[%d] subscript = [%d:%d]\n",me,subscript[0],subscript[1]);
      */
      /*bjp
      printf("p[%d] size is: %d location is [%d:%d] ptr: %p\n",me,size,
          subscript[0],subscript[1], ptr);
          */
      
      if (subscript[0]<lo[0] || subscript[0]>hi[0] || subscript[1]<lo[1] ||
          subscript[1]>hi[1]) {
        printf("p[%d] assign i: %d j: %d lo[0]: %d hi[0]: %d lo[1]: %d hi[1]: %d\n",
            subscript[0],subscript[1],lo[0],hi[0],lo[1],hi[1]);
      }
      GP_Assign_local_element(g_p, subscript, (void*)ptr, size);
      /*bjp
      printf("p[%d] ptr_dim0: %ld ptr_dim[1]: %ld\n",me,ptr[0],ptr[1]);
      */
      /*bjp
      printf("p[%d]  completed assignment of [%d:%d]\n",me, subscript[0],subscript[1]);
      */
    }
  }
  
  /* Guarantee data consistency */
  NGA_Sync();
  GP_Debug(g_p);

  /* Generate bounding coordinates to an arbitrary patch in GP array */
#if 1
  get_range(ndim, dims, lo, hi, g_p);
#else
  idx = (me+4)%NGA_Nnodes();
/*  idx = me; */
  jj = idx%N_J;
  ii = (idx-jj)/N_J;
  lo[0] = ii;
  hi[0] = ii;
  lo[1] = jj;
  hi[1] = jj;
#endif
  /*bjp
  printf("p[%d] Getting patch [%d:%d] [%d:%d]\n",me,lo[0],hi[0],lo[1],hi[1]);
  */

  /* Find the total amount of data contained in the patch */
  nsize = (hi[0]-lo[0]+1)*(hi[1]-lo[1]+1);
  /* bjp
  printf("p[%d] Total size of patch[%d:%d][%d:%d]: %d\n",me,lo[0],hi[0],lo[1],hi[1],nsize);
  */
  GP_Get_size(g_p, lo, hi, &size);
  NGA_Sync();
  /*bjp
  printf("p[%d] Total size of patch data: %d\n",me,size);
  */

  /* Allocate local buffers and retrieve data */
  buf = (void*)malloc(size);
  /*bjp
  printf("p[%d] buf_ptr: %ld\n",me,(long)buf);
  */
  /*bjp
  printf("p[%d] dst_ptr: %p\n",me,buf);
  */
  buf_ptr = (void**)malloc(nsize*sizeof(void*));
  buf_size = (int*) malloc(nsize*sizeof(int));
  ld[1] = hi[0]-lo[0]+1;
  ld[0] = hi[1]-lo[1]+1;
  ld_sz[1] = hi[0]-lo[0]+1;
  ld_sz[0] = hi[1]-lo[1]+1;
  GA_Set_debug(1);
  GP_Get(g_p, lo, hi, buf, buf_ptr, ld, buf_size, ld_sz, &size);
  NGA_Sync();
  if (me==0) printf("\nCompleted GP_Get\n",me);
  GA_Set_debug(0);
  /*bjp
  printf("p[%d] Returned from GP_Get size: %d\n",me,size);
  */
  
  /* Check contents of buffers to see if data is as expected */
  /*bjp
  printf("p[%d] root pointer: %ld\n",me,(long)buf_ptr);
  */
  for (i=lo[0]; i<=hi[0]; i++) {
    ii = i - lo[0];
    for (j=lo[1]; j<=hi[1]; j++) {
      jj = j - lo[1];
      idx = j*N_I + i;
      /*bjp
      printf("p[%d] size of element [%d,%d]: %d offset: %d\n",me,i,j,
             buf_size[ii*ld_sz[0]+jj],ii*ld[0]+jj);
             */
      ptr = (int*)buf_ptr[ii*ld[0]+jj];
      /*bjp
      printf("p[%d] read_ptr: %ld ptr[0]: %d ptr[1]: %d\n",me,(long)ptr,ptr[0],ptr[1]);
      */
      /*bjp
      printf("p[%d] pointer[%d]: %ld\n",me,ii*ld[0]+jj,(long)ptr);
      */
      m_k_ij = i%Q_I + 1;
      m_l_ij = j%Q_J + 1;
      if (buf_size[ii*ld_sz[0]+jj] != 4*(ptr[0]*ptr[1]+2)) {
        printf("p[%d] size expected: %d actual: %d\n",me,buf_size[ii*ld_sz[0]+jj],
            4*(ptr[0]*ptr[1]+2));
      }
      /*
      m_k_ij = 2;
      m_l_ij = 2;
      */
      if (ptr[0] != m_k_ij) {
        printf("p[%d] [%d,%d] Dimension i actual: %d expected: %d\n",me,i,j,ptr[0],m_k_ij);
      }
      if (ptr[1] != m_l_ij) {
        printf("p[%d] [%d,%d] Dimension j actual: %d expected: %d\n",me,i,j,ptr[1],m_l_ij);
      }
      /*bjp
      printf("p[%d] i: %d j: %d m_k_ij: %d m_l_ij: %d\n",me,i,j,m_k_ij,m_l_ij);
      */
      for (k=0; k<ptr[0]; k++) {
        for (l=0; l<ptr[1]; l++) {
          if (ptr[l*ptr[0]+k+2] != l*m_k_ij+k+idx) {
            printf("p[%d] Element i: %d j: %d l: %d k: %d m_k_ij: %d idx: %d does not match: %d %d\n",
                me,i,j,l,k,m_k_ij,idx,ptr[l*ptr[0]+k+2],l*m_k_ij+k+idx);
          }
        }
      }
    }
  }
  NGA_Sync();
  if (me==0) printf("\nCompleted check of GP_Get\n",me);

  /* Clear all bits in GP_Array */
  GP_Memzero(g_p);
  /* Test to see if all bits actually are zero */
  GP_Distribution(g_p, me, lo_t, hi_t);
  for (i=lo_t[0]; i<=hi_t[0]; i++) {
    ii = i - lo_t[0];
    subscript[0] = i;
    for (j=lo_t[1]; j<=hi_t[1]; j++) {
      jj = j - lo_t[1];
      subscript[1] = j;
      GP_Access_element(g_p, subscript, &elem_buf, &size);
      ptr = (int*)elem_buf;
      m_k_ij = i%Q_I + 1;
      m_l_ij = j%Q_J + 1;
      if (size/4 != m_k_ij*m_l_ij+2) {
        printf("p[%d] Mismatched sizes in memzero test\n",me);
      }
      for (k=0; k<m_k_ij*m_l_ij+2; k++) {
        if (ptr[k] != 0) {
          printf("p[%d] Nonzero element %d in memzero test ptr[%d]: %c\n",
              me,idx,k,ptr[k]);
        }
      }
    }
  }
  if (me==0) printf("\nZeroed all bits in GP array\n");

  /* Clean up buffers and Global Pointer array */
  free(buf);
  free(buf_ptr);
  free(buf_size);
  GP_Distribution(g_p, me, lo, hi);
  for (i=lo[0]; i<=hi[0]; i++) {
    subscript[0] = i;
    for (j=lo[1]; j<=hi[1]; j++) {
      subscript[1] = j;
      GP_Free(GP_Free_local_element(g_p, subscript));
    }
  }

  /* destroy Global Pointer array */
  GP_Destroy(g_p);
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
