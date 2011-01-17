/* Test algorithm parameters */

#include "ga.h"
#include "macdecls.h"
#include "mp3.h"

#include <stdlib.h>

#define N_I  567
#define N_J  789

#define Q_I 13
#define Q_J 17

/* get random patch for GP */
void get_range( int ndim, int dims[], int lo[], int hi[])
{
  int dim, nproc, i, itmp;
  nproc = GA_Nnodes();
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
  int dims[2],lo[2],hi[2],ndim;
  int idim, jdim, subscript[2], size;
  int *ptr;

  /* Create Global Pointer array */
  dims[0] = N_I;
  dims[1] = N_J;
  ndim = 2;
  me = GA_Nodeid();

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
      idx = j*N_I + i;
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
      subscript[0] = i;
      subscript[1] = j;
      GP_Assign_local_element(g_p, subscript, (void*)ptr, size);
    }
  }
  
  /* Guarantee data consistency */
  GA_Sync();

  /* Generate bounding coordinates to an arbitrary patch in GP array */
  get_range(ndim, dims, lo, hi);

  /* Clean up Global Pointer array */
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
