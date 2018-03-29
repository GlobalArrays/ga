#include <stdlib.h>
#include <math.h>
#include <stdio.h>

#include "macdecls.h"
#include "ga.h"
#include "mp3.h"

#define NDIM 1024

#define MAX_FACTOR 1024
void grid_factor(int p, int xdim, int ydim, int *idx, int *idy) {
  int i, j; 
  int ip, ifac, pmax, prime[MAX_FACTOR];
  int fac[MAX_FACTOR];
  int ix, iy, ichk;

  i = 1;
/**
 *   factor p completely
 *   first, find all prime numbers, besides 1, less than or equal to 
 *   the square root of p
 */
  ip = (int)(sqrt((double)p))+1;
  pmax = 0;
  for (i=2; i<=ip; i++) {
    ichk = 1;
    for (j=0; j<pmax; j++) {
      if (i%prime[j] == 0) {
        ichk = 0;
        break;
      }
    }
    if (ichk) {
      pmax = pmax + 1;
      if (pmax > MAX_FACTOR) printf("Overflow in grid_factor\n");
      prime[pmax-1] = i;
    }
  }
/**
 *   find all prime factors of p
 */
  ip = p;
  ifac = 0;
  for (i=0; i<pmax; i++) {
    while(ip%prime[i] == 0) {
      ifac = ifac + 1;
      fac[ifac-1] = prime[i];
      ip = ip/prime[i];
    }
  }
/**
 *  p is prime
 */
  if (ifac==0) {
    ifac++;
    fac[0] = p;
  }
/**
 *    find two factors of p of approximately the same size
 */
  *idx = 1;
  *idy = 1;
  for (i = ifac-1; i >= 0; i--) {
    ix = xdim/(*idx);
    iy = ydim/(*idy);
    if (ix >= iy && ix > 1) {
      *idx = fac[i]*(*idx);
    } else if (iy >= ix && iy > 1) {
      *idy = fac[i]*(*idy);
    } else {
      printf("Too many processors in grid factoring routine\n");
    }
  }
}

int main(int argc, char **argv) {
  int s_a, g_v, g_av;
  int one;
  int me, nproc;
  int xdim, ydim, ipx, ipy, idx, idy;
  int ilo, ihi, jlo, jhi;
  int i, j, iproc, ld, ncols, ncnt;
  double val;
  double *vptr;
  double *vbuf, *vsum;
  long *iptr = NULL, *jptr = NULL;
  int ok;
  double r_one = 1.0;
  /* Intitialize a message passing library */
  one = 1;
  MP_INIT(argc,argv);
  /* Initialize GA */
  NGA_Initialize();

  xdim = NDIM;
  ydim = NDIM;
  me = GA_Nodeid();
  nproc = GA_Nnodes();

  /* factor array */
  grid_factor(nproc, xdim, ydim, &ipx, &ipy);
  if (me == 0) {
    printf("Testing sparse array on %d processors\n",nproc);
    printf("\n    Using %d X %d processor grid\n",ipx,ipy);
    printf("\n    Matrix size is %d X %d\n",xdim,ydim);
  }
  /* figure out process location in proc grid */
  idx = me%ipx;
  idy = (me-idx)/ipx;
  /* find bounding indices for this processor */
  ilo = (xdim*idx)/ipx;
  if (idx < ipx-1) {
    ihi = (xdim*(idx+1))/ipx-1;
  } else {
    ihi = xdim-1;
  }
  jlo = (ydim*idy)/ipy;
  if (idy < ipy-1) {
    jhi = (ydim*(idy+1))/ipy-1;
  } else {
    jhi = ydim-1;
  }
 
  /* create sparse array */
  s_a = NGA_Sprs_array_create(xdim, ydim, C_DBL);
  if (ydim%2 == 0) {
    ld = ydim/2;
  } else {
    ld = (ydim-1)/2+1;
  }
  /* add elements to array. Every other element is zero */
  for (i=ilo; i<=ihi; i++) {
    for (j=jlo; j<=jhi; j++) {
      if (i%2 == 0 && j%2 == 0) {
        val = (double)((i/2)*ld+j/2);
        NGA_Sprs_array_add_element(s_a,i,j,&val);
      }
    }
  }
  if (NGA_Sprs_array_assemble(s_a) && me == 0) {
    printf("\n    Sparse array assembly completed\n");
  }

  /* construct vector */
  g_v = NGA_Create_handle();
  NGA_Set_data(g_v,one,&ydim,C_DBL);
  NGA_Allocate(g_v);
  g_av = GA_Duplicate(g_v, "dup");
  GA_Zero(g_av);

  /* set vector values */
  NGA_Distribution(g_v,me,&ilo,&ihi);
  NGA_Access(g_v,&ilo,&ihi,&vptr,&one);
  for (i=ilo;i<=ihi;i++) {
    vptr[i-ilo] = (double)i;
  }
  if (me == 0) {
    printf("\n    Vector initialized\n");
  }
  NGA_Release(g_v,&ilo,&ihi);


  /* access array blocks an check values for correctness */
  NGA_Sprs_array_row_distribution(s_a,me,&ilo,&ihi);
  ok = 1;
  ncnt = 0;
  for (iproc=0; iproc<nproc; iproc++) {
    NGA_Sprs_array_column_distribution(s_a,iproc,&jlo,&jhi);
    NGA_Sprs_array_access_col_block(s_a,iproc,&iptr,&jptr,&vptr);
    if (vptr != NULL) {
      for (i=ilo; i<=ihi; i++) {
        ncols = iptr[i+1-ilo]-iptr[i-ilo];
        for (j=0; j<ncols; j++) {
          ncnt++;
          idy = jptr[iptr[i-ilo]+j];
          if ((i-1)%2 != 0 || (idy-1)%2 != 0) ok = 0;
          val = (double)((i/2)*ld+idy/2);
          if (fabs(val-vptr[iptr[i-ilo]+j]) > 1.0e-5) {
            ok = 0;
            printf("p[%d] i: %d j: %d val: %f\n",me,i,
                jptr[iptr[i-ilo]+j],vptr[iptr[i-ilo]+j]);
          }
        }
      }
    }
  }
  GA_Igop(&ncnt,one,"+");
  if (ncnt != (xdim/2)*(ydim/2)) ok = 0;
  if (ok && me==0) {
    printf("\n    Values in sparse array are correct\n");
  }

  /* multiply sparse matrix by sparse vector */
  vsum = (double*)malloc((ihi-ilo+1)*sizeof(double));
  for (i=ilo; i<=ihi; i++) {
    vsum[i-ilo] = 0.0;
  }
  for (iproc=0; iproc<nproc; iproc++) {
    NGA_Sprs_array_column_distribution(s_a,iproc,&jlo,&jhi);
    NGA_Sprs_array_access_col_block(s_a,iproc,&iptr,&jptr,&vptr);
    if (vptr != NULL) {
      vbuf = (double*)malloc((jhi-jlo+1)*sizeof(double));
      NGA_Get(g_v,&jlo,&jhi,vbuf,&one);
      for (i=ilo; i<=ihi; i++) {
        ncols = iptr[i+1-ilo]-iptr[i-ilo];
        for (j=0; j<ncols; j++) {
          vsum[i-ilo] += vptr[iptr[i-ilo]+j]*vbuf[jptr[iptr[i-ilo]+j]-1-jlo];
          /*
          printf("i: %d j: %d a: %f v: %f j': %d tot: %f\n",i,jptr[iptr[i-ilo]+j],
              vptr[iptr[i-ilo]+j],vbuf[jptr[iptr[i-ilo]+j]-1-jlo],
              jptr[iptr[i-ilo]+j]-1-jlo,vsum[i-ilo]);
              */
        }
        /*
        printf("i: %d vsum: %f\n",i,vsum[i-ilo]);
        */
      }
      free(vbuf);
    }
  }
  ilo--;
  ihi--;
  if (ihi>=ilo) NGA_Acc(g_av,&ilo,&ihi,vsum,&one,&r_one);
  GA_Sync();
  free(vsum);

  /* check product vector */
  ok = 1;
  NGA_Distribution(g_av,me,&ilo,&ihi);
  NGA_Access(g_av,&ilo,&ihi,&vptr,&one);
  /*
  printf("ilo: %d ihi: %d\n",ilo,ihi);
  */
  for (i=ilo; i<=ihi; i++) {
    val = 0.0;
    if (i%2 == 0) {
      for (j=0; j<ydim; j++) {
        if (j%2 == 0) {
          /*
          printf("i: %d j: %d a: %d v: %d\n",i+1,j+1,(i/2)*ld+(j/2),j);
          */
          val += (double)(((i/2)*ld+(j/2))*j);
        }
      }
      if (fabs(val-vptr[i-ilo]) >= 1.0e-5) {
        ok = 0;
        printf("Error for element %d expected: %f actual: %f\n",
            i,val,vptr[i-ilo]);
      }
    } else {
      if (fabs(vptr[i-ilo]) >= 1.0e-5) {
        ok = 0;
        printf("Error for element %d expected: 0.00000 actual: %f\n",
            i,vptr[i-ilo]);
      }
    }
  }
  if (ok && me==0) {
    printf("\n    Matrix-vector product is correct\n\n");
  }

  NGA_Sprs_array_destroy(s_a);
  NGA_Destroy(g_v);
  NGA_Destroy(g_av);

  NGA_Terminate();
  /**
   *  Tidy up after message-passing library
   */
  MP_FINALIZE();
}
