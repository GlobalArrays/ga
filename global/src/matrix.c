/*$Id: matrix.c,v 1.7 2004-04-02 00:27:19 d3h325 Exp $******************************************************
File: matrix.c 

Author: Limin Zhang, Ph.D.
        Mathematics Department
        Columbia Basin College
        Pasco, WA 99301
        Limin.Zhang@cbc2.org
 
Mentor: Jarek Naplocha, Ph.D.
        Environmental Molecular Science Laboratory
        Richland, WA 99352
 
Date: 2/28/2002
 
Purpose:
      matrix interfaces between TAO and
      global arrays.
**************************************************************/

#include "global.h"
#include "globalp.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "message.h"

#define auxi_median(a,b,c,m)                         \
{                                                    \
  if ((c < a) && (a < b))        m = a;              \
  else if ((a <= c) && (c <= b)) m = c;              \
  else                           m = b;              \
}                                                    

#define median(a,b,c,m)                              \
{                                                    \
  if(a == b)     m = a;                              \
  else if(a < b) auxi_median(a, b, c, m)             \
  else           auxi_median(b, a, c, m)             \
}

#define auxi_median_dcpl(na, nb, nc, za, zb, zc, zm) \
{                                                    \
   if ((nc < na) && (na < nb))       zm = za;        \
  else if ((na <= nc) && (nc <= nb)) zm = zc;        \
  else                               zm = zb;        \
}

#define median_dcpl(na, nb, nc, za, zb, zc, zm)                    \
{                                                                  \
  if (na == nb)     zm = za;                                       \
  else if (na < nb) auxi_median_dcpl (na, nb, nc, za, zb, zc, zm)  \
  else              auxi_median_dcpl (nb, na, nc, zb, za, zc, zm)  \
}

/*\ median routine
\*/
void FATR
ga_median_patch_ (g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi, g_m, mlo, mhi)
     Integer *g_a, *alo, *ahi;	/* patch of g_a */
     Integer *g_b, *blo, *bhi;	/* patch of g_b */
     Integer *g_c, *clo, *chi;	/* patch of g_c */
     Integer *g_m, *mlo, *mhi;	/* patch of g_m */
{
  Integer i, j;
  Integer atype, btype, andim, adims[MAXDIM], bndim, bdims[MAXDIM];
  Integer ctype, mtype, cndim, cdims[MAXDIM], mndim, mdims[MAXDIM];
  Integer loA[MAXDIM], hiA[MAXDIM], ldA[MAXDIM];
  Integer loB[MAXDIM], hiB[MAXDIM], ldB[MAXDIM];
  Integer loC[MAXDIM], hiC[MAXDIM], ldC[MAXDIM];
  Integer loM[MAXDIM], hiM[MAXDIM], ldM[MAXDIM];
  Integer g_A = *g_a, g_B = *g_b;
  Integer g_C = *g_c, g_M = *g_m;
  void *A_ptr, *B_ptr;
  void *C_ptr, *M_ptr;
  Integer bvalue[MAXDIM], bunit[MAXDIM], baseldA[MAXDIM];
  Integer idx, n1dim;
  Integer atotal, btotal;
  Integer ctotal, mtotal;
  Integer me = ga_nodeid_ (), b_temp_created = 0, c_temp_created = 0, m_temp_created = 0;
  Integer type = GA_TYPE_GSM;
  char *tempname = "temp", transp = 'n';	/*no transpose */
  double na, nb, nc;		/*norm of a, norm of b, norm of c */
  int ia, ib, ic, im;
  float fa, fb, fc, fm;
  double da, db, dc, dm;
  long la, lb, lc, lm;
  DoubleComplex za, zb, zc, zm;
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  GA_PUSH_NAME ("ga_median_patch_");

  nga_inquire_ (g_a, &atype, &andim, adims);
  nga_inquire_ (g_b, &btype, &bndim, bdims);
  nga_inquire_ (g_c, &ctype, &cndim, cdims);
  nga_inquire_ (g_m, &mtype, &mndim, mdims);


  if (atype != btype)
    ga_error (" ga_median_patch_:type mismatch ", 0L);
  if (atype != ctype)
    ga_error (" ga_median_patch_:type mismatch ", 0L);
  if (atype != mtype)
    ga_error (" ga_median_patch_:type mismatch ", 0L);
  

  /* check if patch indices and g_a dims match */
  for (i = 0; i < andim; i++)
    if (alo[i] <= 0 || ahi[i] > adims[i])
      {
      ga_error ("ga_median_patch_: g_a indices out of range ", *g_a);
      }

  for (i = 0; i < bndim; i++)
    if (blo[i] <= 0 || bhi[i] > bdims[i])
     {
      ga_error ("ga_median_patch_:g_b indices out of range ", *g_b);
     }

  for (i = 0; i < cndim; i++)
    if (clo[i] <= 0 || chi[i] > cdims[i])
      {
      ga_error ("ga_median_patch_:g_c indices out of range ", *g_c);
      }	      
  for (i = 0; i < bndim; i++)
    if (mlo[i] <= 0 || mhi[i] > mdims[i])
      {
      ga_error ("ga_median_patch_:g_m indices out of range ", *g_m);
      }

  /* check if numbers of elements in two patches match each other */

  atotal = 1;
  for (i = 0; i < andim; i++)
    atotal *= (ahi[i] - alo[i] + 1);

  btotal = 1;
  for (i = 0; i < bndim; i++)
    btotal *= (bhi[i] - blo[i] + 1);

  ctotal = 1;
  for (i = 0; i < cndim; i++)
    ctotal *= (chi[i] - clo[i] + 1);

  mtotal = 1;
  for (i = 0; i < mndim; i++)
    mtotal *= (mhi[i] - mlo[i] + 1);


  if (atotal != btotal)
    ga_error ("ga_median_patch_:  capacities of patches do not match ", 0L);

  if (atotal != ctotal)
    ga_error ("ga_median_patch_:  capacities of patches do not match ", 0L);

  if (atotal != mtotal)
    ga_error ("ga_median_patch_:  capacities of patches do not match ", 0L);


  /* find out coordinates of patches of g_A, g_B, g_C, and g_M that I own */
  nga_distribution_ (&g_A, &me, loA, hiA);
  nga_distribution_ (&g_B, &me, loB, hiB);
  nga_distribution_ (&g_C, &me, loC, hiC);
  nga_distribution_ (&g_M, &me, loM, hiM);


  if (!ngai_comp_patch (andim, loA, hiA, bndim, loB, hiB))
    {
      /* either patches or distributions do not match:
       *        - create a temp array that matches distribution of g_a
       *        - copy & reshape patch of g_b into g_B
       */
      if (!ga_duplicate (g_a, &g_B, tempname))
	ga_error ("ga_median_patch_:duplicate failed", 0L);

      nga_copy_patch (&transp, g_b, blo, bhi, &g_B, alo, ahi);
      bndim = andim;
      b_temp_created = 1;
      nga_distribution_ (&g_B, &me, loB, hiB);
    }

  if (!ngai_comp_patch (andim, loA, hiA, cndim, loC, hiC))
    {
      /* either patches or distributions do not match:
       *        - create a temp array that matches distribution of g_a
       *        - copy & reshape patch of g_c into g_C
       */
      if (!ga_duplicate (g_a, &g_C, tempname))
	ga_error ("ga_median_patch_:duplicate failed", 0L);

      nga_copy_patch (&transp, g_c, clo, chi, &g_C, alo, ahi);
      cndim = andim;
      c_temp_created = 1;
      nga_distribution_ (&g_C, &me, loC, hiC);
    }

  if (!ngai_comp_patch (andim, loA, hiA, mndim, loM, hiM))
    {
      /* either patches or distributions do not match:
       *        - create a temp array that matches distribution of g_a
       *        - copy & reshape patch of g_m into g_M
       */
      if (!ga_duplicate (g_a, &g_M, tempname))
	ga_error ("ga_median_patch_:duplicate failed", 0L);

      /*no need to copy g_m since it is the output matrix. */
      mndim = andim;
      m_temp_created = 1;
      nga_copy_patch (&transp, g_m, mlo, mhi, &g_M, alo, ahi);
      nga_distribution_ (&g_M, &me, loM, hiM);
    }


  if (!ngai_comp_patch (andim, loA, hiA, bndim, loB, hiB))
    ga_error (" patches mismatch ", 0);
  if (!ngai_comp_patch (andim, loA, hiA, cndim, loC, hiC))
    ga_error (" patches mismatch ", 0);
  if (!ngai_comp_patch (andim, loA, hiA, mndim, loM, hiM))
    ga_error (" patches mismatch ", 0);


  /* A[83:125,1:1]  <==> B[83:125] */
  if (andim > bndim)
    andim = bndim;		/* need more work */
  if (andim > cndim)
    andim = cndim;		/* need more work */
  if (andim > mndim)
    andim = mndim;		/* need more work */


  /*  determine subsets of my patches to access  */
  if (ngai_patch_intersect (alo, ahi, loA, hiA, andim))
    {
      nga_access_ptr (&g_A, loA, hiA, &A_ptr, ldA);
      nga_access_ptr (&g_B, loA, hiA, &B_ptr, ldB);
      nga_access_ptr (&g_C, loA, hiA, &C_ptr, ldC);
      nga_access_ptr (&g_M, loA, hiA, &M_ptr, ldM);

      /* number of n-element of the first dimension */
      n1dim = 1;
      for (i = 1; i < andim; i++)
	n1dim *= (hiA[i] - loA[i] + 1);

      /* calculate the destination indices */
      bvalue[0] = 0;
      bvalue[1] = 0;
      bunit[0] = 1;
      bunit[1] = 1;
      /* baseldA[0] = ldA[0]
       * baseldA[1] = ldA[0] * ldA[1]
       * baseldA[2] = ldA[0] * ldA[1] * ldA[2] .....
       */
      baseldA[0] = ldA[0];
      baseldA[1] = baseldA[0] * ldA[1];
      for (i = 2; i < andim; i++)
	{
	  bvalue[i] = 0;
	  bunit[i] = bunit[i - 1] * (hiA[i - 1] - loA[i - 1] + 1);
	  baseldA[i] = baseldA[i - 1] * ldA[i];
	}


      /*compute elementwise median */
      /*I have to inquire the data type again since ng_inquire and nga_inquire_internal_ treat data type differently */
      nga_inquire_internal_ (g_a, &type, &andim, adims);

      switch (type) {
	
      case C_INT: 
	for (i = 0; i < n1dim; i++) {
	  idx = 0;
	  for (j = 1; j < andim; j++) {
	    idx += bvalue[j] * baseldA[j - 1];
	    if (((i + 1) % bunit[j]) == 0)     bvalue[j]++;
	    if (bvalue[j] > (hiA[j] - loA[j])) bvalue[j] = 0;
	  }
	  for (j = 0; j < (hiA[0] - loA[0] + 1); j++) {
	    ia = ((int *) A_ptr)[idx + j];
	    ib = ((int *) B_ptr)[idx + j];
	    ic = ((int *) C_ptr)[idx + j];
	    im = ((int *) M_ptr)[idx + j];
	    median(ia, ib, ic, im);
	    ((int *) M_ptr)[idx + j] = im;
	  }
	}
	break;
	case C_LONG: 
	for (i = 0; i < n1dim; i++) {
	  idx = 0;
	  for (j = 1; j < andim; j++) {
	    idx += bvalue[j] * baseldA[j - 1];
	    if (((i + 1) % bunit[j]) == 0)     bvalue[j]++;
	    if (bvalue[j] > (hiA[j] - loA[j])) bvalue[j] = 0;
	  }
	  for (j = 0; j < (hiA[0] - loA[0] + 1); j++) {
	    la = ((long *) A_ptr)[idx + j];
	    lb = ((long *) B_ptr)[idx + j];
	    lc = ((long *) C_ptr)[idx + j];
	    lm = ((long *) M_ptr)[idx + j];
	    median(la, lb, lc, lm);
	    ((long *) M_ptr)[idx + j] = lm;
	  }
	}
	break;
      case C_FLOAT: 
	for (i = 0; i < n1dim; i++) {
	  idx = 0;
	  for (j = 1; j < andim; j++) {
	    idx += bvalue[j] * baseldA[j - 1];
	    if (((i + 1) % bunit[j]) == 0)     bvalue[j]++;
	    if (bvalue[j] > (hiA[j] - loA[j])) bvalue[j] = 0;
	  }
	  for (j = 0; j < (hiA[0] - loA[0] + 1); j++) {
	    fa = ((float *) A_ptr)[idx + j];
	    fb = ((float *) B_ptr)[idx + j];
	    fc = ((float *) C_ptr)[idx + j];
	    fm = ((float *) M_ptr)[idx + j];
	    median(fa, fb, fc, fm);
	    ((float *) M_ptr)[idx + j] = fm;
	  }
	}
	break;
      case C_DBL: 
	for (i = 0; i < n1dim; i++) {
	  idx = 0;
	  for (j = 1; j < andim; j++) {
	    idx += bvalue[j] * baseldA[j - 1];
	    if (((i + 1) % bunit[j]) == 0)     bvalue[j]++;
	    if (bvalue[j] > (hiA[j] - loA[j])) bvalue[j] = 0;
	  }
	  for (j = 0; j < (hiA[0] - loA[0] + 1); j++) {
	    da = ((double *) A_ptr)[idx + j];
	    db = ((double *) B_ptr)[idx + j];
	    dc = ((double *) C_ptr)[idx + j];
	    dm = ((double *) M_ptr)[idx + j];
	    median(da, db, dc, dm);
	    ((double *) M_ptr)[idx + j] = dm;
	  }
	}
	break;
      case C_DCPL:
	for (i = 0; i < n1dim; i++) {
	  idx = 0;
	  for (j = 1; j < andim; j++) {
	    idx += bvalue[j] * baseldA[j - 1];
	    if (((i + 1) % bunit[j]) == 0)     bvalue[j]++;
	    if (bvalue[j] > (hiA[j] - loA[j])) bvalue[j] = 0;
	  }
	  for (j = 0; j < (hiA[0] - loA[0] + 1); j++) {
	    za = ((DoubleComplex *) A_ptr)[idx + j];
	    zb = ((DoubleComplex *) B_ptr)[idx + j];
	    zc = ((DoubleComplex *) C_ptr)[idx + j];
	    zm = ((DoubleComplex *) M_ptr)[idx + j];
	    na = sqrt ((za.real) * (za.real) + (za.imag) * (za.imag));
	    nb = sqrt ((zb.real) * (zb.real) + (zb.imag) * (zb.imag));
	    nc = sqrt ((zc.real) * (zc.real) + (zc.imag) * (zc.imag));
	    median_dcpl(na, nb, nc, za, zb, zc, zm);
	    ((DoubleComplex *) M_ptr)[idx + j] = zm;
	  }
	}
	break;
      default:
	ga_error ("median: wrong data type", type);
      }


      /* release access to the data */
      nga_release_ (&g_A, loA, hiA);
      nga_release_ (&g_B, loA, hiA);
      nga_release_ (&g_C, loA, hiA);
      nga_release_ (&g_M, loA, hiA);
    }

  ga_sync_ ();


  if (b_temp_created)
    ga_destroy_ (&g_B);
  if (c_temp_created)
    ga_destroy_ (&g_C);
  if (m_temp_created)
    ga_destroy_ (&g_M);
  GA_POP_NAME;
  if(local_sync_end)ga_sync_();
}



void FATR
ga_median_ (Integer * g_a, Integer * g_b, Integer * g_c, Integer * g_m){


   Integer atype, andim;
   Integer alo[MAXDIM],ahi[MAXDIM];
   Integer btype, bndim;
   Integer blo[MAXDIM],bhi[MAXDIM];
   Integer ctype, cndim;
   Integer clo[MAXDIM],chi[MAXDIM];
   Integer mtype, mndim;
   Integer mlo[MAXDIM],mhi[MAXDIM];

    nga_inquire_internal_(g_a,  &atype, &andim, ahi);
    nga_inquire_internal_(g_b,  &btype, &bndim, bhi);
    nga_inquire_internal_(g_c,  &ctype, &cndim, chi);
    nga_inquire_internal_(g_m,  &mtype, &mndim, mhi);

    while(andim){
        alo[andim-1]=1;
        andim--;
    }

    while(bndim){
        blo[bndim-1]=1;
        bndim--;
    }

    while(cndim){
        clo[cndim-1]=1;
        cndim--;
    }

    while(mndim){
        mlo[mndim-1]=1;
        mndim--;
    }
    _ga_sync_begin = 1;
    ga_median_patch_(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi, g_m, mlo, mhi);

}


void FATR
ga_norm_infinity_ (Integer * g_a, double *nm)
{
  Integer dim1, dim2, type, size, nelem;
  Integer iloA, ihiA, jloA, jhiA, ld, lo[2], hi[2];
  Integer me = ga_nodeid_ (), i, j;
  void *ptr;

  int imax, *isum;
  long lmax, *lsum;
  double dmax, zmax, *dsum;
  float fmax, *fsum;
  DoubleComplex *zsum;
  void *buf;			/*temporary buffer */

  Integer ndim, dims[MAXDIM]; 
  Integer ga_type = GA_TYPE_GSM;
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  ga_check_handle (g_a, "ga_norm_infinity_");
  GA_PUSH_NAME ("ga_norm_infinity_");

/*  ga_inquire (g_a, &type, &dim1, &dim2); */
  nga_inquire_internal_ (g_a, &type, &ndim, dims);

  dim1 = dims[0];
  if(ndim<=0)
        ga_error("ga_norm_infinity: wrong dimension", ndim);
  else if(ndim == 1)
        dim2 = 1;
  else if(ndim==2)  
        dim2 = dims[1];
  else
        ga_error("ga_norm_infinity: wrong dimension", ndim);


  /*allocate a temporary buffer of size equal to the number of rows */
  size = GAsizeof (type);
  nelem = dim1;
  buf = malloc (nelem * size);

  if (buf == NULL)
    ga_error ("ga_norm_infinity_: no more memory for the buffer.\n", 0);

  switch (type)
    {
    case C_INT:
      isum = (int *) buf;
      break;
    case C_LONG:
      lsum = (long *) buf;
      break;
    case C_FLOAT:
      fsum = (float *) buf;
      break;
    case C_DBL:
      dsum = (double *) buf;
      break;
    case C_DCPL:
      zsum = (DoubleComplex *) buf;
      break;
    default:
      ga_error ("ga_norm_infinity_: wrong data type:", type);
    }


  /*zero the buffer */
  memset (buf, 0, nelem * size);

  /* ga_distribution_ (g_a, &me, &iloA, &ihiA, &jloA, &jhiA); */

  nga_distribution_(g_a, &me, lo, hi);
  if(ndim<=0)
        ga_error("ga_norm_infinity: wrong dimension", ndim);
  else if(ndim == 1){
      iloA=lo[0];
      ihiA=hi[0];
      jloA=1;
      jhiA=1;
  }
  else if(ndim == 2)
 {
      iloA=lo[0];
      ihiA=hi[0];
      jloA=lo[1];
      jhiA=hi[1];
  }
  else
        ga_error("ga_norm_infinity: wrong dimension", ndim);

  /* determine subset of my patch to access */
  if (ihiA > 0 && jhiA > 0)
    {
      /* lo[0] = iloA; */
      /* lo[1] = jloA; */
      /* hi[0] = ihiA; */
      /* hi[1] = jhiA; */
      nga_access_ptr (g_a, lo, hi, &ptr, &ld);

      switch (type)
	{
	  int *pi;
	  double *pd;
	  long *pl;
	  float *pf;
	  DoubleComplex *pz;
	case C_INT:
	  pi = (int *) ptr;
	  for (i = 0; i < ihiA - iloA + 1; i++)
	     for (j = 0; j < jhiA - jloA + 1; j++)
	      isum[iloA + i - 1] += ABS (pi[j * ld + i]);
	  break;
	case C_LONG:
	  pl = (long *) ptr;
	  for (i = 0; i < ihiA - iloA + 1; i++)
	    for (j = 0; j < jhiA - jloA + 1; j++)
	      lsum[iloA + i - 1] += ABS (pl[j * ld + i]);
	  break;
	case C_DCPL:
	  pz = (DoubleComplex *) ptr;
	  for (i = 0; i < ihiA - iloA + 1; i++)
	    for (j = 0; j < jhiA - jloA + 1; j++)
	      {
		DoubleComplex zval = pz[j * ld + i];
		double temp =
		  sqrt (zval.real * zval.real + zval.imag * zval.imag);
		(zsum[iloA + i - 1]).real += temp;
	      }
	  break;
	case C_FLOAT:
	  pf = (float *) ptr;
	  for (i = 0; i < ihiA - iloA + 1; i++)
	     for (j = 0; j < jhiA - jloA + 1; j++)
	      fsum[iloA + i - 1] += ABS (pf[j * ld + i]);
	  break;
	case C_DBL:
	  pd = (double *) ptr;
	  for (i = 0; i < ihiA - iloA + 1; i++)
	    for (j = 0; j < jhiA - jloA + 1; j++)
	      dsum[iloA + i - 1] += ABS (pd[j * ld + i]);
	  break;
	default:
	  ga_error ("ga_norm_infinity_: wrong data type ", type);
	}

      /* release access to the data */
      ga_release_update_ (g_a, &iloA, &ihiA, &jloA, &jhiA);
    }

  /*calculate the global value buf[j] for each column */
  switch (type)
    {
    case C_INT:
      armci_msg_igop (isum, nelem, "+");
      break;
    case C_DBL:
      armci_msg_dgop (dsum, nelem, "+");
      break;
    case C_DCPL:
      armci_msg_dgop ((double *) zsum, 2 * nelem, "+");
      break;
    case C_FLOAT:
      armci_msg_fgop (fsum, nelem, "+");
      break;
    case C_LONG:
      armci_msg_lgop (lsum, nelem, "+");
      break;
    default:
      ga_error ("ga_norm_infinity_: wrong data type ", type);
    }

  /*evaluate the norm infinity for the matrix g_a */
  switch (type)
    {
    case C_INT:
      imax = ((int *) buf)[0];
      for (j = 1; j < nelem; j++)
	if (imax < ((int *) buf)[j])
	  imax = ((int *) buf)[j];
      *((double *) nm) = (double) imax;
      break;
    case C_LONG:
      lmax = ((long *) buf)[0];
      for (j = 1; j < nelem; j++)
	if (lmax < ((long *) buf)[j])
	  lmax = ((long *) buf)[j];
      *((double *) nm) = (double) lmax;
      break;
    case C_FLOAT:
      fmax = ((float *) buf)[0];
      for (j = 1; j < nelem; j++)
	if (fmax < ((float *) buf)[j])
	  fmax = ((float *) buf)[j];
      *((double *) nm) = (double) fmax;
      break;
    case C_DBL:
      dmax = ((double *) buf)[0];
      for (j = 1; j < nelem; j++)
	if (dmax < ((double *) buf)[j])
	  dmax = ((double *) buf)[j];
      *((double *) nm) = dmax;
      break;
    case C_DCPL:
      zmax = (((DoubleComplex *) buf)[0]).real;
      for (j = 1; j < nelem; j++)
	if (zmax < (((DoubleComplex *) buf)[j]).real)
	  zmax = (((DoubleComplex *) buf)[j]).real;
      *((double *) nm) = zmax;
      break;
    default:
      ga_error ("ga_norm_infinity_:wrong data type.", type);
    }

  GA_POP_NAME;
  ga_sync_ ();

  /*free the memory allocated to buf */
  free (buf);
  buf = NULL;
}

void FATR
ga_norm1_ (Integer * g_a, double *nm)
{
  Integer dim1, dim2, type, size, nelem;
  Integer iloA, ihiA, jloA, jhiA, ld, lo[2], hi[2];
  Integer me = ga_nodeid_ (), i, j;
  void *ptr;

  int imax, *isum;
  long lmax, *lsum;
  double dmax, zmax, *dsum;
  float fmax, *fsum;
  DoubleComplex *zsum;
  void *buf;			/*temporary buffer */
  Integer ndim, dims[MAXDIM]; 

  Integer ga_type = GA_TYPE_GSM;
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  ga_check_handle (g_a, "ga_norm1_");
  GA_PUSH_NAME ("ga_norm1_");

  nga_inquire_internal_ (g_a, &type, &ndim, dims);
  dim1 = dims[0];
  if(ndim<=0)
        ga_error("ga_norm1: wrong dimension", ndim);
  else if(ndim == 1) 
	dim2 = 1;
  else if(ndim == 2) 
        dim2 = dims[1];
  else
        ga_error("ga_norm1: wrong dimension", ndim);
 /* ga_inquire (g_a, &type, &dim1, &dim2); */

  /*allocate a temporary buffer of size equal to the number of columns */
  size = GAsizeof (type);
  nelem = dim2;
  buf = malloc (nelem * size);

  if (buf == NULL)
    ga_error ("ga_norm1: no more memory for the buffer.\n", 0);

  switch (type)
    {
    case C_INT:
      isum = (int *) buf;
      break;
    case C_LONG:
      lsum = (long *) buf;
      break;
    case C_FLOAT:
      fsum = (float *) buf;
      break;
    case C_DBL:
      dsum = (double *) buf;
      break;
    case C_DCPL:
      zsum = (DoubleComplex *) buf;
      break;
    default:
      ga_error ("ga_norm1_: wrong data type:", type);
    }


  /*zero the buffer */
  memset (buf, 0, nelem * size);

  nga_distribution_(g_a, &me, lo, hi);
  if(ndim<=0)
        ga_error("ga_norm1: wrong dimension", ndim);
  else if(ndim == 1) { 
       iloA=lo[0];
       ihiA=hi[0];
       jloA=1;
       jhiA=1;
  }
  else if(ndim == 2)
 {
       iloA=lo[0];
       ihiA=hi[0];
       jloA=lo[1];
       jhiA=hi[1];
  }
  else
        ga_error("ga_norm1: wrong dimension", ndim);

  /* ga_distribution_ (g_a, &me, &iloA, &ihiA, &jloA, &jhiA); */


  /* determine subset of my patch to access */
  if (ihiA > 0 && jhiA > 0)
    {
     /* lo[0] = iloA; */
     /* lo[1] = jloA; */
     /* hi[0] = ihiA; */
      hi[1] = jhiA;

      nga_access_ptr (g_a, lo, hi, &ptr, &ld);

      switch (type)
	{
	  int *pi;
	  double *pd;
	  long *pl;
	  float *pf;
	  DoubleComplex *pz;
	case C_INT:
	  pi = (int *) ptr;
	  for (j = 0; j < jhiA - jloA + 1; j++)
	     for (i = 0; i < ihiA - iloA + 1; i++)
	      isum[jloA + j - 1 ] += ABS (pi[j * ld + i]);
	  break;
	case C_LONG:
	  pl = (long *) ptr;
	  for (j = 0; j < jhiA - jloA + 1; j++)
	    for (i = 0; i < ihiA - iloA + 1; i++)
	      lsum[jloA + j  - 1] += ABS (pl[j * ld + i]);
	  break;
	case C_DCPL:
	  pz = (DoubleComplex *) ptr;
	  for (j = 0; j < jhiA - jloA + 1; j++)
	    for (i = 0; i < ihiA - iloA + 1; i++)
	      {
		DoubleComplex zval = pz[j * ld + i];
		double temp =
		  sqrt (zval.real * zval.real + zval.imag * zval.imag);
		(zsum[jloA + j  - 1 ]).real += temp;
	      }
	  break;
	case C_FLOAT:
	  pf = (float *) ptr;
	  for (j = 0; j < jhiA - jloA + 1; j++)
	    for (i = 0; i < ihiA - iloA + 1; i++)
	      fsum[jloA + j  - 1 ] += ABS (pf[j * ld + i]);
	  break;
	case C_DBL:
	  pd = (double *) ptr;
	  for (j = 0; j < jhiA - jloA + 1; j++)
	    for (i = 0; i < ihiA - iloA + 1; i++)
	      dsum[jloA + j - 1 ] += ABS (pd[j * ld + i]);
	  break;
	default:
	  ga_error ("ga_norm1_: wrong data type ", type);
	}

      /* release access to the data */
      ga_release_update_ (g_a, &iloA, &ihiA, &jloA, &jhiA);
    }

  /*calculate the global value buf[j] for each column */
  switch (type)
    {
    case C_INT:
      armci_msg_igop (isum, nelem, "+");
      break;
    case C_DBL:
      armci_msg_dgop (dsum, nelem, "+");
      break;
    case C_DCPL:
      armci_msg_dgop ((double *) zsum, 2 * nelem, "+");
      break;
    case C_FLOAT:
      armci_msg_fgop (fsum, nelem, "+");
      break;
    case C_LONG:
      armci_msg_lgop (lsum, nelem, "+");
      break;
    default:
      ga_error ("ga_norm1_: wrong data type ", type);
    }

  /*evaluate the norm1 for the matrix g_a */
  switch (type)
    {
    case C_INT:
      imax = ((int *) buf)[0];
      for (j = 1; j < nelem; j++)
	if (imax < ((int *) buf)[j])
	  imax = ((int *) buf)[j];
      *((double *) nm) = (double) imax;
      break;
    case C_LONG:
      lmax = ((long *) buf)[0];
      for (j = 1; j < nelem; j++)
	if (lmax < ((long *) buf)[j])
	  lmax = ((long *) buf)[j];
      *((double *) nm) = (double) lmax;
      break;
    case C_FLOAT:
      fmax = ((float *) buf)[0];
      for (j = 1; j < nelem; j++)
	if (fmax < ((float *) buf)[j])
	  fmax = ((float *) buf)[j];
      *((double *) nm) = (double) fmax;
      break;
    case C_DBL:
      dmax = ((double *) buf)[0];
      for (j = 1; j < nelem; j++)
	if (dmax < ((double *) buf)[j])
	  dmax = ((double *) buf)[j];
      *((double *) nm) = dmax;
      break;
    case C_DCPL:
      zmax = (((DoubleComplex *) buf)[0]).real;
      for (j = 1; j < nelem; j++)
	if (zmax < (((DoubleComplex *) buf)[j]).real)
	  zmax = (((DoubleComplex *) buf)[j]).real;
      *((double *) nm) = zmax;
      break;
    default:
      ga_error ("ga_norm1_:wrong data type.", type);
    }

  GA_POP_NAME;
  ga_sync_ ();

  /*free the memory allocated to buf */
  free (buf);
  buf = NULL;
}

void FATR
ga_get_diag_ (Integer * g_a, Integer * g_v)
{
  Integer vndim, vdims, dim1, dim2, vtype, atype, type, nelem, size;
  Integer vlo, vhi, iloA, ihiA, jloA, jhiA, index, ld, lo[2], hi[2];
  Integer me = ga_nodeid_ (), i;
  void *buf, *ptr;
  int *ia;
  float *fa;
  double *da;
  long *la;
  DoubleComplex *dca;
  Integer andim, adims[2];
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  ga_check_handle (g_a, "ga_get_diag_");
  ga_check_handle (g_v, "ga_get_diag_");
  GA_PUSH_NAME ("ga_get_diag_");

  ga_inquire (g_a, &type, &dim1, &dim2);

  ga_distribution_ (g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

  /*Make sure to use nga_inquire to query for the data type since ga_inquire and nga_inquire treat data type differently */
  nga_inquire_ (g_a, &atype, &andim, adims);
  nga_inquire_ (g_v, &vtype, &vndim, &vdims);

  /* Perform some error checking */
  if (vndim != 1)
    ga_error ("ga_get_diag: wrong dimension for g_v.", vndim);


  if (vdims != MIN (dim1, dim2))
    ga_error
      ("ga_get_diag: The size of the first array's diagonal is greater than the size of the second array.",
       type);

  if (vtype != atype)
    {
      ga_error
	("ga_get_diag: input global arrays do not have the same data type. Global array type =",
	 atype);
    }


  /* determine subset of my patch to access */
  if (iloA > 0)
    {
      lo[0] = MAX (iloA, jloA);
      lo[1] = MAX (iloA, jloA);
      hi[0] = MIN (ihiA, jhiA);
      hi[1] = MIN (ihiA, jhiA);



      if (hi[0] >= lo[0])	/*make sure the equality symbol is there!!! */
	{			/* we got a block containing diagonal elements */

	  /*allocate a buffer for the given vector g_v */
	  size = GAsizeof (type);
	  vlo = MAX (iloA, jloA);
	  vhi = MIN (ihiA, jhiA);
	  nelem = vhi - vlo + 1;
	  buf = malloc (nelem * size);
	  if (buf == NULL)
	    ga_error
	      ("ga_get_diag_:failed to allocate memory for the local buffer.",
	       9999);

	  nga_access_ptr (g_a, lo, hi, &ptr, &ld);

	  /* get the vector from the global array g_a, put that in the the local memory buffer buf */
	  switch (type)
	    {
	    case C_INT:
	      ia = (int *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  ((int *) buf)[i] = *ia;
		  ia += ld + 1;
		}
	      break;
	    case C_LONG:
	      la = (long *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  ((long *) buf)[i] = *la;
		  la += ld + 1;
		}
	      break;
	    case C_FLOAT:
	      fa = (float *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  ((float *) buf)[i] = *fa;
		  fa += ld + 1;
		}
	      break;
	    case C_DBL:
	      da = (double *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  ((double *) buf)[i] = *da;
		  da += ld + 1;
		}
	      break;
	    case C_DCPL:
	      dca = (DoubleComplex *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  (((DoubleComplex *) buf)[i]).real = (*dca).real;
		  (((DoubleComplex *) buf)[i]).imag = (*dca).imag;
		  dca += ld + 1;
		}
	      break;

	    default:
	      ga_error ("get_diagonal_zero: wrong data type:", type);
	    }

	  /* copy the local memory buffer buf to g_v */
	  nga_put_ (g_v, &vlo, &vhi, buf, &vhi);

	  /*free the memory */
	  free (buf);

	  /* release access to the data */
	  ga_release_update_ (g_a, &iloA, &ihiA, &jloA, &jhiA);
	}
    }
  GA_POP_NAME;
  if(local_sync_end)ga_sync_();
}




void FATR
ga_add_diagonal_ (Integer * g_a, Integer * g_v)
{
  Integer vndim, vdims, dim1, dim2, vtype, atype, type, nelem, size;
  Integer vlo, vhi, iloA, ihiA, jloA, jhiA, index, ld, lo[2], hi[2];
  Integer me = ga_nodeid_ (), i;
  void *buf, *ptr;
  int *ia;
  float *fa;
  double *da;
  long *la;
  DoubleComplex *dca;
  Integer andim, adims[2];
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  ga_check_handle (g_a, "ga_add_diagonal_");
  ga_check_handle (g_v, "ga_add_diagonal_");
  GA_PUSH_NAME ("ga_add_diagonal_");

  ga_inquire (g_a, &type, &dim1, &dim2);

  ga_distribution_ (g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

  /*Make sure to use nga_inquire to query for the data type since ga_inquire and nga_inquire treat data type differently */
  nga_inquire_ (g_a, &atype, &andim, adims);
  nga_inquire_ (g_v, &vtype, &vndim, &vdims);

  /* Perform some error checking */
  if (vndim != 1)
    ga_error ("ga_add_diagonal: wrong dimension for g_v.", vndim);


  if (vdims != MIN (dim1, dim2))
    ga_error
      ("ga_add_diagonal: The size of the first array's diagonal is greater than the size of the second array.",
       type);

  if (vtype != atype)
    {
      ga_error
	("ga_add_diagonal: input global arrays do not have the same data type. Global array type =",
	 atype);
    }


  /* determine subset of my patch to access */
  if (iloA > 0)
    {
      lo[0] = MAX (iloA, jloA);
      lo[1] = MAX (iloA, jloA);
      hi[0] = MIN (ihiA, jhiA);
      hi[1] = MIN (ihiA, jhiA);



      if (hi[0] >= lo[0])	/*make sure the equality symbol is there!!! */
	{			/* we got a block containing diagonal elements */

	  /*allocate a buffer for the given vector g_v */
	  size = GAsizeof (type);
	  vlo = MAX (iloA, jloA);
	  vhi = MIN (ihiA, jhiA);
	  nelem = vhi - vlo + 1;
	  buf = malloc (nelem * size);
	  if (buf == NULL)
	    ga_error
	      ("ga_add_diagonal_:failed to allocate memory for the local buffer.",
	       0);

	  /* get the vector from the global array to the local memory buffer */
	  nga_get_ (g_v, &vlo, &vhi, buf, &vhi);

	  nga_access_ptr (g_a, lo, hi, &ptr, &ld);

	  switch (type)
	    {
	    case C_INT:
	      ia = (int *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *ia += ((int *) buf)[i];
		  ia += ld + 1;
		}
	      break;
	    case C_LONG:
	      la = (long *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *la += ((long *) buf)[i];
		  la += ld + 1;
		}
	      break;
	    case C_FLOAT:
	      fa = (float *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *fa += ((float *) buf)[i];
		  fa += ld + 1;
		}
	      break;
	    case C_DBL:
	      da = (double *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *da += ((double *) buf)[i];
		  da += ld + 1;
		}
	      break;
	    case C_DCPL:
	      dca = (DoubleComplex *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  (*dca).real += (((DoubleComplex *) buf)[i]).real;
		  (*dca).imag += (((DoubleComplex *) buf)[i]).imag;
		  dca += ld + 1;
		}
	      break;

	    default:
	      ga_error ("ga_add_diagonal_: wrong data type:", type);
	    }

	  /*free the memory */
	  free (buf);

	  /* release access to the data */
	  ga_release_update_ (g_a, &iloA, &ihiA, &jloA, &jhiA);
	}
    }
  GA_POP_NAME;
  if(local_sync_end)ga_sync_();
}



void FATR
ga_set_diagonal_ (Integer * g_a, Integer * g_v)
{
  Integer vndim, vdims, dim1, dim2, vtype, atype, type, nelem, size;
  Integer vlo, vhi, iloA, ihiA, jloA, jhiA, index, ld, lo[2], hi[2];
  Integer me = ga_nodeid_ (), i;
  void *buf, *ptr;
  int *ia;
  float *fa;
  double *da;
  long *la;
  DoubleComplex *dca;
  Integer andim, adims[2];
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  ga_check_handle (g_a, "ga_set_diagonal_");
  ga_check_handle (g_v, "ga_set_diagonal_");
  GA_PUSH_NAME ("ga_set_diagonal_");

  ga_inquire (g_a, &type, &dim1, &dim2);

  ga_distribution_ (g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

  /*Make sure to use nga_inquire to query for the data type since ga_inquire and nga_inquire treat data type differently */
  nga_inquire_ (g_a, &atype, &andim, adims);
  nga_inquire_ (g_v, &vtype, &vndim, &vdims);

  /* Perform some error checking */
  if (vndim != 1)
    ga_error ("ga_set_diagonal: wrong dimension for g_v.", vndim);


  if (vdims != MIN (dim1, dim2))
    ga_error
      ("ga_set_diagonal: The size of the first array's diagonal is greater than the size of the second array.",
       type);

  if (vtype != atype)
    {
      ga_error
	("ga_set_diagonal: input global arrays do not have the same data type. Global array type =",
	 atype);
    }


  /* determine subset of my patch to access */
  if (iloA > 0)
    {
      lo[0] = MAX (iloA, jloA);
      lo[1] = MAX (iloA, jloA);
      hi[0] = MIN (ihiA, jhiA);
      hi[1] = MIN (ihiA, jhiA);



      if (hi[0] >= lo[0])	/*make sure the equality symbol is there!!! */
	{			/* we got a block containing diagonal elements*/

	  /*allocate a buffer for the given vector g_v */
	  size = GAsizeof (type);
	  vlo = MAX (iloA, jloA);
	  vhi = MIN (ihiA, jhiA);
	  nelem = vhi - vlo + 1;
	  buf = malloc (nelem * size);
	  if (buf == NULL)
	    ga_error
	      ("ga_set_diagonal_:failed to allocate memory for local buffer",0);

	  /* get the vector from the global array to the local memory buffer */
	  nga_get_ (g_v, &vlo, &vhi, buf, &vhi);

	  nga_access_ptr (g_a, lo, hi, &ptr, &ld);

	  switch (type)
	    {
	    case C_INT:
	      ia = (int *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *ia = ((int *) buf)[i];
		  ia += ld + 1;
		}
	      break;
	    case C_LONG:
	      la = (long *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *la = ((long *) buf)[i];
		  la += ld + 1;
		}
	      break;
	    case C_FLOAT:
	      fa = (float *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *fa = ((float *) buf)[i];
		  fa += ld + 1;
		}
	      break;
	    case C_DBL:
	      da = (double *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *da = ((double *) buf)[i];
		  da += ld + 1;
		}
	      break;
	    case C_DCPL:
	      dca = (DoubleComplex *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  (*dca).real = (((DoubleComplex *) buf)[i]).real;
		  (*dca).imag = (((DoubleComplex *) buf)[i]).imag;
		  dca += ld + 1;
		}
	      break;

	    default:
	      ga_error ("ga_set_diagonal_: wrong data type:", type);
	    }

	  /*free the memory */
	  free (buf);

	  /* release access to the data */
	  ga_release_update_ (g_a, &iloA, &ihiA, &jloA, &jhiA);
	}
    }
  GA_POP_NAME;
  if(local_sync_begin)ga_sync_();
}



void FATR
ga_shift_diagonal_ (Integer * g_a, void *c)
{
  Integer dim1, dim2, type;
  Integer iloA, ihiA, jloA, jhiA, index, ld, lo[2], hi[2];
  Integer me = ga_nodeid_ (), i;
  void *ptr;
  int *ia;
  float *fa;
  double *da;
  long *la;
  DoubleComplex *dca;
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  ga_check_handle (g_a, "ga_shift_diagonal_");
  GA_PUSH_NAME ("ga_shift_diagonal_");

  ga_inquire (g_a, &type, &dim1, &dim2);

  ga_distribution_ (g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

  /* determine subset of my patch to access */
  if (iloA > 0)
    {
      lo[0] = MAX (iloA, jloA);
      lo[1] = MAX (iloA, jloA);
      hi[0] = MIN (ihiA, jhiA);
      hi[1] = MIN (ihiA, jhiA);
      if (hi[0] >= lo[0])	/*make sure the equality sign is there since it is the singleton case */
	{			/* we got a block containing diagonal elements */
	  nga_access_ptr (g_a, lo, hi, &ptr, &ld);

	  switch (type)
	    {
	    case C_INT:
	      ia = (int *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *ia += *((int *) c);
		  ia += ld + 1;
		}
	      break;
	    case C_LONG:
	      la = (long *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *la += *((long *) c);
		  la += ld + 1;
		}
	      break;
	    case C_FLOAT:
	      fa = (float *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *fa += *((float *) c);
		  fa += ld + 1;
		}
	      break;
	    case C_DBL:
	      da = (double *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *da += *((double *) c);
		  da += ld + 1;
		}
	      break;
	    case C_DCPL:
	      dca = (DoubleComplex *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  (*dca).real += (*((DoubleComplex *) c)).real;
		  (*dca).imag += (*((DoubleComplex *) c)).imag;
		  dca += ld + 1;
		}
	      break;

	    default:
	      ga_error ("ga_shift_diagonal_: wrong data type:", type);
	    }

	  /* release access to the data */
	  ga_release_update_ (g_a, &iloA, &ihiA, &jloA, &jhiA);
	}
    }
  GA_POP_NAME;
  if(local_sync_end)ga_sync_();
}

void FATR ga_zero_diagonal_(Integer * g_a)
{
  Integer dim1, dim2, type;
  Integer iloA, ihiA, jloA, jhiA, index, ld, lo[2], hi[2];
  Integer me = ga_nodeid_ (), i;
  void *ptr;
  int *ia;
  float *fa;
  double *da;
  long *la;
  DoubleComplex *dca;
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  GA_PUSH_NAME ("ga_zero_diagonal_");

  ga_inquire (g_a, &type, &dim1, &dim2);

  ga_distribution_ (g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

  /* determine subset of my patch to access */
  if (iloA > 0)
    {
      lo[0] = MAX (iloA, jloA);
      lo[1] = MAX (iloA, jloA);
      hi[0] = MIN (ihiA, jhiA);
      hi[1] = MIN (ihiA, jhiA);
      if (hi[0] > lo[0])
	{			/* we got a block containing diagonal elements */
	  nga_access_ptr (g_a, lo, hi, &ptr, &ld);

	  switch (type)
	    {
	    case C_INT:
	      ia = (int *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *ia = 0;
		  ia += ld + 1;
		}
	      break;
	    case C_LONG:
	      la = (long *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *la = 0;
		  la += ld + 1;
		}
	      break;
	    case C_FLOAT:
	      fa = (float *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *fa = 0.0;
		  fa += ld + 1;
		}
	      break;
	    case C_DBL:
	      da = (double *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  *da = 0.0;
		  da += ld + 1;
		}
	      break;
	    case C_DCPL:
	      dca = (DoubleComplex *) ptr;
	      for (i = 0; i < hi[0] - lo[0] + 1; i++)
		{
		  (*dca).real = 0.0;
		  (*dca).imag = 0.0;
		  dca += ld + 1;
		}
	      break;


	    default:
	      ga_error ("set_diagonal_zero: wrong data type:", type);
	    }

	  /* release access to the data */
	  ga_release_update_ (g_a, &iloA, &ihiA, &jloA, &jhiA);
	}
    }
  GA_POP_NAME;
  if(local_sync_end)ga_sync_();
}



void FATR ga_scale_rows_(Integer *g_a, Integer *g_v)
{
  Integer vndim, vdims, dim1, dim2, vtype, atype, type, nelem, size;
  Integer vlo, vhi, iloA, ihiA, jloA, jhiA, index, ld, lo[2], hi[2];
  Integer me = ga_nodeid_ (), i;
  void *buf, *ptr;
  int *ia;
  float *fa;
  double *da;
  long *la;
  DoubleComplex *dca;
  Integer andim, adims[2];
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  ga_check_handle (g_a, "ga_scale_rows_");
  ga_check_handle (g_v, "ga_scale_rows_");
  GA_PUSH_NAME ("ga_scale_rows_");

  ga_inquire (g_a, &type, &dim1, &dim2);

  ga_distribution_ (g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

  /*Make sure to use nga_inquire to query for the data type since ga_inquire and nga_inquire treat data type differently */
  nga_inquire_ (g_a, &atype, &andim, adims);
  nga_inquire_ (g_v, &vtype, &vndim, &vdims);

  /* Perform some error checking */
  if (vndim != 1)
    ga_error ("ga_scale_rows_: wrong dimension for g_v.", vndim);



  /*in internal functions, dim1 = number of rows of the matrix g_a*/
  /*in internal functions, dim2 = number of columns of the matrix g_a*/
  if (vdims != dim1)
    ga_error
      ("ga_scale_rows_: The size of the scalar array is not the same as the number of the rows of g_a.",
       vdims);

  if (vtype != atype)
    {
      ga_error
	("ga_scale_rows_: input global arrays do not have the same data type. Global array type =",
	 atype);
    }


  /* determine subset of my patch to access */
  if (iloA > 0)
    {
      lo[0] = iloA;
      lo[1] = jloA;
      hi[0] = ihiA;
      hi[1] = jhiA;


      if (hi[0] >= lo[0])	/*make sure the equality symbol is there!!! */
	{			/* we got a block containing diagonal elements */

          Integer myrows = hi[0] - lo[0] + 1;
          Integer mycols = hi[1] - lo[1] + 1;
          Integer j;
          /*number of rows on the patch is jhiA - jloA + 1 */
          vlo =iloA ;
          vhi = ihiA;

	  /*allocate a buffer for the given vector g_v */
	  size = GAsizeof (type);
         
	  buf = malloc (myrows * size);
	  if (buf == NULL)
	    ga_error
	      ("ga_scale_rows_:failed to allocate memory for the local buffer.",
	       0);

	  /* get the vector from the global array to the local memory buffer */
	  nga_get_ (g_v, &vlo, &vhi, buf, &vhi);

	  nga_access_ptr (g_a, lo, hi, &ptr, &ld);

	  switch (type)
	    {
	    case C_INT:
	      ia = (int *) ptr;
	      for (i = 0; i < hi[0]-lo[0]+1; i++) /*for each row */
              for(j=0;j<hi[1]-lo[1]+1;j++) /*for each column*/
		  ia[j*ld+i] *= ((int *) buf)[i];
	      break;
	    case C_LONG:
	      la = (long *) ptr;
	      for (i = 0; i < hi[0]-lo[0]+1; i++)
              for(j=0;j<hi[1]-lo[1]+1;j++)
		  la[j*ld+i] *= ((long *) buf)[i];
	      break;
	    case C_FLOAT:
	      fa = (float *) ptr;
	      for (i = 0; i < hi[0]-lo[0]+1; i++)
              for(j=0;j<hi[1]-lo[1]+1;j++)
		  fa[j*ld+i] *= ((float *) buf)[i];
	      break;
	    case C_DBL:
	      da = (double *) ptr;
	      for (i = 0; i < hi[0]-lo[0]+1; i++)
              for(j=0;j<hi[1]-lo[1]+1;j++)
		  da[j*ld+i] *= ((double *) buf)[i];
	      break;
	    case C_DCPL:
	      dca = (DoubleComplex *) ptr;
	      for (i = 0; i < hi[0]-lo[0]+1; i++)
              for(j=0;j<hi[1]-lo[1]+1;j++)
		{
		  (dca[j*ld+i]).real *= (((DoubleComplex *) buf)[i]).real;
		  (dca[j*ld+i]).imag *= (((DoubleComplex *) buf)[i]).imag;
		}
	      break;

	    default:
	      ga_error ("ga_scale_rows_: wrong data type:", type);
	    }

	  /*free the memory */
	  free (buf);

	  /* release access to the data */
	  ga_release_update_ (g_a, &iloA, &ihiA, &jloA, &jhiA);
	}
    }
  GA_POP_NAME;
  if(local_sync_end)ga_sync_();
}





 


void FATR ga_scale_cols_(Integer *g_a, Integer *g_v)
{
  Integer vndim, vdims, dim1, dim2, vtype, atype, type, nelem, size;
  Integer vlo, vhi, iloA, ihiA, jloA, jhiA, index, ld, lo[2], hi[2];
  Integer me = ga_nodeid_ (), i;
  void *buf, *ptr;
  int *ia;
  float *fa;
  double *da;
  long *la;
  DoubleComplex *dca;
  Integer andim, adims[2];
  int local_sync_begin,local_sync_end;

  local_sync_begin = _ga_sync_begin; local_sync_end = _ga_sync_end;
  _ga_sync_begin = 1; _ga_sync_end=1; /*remove any previous masking*/
  if(local_sync_begin)ga_sync_();

  ga_check_handle (g_a, "ga_scale_cols_");
  ga_check_handle (g_v, "ga_scale_cols_");
  GA_PUSH_NAME ("ga_scale_cols_");

  ga_inquire (g_a, &type, &dim1, &dim2);

  ga_distribution_ (g_a, &me, &iloA, &ihiA, &jloA, &jhiA);

  /*Make sure to use nga_inquire to query for the data type since ga_inquire and nga_inquire treat data type differently */
  nga_inquire_ (g_a, &atype, &andim, adims);
  nga_inquire_ (g_v, &vtype, &vndim, &vdims);

  /* Perform some error checking */
  if (vndim != 1)
    ga_error ("ga_scale_cols_: wrong dimension for g_v.", vndim);



  /*in internal functions, dim1 = number of rows of the matrix g_a*/
  /*in internal functions, dim2 = number of columns of the matrix g_a*/
  if (vdims != dim2)
    ga_error
      ("ga_scale_cols_: The size of the scalar array is not the same as the number of the rows of g_a.",
       vdims);

  if (vtype != atype)
    {
      ga_error
	("ga_scale_cols_: input global arrays do not have the same data type. Global array type =",
	 atype);
    }


  /* determine subset of my patch to access */
  if (iloA > 0)
    {
      lo[0] = iloA;
      lo[1] = jloA;
      hi[0] = ihiA;
      hi[1] = jhiA;


      if (hi[0] >= lo[0])	/*make sure the equality symbol is there!!! */
	{			/* we got a block containing diagonal elements*/

          Integer mycols = hi[1] - lo[1] + 1;
          Integer j;
          /*number of rows on the patch is jhiA - jloA + 1 */
          vlo =jloA ;
          vhi = jhiA;

	  /*allocate a buffer for the given vector g_v */
	  size = GAsizeof (type);
         
	  buf = malloc (mycols * size);
	  if (buf == NULL)
	    ga_error
	      ("ga_scale_cols_:failed to allocate memory for the local buffer.",
	       0);

	  /* get the vector from the global array to the local memory buffer */
	  nga_get_ (g_v, &vlo, &vhi, buf, &vhi);

	  nga_access_ptr (g_a, lo, hi, &ptr, &ld);

	  switch (type)
	    {
	    case C_INT:
	      ia = (int *) ptr;
              for(j=0;j<hi[1]-lo[1]+1;j++) /*for each column*/
	      for (i = 0; i < hi[0]-lo[0]+1; i++) /*for each row */
		  ia[j*ld+i] *= ((int *) buf)[j];
	      break;
	    case C_LONG:
	      la = (long *) ptr;
              for(j=0;j<hi[1]-lo[1]+1;j++)
	      for (i = 0; i < hi[0]-lo[0]+1; i++)
		  la[j*ld+i] *= ((long *) buf)[j];
	      break;
	    case C_FLOAT:
	      fa = (float *) ptr;
              for(j=0;j<hi[1]-lo[1]+1;j++)
	      for (i = 0; i < hi[0]-lo[0]+1; i++)
		  fa[j*ld+i] *= ((float *) buf)[j];
	      break;
	    case C_DBL:
	      da = (double *) ptr;
              for(j=0;j<hi[1]-lo[1]+1;j++)
	      for (i = 0; i < hi[0]-lo[0]+1; i++)
		  da[j*ld+i] *= ((double *) buf)[j];
	      break;
	    case C_DCPL:
	      dca = (DoubleComplex *) ptr;
              for(j=0;j<hi[1]-lo[1]+1;j++)
	      for (i = 0; i < hi[0]-lo[0]+1; i++)
		{
		  (dca[j*ld+i]).real *= (((DoubleComplex *) buf)[j]).real;
		  (dca[j*ld+i]).imag *= (((DoubleComplex *) buf)[j]).imag;
		}
	      break;

	    default:
	      ga_error ("ga_scale_cols_: wrong data type:", type);
	    }

	  /*free the memory */
	  free (buf);

	  /* release access to the data */
	  ga_release_update_ (g_a, &iloA, &ihiA, &jloA, &jhiA);
	}
    }
  GA_POP_NAME;
  if(local_sync_end)ga_sync_();
}





 


