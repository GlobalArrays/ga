/**\
File: elempatch.c
Purpose: test the interfaces:

	GA_Abs_value_patch(g_a)
	GA_Add_constant_patch(g_a, alpha)
	GA_Recip_patch_patch(g_a)
	GA_Elem_multiply_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi)
	GA_Elem_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi)
	GA_Elem_maximum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, ch
	GA_Elem_minimum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi)
        
        that are for TAO/Global Array Project
 
Author:

Limin Zhang, Ph.D.
Mathematics Department
Columbia Basin College
Pasco, WA 99301

Mentor:

Jarek Nieplocha
Pacific Northwest National Laboratory

Date: Jauary 30, 2002
Revised on February 26, 2002.

\**/

#include "ga.h"
#include "macdecls.h"
#include "../src/globalp.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#ifdef MPI
#include <mpi.h>
#else
#include "sndrcv.h"
#endif


# define THRESH 1e-5
#define MISMATCHED(x,y) ABS((x)-(y))>=THRESH

#define N 8
#define OP_ELEM_MULT 0
#define OP_ELEM_DIV 1
#define OP_ELEM_MAX 2
#define OP_ELEM_MIN 3
#define OP_ABS 4
#define OP_ADD_CONST 5
#define OP_RECIP 6
#define MY_TYPE 2002

Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM], _ga_work[MAXDIM];
#  define COPYINDEX_C2F(carr, farr, n){\
   int i; for(i=0; i< (n); i++)(farr)[n-i-1]=(Integer)(carr)[i]+1;}


int
ifun (int k)
{
  int result;
  result = -k - 1;
  result = -2;
  return result;
}

int
ifun2 (int k)
{
  int result;
  result = k + 1;
  result = -3;
  return result;
}

void
fill_func (int nelem, int type, void *buf)
{
  int i;


  switch (type)
    {
    case C_FLOAT:
      for (i = 0; i < nelem; i++)
	((float *) buf)[i] = (float) ifun (i);
      break;
    case C_LONG:
      for (i = 0; i < nelem; i++)
	((long *) buf)[i] = (long) ifun (i);
      break;
    case C_DBL:
      for (i = 0; i < nelem; i++)
	((double *) buf)[i] = (double) ifun (i);
      break;
    case C_DCPL:
      for (i = 0; i < 2 * nelem; i++)
	((double *) buf)[i] = (double) ifun (i);
      break;
    case C_INT:
      for (i = 0; i < nelem; i++)
	((int *) buf)[i] = ifun (i);
      break;
    default:
      ga_error (" wrong data type ", type);

    }
}

void
fill_func2 (int nelem, int type, void *buf)
{
  /* int i,size=MA_sizeof(MT_CHAR,type,1);*/

  int i;

  switch (type)
    {
    case C_FLOAT:
      for (i = 0; i < nelem; i++)
	((float *) buf)[i] = (float) ifun2 (i);
      break;
    case C_LONG:
      for (i = 0; i < nelem; i++)
	((long *) buf)[i] = (long) ifun2 (i);
      break;
    case C_DBL:
      for (i = 0; i < nelem; i++)
	((double *) buf)[i] = (double) ifun2 (i);
      break;
    case C_DCPL:
      for (i = 0; i < 2 * nelem; i++)
	((double *) buf)[i] = (double) ifun2 (i);
      break;
    case C_INT:
      for (i = 0; i < nelem; i++)
	((int *) buf)[i] = ifun2 (i);
      break;
    default:
      ga_error (" wrong data type ", type);

    }
}

void
fill_func3 (int nelem, int type, void *buf)
/*taking the absolute of the ifun() */
{
/*int i,size=MA_sizeof(MT_CHAR,type,1);*/

  int i;

  switch (type)
    {
    case C_FLOAT:
      for (i = 0; i < nelem; i++)
	((float *) buf)[i] = (float) ABS (ifun (i));
      break;
    case C_LONG:
      for (i = 0; i < nelem; i++)
	((long *) buf)[i] = (long) ABS (ifun (i));
      break;
    case C_DBL:
      for (i = 0; i < nelem; i++)
	((double *) buf)[i] = (double) ABS (ifun (i));
      break;
    case C_DCPL:
      for (i = 0; i < 2 * nelem - 1; i = i + 2)
	{
	  ((double *) buf)[i] =
	    sqrt ((double)
		  (ifun (i) * ifun (i) + ifun (i + 1) * ifun (i + 1)));
	  ((double *) buf)[i + 1] = 0.0;
	}
      break;
    case C_INT:
      for (i = 0; i < nelem; i++)
	((int *) buf)[i] = ABS (ifun (i));
      break;
    default:
      ga_error (" wrong data type ", type);

    }
}






int
test_fun (int type, int dim, int OP)
{
  int ONE = 1, ZERO = 0;	/* useful constants */
  int g_a, g_b, g_c, g_d, g_e;
  int n = N;
  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  int col, i, row;
  int dims[MAXDIM];
  int lo[MAXDIM], hi[MAXDIM];
  int index[MAXDIM];
  void *val;
  int ival = -2;
  double dval = -2.0;
  float fval = -2.0;
  long lval = -2;
  DoubleComplex dcval;
  void *val2;
  int ival2 = -3;
  double dval2 = -3.0;
  float fval2 = -3.0;
  long lval2 = -3;
  DoubleComplex dcval2;
  int ok = 1;
  int result;
  void *min, *max;
  int imin, imax;
  float fmin, fmax;
  long lmin, lmax;
  double dmin, dmax;
  DoubleComplex dcmin, dcmax;


  void *alpha, *beta;
  int ai = 1, bi = -1;
  long al = 1, bl = -1;
  float af = 1.0, bf = -1.0;
  double ad = 1.0, bd = -1.0;
  DoubleComplex adc, bdc;

  adc.real = 1.0;
  adc.imag = 0.0;
  bdc.real = -1.0;
  bdc.imag = 0.0;


  dcval.real = -sin (3.0);
  dcval.imag = -cos (3.0);
  dcval2.real = 2 * sin (3.0);
  dcval2.imag = 2 * cos (3.0);

  for (i = 0; i < dim; i++)
    dims[i] = N;

  for (i = 0; i < dim; i++)
    {
      lo[i] = 0;
      hi[i] = N - 1;
    }
  g_a = NGA_Create (type, dim, dims, "A", NULL);
  if (!g_a)
    GA_Error ("create failed: A", n);

  g_b = GA_Duplicate (g_a, "B");
  if (!g_b)
    GA_Error ("duplicate failed: B", n);

  g_c = GA_Duplicate (g_a, "C");
  if (!g_c)
    GA_Error ("duplicate failed: C", n);

  g_d = GA_Duplicate (g_a, "D");
  if (!g_d)
    GA_Error ("duplicate failed: D", n);

  g_e = GA_Duplicate (g_a, "E");
  if (!g_e)
    GA_Error ("duplicate failed: E", n);

  /*initialize  with zero */
  GA_Zero (g_a);
  GA_Zero (g_b);
  GA_Zero (g_c);
  GA_Zero (g_d);
  GA_Zero (g_e);

  switch (type)
    {
    case C_INT:
      val = &ival;
      val2 = &ival2;
      break;
    case C_DCPL:
      val = &dcval;
      val2 = &dcval2;
      break;

    case C_DBL:
      val = &dval;
      val2 = &dval2;
      break;
    case C_FLOAT:
      val = &fval;
      val2 = &fval2;
      break;
    case C_LONG:
      val = &lval;
      val2 = &lval2;
      break;
    default:
      ga_error ("wrong data type.", type);
    }


  NGA_Fill_patch (g_a, lo, hi, val);

  switch (OP)
    {
      double tmp, tmp2;
      DoubleComplex dctemp;
    case OP_ABS:
      if (me == 0)
	printf ("Testing GA_Abs_value...");
      GA_Abs_value_patch (g_a, lo, hi);
      ival = ABS (ival);
      dval = ABS (dval);
      fval = ABS (fval);
      lval = ABS (lval);
      dcval.real = dcval.real * dcval.real + dcval.imag * dcval.imag;
      dcval.imag = 0.0;
      NGA_Fill_patch (g_d, lo, hi, val);
      break;
    case OP_ADD_CONST:
      if (me == 0)
	printf ("Testing GA_Add_const...");
      GA_Add_constant_patch (g_a, lo, hi, val2);
      ival = ival + ival2;
      dval = dval + dval2;
      fval = fval + fval2;
      lval = lval + lval2;
      dcval.real = dcval.real + dcval2.real;
      dcval.imag = dcval.imag + dcval2.imag;
      NGA_Fill_patch (g_d, lo, hi, val);
      break;
    case OP_RECIP:
      if (me == 0)
	printf ("Testing GA_Recip...");
      GA_Recip_patch (g_a, lo, hi);
      ival = 1 / ival;
      dval = 1.0 / dval;
      fval = 1.0 / fval;
      lval = 1 / lval;
      tmp = dcval.real * dcval.real + dcval.imag * dcval.imag;
      dcval.real = dcval.real / tmp;
      dcval.imag = -dcval.imag / tmp;
      NGA_Fill_patch (g_d, lo, hi, val);
      break;
    case OP_ELEM_MULT:
      if (me == 0)
	printf ("Testin GA_Elem_multiply...");
      NGA_Fill_patch (g_b, lo, hi, val2);
#if 0
      /* g_c is different from g_a or g_b*/
      GA_Elem_multiply_patch (g_a, lo, hi, g_b, lo, hi, g_c, lo, hi);
#else
      /* g_c is g_b */
      GA_Elem_multiply_patch (g_a, lo, hi, g_b, lo, hi, g_b, lo, hi);

#endif
      ival = ival * ival2;
      dval = dval * dval2;
      fval = fval * fval2;
      lval = lval * lval2;
      dctemp.real = dcval.real * dcval2.real - dcval.imag * dcval2.imag;
      dctemp.imag = dcval.real * dcval2.imag + dcval2.real * dcval.imag;
      dcval = dctemp;
      NGA_Fill_patch (g_d, lo, hi, val);
      break;
    case OP_ELEM_DIV:
      if (me == 0)
	printf ("Testin GA_Elem_divide...");
      NGA_Fill_patch (g_b, lo, hi, val2);
      GA_Elem_divide_patch (g_a, lo, hi, g_b, lo, hi, g_c, lo, hi);
      ival = ival / ival2;
      dval = dval / dval2;
      fval = fval / fval2;
      lval = lval / lval2;
      tmp = dcval2.real * dcval2.real + dcval2.imag * dcval2.imag;
      dctemp.real =
	(dcval.real * dcval2.real + dcval.imag * dcval2.imag) / tmp;
      dctemp.imag =
	(-dcval.real * dcval2.imag + dcval2.real * dcval.imag) / tmp;
      dcval = dctemp;
      NGA_Fill_patch (g_d, lo, hi, val);
      break;

    case OP_ELEM_MAX:
      if (me == 0)
	printf ("Testin GA_Elem_maximum...");
      NGA_Fill_patch (g_b, lo, hi, val2);
      GA_Elem_maximum_patch (g_a, lo, hi, g_b, lo, hi, g_c, lo, hi);
      ival = MAX (ival, ival2);
      dval = MAX (dval, dval2);
      fval = MAX (fval, fval2);
      lval = MAX (lval, lval2);
      tmp = dcval.real * dcval.real + dcval.imag * dcval.imag;
      tmp2 = dcval2.real * dcval2.real + dcval2.imag * dcval2.imag;
      if (tmp2 > tmp)
	dcval = dcval2;
      NGA_Fill_patch (g_d, lo, hi, val);
      break;
    case OP_ELEM_MIN:
      if (me == 0)
	printf ("Testin GA_Elem_minimum...");
      NGA_Fill_patch (g_b, lo, hi, val2);
      GA_Elem_minimum_patch (g_a, lo, hi, g_b, lo, hi, g_c, lo, hi);
      ival = MIN (ival, ival2);
      dval = MIN (dval, dval2);
      fval = MIN (fval, fval2);
      lval = MIN (lval, lval2);
      tmp = dcval.real * dcval.real + dcval.imag * dcval.imag;
      tmp2 = dcval2.real * dcval2.real + dcval2.imag * dcval2.imag;
      if (tmp2 < tmp)
	dcval = dcval2;
      NGA_Fill_patch (g_d, lo, hi, val);
      break;
    default:
      GA_Error ("test_function: wrong operation.", OP);

    }
  switch (type)
    {
    case C_INT:
      alpha = &ai;
      beta = &bi;
      break;
    case C_DCPL:
      alpha = &adc;
      beta = &bdc;
      break;

    case C_DBL:
      alpha = &ad;
      beta = &bd;
      break;
    case C_FLOAT:
      alpha = &af;
      beta = &bf;
      break;
    case C_LONG:
      alpha = &al;
      beta = &bl;
      break;
    default:
      ga_error ("wrong data type.", type);
    }

  if (OP < 4)
    NGA_Add_patch (alpha, g_c, lo, hi, beta, g_d, lo, hi, g_e, lo, hi);
  else
    NGA_Add_patch (alpha, g_a, lo, hi, beta, g_d, lo, hi, g_e, lo, hi);

  switch (type)
    {
    case C_INT:
      max = &lmax;
      min = &lmin;
      break;
    case C_DCPL:
      max = &dcmax;
      min = &dcmin;
      break;
    case C_DBL:
      max = &dmax;
      min = &dmin;
      break;
    case C_FLOAT:
      max = &fmax;
      min = &fmin;
      break;
    case C_LONG:
      max = &lmax;
      min = &lmin;
      break;
    default:
      ga_error ("wrong data type.", type);
    }

  NGA_Select_elem (g_e, "max", max, index);
  NGA_Select_elem (g_e, "min", min, index);

  switch (type)
    {
      double r, im, tmp;
    case C_INT:
      result = (int)(lmax - lmin);
      break;
    case C_DCPL:
      r = dcmax.real - dcmin.real;
      im = dcmax.imag - dcmin.imag;
      result = (int) (ABS (r) + ABS (im));
      break;
    case C_DBL:
      result = (int) (dmax - dmin);
      break;
    case C_FLOAT:
      result = (int) (fmax - fmin);
      break;
    case C_LONG:
      result = (int) (lmax - lmin);
      break;
    default:
      ga_error ("wrong data type.", type);
    }


  if (me == 0)
    {
      if (MISMATCHED (result, 0))
	printf ("is not ok\n");
      else
	printf ("is ok.\n");
    }

/*
 NGA_Print_patch(g_a, lo, hi, 1);
 NGA_Print_patch(g_d, lo, hi, 1);
 NGA_Print_patch(g_e, lo, hi, 1);
*/

  GA_Destroy (g_a);
  GA_Destroy (g_b);
  GA_Destroy (g_c);
  GA_Destroy (g_d);

  return ok;
}

int
main (argc, argv)
     int argc;
     char **argv;
{
  int heap = 20000, stack = 20000;
  int me, nproc;
  int d, op;
  int ok = 1;

#ifdef MPI
  MPI_Init (&argc, &argv);	/* initialize MPI */
#else
  PBEGIN_ (argc, argv);		/* initialize TCGMSG */
#endif

  GA_Initialize ();		/* initialize GA */
  me = GA_Nodeid ();
  nproc = GA_Nnodes ();
  if (me == 0)
    {
      if (GA_Uses_fapi ())
	GA_Error ("Program runs with C array API only", 0);
      printf ("Using %ld processes\n", (long) nproc);
      fflush (stdout);
    }

  heap /= nproc;
  stack /= nproc;
  if (!MA_init (C_DBL, stack, heap))
    GA_Error ("MA_init failed", stack + heap);	/* initialize memory allocator */


  /* op = 6;*/
  for (op = 0; op < 7; op++)
    {
      for (d = 1; d < 4; d++)
	{
	  if (me == 0)
	    printf ("\n\ndim =%d\n\n", d);
	  if (me == 0)
	    printf ("\ndata type: INT\t\t");
	  ok = test_fun (C_INT, d, op);
	  if (me == 0)
	    printf ("\ndata type: double\t");
	  ok = test_fun (C_DBL, d, op);
	  if (me == 0)
	    printf ("\ndata type: float\t");
	  ok = test_fun (C_FLOAT, d, op);

	  if (me == 0)
	    printf ("\ndata type: long\t\t");
	  ok = test_fun (C_LONG, d, op);
	  if (me == 0)
	    printf ("\ndata type: complex\t");
	  test_fun (C_DCPL, d, op);
	}
    }

  GA_Terminate();
  
#ifdef MPI
  MPI_Finalize ();
#else
  PEND_ ();
#endif

  return 0;
}
