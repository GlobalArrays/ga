/*File: mtest.c 

Anthor: Limin Zhang
purpose: testing those matrix functions developed for TAO/Global Array project.

Date: 2/28/2002

*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "ga.h"
#include "../src/globalp.h"
#ifdef MPI
#include <mpi.h>
#else
#include "sndrcv.h"
#endif

#define N 4			/* dimension of matrices */

#define OP_SHIFT_DIAGONAL 1
#define OP_SET_DIAGONAL 2
#define OP_ADD_DIAGONAL         3
#define OP_GET_DIAGONAL 4
#define OP_NORM1	5
#define OP_NORM_INFINITY	6
#define OP_MEDIAN               7
#define OP_MEDIAN_PATCH         8
#define OP_SCALE_ROWS           9
#define OP_SCALE_COLS           10

# define THRESH 1e-5
#define MISMATCHED(x,y) ABS((x)-(y))>=THRESH


void  test_scale_cols (int g_a, int g_v)
{

  int index[MAXDIM];
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
  DoubleComplex adc = { 1.0, 0.0 }, bdc =
    {
      -1.0, 0.0};

  int g_b, g_c;
  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  void *val;
  int ival = -2;
  double dval = -2.0;
  float fval = -2.0;
  long lval = -2;
  DoubleComplex dcval = { -2.0, 0.0 };
  void *val2;
  int ival2 = 4;
  double dval2 = 4.0;
  float fval2 = 4.0;
  long lval2 = 4;
  DoubleComplex dcval2 = { 4.0, 0.0 };

  int idot, iresult, ldot, lresult;
  double fdot, ddot, fresult, dresult;
  DoubleComplex zdot, zresult;

  int type, ndim, dims[MAXDIM];
  int vtype, vndim, vdims[MAXDIM];


  int lo[2], hi[2], n, col, i, size;
  int vlo, vhi;
  void *buf;



  NGA_Inquire (g_a, &type, &ndim, dims);
  NGA_Inquire (g_v, &vtype, &vndim, vdims);


  switch (type)
    {
    case C_INT:
      alpha = (void *) &ai;
      beta = (void *) &bi;
      break;
    case C_DCPL:
      alpha = (void *)&adc;
      beta = (void *)&bdc;
      break;

    case C_DBL:
      alpha = (void *) &ad;
      beta = (void *) &bd;
      break;
    case C_FLOAT:
      alpha = (void *) &af;
      beta =(void *)  &bf;
      break;
    case C_LONG:
      alpha = (void *) &al;
      beta = (void *) &bl;
      break;
    default:
      ga_error ("test_scale_cols:wrong data type.", type);
    }

  switch (type)
    {
    case C_INT:
      val = (void *)&ival;
      val2 = (void *)&ival2;
      break;
    case C_DCPL:
      val = (void *)&dcval;
      val2 = (void *)&dcval2;
      break;
    case C_DBL:
      val = (void *)&dval;
      val2 = (void *)&dval2;
      break;
    case C_FLOAT:
      val = (void *)&fval;
      val2 = (void *)&fval2;
      break;
    case C_LONG:
      val = (void *)&lval;
      val2 = (void *)&lval2;
      break;
    default:
      ga_error ("test_scale_cols:wrong data type.", type);
    }

  if (me == 0)
    printf ("Testing GA_Scale_cols...");


  /*
    n = vdims[0];
    printf("n=%d\n", n);
    size = GAsizeof (type);
    buf = malloc(n*size);
    if(me==0)printf("Initializing matrix A\n");
    vlo = 0;
    vhi = n-1;
    col = 0; 
    for(i=0; i<n; i++)
    switch(type){
    case C_INT:
    ((int*)buf)[i]=col+i;
    break;
    case C_LONG:
    ((long*)buf)[i]=(long)(col+i);
    break;
    case C_DBL:
    ((double*)buf)[i]=(double)(i+col);
    break;
    case C_FLOAT:
    ((float*)buf)[i]=(float)(col+i);
    break;
    case C_DCPL:
    ((DoubleComplex*)buf)[i].real=(double)(i+col);
    ((DoubleComplex*)buf)[i].imag=(double)0.0;
    break;
    default:
    ga_error("test_scale:wrong data type.", type);
    }
    NGA_Put(g_v, &vlo, &vhi, buf, &n);


    n = dims[1];
    printf("dims[0]=%d \t dims[1]=%d\n", dims[0], dims[1]);
    printf("n=%d\n", n);
    size = GAsizeof (type);
    buf = malloc(n*size);
    if(me==0)printf("Initializing matrix A\n");
    lo[1]=0; hi[1]=n-1;
    for(col=me; col<dims[0]; col+= nproc){
    lo[0]=hi[0]=col;
    for(i=0; i<n; i++)
    switch(type){
    case C_INT:
    ((int*)buf)[i]=col+i;
    break;
    case C_LONG:
    ((long*)buf)[i]=(long)(col+i);
    break;
    case C_DBL:
    ((double*)buf)[i]=(double)(i+col);
    break;
    case C_FLOAT:
    ((float*)buf)[i]=(float)(col+i);
    break;
    case C_DCPL:
    ((DoubleComplex*)buf)[i].real=(double)(i+col);
    ((DoubleComplex*)buf)[i].imag=0.0;
    break;
    default:
    ga_error("test_scale_cols:wrong data type.", type);
    }
    NGA_Put(g_a, lo, hi, buf, &n);
    }

 
 
    free(buf);
    buf = NULL; 
  */

  GA_Fill (g_a, val);
  GA_Fill (g_v, val);
  GA_Scale_cols (g_a, g_v);
  /*the result is the same same as g_b filled with val2 */
  g_b = GA_Duplicate (g_a, "B");
  if (!g_b)
    GA_Error ("test_scale_cols:duplicate failed: B", n);

  g_c = GA_Duplicate (g_a, "C");
  if (!g_c)
    GA_Error ("duplicate failed: C", n);

  GA_Fill (g_b, val2);

  GA_Add (alpha, g_a, beta, g_b, g_c);

  switch (type)
    {
    case C_INT:
      max = (void *)&lmax;
      min = (void *)&lmin;
      break;
    case C_DCPL:
      max = (void *)&dcmax;
      min = (void *)&dcmin;
      break;
    case C_DBL:
      max = (void *)&dmax;
      min = (void *)&dmin;
      break;
    case C_FLOAT:
      max = (void *)&fmax;
      min = (void *)&fmin;
      break;
    case C_LONG:
      max = (void *)&lmax;
      min = (void *)&lmin;
      break;
    default:
      ga_error ("test_scale_rows:wrong data type.", type);
    }

  NGA_Select_elem (g_c, "max", max, index);
  NGA_Select_elem (g_c, "min", min, index);


  switch (type)
    {
      double r, m, tmp;
    case C_INT:
      if (me == 0)
	{
	  if (MISMATCHED (lmax, lmin) || (lmax != 0) || (lmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DCPL:
      r = dcmax.real - dcmin.real;
      m = dcmax.imag - dcmin.imag;
      if (me == 0)
	{
	  if (MISMATCHED (dcmax.real, dcmin.real) || (dcmax.real != 0.0)
	      || (dcmin.real != 0.0) || MISMATCHED (dcmax.imag, dcmin.imag)
	      || (dcmax.imag != 0.0) || (dcmin.imag != 0.0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DBL:
      if (me == 0)
	{
	  if (MISMATCHED (dmax, dmin) || (dmax != 0) || (dmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_FLOAT:
      if (me == 0)
	{
	  if (MISMATCHED (fmax, fmin) || (fmax != 0) || (fmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_LONG:
      if (me == 0)
	{
	  if (MISMATCHED (lmax, lmin) || (lmax != 0) || (lmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    default:
      ga_error ("test_scale_rows:wrong data type.", type);
    }
}



void
test_scale_rows (int g_a, int g_v)
{

  int index[MAXDIM];
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
  DoubleComplex adc = { 1.0, 0.0 }, bdc =
    {
      -1.0, 0.0};

  int g_b, g_c;
  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  void *val;
  int ival = -2;
  double dval = -2.0;
  float fval = -2.0;
  long lval = -2;
  DoubleComplex dcval = { -2.0, 0.0 };
  void *val2;
  int ival2 = 4;
  double dval2 = 4.0;
  float fval2 = 4.0;
  long lval2 = 4;
  DoubleComplex dcval2 = { 4.0, 0.0 };

  int idot, iresult, ldot, lresult;
  double fdot, ddot, fresult, dresult;
  DoubleComplex zdot, zresult;

  int type, ndim, dims[MAXDIM];
  int vtype, vndim, vdims[MAXDIM];


  int lo[2], hi[2], n, col, i, size;
  int vlo, vhi;
  void *buf;



  NGA_Inquire (g_a, &type, &ndim, dims);
  NGA_Inquire (g_v, &vtype, &vndim, vdims);


  switch (type)
    {
    case C_INT:
      alpha = (void *) &ai;
      beta = (void *) &bi;
      break;
    case C_DCPL:
      alpha = (void *)&adc;
      beta = (void *)&bdc;
      break;

    case C_DBL:
      alpha = (void *) &ad;
      beta = (void *) &bd;
      break;
    case C_FLOAT:
      alpha = (void *)&af;
      beta = (void *)&bf;
      break;
    case C_LONG:
      alpha = (void *)&al;
      beta = (void *)&bl;
      break;
    default:
      ga_error ("test_scale_rows:wrong data type.", type);
    }

  switch (type)
    {
    case C_INT:
      val = (void *) &ival;
      val2 = (void *) &ival2;
      break;
    case C_DCPL:
      val = (void *) &dcval;
      val2 = (void *) &dcval2;
      break;
    case C_DBL:
      val = (void *) &dval;
      val2 = (void *) &dval2;
      break;
    case C_FLOAT:
      val = (void *) &fval;
      val2 = (void *) &fval2;
      break;
    case C_LONG:
      val = (void *) &lval;
      val2 = (void *) &lval2;
      break;
    default:
      ga_error ("test_scale_rows:wrong data type.", type);
    }

  if (me == 0)
    printf ("Testing GA_Scale_rows...");


  /*
    n = vdims[0];
    printf("n=%d\n", n);
    size = GAsizeof (type);
    buf = malloc(n*size);
    if(me==0)printf("Initializing matrix A\n");
    vlo = 0;
    vhi = n-1;
    col = 0; 
    for(i=0; i<n; i++)
    switch(type){
    case C_INT:
    ((int*)buf)[i]=col+i;
    break;
    case C_LONG:
    ((long*)buf)[i]=(long)(col+i);
    break;
    case C_DBL:
    ((double*)buf)[i]=(double)(i+col);
    break;
    case C_FLOAT:
    ((float*)buf)[i]=(float)(col+i);
    break;
    case C_DCPL:
    ((DoubleComplex*)buf)[i].real=(double)(i+col);
    ((DoubleComplex*)buf)[i].imag=(double)0.0;
    break;
    default:
    ga_error("test_scale:wrong data type.", type);
    }
    NGA_Put(g_v, &vlo, &vhi, buf, &n);


    n = dims[1];
    printf("dims[0]=%d \t dims[1]=%d\n", dims[0], dims[1]);
    printf("n=%d\n", n);
    size = GAsizeof (type);
    buf = malloc(n*size);
    if(me==0)printf("Initializing matrix A\n");
    lo[1]=0; hi[1]=n-1;
    for(col=me; col<dims[0]; col+= nproc){
    lo[0]=hi[0]=col;
    for(i=0; i<n; i++)
    switch(type){
    case C_INT:
    ((int*)buf)[i]=col+i;
    break;
    case C_LONG:
    ((long*)buf)[i]=(long)(col+i);
    break;
    case C_DBL:
    ((double*)buf)[i]=(double)(i+col);
    break;
    case C_FLOAT:
    ((float*)buf)[i]=(float)(col+i);
    break;
    case C_DCPL:
    ((DoubleComplex*)buf)[i].real=(double)(i+col);
    ((DoubleComplex*)buf)[i].imag=(double)0.0;
    break;
    default:
    ga_error("test_scale:wrong data type.", type);
    }
    NGA_Put(g_a, lo, hi, buf, &n);
    }

 
 
    free(buf);
    buf = NULL; 
    GA_Print(g_v);

  */
  GA_Fill (g_a, val);
  GA_Fill (g_v, val);
  GA_Scale_rows (g_a, g_v);

  /*the result is the same same as g_b filled with val2 */
  g_b = GA_Duplicate (g_a, "B");
  if (!g_b)
    GA_Error ("duplicate failed: B", n);

  g_c = GA_Duplicate (g_a, "C");
  if (!g_c)
    GA_Error ("duplicate failed: C", n);

  GA_Fill (g_b, val2);

  GA_Add (alpha, g_a, beta, g_b, g_c);
  switch (type)
    {
    case C_INT:
      max = (void *) &lmax;
      min = (void *) &lmin;
      break;
    case C_DCPL:
      max = (void *) &dcmax;
      min = (void *) &dcmin;
      break;
    case C_DBL:
      max = (void *) &dmax;
      min = (void *) &dmin;
      break;
    case C_FLOAT:
      max = (void *) &fmax;
      min = (void *) &fmin;
      break;
    case C_LONG:
      max = (void *) &lmax;
      min = (void *) &lmin;
      break;
    default:
      ga_error ("test_scale_rows:wrong data type.", type);
    }

  NGA_Select_elem (g_c, "max", max, index);
  NGA_Select_elem (g_c, "min", min, index);


  switch (type)
    {
      double r, m, tmp;
    case C_INT:
      if (me == 0)
	{
	  if (MISMATCHED (lmax, lmin) || (lmax != 0) || (lmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DCPL:
      r = dcmax.real - dcmin.real;
      m = dcmax.imag - dcmin.imag;
      if (me == 0)
	{
	  if (MISMATCHED (dcmax.real, dcmin.real) || (dcmax.real != 0.0)
	      || (dcmin.real != 0.0) || MISMATCHED (dcmax.imag, dcmin.imag)
	      || (dcmax.imag != 0.0) || (dcmin.imag != 0.0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DBL:
      if (me == 0)
	{
	  if (MISMATCHED (dmax, dmin) || (dmax != 0) || (dmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_FLOAT:
      if (me == 0)
	{
	  if (MISMATCHED (fmax, fmin) || (fmax != 0) || (fmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_LONG:
      if (me == 0)
	{
	  if (MISMATCHED (lmax, lmin) || (lmax != 0) || (lmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    default:
      ga_error ("test_scale_rows:wrong data type.", type);
    }

}

void
test_median_patch (int g_a, int *alo, int *ahi, int g_b, int *blo, int *bhi,
		   int g_c, int *clo, int *chi, int g_m, int *mlo, int *mhi)
{

  int g_e;
  int index[MAXDIM];
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
  DoubleComplex adc = { 1.0, 0.0 }, bdc =
    {
      -1.0, 0.0};

  void *val;
  int ival = -2;
  double dval = -2.0;
  float fval = -2.0;
  long lval = -2;
  DoubleComplex dcval;

  void *val2;
  int ival2 = 6;
  double dval2 = 6.0;
  float fval2 = 6.0;
  long lval2 = 6;
  DoubleComplex dcval2;


  void *val3;
  int ival3 = 4;
  double dval3 = 4.0;
  float fval3 = 4.0;
  long lval3 = 4;
  DoubleComplex dcval3;

  int idot, iresult, ldot, lresult;
  double fdot, ddot, fresult, dresult;
  DoubleComplex zdot, zresult;

  double norm_infinity = -1.0, result = -1.0;

  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  int type, ndim, dims[MAXDIM];

  NGA_Inquire (g_a, &type, &ndim, dims);


  switch (type)
    {
    case C_INT:
      alpha = (void *) &ai;
      beta = (void *) &bi;
      break;
    case C_DCPL:
      alpha = (void *) &adc;
      beta = (void *) &bdc;
      break;

    case C_DBL:
      alpha = (void *) (void *) &ad;
      beta = (void *) (void *) &bd;
      break;
    case C_FLOAT:
      alpha = (void *) &af;
      beta = (void *) &bf;
      break;
    case C_LONG:
      alpha = (void *) &al;
      beta = (void *) &bl;
      break;
    default:
      ga_error ("test_median:wrong data type.", type);
    }

  dcval.real = -2.0;
  dcval.imag = -0.0;

  dcval2.real = 6.0;
  dcval2.imag = 0.0;


  dcval3.real = 4.0;
  dcval3.imag = 0.0;


  switch (type)
    {
    case C_INT:
      val = (void *) &ival;
      val2 = (void *) &ival2;
      val3 = (void *) &ival3;
      break;
    case C_DCPL:
      val = (void *) &dcval;
      val2 = (void *) &dcval2;
      val3 = (void *) &dcval3;
      break;
    case C_DBL:
      val = (void *) &dval;
      val2 = (void *) &dval2;
      val3 = (void *) &dval3;
      break;
    case C_FLOAT:
      val = (void *) &fval;
      val2 = (void *) &fval2;
      val3 = (void *) &fval3;
      break;
    case C_LONG:
      val = (void *) &lval;
      val2 = (void *) &lval2;
      val3 = (void *) &lval3;
      break;
    default:
      ga_error ("test_median:test_median:wrong data type.", type);
    }

  if (me == 0)
    printf ("Testing GA_Median_patch...");

  GA_Zero (g_a);
  GA_Zero (g_b);
  GA_Zero (g_c);
  GA_Zero (g_m);

  NGA_Fill_patch (g_a, alo, ahi, val);
  NGA_Fill_patch (g_b, blo, bhi, val2);
  NGA_Fill_patch (g_c, alo, bhi, val3);

  GA_Median_patch (g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi, g_m, mlo,
		   mhi);

  /*
    The result array should        be g_c due to the value I chose: 
    val3 is the median of the three values val, val2, and val3
  */

  /* g_e = g_c - g_m */
  g_e = GA_Duplicate (g_a, "E");
  if (!g_e)
    GA_Error ("duplicate failed: E", 4);
  GA_Zero (g_e);

  NGA_Add_patch (alpha, g_c, clo, chi, beta, g_m, mlo, mhi, g_e, alo, ahi);
  switch (type)
    {
    case C_INT:
      max = (void *)&lmax;
      min = (void *)&lmin;
      break;
    case C_DCPL:
      max = (void *)&dcmax;
      min = (void *)&dcmin;
      break;
    case C_DBL:
      max = (void *)&dmax;
      min = (void *)&dmin;
      break;
    case C_FLOAT:
      max = (void *)&fmax;
      min = (void *)&fmin;
      break;
    case C_LONG:
      max = (void *)&lmax;
      min = (void *)&lmin;
      break;
    default:
      ga_error ("test_median:wrong data type.", type);
    }

  NGA_Select_elem (g_e, "max", max, index);
  NGA_Select_elem (g_e, "min", min, index);


  switch (type)
    {
      double r, m, tmp;
    case C_INT:
      if (me == 0)
	{
	  if (MISMATCHED (lmax, lmin) || (lmax != 0) || (lmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DCPL:
      r = dcmax.real - dcmin.real;
      m = dcmax.imag - dcmin.imag;
      if (me == 0)
	{
	  if (MISMATCHED (dcmax.real, dcmin.real) || (dcmax.real != 0.0)
	      || (dcmin.real != 0.0) || MISMATCHED (dcmax.imag, dcmin.imag)
	      || (dcmax.imag != 0.0) || (dcmin.imag != 0.0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DBL:
      if (me == 0)
	{
	  if (MISMATCHED (dmax, dmin) || (dmax != 0) || (dmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_FLOAT:
      if (me == 0)
	{
	  if (MISMATCHED (fmax, fmin) || (fmax != 0) || (fmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_LONG:
      if (me == 0)
	{
	  if (MISMATCHED (lmax, lmin) || (lmax != 0) || (lmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    default:
      ga_error ("test_median:wrong data type.", type);
    }


}


void
test_median (int g_a, int g_b, int g_c, int g_m)
{

  int g_e;
  int index[MAXDIM];
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
  DoubleComplex adc = { 1.0, 0.0 }, bdc =
    {
      -1.0, 0.0};

  void *val;
  int ival = -2;
  double dval = -2.0;
  float fval = -2.0;
  long lval = -2;
  DoubleComplex dcval;

  void *val2;
  int ival2 = 6;
  double dval2 = 6.0;
  float fval2 = 6.0;
  long lval2 = 6;
  DoubleComplex dcval2;


  void *val3;
  int ival3 = 4;
  double dval3 = 4.0;
  float fval3 = 4.0;
  long lval3 = 4;
  DoubleComplex dcval3;

  int idot, iresult, ldot, lresult;
  double fdot, ddot, fresult, dresult;
  DoubleComplex zdot, zresult;

  double norm_infinity = -1.0, result = -1.0;

  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  int type, ndim, dims[MAXDIM];

  NGA_Inquire (g_a, &type, &ndim, dims);

  switch (type)
    {
    case C_INT:
      alpha = (void *) &ai;
      beta = (void *) &bi;
      break;
    case C_DCPL:
      alpha = (void *)&adc;
      beta =(void *) &bdc;
      break;

    case C_DBL:
      alpha = (void *) &ad;
      beta = (void *) &bd;
      break;
    case C_FLOAT:
      alpha = (void *)&af;
      beta =(void *) &bf;
      break;
    case C_LONG:
      alpha = (void *)&al;
      beta = (void *)&bl;
      break;
    default:
      ga_error ("test_median:wrong data type.", type);
    }

  dcval.real = -2.0;
  dcval.imag = -0.0;

  dcval2.real = 6.0;
  dcval2.imag = 0.0;


  dcval3.real = 4.0;
  dcval3.imag = 0.0;


  switch (type)
    {
    case C_INT:
      val = (void *)&ival;
      val2 = (void *)&ival2;
      val3 = (void *)&ival3;
      break;
    case C_DCPL:
      val = (void *)&dcval;
      val2 = (void *)&dcval2;
      val3 = (void *)&dcval3;
      break;
    case C_DBL:
      val = (void *)&dval;
      val2 = (void *)&dval2;
      val3 = (void *)&dval3;
      break;
    case C_FLOAT:
      val = (void *)&fval;
      val2 = (void *)&fval2;
      val3 = (void *)&fval3;
      break;
    case C_LONG:
      val = (void *)&lval;
      val2 = (void *)&lval2;
      val3 = (void *)&lval3;
      break;
    default:
      ga_error ("test_median:test_median:wrong data type.", type);
    }

  if (me == 0)
    printf ("Testing GA_Median...");

  GA_Zero (g_a);
  GA_Zero (g_b);
  GA_Zero (g_c);
  GA_Zero (g_m);

  GA_Fill (g_a, val);
  GA_Fill (g_b, val2);
  GA_Fill (g_c, val3);


  GA_Median (g_a, g_b, g_c, g_m);

  /*
    The result array should        be g_c due to the value I chose: 
    val3 is the median of the three values val, val2, and val3
  */

  /* g_e = g_c - g_m */
  g_e = GA_Duplicate (g_a, "E");
  if (!g_e)
    GA_Error ("duplicate failed: E", 4);

  GA_Add (alpha, g_c, beta, g_m, g_e);
  switch (type)
    {
    case C_INT:
      max = (void *)&lmax;
      min = (void *)&lmin;
      break;
    case C_DCPL:
      max = (void *)&dcmax;
      min = (void *)&dcmin;
      break;
    case C_DBL:
      max = (void *)&dmax;
      min = (void *)&dmin;
      break;
    case C_FLOAT:
      max = (void *)&fmax;
      min = (void *)&fmin;
      break;
    case C_LONG:
      max = (void *)&lmax;
      min = (void *)&lmin;
      break;
    default:
      ga_error ("test_median:wrong data type.", type);
    }

  NGA_Select_elem (g_e, "max", max, index);
  NGA_Select_elem (g_e, "min", min, index);


  switch (type)
    {
      double r, m, tmp;
    case C_INT:
      if (me == 0)
	{
	  if (MISMATCHED (lmax, lmin) || (lmax != 0) || (lmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DCPL:
      r = dcmax.real - dcmin.real;
      m = dcmax.imag - dcmin.imag;
      if (me == 0)
	{
	  if (MISMATCHED (dcmax.real, dcmin.real) || (dcmax.real != 0.0)
	      || (dcmin.real != 0.0) || MISMATCHED (dcmax.imag, dcmin.imag)
	      || (dcmax.imag != 0.0) || (dcmin.imag != 0.0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DBL:
      if (me == 0)
	{
	  if (MISMATCHED (dmax, dmin) || (dmax != 0) || (dmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_FLOAT:
      if (me == 0)
	{
	  if (MISMATCHED (fmax, fmin) || (fmax != 0) || (fmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_LONG:
      if (me == 0)
	{
	  if (MISMATCHED (lmax, lmin) || (lmax != 0) || (lmin != 0))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    default:
      ga_error ("test_median:wrong data type.", type);
    }


}


void
test_norm_infinity (int g_a)
{

  void *val;
  int ival = -2;
  double dval = -2.0;
  float fval = -2.0;
  long lval = -2;
  DoubleComplex dcval;

  int idot, iresult, ldot, lresult;
  double fdot, ddot, fresult, dresult;
  DoubleComplex zdot, zresult;

  double norm_infinity = -1.0, result = -1.0;

  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  int type, ndim, dims[MAXDIM];

  NGA_Inquire (g_a, &type, &ndim, dims);
  dcval.real = -2.0;
  dcval.imag = -0.0;

  switch (type)
    {
    case C_INT:
      val = (void *)&ival;
      break;
    case C_DCPL:
      val = (void *)&dcval;
      break;
    case C_DBL:
      val = (void *)&dval;
      break;
    case C_FLOAT:
      val = (void *)&fval;
      break;
    case C_LONG:
      val = (void *)&lval;
      break;
    default:
      ga_error ("test_norm_infinity:wrong data type.", type);
    }

  if (me == 0)
    printf ("Testing GA_Norm_infinity...");
  GA_Fill (g_a, val);
  GA_Norm_infinity (g_a, &norm_infinity);
  /* GA_Print(g_a);
     printf("norm_infinity = %lf\n",norm_infinity);*/
  switch (type)
    {
    case C_INT:
      result = (double) ABS (ival);
      break;
    case C_LONG:
      result = (double) ABS (lval);
      break;
    case C_FLOAT:
      result = (double) ABS (fval);
      break;

    case C_DBL:
      result = ABS (dval);
      break;

    case C_DCPL:
      result = sqrt (dcval.real * dcval.real + dcval.imag * dcval.imag);
      break;
    default:
      ga_error ("test_norm_infinity: wrong data type.\n", type);
    }
  result = result * dims[0];
  if (me == 0)
    {
      if (MISMATCHED (result, norm_infinity))
	printf ("not ok.\n");
      else
	printf ("ok.\n");
    }
}


void
test_norm1 (int g_a)
{

  void *val;
  int ival = -2;
  double dval = -2.0;
  float fval = -2.0;
  long lval = -2;
  DoubleComplex dcval;

  int idot, iresult, ldot, lresult;
  double fdot, ddot, fresult, dresult;
  DoubleComplex zdot, zresult;

  double norm1 = 0.0, result = -1.0;

  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  int type, ndim, dims[MAXDIM];

  NGA_Inquire (g_a, &type, &ndim, dims);
  dcval.real = -2.0;
  dcval.imag = -0.0;

  switch (type)
    {
    case C_INT:
      val = (void *) &ival;
      break;
    case C_DCPL:
      val = (void *)&dcval;
      break;
    case C_DBL:
      val = (void *)&dval;
      break;
    case C_FLOAT:
      val = (void *)&fval;
      break;
    case C_LONG:
      val = (void *)&lval;
      break;
    default:
      ga_error ("test_norm1:wrong data type.", type);
    }

  if (me == 0)
    printf ("Testing GA_Norm1...");
  GA_Fill (g_a, val);
  GA_Norm1 (g_a, &norm1);
  /* GA_Print(g_a);
     printf("norm1=%lf\n", norm1);*/
  switch (type)
    {
    case C_INT:
      result = (double) ABS (ival);
      break;
    case C_LONG:
      result = (double) ABS (lval);
      break;
    case C_FLOAT:
      result = (double) ABS (fval);
      break;

    case C_DBL:
      result = ABS (dval);
      break;

    case C_DCPL:
      result = sqrt (dcval.real * dcval.real + dcval.imag * dcval.imag);
      break;
    default:
      ga_error ("test_norm1: wrong data type.\n", type);
    }
  result = result * dims[1];
  if (me == 0)
    {
      if (MISMATCHED (result, norm1))
	printf ("not ok.\n");
      else
	printf ("ok.\n");
    }
}


void
test_get_diagonal (int g_a, int g_v)
{

  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  void *val;
  int ival = -2;
  double dval = -2.0;
  float fval = -2.0;
  long lval = -2;
  DoubleComplex dcval;

  int idot, iresult, ldot, lresult;
  double fdot, ddot, fresult, dresult;
  DoubleComplex zdot, zresult;

  int type, ndim, dims[MAXDIM];
  int vtype, vndim, vdims[MAXDIM];

  NGA_Inquire (g_a, &type, &ndim, dims);
  NGA_Inquire (g_v, &vtype, &vndim, vdims);
  dcval.real = -2.0;
  dcval.imag = -0.0;

  switch (type)
    {
    case C_INT:
      val = (void *)&ival;
      break;
    case C_DCPL:
      val = (void *)&dcval;
      break;
    case C_DBL:
      val = (void *)&dval;
      break;
    case C_FLOAT:
      val = (void *)&fval;
      break;
    case C_LONG:
      val = (void *)&lval;
      break;
    default:
      ga_error ("test_get_diagonal:wrong data type.", type);
    }

  if (me == 0)
    printf ("Testing GA_Get_diag...");
  GA_Zero (g_v);
  GA_Fill (g_a, val);
  GA_Get_diag (g_a, g_v);
  switch (type)
    {
    case C_INT:
      idot = vdims[0] * ival * ival;
      iresult = GA_Idot (g_v, g_v);
      if (me == 0)
	{
	  if (MISMATCHED (idot, iresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_LONG:
      ldot = ((long) vdims[0]) * lval * lval;
      lresult = GA_Ldot (g_v, g_v);
      if (me == 0)
	{
	  if (MISMATCHED (ldot, lresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_FLOAT:
      fdot = ((float) vdims[0]) * fval * fval;
      fresult = GA_Fdot (g_v, g_v);
      if (me == 0)
	{
	  if (MISMATCHED (fdot, fresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DBL:
      ddot = ((double) vdims[0]) * dval * dval;
      dresult = GA_Ddot (g_v, g_v);
      if (me == 0)
	{
	  if (MISMATCHED (ddot, dresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DCPL:
      zdot.real =
	((double) vdims[0]) * (dcval.real * dcval.real -
			       dcval.imag * dcval.imag);
      zdot.imag = ((double) vdims[0]) * (2.0 * dcval.real * dcval.imag);
      zresult = GA_Zdot (g_v, g_v);
      if (me == 0)
	{
	  if (MISMATCHED (zdot.real, zresult.real)
	      || MISMATCHED (zdot.imag, zresult.imag))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    default:
      ga_error ("test_get_diagonal:wrong data type:", type);
    }




}


void
test_add_diagonal (int g_a, int g_v)
{


  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  void *val;
  int ival = -2;
  double dval = -2.0;
  float fval = -2.0;
  long lval = -2;
  DoubleComplex dcval;

  int idot, iresult, ldot, lresult;
  double fdot, ddot, fresult, dresult;
  DoubleComplex zdot, zresult;

  int type, ndim, dims[MAXDIM];
  int vtype, vndim, vdims[MAXDIM];


  NGA_Inquire (g_a, &type, &ndim, dims);
  dcval.real = -2.0;
  dcval.imag = -0.0;

  NGA_Inquire (g_v, &vtype, &vndim, vdims);

  switch (type)
    {
    case C_INT:
      val = (void *)&ival;
      break;
    case C_DCPL:
      val = (void *)&dcval;
      break;
    case C_DBL:
      val =(void *) &dval;
      break;
    case C_FLOAT:
      val = (void *)&fval;
      break;
    case C_LONG:
      val = (void *)&lval;
      break;
    default:
      ga_error ("test_add_diagonal:wrong data type.", type);
    }


  if (me == 0)
    printf ("Testing GA_Add_diagonal...");
  GA_Zero (g_a);
  GA_Fill (g_v, val);
  GA_Set_diagonal (g_a, g_v);

  /*reassign value to val */
  ival = 3;
  dval = 3.0;
  fval = 3.0;
  lval = 3;
  dcval.real = 3.0;
  dcval.imag = -0.0;

  /*refile the global array g_v */
  GA_Fill (g_v, val);

  /*Add g_v to the diagonal of g_a */
  GA_Add_diagonal (g_a, g_v);
  /*after this line, the g_a should only have 1 on the diagonal and zeros every where else */

  switch (type)
    {
    case C_INT:
      idot = vdims[0];
      iresult = GA_Idot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (idot, iresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_LONG:
      ldot = ((long) vdims[0]);
      lresult = GA_Ldot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (ldot, lresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_FLOAT:
      fdot = ((float) vdims[0]);
      fresult = GA_Fdot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (fdot, fresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DBL:
      ddot = (double) vdims[0];
      dresult = GA_Ddot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (ddot, dresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DCPL:
      zdot.real = ((double) vdims[0]);
      zdot.imag = 0.0;
      zresult = GA_Zdot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (zdot.real, zresult.real)
	      || MISMATCHED (zdot.imag, zresult.imag))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    default:
      ga_error ("test_add_diagonal:wrong data type:", type);
    }





}

void
test_set_diagonal (int g_a, int g_v)
{


  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  void *val;
  int ival = -2;
  double dval = -2.0;
  float fval = -2.0;
  long lval = -2;
  DoubleComplex dcval;

  int idot, iresult, ldot, lresult;
  double fdot, ddot, fresult, dresult;
  DoubleComplex zdot, zresult;

  int type, ndim, dims[MAXDIM];
  int vtype, vndim, vdims[MAXDIM];

  NGA_Inquire (g_a, &type, &ndim, dims);
  NGA_Inquire (g_v, &vtype, &vndim, vdims);
  dcval.real = -2.0;
  dcval.imag = -0.0;

  switch (type)
    {
    case C_INT:
      val = (void *)&ival;
      break;
    case C_DCPL:
      val = (void *)&dcval;
      break;
    case C_DBL:
      val = (void *)&dval;
      break;
    case C_FLOAT:
      val = (void *)&fval;
      break;
    case C_LONG:
      val = (void *)&lval;
      break;
    default:
      ga_error ("test_set_diagonal:wrong data type.", type);
    }


  if (me == 0)
    printf ("Testing GA_Set_diagonal...");
  GA_Zero (g_a);
  GA_Fill (g_v, val);
  GA_Set_diagonal (g_a, g_v);
  switch (type)
    {
    case C_INT:
      idot = vdims[0] * ival * ival;
      iresult = GA_Idot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (idot, iresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_LONG:
      ldot = ((long) vdims[0]) * lval * lval;
      lresult = GA_Ldot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (ldot, lresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_FLOAT:
      fdot = ((float) vdims[0]) * fval * fval;
      fresult = GA_Fdot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (fdot, fresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DBL:
      ddot = ((double) vdims[0]) * dval * dval;
      dresult = GA_Ddot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (ddot, dresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DCPL:
      zdot.real =
	((double) vdims[0]) * (dcval.real * dcval.real -
			       dcval.imag * dcval.imag);
      zdot.imag = ((double) dims[0]) * (2.0 * dcval.real * dcval.imag);
      zresult = GA_Zdot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (zdot.real, zresult.real)
	      || MISMATCHED (zdot.imag, zresult.imag))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    default:
      ga_error ("test_set_diagonal:wrong data type:", type);
    }

}

void
test_shift_diagonal (int g_a)
{

  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  void *val;
  int ival = -2;
  double dval = -2.0;
  float fval = -2.0;
  long lval = -2;
  DoubleComplex dcval;

  int idot, iresult, ldot, lresult;
  double fdot, ddot, fresult, dresult;
  DoubleComplex zdot, zresult;
  int type, ndim, dims[MAXDIM];
  int dim;			/*the length of the diagonal */


  NGA_Inquire (g_a, &type, &ndim, dims);

  dim = MIN (dims[0], dims[1]);

  dcval.real = -2.0;
  dcval.imag = -0.0;
  switch (type)
    {
    case C_INT:
      val = (void *)&ival;
      break;
    case C_DCPL:
      val = (void *)&dcval;
      break;
    case C_DBL:
      val = (void *)&dval;
      break;
    case C_FLOAT:
      val = (void *)&fval;
      break;
    case C_LONG:
      val = (void *)&lval;
      break;
    default:
      ga_error ("test_shift_diagonal:wrong data type.", type);
    }


  if (me == 0)
    printf ("Testing GA_Shift_diagonal...");
  GA_Zero (g_a);
  GA_Shift_diagonal (g_a, val);

  switch (type)
    {
    case C_INT:
      idot = dim * ival * ival;
      iresult = GA_Idot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (idot, iresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_LONG:
      ldot = ((long) dim) * lval * lval;
      lresult = GA_Ldot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (ldot, lresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_FLOAT:
      fdot = ((float) dim) * fval * fval;
      fresult = GA_Fdot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (fdot, fresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DBL:
      ddot = ((double) dim) * dval * dval;
      dresult = GA_Ddot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (ddot, dresult))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    case C_DCPL:
      zdot.real =
	((double) dim) * (dcval.real * dcval.real - dcval.imag * dcval.imag);
      zdot.imag = ((double) dim) * (2.0 * dcval.real * dcval.imag);
      zresult = GA_Zdot (g_a, g_a);
      if (me == 0)
	{
	  if (MISMATCHED (zdot.real, zresult.real)
	      || MISMATCHED (zdot.imag, zresult.imag))
	    printf ("not ok.\n");
	  else
	    printf ("ok.\n");
	}
      break;
    default:
      ga_error ("test_shift_diagonal:wrong data type: ", type);
    }



}

void
do_work (int type, int op)
{
  int g_a, g_b, g_c, g_m, g_v, g_vv;
  int n = N;
  int me = GA_Nodeid (), nproc = GA_Nnodes ();
  int dims[2] = { N,		/*N columns */
		  N + 2			/*N+2 rows */
  };
  int vdim;
  int lo[2], hi[2];


  int atype, andim, adims[2];
  int vtype, vndim, vdims;


  lo[0] = 1;
  hi[0] = dims[0] - 1;
  lo[1] = 1;
  hi[1] = dims[1] - 1;


  switch (op)
    {

    case OP_SHIFT_DIAGONAL:
      g_a = NGA_Create (type, 2, dims, "A", NULL);
      if (!g_a)
	GA_Error ("create failed: A", n);
      test_shift_diagonal (g_a);
      GA_Destroy (g_a);
      break;
    case OP_SET_DIAGONAL:
      g_a = NGA_Create (type, 2, dims, "A", NULL);
      if (!g_a)
	GA_Error ("create failed: A", n);
      /*find out the diagonal length of the matrix A */
      vdim = MIN (dims[0], dims[1]);
      g_v = NGA_Create (type, 1, &vdim, "V", NULL);
      if (!g_v)
	GA_Error ("create failed:V", n);
      test_set_diagonal (g_a, g_v);
      GA_Destroy (g_a);
      GA_Destroy (g_v);
      break;
    case OP_ADD_DIAGONAL:
      g_a = NGA_Create (type, 2, dims, "A", NULL);
      if (!g_a)
	GA_Error ("create failed: A", n);
      /*find out the diagonal length of the matrix A */
      vdim = MIN (dims[0], dims[1]);
      g_v = NGA_Create (type, 1, &vdim, "V", NULL);
      if (!g_v)
	GA_Error ("create failed:V", n);
      test_add_diagonal (g_a, g_v);
      GA_Destroy (g_a);
      GA_Destroy (g_v);
      break;
    case OP_GET_DIAGONAL:
      g_a = NGA_Create (type, 2, dims, "A", NULL);
      if (!g_a)
	GA_Error ("create failed: A", n);
      /*find out the diagonal length of the matrix A */
      vdim = MIN (dims[0], dims[1]);
      g_v = NGA_Create (type, 1, &vdim, "V", NULL);
      if (!g_v)
	GA_Error ("create failed:V", n);
      test_get_diagonal (g_a, g_v);
      GA_Destroy (g_a);
      GA_Destroy (g_v);
      break;
    case OP_NORM1:
      g_a = NGA_Create (type, 2, dims, "A", NULL);
      if (!g_a)
	GA_Error ("create failed: A", n);
      test_norm1 (g_a);
      GA_Destroy (g_a);
      break;

    case OP_NORM_INFINITY:
      g_a = NGA_Create (type, 2, dims, "A", NULL);
      if (!g_a)
	GA_Error ("create failed: A", n);
      test_norm_infinity (g_a);
      GA_Destroy (g_a);
      break;

    case OP_MEDIAN:
      g_a = NGA_Create (type, 2, dims, "A", NULL);
      if (!g_a)
	GA_Error ("create failed: A", n);
      /*duplicate g_a */
      g_b = GA_Duplicate (g_a, "B");
      if (!g_b)
	GA_Error ("duplicate failed: B", n);

      g_c = GA_Duplicate (g_a, "C");
      if (!g_c)
	GA_Error ("duplicate failed: C", n);
#if 0 /* test g_m is different from g_a, g_b, amd g_c */
      g_m = GA_Duplicate (g_a, "M");
      if (!g_m)
	GA_Error ("duplicate failed: M", n);
      test_median (g_a, g_b, g_c, g_m);
#else /* test g_m = g_c */
      test_median (g_a, g_b, g_c, g_a);
#endif
      GA_Destroy (g_a);
      GA_Destroy (g_b);
      GA_Destroy (g_c);
#if 0  /* test g_m is different from g_a, g_b, g_c */
      GA_Destroy (g_m);
#endif
      break;

    case OP_MEDIAN_PATCH:
      g_a = NGA_Create (type, 2, dims, "A", NULL);
      if (!g_a)
	GA_Error ("create failed: A", n);
      /*duplicate g_a */
      g_b = GA_Duplicate (g_a, "B");
      if (!g_b)
	GA_Error ("duplicate failed: B", n);

      g_c = GA_Duplicate (g_a, "C");
      if (!g_c)
	GA_Error ("duplicate failed: C", n);

      g_m = GA_Duplicate (g_a, "M");
      if (!g_m)
	GA_Error ("duplicate failed: M", n);

      test_median_patch (g_a, lo, hi, g_b, lo, hi, g_c, lo, hi, g_m, lo, hi);
      GA_Destroy (g_a);
      GA_Destroy (g_b);
      GA_Destroy (g_c);
      GA_Destroy (g_m);
      break;
    case OP_SCALE_ROWS:
      g_a = NGA_Create (type, 2, dims, "A", NULL);
      if (!g_a)
	GA_Error ("create failed: A", n);
      /*find out the diagonal length of the matrix A */
      vdim = dims[1];
      g_v = NGA_Create (type, 1, &vdim, "V", NULL);
      if (!g_v)
	GA_Error ("create failed:V", n);
      test_scale_rows (g_a, g_v);
      GA_Destroy (g_a);
      GA_Destroy (g_v);
      break;
    case OP_SCALE_COLS:
      g_a = NGA_Create (type, 2, dims, "A", NULL);
      if (!g_a)
        GA_Error ("create failed: A", n);
      /*find out the diagonal length of the matrix A */
      vdim = dims[0];
      g_v = NGA_Create (type, 1, &vdim, "V", NULL);
      if (!g_v)
        GA_Error ("create failed:V", n);
      test_scale_cols (g_a, g_v);
      GA_Destroy (g_a);
      GA_Destroy (g_v);
      break;

    default:
      GA_Error ("test_function: wrong operation.", op);
    }


}



int
main (argc, argv)
     int argc;
     char **argv;
{
  int heap = 20000, stack = 20000;
  int me, nproc;
  int type, op;

  type = C_DBL;

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
  if (!MA_init (MT_F_DBL, stack, heap))
    GA_Error ("MA_init failed", stack + heap);	/* initialize memory allocator */


  for (op = 1; op < 11; op++)
    {
      if(me == 0) printf ("\n\n");
      if (me == 0)
	printf ("type = C_INT \t ");
      do_work (C_INT, op);
      if (me == 0)
	printf ("type = C_LONG \t ");
      do_work (C_LONG, op);
      if (me == 0)
	printf ("type = C_FLOAT \t ");
      do_work (C_FLOAT, op);
      if (me == 0)
	printf ("type = C_DBL \t ");
      do_work (C_DBL, op);
      if (me == 0)
	printf ("type = C_DCPL \t ");
      do_work (C_DCPL, op);
    }

  GA_Terminate ();

#ifdef MPI
  MPI_Finalize ();
#else
  PEND_ ();
#endif

  return 0;
}
