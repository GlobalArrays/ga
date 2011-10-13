/***************************************************************************

                  COPYRIGHT

The following is a notice of limited availability of the code, and disclaimer
which must be included in the prologue of the code and in all source listings
of the code.

Copyright Notice
 + 2009 University of Chicago

Permission is hereby granted to use, reproduce, prepare derivative works, and
to redistribute to others.  This software was authored by:

Jeff R. Hammond
Leadership Computing Facility
Argonne National Laboratory
Argonne IL 60439 USA
phone: (630) 252-5381
e-mail: jhammond@anl.gov

                  GOVERNMENT LICENSE

Portions of this material resulted from work developed under a U.S.
Government Contract and are subject to the following license: the Government
is granted for itself and others acting on its behalf a paid-up, nonexclusive,
irrevocable worldwide license in this computer software to reproduce, prepare
derivative works, and perform publicly and display publicly.

                  DISCLAIMER

This computer code material was prepared, in part, as an account of work
sponsored by an agency of the United States Government.  Neither the United
States, nor the University of Chicago, nor any of their employees, makes any
warranty express or implied, or assumes any legal liability or responsibility
for the accuracy, completeness, or usefulness of any information, apparatus,
product, or process disclosed, or represents that its use would not infringe
privately owned rights.

 ***************************************************************************/
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include <stdlib.h>
#include <stdio.h>
#include <assert.h>

#include "copy.h"
#include "timer.h"

#define F_TESTS 1
#define C_TESTS 1
#define M_TESTS 1
#define ALLOW_FREE 1

typedef void (*func1d)(const double* const restrict,
                             double* const restrict,
                       const int*    const restrict);
typedef void (*func2d)(const int*    const restrict,
                       const int*    const restrict,
                       const double* const restrict,
                       const int*    const restrict,
                             double* const restrict,
                       const int*    const restrict);
typedef void (*func21)(const int*    const restrict,
                       const int*    const restrict,
                       const double* const restrict,
                       const int*    const restrict,
                             double* const restrict,
                             int*    const restrict);
typedef void (*func12)(const int*    const restrict,
                       const int*    const restrict,
                             double* const restrict,
                       const int*    const restrict,
                       const double* const restrict,
                             int*    const restrict);
typedef void (*func31)(const int*    const restrict,
                       const int*    const restrict,
                       const int*    const restrict,
                       const double* const restrict,
                       const int*    const restrict,
                       const int*    const restrict,
                             double* const restrict,
                             int*    const restrict);
typedef void (*func13)(const int*    const restrict,
                       const int*    const restrict,
                       const int*    const restrict,
                             double* const restrict,
                       const int*    const restrict,
                       const int*    const restrict,
                       const double* const restrict,
                             int*    const restrict);

static void test_free(void *pointer)
{
#if ALLOW_FREE
  free(pointer);
#endif
}


static void timer_print(char *name, unsigned long long timer, char l)
{
  printf("%10s = %10llu %c\n", name, timer, l);
}


static void init_in(double *in, int n)
{
    int i=0;
    for (i=0; i<n; ++i) {
        in[i] = (double)i;
    }
}


static void init_out(double *out, int n)
{
    int i=0;
    for (i=0; i<n; ++i) {
        out[i] = -1.0f;
    }
}


static void test1d(
    char *name, func1d f, func1d c, int dim1, int correctness)
{
  unsigned long long timer;
  int i;
#if F_TESTS
  double *fin  = malloc(dim1 * sizeof(double));
  double *fout = malloc(dim1 * sizeof(double));
#endif
#if C_TESTS
  double *cin  = malloc(dim1 * sizeof(double));
  double *cout = malloc(dim1 * sizeof(double));
#endif
#if M_TESTS
  double *min  = malloc(dim1 * sizeof(double));
  double *mout = malloc(dim1 * sizeof(double));
#endif

#if F_TESTS
  if (correctness) {
    init_in(fin, dim1);
    init_out(fout, dim1);
  }
  timer = timer_start();
  (*f)(fin, fout, &dim1);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'F');
  }
#endif

#if C_TESTS
  if (correctness) {
    init_in(cin, dim1);
    init_out(cout, dim1);
  }
  timer = timer_start();
  (*c)(cin, cout, &dim1);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'C');
  }
#endif

#if M_TESTS
  if (correctness) {
    init_in(min, dim1);
    init_out(mout, dim1);
  }
  timer = timer_start();
  memcpy(mout, min, dim1*sizeof(double));
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'M');
  }
#endif

#if F_TESTS && C_TESTS && M_TESTS
  if (correctness) {
    for (i = 0 ; i < dim1; i++) {
      assert(cout[i] == fout[i]);
    }
    for (i = 0 ; i < dim1; i++) {
      assert(mout[i] == fout[i]);
    }
  }
#endif

#if F_TESTS
  test_free(fin);
  test_free(fout);
#endif
#if C_TESTS
  test_free(cin);
  test_free(cout);
#endif
#if M_TESTS
  test_free(min);
  test_free(mout);
#endif
}


static void test2d(
    char *name, func2d f, func2d c, int dim1, int dim2, int correctness)
{
  unsigned long long timer;
  int i;
#if F_TESTS
  double *fin  = malloc(dim1 * dim2 * sizeof(double));
  double *fout = malloc(dim1 * dim2 * sizeof(double));
#endif
#if C_TESTS
  double *cin  = malloc(dim1 * dim2 * sizeof(double));
  double *cout = malloc(dim1 * dim2 * sizeof(double));
#endif

#if F_TESTS
  if (correctness) {
    init_in(fin, dim1 * dim2);
    init_out(fout, dim1 * dim2);
  }
  timer = timer_start();
  (*f)(&dim1, &dim2, fin, &dim1, fout, &dim1);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'F');
  }
#endif

#if C_TESTS
  if (correctness) {
    init_in(cin, dim1 * dim2);
    init_out(cout, dim1 * dim2);
  }
  timer = timer_start();
  (*c)(&dim1, &dim2, cin, &dim1, cout, &dim1);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'C');
  }
#endif

#if F_TESTS && C_TESTS
  if (correctness) {
    for (i = 0 ; i < dim1 * dim2; i++) {
      assert(cout[i] == fout[i]);
    }
  }
#endif

#if F_TESTS
  test_free(fin);
  test_free(fout);
#endif
#if C_TESTS
  test_free(cin);
  test_free(cout);
#endif
}


static void test21(
    char *name, func21 f, func21 c, int dim1, int dim2, int correctness)
{
  unsigned long long timer;
  int i;
#if F_TESTS
  int fcur;
  double *fin  = malloc(dim1 * dim2 * sizeof(double));
  double *fout = malloc(dim1 * dim2 * sizeof(double));
#endif
#if C_TESTS
  int ccur;
  double *cin  = malloc(dim1 * dim2 * sizeof(double));
  double *cout = malloc(dim1 * dim2 * sizeof(double));
#endif

#if F_TESTS
  if (correctness) {
    init_in(fin, dim1 * dim2);
    init_out(fout, dim1 * dim2);
  }
  timer = timer_start();
  (*f)(&dim1, &dim2, fin, &dim1, fout, &fcur);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'F');
  }
#endif

#if C_TESTS
  if (correctness) {
    init_in(cin, dim1 * dim2);
    init_out(cout, dim1 * dim2);
  }
  timer = timer_start();
  (*c)(&dim1, &dim2, cin, &dim1, cout, &ccur);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'C');
  }
#endif

#if F_TESTS && C_TESTS
  if (correctness) {
    for (i = 0 ; i < dim1 * dim2; i++) {
      assert(cout[i] == fout[i]);
    }
    assert(fcur == ccur);
  }
#endif

#if F_TESTS
  test_free(fin);
  test_free(fout);
#endif
#if C_TESTS
  test_free(cin);
  test_free(cout);
#endif
}


static void test12(
    char *name, func12 f, func12 c, int dim1, int dim2, int correctness)
{
  unsigned long long timer;
  int i;
#if F_TESTS
  int fcur;
  double *fin  = malloc(dim1 * dim2 * sizeof(double));
  double *fout = malloc(dim1 * dim2 * sizeof(double));
#endif
#if C_TESTS
  int ccur;
  double *cin  = malloc(dim1 * dim2 * sizeof(double));
  double *cout = malloc(dim1 * dim2 * sizeof(double));
#endif

#if F_TESTS
  if (correctness) {
    init_in(fin, dim1 * dim2);
    init_out(fout, dim1 * dim2);
  }
  timer = timer_start();
  (*f)(&dim1, &dim2, fout, &dim1, fin, &fcur);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'F');
  }
#endif

#if C_TESTS
  if (correctness) {
    init_in(cin, dim1 * dim2);
    init_out(cout, dim1 * dim2);
  }
  timer = timer_start();
  (*c)(&dim1, &dim2, cout, &dim1, cin, &ccur);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'C');
  }
#endif

#if F_TESTS && C_TESTS
  if (correctness) {
    for (i = 0 ; i < dim1 * dim2; i++) {
      assert(cout[i] == fout[i]);
    }
    assert(fcur == ccur);
  }
#endif

#if F_TESTS
  test_free(fin);
  test_free(fout);
#endif
#if C_TESTS
  test_free(cin);
  test_free(cout);
#endif
}


static void test31(
    char *name, func31 f, func31 c, int dim1, int dim2, int dim3, int correctness)
{
  unsigned long long timer;
  int i;
#if F_TESTS
  int fcur;
  double *fin  = malloc(dim1 * dim2 * dim3 * sizeof(double));
  double *fout = malloc(dim1 * dim2 * dim3 * sizeof(double));
#endif
#if C_TESTS
  int ccur;
  double *cin  = malloc(dim1 * dim2 * dim3 * sizeof(double));
  double *cout = malloc(dim1 * dim2 * dim3 * sizeof(double));
#endif

#if F_TESTS
  if (correctness) {
    init_in(fin, dim1 * dim2 * dim3);
    init_out(fout, dim1 * dim2 * dim3);
  }
  timer = timer_start();
  (*f)(&dim1, &dim2, &dim3, fin, &dim1, &dim2, fout, &fcur);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'F');
  }
#endif

#if C_TESTS
  if (correctness) {
    init_in(cin, dim1 * dim2 * dim3);
    init_out(cout, dim1 * dim2 * dim3);
  }
  timer = timer_start();
  (*c)(&dim1, &dim2, &dim3, cin, &dim1, &dim2, cout, &ccur);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'C');
  }
#endif

#if F_TESTS && C_TESTS
  if (correctness) {
    for (i = 0 ; i < dim1 * dim2 * dim3; i++) {
      assert(cout[i] == fout[i]);
    }
    assert(fcur == ccur);
  }
#endif

#if F_TESTS
  test_free(fin);
  test_free(fout);
#endif
#if C_TESTS
  test_free(cin);
  test_free(cout);
#endif
}


static void test13(
    char *name, func13 f, func13 c, int dim1, int dim2, int dim3, int correctness)
{
  unsigned long long timer;
  int i;
#if F_TESTS
  int fcur;
  double *fin  = malloc(dim1 * dim2 * dim3 * sizeof(double));
  double *fout = malloc(dim1 * dim2 * dim3 * sizeof(double));
#endif
#if C_TESTS
  int ccur;
  double *cin  = malloc(dim1 * dim2 * dim3 * sizeof(double));
  double *cout = malloc(dim1 * dim2 * dim3 * sizeof(double));
#endif

#if F_TESTS
  if (correctness) {
    init_in(fin, dim1 * dim2 * dim3);
    init_out(fout, dim1 * dim2 * dim3);
  }
  timer = timer_start();
  (*f)(&dim1, &dim2, &dim3, fout, &dim1, &dim2, fin, &fcur);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'F');
  }
#endif

#if C_TESTS
  if (correctness) {
    init_in(cin, dim1 * dim2 * dim3);
    init_out(cout, dim1 * dim2 * dim3);
  }
  timer = timer_start();
  (*c)(&dim1, &dim2, &dim3, cout, &dim1, &dim2, cin, &ccur);
  timer = timer_end(timer);
  if (!correctness) {
    timer_print(name, timer, 'C');
  }
#endif

#if F_TESTS && C_TESTS
  if (correctness) {
    for (i = 0 ; i < dim1 * dim2 * dim3; i++) {
      assert(cout[i] == fout[i]);
    }
    assert(fcur == ccur);
  }
#endif

#if F_TESTS
  test_free(fin);
  test_free(fout);
#endif
#if C_TESTS
  test_free(cin);
  test_free(cout);
#endif
}


int main(int argc, char **argv)
{
#if ALLOW_FREE
  int dim1  = (argc > 1 ? atoi(argv[1]) : 179);
  int dim2  = (argc > 2 ? atoi(argv[2]) : 233);
  int dim3  = (argc > 3 ? atoi(argv[3]) : 283);
#else
  int dim1  = (argc > 1 ? atoi(argv[1]) :  31);
  int dim2  = (argc > 2 ? atoi(argv[2]) :  73);
  int dim3  = (argc > 3 ? atoi(argv[3]) : 127);
#endif

  /*********************************************************/

  timer_init();

  printf("\ntesting ARMCI copy routines\n");
#if __STDC_VERSION__ >= 199901L
  printf("\nrestrict keyword is used for C routines\n");
#endif
  printf("\ntimer name '%s'\n", timer_name());

  /*********************************************************/

  test1d("dcopy1d_n", dcopy1d_n_, c_dcopy1d_n_, dim1, 0);
  test1d("dcopy1d_n", dcopy1d_n_, c_dcopy1d_n_, dim1, 1);
  test1d("dcopy1d_u", dcopy1d_u_, c_dcopy1d_u_, dim1, 0);
  test1d("dcopy1d_u", dcopy1d_u_, c_dcopy1d_u_, dim1, 1);

  /*********************************************************/

  test2d("dcopy2d_n", dcopy2d_n_, c_dcopy2d_n_, dim1, dim2, 0);
  test2d("dcopy2d_n", dcopy2d_n_, c_dcopy2d_n_, dim1, dim2, 1);
  test2d("dcopy2d_u", dcopy2d_u_, c_dcopy2d_u_, dim1, dim2, 0);
  test2d("dcopy2d_u", dcopy2d_u_, c_dcopy2d_u_, dim1, dim2, 1);

  /*********************************************************/

  test21("dcopy21", dcopy21_, c_dcopy21_, dim1, dim2, 0);
  test21("dcopy21", dcopy21_, c_dcopy21_, dim1, dim2, 1);
  test12("dcopy12", dcopy12_, c_dcopy12_, dim1, dim2, 0);
  test12("dcopy12", dcopy12_, c_dcopy12_, dim1, dim2, 1);

  /*********************************************************/

  test31("dcopy31", dcopy31_, c_dcopy31_, dim1, dim2, dim3, 0);
  test31("dcopy31", dcopy31_, c_dcopy31_, dim1, dim2, dim3, 1);
  test13("dcopy13", dcopy13_, c_dcopy13_, dim1, dim2, dim3, 0);
  test13("dcopy13", dcopy13_, c_dcopy13_, dim1, dim2, dim3, 1);

  /*********************************************************/

  return(0);
}
