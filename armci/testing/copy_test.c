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

#define DCOPY2D_N_ F77_FUNC_(dcopy2d_n,DCOPY2D_N)
#define DCOPY2D_U_ F77_FUNC_(dcopy2d_u,DCOPY2D_U)
#define DCOPY1D_N_ F77_FUNC_(dcopy1d_n,DCOPY1D_N)
#define DCOPY1D_U_ F77_FUNC_(dcopy1d_u,DCOPY1D_U)

int main(int argc, char** argv)
{
    int dim1  = ( argc>1 ? atoi(argv[1]) : 17 );
    int dim2  = ( argc>2 ? atoi(argv[2]) : 31 );
    int dim3  = ( argc>3 ? atoi(argv[3]) : 73 );

    printf("testing ARMCI copy routines\n");

    /*********************************************************/

    double* in1 = malloc( (dim1)           * sizeof(double) );
    double* in2 = malloc( (dim1*dim2)      * sizeof(double) );
    double* in3 = malloc( (dim1*dim2*dim3) * sizeof(double) );

    double* cout1 = malloc( (dim1)           * sizeof(double) );
    double* cout2 = malloc( (dim1*dim2)      * sizeof(double) );
    double* cout3 = malloc( (dim1*dim2*dim3) * sizeof(double) );

    double* fout1 = malloc( (dim1)           * sizeof(double) );
    double* fout2 = malloc( (dim1*dim2)      * sizeof(double) );
    double* fout3 = malloc( (dim1*dim2*dim3) * sizeof(double) );

    int i;

    for ( i=0 ; i<(dim1)           ; i++ ) in1[i] = (double)i;
    for ( i=0 ; i<(dim1*dim2)      ; i++ ) in2[i] = (double)i;
    for ( i=0 ; i<(dim1*dim2*dim3) ; i++ ) in3[i] = (double)i;

    int ccur, fcur;

    /*********************************************************/

    printf("c_dcopy1d_n_ \n");

    for (i=0;i<dim1;i++) cout1[i] = -1.0f;
    for (i=0;i<dim1;i++) fout1[i] = -1.0f;

      DCOPY1D_N_(in1, fout1, &dim1);
    c_dcopy1d_n_(in1, cout1, &dim1);
    for (i=0 ;i<dim1;i++) assert(cout1[i]==fout1[i]);

    printf("c_dcopy1d_u_ \n");

    for (i=0;i<dim1;i++) cout1[i] = -1.0f;
    for (i=0;i<dim1;i++) fout1[i] = -1.0f;

      DCOPY1D_U_(in1, fout1, &dim1);
    c_dcopy1d_u_(in1, cout1, &dim1);
    for (i=0 ;i<dim1;i++) assert(cout1[i]==fout1[i]);

    printf("all 1d tests have passed!\n");

    /*********************************************************/

    printf("c_dcopy2d_n \n");

    for (i=0;i<(dim1*dim2);i++) cout2[i] = -1.0f;
    for (i=0;i<(dim1*dim2);i++) fout2[i] = -1.0f;

      DCOPY2D_N_(&dim1,&dim2,in2,&dim1,fout2,&dim1);
    c_dcopy2d_n_(&dim1,&dim2,in2,&dim1,cout2,&dim1);
    for (i=0 ;i<(dim1*dim2);i++) assert(cout2[i]==fout2[i]);

    printf("c_dcopy2d_u \n");

    for (i=0;i<(dim1*dim2);i++) cout2[i] = -1.0f;
    for (i=0;i<(dim1*dim2);i++) fout2[i] = -1.0f;

      DCOPY2D_U_(&dim1,&dim2,in2,&dim1,fout2,&dim1);
    c_dcopy2d_u_(&dim1,&dim2,in2,&dim1,cout2,&dim1);
    for (i=0 ;i<(dim1*dim2);i++) assert(cout2[i]==fout2[i]);

    printf("c_dcopy21 \n");

    for (i=0;i<(dim1*dim2);i++) cout2[i] = -1.0f;
    for (i=0;i<(dim1*dim2);i++) fout2[i] = -1.0f;

      DCOPY21 (&dim1,&dim2,in2,&dim1,fout2,&fcur);
    c_dcopy21_(&dim1,&dim2,in2,&dim1,cout2,&ccur);
    for (i=0 ;i<(dim1*dim2);i++) assert(cout2[i]==fout2[i]);
    assert(ccur==fcur);

    printf("c_dcopy12 \n");

    for (i=0;i<(dim1*dim2);i++) cout2[i] = -1.0f;
    for (i=0;i<(dim1*dim2);i++) fout2[i] = -1.0f;

      DCOPY12 (&dim1,&dim2,fout2,&dim1,in2,&fcur);
    c_dcopy12_(&dim1,&dim2,cout2,&dim1,in2,&ccur);
    for (i=0 ;i<(dim1*dim2);i++) assert(cout2[i]==fout2[i]);
    assert(ccur==fcur);

    printf("all 2d tests have passed!\n");

    /*********************************************************/

    printf("c_dcopy31 \n");

    for (i=0;i<(dim1*dim2*dim3);i++) cout3[i] = -1.0f;
    for (i=0;i<(dim1*dim2*dim3);i++) fout3[i] = -1.0f;

      DCOPY31 (&dim1,&dim2,&dim3,in3,&dim1,&dim2,fout3,&fcur);
    c_dcopy31_(&dim1,&dim2,&dim3,in3,&dim1,&dim2,cout3,&ccur);
    for (i=0 ;i<(dim1*dim2*dim3);i++) assert(cout3[i]==fout3[i]);
    assert(ccur==fcur);

    printf("c_dcopy13 \n");

    for (i=0;i<(dim1*dim2*dim3);i++) cout3[i] = -1.0f;
    for (i=0;i<(dim1*dim2*dim3);i++) fout3[i] = -1.0f;

      DCOPY13 (&dim1,&dim2,&dim3,fout3,&dim1,&dim2,in3,&fcur);
    c_dcopy13_(&dim1,&dim2,&dim3,cout3,&dim1,&dim2,in3,&ccur);
    for (i=0 ;i<(dim1*dim2*dim3);i++) assert(cout3[i]==fout3[i]);
    assert(ccur==fcur);

    printf("all 3d tests have passed!\n");

    /*********************************************************/

    free(in1);
    free(in2);
    free(in3);
    free(cout1);
    free(cout2);
    free(cout3);
    free(fout1);
    free(fout2);
    free(fout3);

    return(0);
}
