 
/* This function loads GA arrays into SUMMA arrays.  */

#include <math.h>
#include <stdio.h>

#include "global.h"
#include "f2cblas.h"
 
void ga_summa_layout2( Integer trans, Integer my_row, Integer *kk_rows, Integer my_col, Integer *kk_cols,
                      DoublePrecision *summa_matrix, Integer lda,  Integer *ga_matrix, DoublePrecision *work)

/* PARAMETERS
   ==========
   trans ............ transpose matrix if (trans) is true
   my_row ........... row coordinate in 2d topology
   kk_rows[i] ....... number of rows of matrix owned by i-th row of processors
   my_col ........... row coordinate in 2d topology
   kk_cols[i] ....... number of columns of matrix owned by i-th column of processors
   *summa_matrix ... pointer to the summa version of matrix
   lda ............. leading dim of summa_matrix on the node
   ga_matrix ....... handle for GA matrix to load into summa_matrix
   *work ........... work array big enough to hold all of matrix, i.e., basically
                     the same size as summa_matrix, used for transposing, needed
                     since matrix is generally not square.
*/

{
  /* Local variables */
  static Integer   i_one = 1;
  Integer          i, j, ldw, n_col, fst_row, lst_row, fst_col, lst_col;
  DoublePrecision  *p;

  fst_row = 0;
  for( i=0; i<my_row; i++)
    fst_row += kk_rows[i];

  /* accomodate fortran indexing */
  fst_row++;

  lst_row = fst_row + kk_rows[my_row] - 1;

  fst_col = 0;
  for( i=0; i<my_col; i++)
    fst_col += kk_cols[i];

  /* accomodate fortran indexing */
  fst_col++;

  lst_col = fst_col + kk_cols[my_col] - 1;

  if( lst_col >= fst_col  && lst_row >= fst_row ) {

    if( !trans )

      ga_get_( ga_matrix, &fst_row, &lst_row, &fst_col, &lst_col, summa_matrix, &lda );

    else {

      ldw = lst_col - fst_col + 1;
      ga_get_( ga_matrix, &fst_col, &lst_col, &fst_row, &lst_row, work, &ldw );

      p = work;
      n_col = lst_row - fst_row + 1;

/*
         fprintf( stderr, " pre trans \n");
         for( j=0; j<n_col; j++) {
           fprintf( stderr, " \n");
           for( i=0; i<ldw; i++)
             fprintf( stderr, " matrix[%d][%d] = %f \n", i, j, work[i+j*ldw]);
         }
 */

      for( j=0; j<n_col; j++) {
        dcopy_( &ldw, p, &i_one, summa_matrix+j, &lda );
        p += ldw;
      }


/*
         fprintf( stderr, " \n");
         fprintf( stderr, " \n");
         fprintf( stderr, " post trans \n");
         for( j=0; j<ldw; j++) {
           fprintf( stderr, " \n");
           for( i=0; i<n_col; i++)
             fprintf( stderr, " tmatrix[%d][%d] = %f \n", i, j, summa_matrix[i+j*n_col]);
         }
 */

    }
  }
}
