
/* This function loads GA arrays into SUMMA arrays.  */

#include <math.h>
#include <stdio.h>

#include "global.h"
#include "f2cblas.h"

void ga_summa_layout( Integer my_row, Integer *kk_rows, Integer my_col, Integer *kk_cols,
                      DoublePrecision *summa_matrix, Integer lda,  Integer *ga_matrix )

/* PARAMETERS
   ==========
   my_row ........... row coordinate in 2d topology
   kk_rows[i] ....... number of rows of matrix owned by i-th row of processors
   my_col ........... row coordinate in 2d topology
   kk_cols[i] ....... number of columns of matrix owned by i-th column of processors
   *summa_matrix ... pointer to the summa version of matrix
   lda ............. leading dim of summa_matrix on the node
   ga_matrix ....... handle for GA matrix to load into summa_matrix
*/

{
  /* Local variables */
  Integer       i, fst_row, lst_row, fst_col, lst_col;

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

  if( lst_col >= fst_col  && lst_row >= fst_row )
    ga_get_( ga_matrix, &fst_row, &lst_row, &fst_col, &lst_col, summa_matrix, &lda );

}
