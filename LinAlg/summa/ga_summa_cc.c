/*
 *  Routine to interface GA to SUMMA matrix-matrix multiply.
 *  This version makes copies of g_a,g_b, and g_c.
 *
 *  returns: 2 if insufficient memory in MA for this routine
 *           1 if GA array shapes are incompatible with this routine
 *             This will never happen. Here for compatibility with ga_summa_c_. 
 *           0 if things went ok
 */


#include "global.h"
#include "macommon.h"
#include "sndrcv.h"

#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <fcntl.h>
#include <math.h>


typedef Integer Boolean;        /* MA_TRUE or MA_FALSE */

extern Boolean MA_alloc_get();
extern Boolean MA_allocate_heap();
extern Boolean MA_chop_stack();
extern Boolean MA_free_heap();
extern Boolean MA_get_index();
extern Boolean MA_get_next_memhandle();
extern Boolean MA_get_pointer();
extern Boolean MA_init();
extern Boolean MA_init_memhandle_iterator();
extern Integer MA_inquire_avail();
extern Integer MA_inquire_heap();
extern Integer MA_inquire_stack();
extern Boolean MA_pop_stack();
extern void MA_print_stats();
extern Boolean MA_push_get();
extern Boolean MA_push_stack();
extern Boolean MA_set_auto_verify();
extern Boolean MA_set_error_print();
extern Boolean MA_set_hard_fail();
extern Integer MA_sizeof();
extern Integer MA_sizeof_overhead();
extern void MA_summarize_allocated_blocks();
extern Boolean MA_verify_allocator_stuff();




#undef MIN
#undef MAX
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define MAX(a,b) (((a) <= (b)) ? (b) : (a))


#define PANEL_SIZE 32

/*
    #define SUMMA_TIMING
*/


#define SUCCESSFUL_EXIT       0
#define GA_ARRAYS_WRONG_SHAPE 1
#define INSUFFICIENT_MEMORY   2

ga_summa_cc_( itransa, itransb, m, n, k, alpha, g_a, g_b, beta, g_c)

Integer          *itransa, *itransb, *m, *n, *k, *g_a, *g_b, *g_c;
DoublePrecision  *alpha, *beta;

{

  /* --------------- */
  /* Local variables */
  /* --------------- */

  Integer nb, itmp;

  Integer    ma_tmp_handle,
             num_compute_nodes,
             ma_int_arg,
             ma_handles[10],
             n_handles,
             *proclist,
             nele, nmax,
             status;

  Void       *ma_pointer;

  Integer
    me,
    transa,
    transb,
    my_row,          /* row coord in 2D comm used for multiply */
    my_col,          /* column coord in 2D comm used for multiply */
    i_comm_size,       /* holds # nodes */
    i_initialized,     /* to see it MPI is initialized */
    i_world_id,        /* rank in MPI_COMM_WORLD */
    i_node_id,         /* rank in 2D topology */
    i, j,
    index,
    i_mdim, i_ndim,
    i_kdim,
    np_rows,      /* num of rows in the processor grid */
    np_cols,      /* num of cols in the processor grid */
    lda, ldb, ldc,
    max_rows, max_cols,
    n_work1,
    n_work2,
    *mm_rows,     /* number of row/columns owned by each row/col of  */
    *mm_cols,     /* processors */
    *nn_rows,
    *nn_cols,
    *kk_rows,
    *kk_cols
      ;

  DoublePrecision d_alpha, d_beta;
 
#ifdef SUMMA_TIMING
  DoublePrecision times[10];
#endif
 
  extern DoublePrecision *ga_summa_alloc();
  extern void   ga_summa_dealloc();
  extern void   ga_summa_numrows();
  extern void   summa_ab();
  extern Integer ga_summa_ok();

  DoublePrecision *summa_a, *summa_b, *summa_c, *work1, *work2;


#ifdef SUMMA_TIMING
  times[0] = TCGTIME_();
#endif

  n_handles = 0;

  me = ga_nodeid_();

  i_mdim = *m;
  i_ndim = *n;
  i_kdim = *k;

  d_alpha = *alpha;
  d_beta  = *beta;

  transa = *itransa;
  transb = *itransb;

  num_compute_nodes = ga_nnodes_();

  /* Get list of "real, i.e., tcgmsg" node ids. */

  if( MA_push_stack(MT_F_INT, num_compute_nodes, "proclist", ma_handles+n_handles ) ) {
     MA_get_pointer( ma_handles[n_handles], &ma_pointer );
     n_handles++;
  }
  else
     ga_error("ga_summa_cc_: MA_push_stack, failed for proclist!", 0);

  proclist = (Integer *) ma_pointer;

  ga_list_nodeid_( proclist, &num_compute_nodes );

  if( proclist[me] != NODEID_() )
      ga_error("ga_summa_cc_-bug: proclist is wrong!", 0);


  /* Need 2D grid. The first dimension is logically the rows and the second
   * dimension is the columns.  Set dimension of processor mesh as close to
   * square as possible using num_compute_nodes nodes.
   */

  np_rows = ( Integer ) ( sqrt( (double) (num_compute_nodes) ) );
  np_rows++;

  while( ( num_compute_nodes % np_rows ) != 0 )
   np_rows--;

  np_cols = num_compute_nodes / np_rows;

  /* Determine panel size, and possibly reduce number of processors */

  max_rows = MIN( i_mdim, i_kdim );
  max_cols = MIN( i_ndim, i_kdim );

  nb = PANEL_SIZE;

  np_rows = MIN( np_rows, max_rows );
  np_cols = MIN( np_cols, max_cols );

  num_compute_nodes = np_cols * np_rows;

  /* Determine my processor column and row. */

  my_col = me / np_rows;
  my_row = me - my_col * np_rows;

  if( me >= num_compute_nodes ) {

    /* Extra nodes wait till others finish matrix multi. before returning. */

    if( (status = ga_summa_ok( proclist, SUCCESSFUL_EXIT ) ) != SUCCESSFUL_EXIT ) {
      ga_summa_dealloc( ma_handles, n_handles );
      return( status );
    }

    ga_sync_();
    ga_sync_();

    ga_summa_dealloc( ma_handles, n_handles );

    ga_sync_();

    return( SUCCESSFUL_EXIT );
  }

  /* ---------------------------- */
  /* Allocate memory for matrices */
  /* ---------------------------- */

  ma_int_arg = 3*(np_rows+np_cols);
  if( MA_push_stack(MT_F_INT, ma_int_arg, "rowdata", ma_handles+n_handles ) ) {
     MA_get_pointer( ma_handles[n_handles], &ma_pointer );
     n_handles++;
  }
  else
     ga_error("ga_summa_cc_: MA_push_stack, failed for rowdata!", 0);

  mm_rows = (Integer *) ma_pointer;
  mm_cols = mm_rows + np_rows;

  nn_rows = mm_cols + np_cols;
  nn_cols = nn_rows + np_rows;

  kk_rows = nn_cols + np_cols;
  kk_cols = kk_rows + np_rows;


  /* calculate number of rows/cols per node for A, B and C */
  
  ga_summa_numrows( i_mdim, nb, np_rows, mm_rows );
  ga_summa_numrows( i_mdim, nb, np_cols, mm_cols );

  ga_summa_numrows( i_ndim, nb, np_rows, nn_rows );
  ga_summa_numrows( i_ndim, nb, np_cols, nn_cols );

  ga_summa_numrows( i_kdim, nb, np_rows, kk_rows );
  ga_summa_numrows( i_kdim, nb, np_cols, kk_cols );


  lda = mm_rows[my_row];
  ldb = kk_rows[my_row];
  ldc = mm_rows[my_row];

  /* Size of work buffers */

  n_work1 = nb * lda;
  n_work2 = nb * nn_cols[my_col];


  /* Test for sufficient workspace BEFORE making copies of any of the GA arrays. */

  nele = lda*kk_cols[my_col];

  nmax = nele;
  if ( transa ) nmax += nele;

  nele += ldb * nn_cols[my_col];

  if( transb ) nmax = MAX( nmax, nele + ldb*nn_cols[my_col] );

  nele += ldc * nn_cols[my_col] + n_work1 + n_work2;

  nmax = MAX( nmax, nele ) + 5 * 20;   /* 5*20= fudge factor to account for ma guards*/

  if( MA_inquire_avail( MT_F_DBL ) < nmax ) {
     status = ga_summa_ok( proclist, INSUFFICIENT_MEMORY );
     ga_summa_dealloc( ma_handles, n_handles );
     return( status );
  }

#ifdef SUMMA_TIMING
  times[1] = TCGTIME_();
#endif

  /* Allocate memory for summa a, b, and c matrices and load them from GA. */

  summa_a = ga_summa_alloc( lda*kk_cols[my_col], "ga_summa_a", ma_handles+n_handles ); 
  n_handles++;

  if( summa_a == NULL ) {
     status = ga_summa_ok( proclist, INSUFFICIENT_MEMORY );
     ga_summa_dealloc( ma_handles, n_handles-1 );
     return( status );
  }

  if( transa ) {
    work1 = ga_summa_alloc( lda*kk_cols[my_col], "ga_summa_trana", &ma_tmp_handle ); 

    if( work1 == NULL ) {
       status = ga_summa_ok( proclist, INSUFFICIENT_MEMORY );
       ga_summa_dealloc( ma_handles, n_handles );
       return( status );
    }

    ga_summa_layout2( transa, my_row, mm_rows, my_col, kk_cols, summa_a, lda, g_a, work1 );

    if( ! MA_pop_stack( ma_tmp_handle ) )
      ga_error("ga_summa_cc_:  pop stack failed for trana!", 0);

  }
  else
    ga_summa_layout( my_row, mm_rows, my_col, kk_cols, summa_a, lda, g_a );


  summa_b = ga_summa_alloc( ldb*nn_cols[my_col], "ga_summa_b", ma_handles+n_handles );
  n_handles++;

  if( summa_b == NULL ) {
     status = ga_summa_ok( proclist, INSUFFICIENT_MEMORY );
     ga_summa_dealloc( ma_handles, n_handles-1 );
     return( status );
  }

  if( transb ) {
    work1 = ga_summa_alloc( ldb*nn_cols[my_col], "ga_summa_tranb", &ma_tmp_handle );

    if( work1 == NULL ) {
       status = ga_summa_ok( proclist, INSUFFICIENT_MEMORY );
       ga_summa_dealloc( ma_handles, n_handles );
       return( status );
    }

    ga_summa_layout2( transb, my_row, kk_rows, my_col, nn_cols, summa_b, ldb, g_b, work1 );

    if( ! MA_pop_stack( ma_tmp_handle ) )
      ga_error("ga_summa_cc_:  pop stack failed for tranb!", 0);

  }
  else
    ga_summa_layout( my_row, kk_rows, my_col, nn_cols, summa_b, ldb, g_b );


  summa_c = ga_summa_alloc( ldc*nn_cols[my_col], "ga_summa_c", ma_handles+n_handles );
  n_handles++;

  if( summa_c == NULL ) {
     status = ga_summa_ok( proclist, INSUFFICIENT_MEMORY );
     ga_summa_dealloc( ma_handles, n_handles-1 );
     return( status );
  }

  ga_summa_layout( my_row, mm_rows, my_col, nn_cols, summa_c, ldc, g_c );


#ifdef SUMMA_TIMING
  times[2] = TCGTIME_();
#endif

  /* Allocate memory for work buffers */

  work1 = ga_summa_alloc( n_work1, "ga_summa_work1", ma_handles+n_handles );
  n_handles++;

  if( work1 == NULL ) {
     status = ga_summa_ok( proclist, INSUFFICIENT_MEMORY );
     ga_summa_dealloc( ma_handles, n_handles-1 );
     return( status );
  }

  work2 = ga_summa_alloc( n_work2, "ga_summa_work2", ma_handles+n_handles );
  n_handles++;

  if( work2 == NULL ) {
     status = ga_summa_ok( proclist, INSUFFICIENT_MEMORY );
     ga_summa_dealloc( ma_handles, n_handles-1 );
     return( status );
  }


  /* Make sure everyone is ok. */

  if( (status = ga_summa_ok( proclist, SUCCESSFUL_EXIT ) ) != SUCCESSFUL_EXIT ) {
    ga_summa_dealloc( ma_handles, n_handles );
    return( status );
  }

  /* --------------------- */
  /* Matrix multiplication */
  /* --------------------- */

  /* Synchronize processors to make sure they start together */
  ga_sync_();

#ifdef SUMMA_TIMING
  times[3] = TCGTIME_();
#endif

  summa_ab( i_mdim, i_ndim, i_kdim, nb, d_alpha, summa_a, lda, summa_b, ldb,
              d_beta, summa_c, ldc, 
              mm_rows, kk_cols,  kk_rows, nn_cols, mm_rows, nn_cols,
              my_row, my_col, np_rows, np_cols, proclist, work1, work2 );

  /* Synchronize processors to make sure they end together */
  ga_sync_();

#ifdef SUMMA_TIMING
  times[4] = TCGTIME_();
#endif

  /* Copy SUMMA results to GA c array. */
  ga_summa_to_ga( my_row, mm_rows, my_col, nn_cols, summa_c, ldc, g_c );


#ifdef SUMMA_TIMING
  if( me == 0 ) 
    fprintf( stderr, "                       %d-by-%d nb = %d   %f  \n",
             np_rows, np_cols,  nb, times[4]-times[3] );
#endif

  ga_summa_dealloc( ma_handles, n_handles );

  /* Synchronize ALL processors coming out of here */
  ga_sync_();

  return( SUCCESSFUL_EXIT );
}
void ga_summa_numrows( n, nb, np_rows, n_rows )
  Integer n, nb, np_rows, *n_rows;

/*
 * Set number of rows (columns) owned by each processor.
 * 
 * Input
 * -----
 *   n ......... row (column) dimension of matrix
 *   nb ........ block size
 *   np_rows ... number of rows (columns) of processors
 *
 * Output
 * ------
 *   n_rows .... n_rows[i] is number of rows (columns) owned
 *               by i-th row (column) of processors.
 */

{
  Integer n_all, n_extra, nrows, i;

  n_all = n / np_rows;
  n_extra = n - n_all * np_rows;

  nrows = n_all + 1;
  for( i=0; i< n_extra; i++)
    n_rows[i] = nrows;
  
  for( i=n_extra; i< np_rows; i++)
    n_rows[i] = n_all;
}
