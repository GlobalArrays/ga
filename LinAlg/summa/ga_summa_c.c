/*
 *  Routine to interface GA to SUMMA matrix-matrix multiply.
 *  This version uses ga arrays, rather than copying them to workspace.
 *
 *  A copy of the smallest of A, B, and C is made for the case A'*B'.
 *
 *  returns: 2 if insufficient memory in MA for this routine
 *           1 if GA array shapes are incompatible with this routine
 *           0 if things went ok
 *
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


#define PANEL_SIZE 25 

/*
    #define SUMMA_TIMING
*/


#define SUCCESSFUL_EXIT       0
#define GA_ARRAYS_WRONG_SHAPE 1
#define INSUFFICIENT_MEMORY   2

Integer ga_summa_c_( itransa, itransb, mm, nn, kk, alpha, g_a, g_b, beta, g_c)

Integer          *itransa, *itransb, *mm, *nn, *kk, *g_a, *g_b, *g_c;
DoublePrecision  *alpha, *beta;

{

  DoublePrecision     d_alpha, d_beta;
  DoublePrecision     *a, *b, *c, *work1, *work2;
  DoublePrecision     *dummy;

  Integer    lda, nprow_a, npcol_a, my_row_a, my_col_a, 
             ldb, nprow_b, npcol_b, my_row_b, my_col_b, 
             ldc, nprow_c, npcol_c, my_row_c, my_col_c; 

  Integer    *ms_a, *ns_a, *proclist_a,
             *ms_b, *ns_b, *proclist_b,
             *ms_c, *ns_c, *proclist_c,
             num_nodes;

  Integer    ma_handles[10], n_handles, status;

  Integer 
    nb,
    transa, transb, transab,
    copy_a, copy_b, copy_c,
    my_row, my_col,          /* row/column coord in 2D comm used for multiply */
    i, j,
    m, n, k,
    np_rows, np_cols,      /* num of rows/cols in the processor grid */
    n_work1,
    n_work2;
 
  extern DoublePrecision *ga_summa_alloc();
  extern void   summa_ab(), summa_abt(), summa_atb(),
                summa_ab2(), summa_abt2(), summa_atb2();
  extern void   ga_to_summa(), ga_summa_dealloc();
  extern Integer ga_summa_ok();

#ifdef SUMMA_TIMING
  DoublePrecision times[10];

  times[0] = TCGTIME_();
#endif

  /*
   * Synchronize processors to make sure they start together
   * Especially important for A'*B' since we use ga_access in this case.
   */
  ga_sync_();

  num_nodes = ga_nnodes_(); 

  dummy = NULL;

  nb = PANEL_SIZE;

  m = *mm;
  n = *nn;
  k = *kk;

  d_alpha = *alpha;
  d_beta  = *beta;

  transa = *itransa;
  transb = *itransb;

  /* Get info about ga arrays.  */

  if( transa )
    ga_to_summa( *g_a, k, m, &a, &lda, &nprow_a, &npcol_a, &my_row_a, &my_col_a,
                 &ms_a, &ns_a, &proclist_a, ma_handles  );
  else
    ga_to_summa( *g_a, m, k, &a, &lda, &nprow_a, &npcol_a, &my_row_a, &my_col_a,
                 &ms_a, &ns_a, &proclist_a, ma_handles  );

  if( transb )
    ga_to_summa( *g_b, n, k, &b, &ldb, &nprow_b, &npcol_b, &my_row_b, &my_col_b,
                 &ms_b, &ns_b, &proclist_b, ma_handles+1 );
  else
    ga_to_summa( *g_b, k, n, &b, &ldb, &nprow_b, &npcol_b, &my_row_b, &my_col_b,
                 &ms_b, &ns_b, &proclist_b, ma_handles+1 );

  ga_to_summa( *g_c, m, n, &c, &ldc, &nprow_c, &npcol_c, &my_row_c, &my_col_c,
                 &ms_c, &ns_c, &proclist_c, ma_handles+2 );

  n_handles = 3;

  if( nprow_a != nprow_b || nprow_a != nprow_c ||
      npcol_a != npcol_b || npcol_a != npcol_c ||
      my_row_a != my_row_b || my_row_a != my_row_c ||
      my_col_a != my_col_b || my_col_a != my_col_c ) {
    status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
    ga_summa_dealloc( ma_handles, n_handles );
    return( status );
  }


  for( i=0; i<nprow_a*npcol_a; i++ )
    if( proclist_a[i] != proclist_b[i]  ||  proclist_a[i] != proclist_c[i] ) {
      status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
      ga_summa_dealloc( ma_handles, n_handles );
      return( status );
    }

  /*
   * If copy_X, X=a,b,c, is set, then proclist_X is currently ok,
   *  but other yyyy_X data is incorrect.
   */

  transab = 0;

  copy_a = 0;
  copy_b = 0;
  copy_c = 0;

  if( transa && transb ) {
    transab = 1;
    if( k <= n  && m <= n ) {
      copy_a = 1;
      transa = 0;
    }
    else if( k <= m && n <= m ) {
      copy_b = 1;
      transb = 0;
    }
    else
      copy_c = 1;
  }

  if( copy_a ) {

    nprow_a  = nprow_c;
    my_row_a = my_row_c;

    npcol_a  = npcol_b;
    my_col_a = my_col_b;

    ms_a = ms_c;
    ns_a = ns_b;

    if( my_row_a > -1 && my_col_a > -1 ) {
      lda = ns_a[my_col_a];
      a   = ga_summa_alloc( ms_a[my_row_a]*lda, "ga_summa_a", ma_handles+n_handles ); 
      n_handles++;

      if( a == NULL ) {
        status = ga_summa_ok( proclist_a, INSUFFICIENT_MEMORY );
        ga_summa_dealloc( ma_handles, n_handles-1 );
        return( status );
      }

      /* Delay actually copying g_a to a until know everything is ok. */
    }
    else {
      lda = 0;
      a   = NULL;
    }

  }
  else if( copy_b ) {

    nprow_b  = nprow_a;
    my_row_b = my_row_a;

    npcol_b  = npcol_c;
    my_col_b = my_col_c;

    ms_b = ms_a;
    ns_b = ns_c;

    if( my_row_b > -1 && my_col_b > -1 ) {
      ldb = ns_b[my_col_b];
      b   = ga_summa_alloc( ms_b[my_row_b]*ldb, "ga_summa_b", ma_handles+n_handles ); 
      n_handles++;

      if( b == NULL ) {
        status = ga_summa_ok( proclist_a, INSUFFICIENT_MEMORY );
        ga_summa_dealloc( ma_handles, n_handles-1 );
        return( status );
      }

      /* Delay actually copying g_b to b until know everything is ok. */
    }
    else {
      ldb = 0;
      b   = NULL;
    }
  }
  else if( copy_c ) {

    nprow_c  = nprow_b;
    my_row_c = my_row_b;

    npcol_c  = npcol_a;
    my_col_c = my_col_a;

    ms_c = ms_b;
    ns_c = ns_a;

    if( my_row_c > -1 && my_col_c > -1 ) {
      ldc = ns_c[my_col_c];
      c   = ga_summa_alloc( ms_c[my_row_c]*ldc, "ga_summa_c", ma_handles+n_handles ); 
      n_handles++;

      if( c == NULL ) {
        status = ga_summa_ok( proclist_a, INSUFFICIENT_MEMORY );
        ga_summa_dealloc( ma_handles, n_handles-1 );
        return( status );
      }

      /* Delay actually copying g_c to c until know everything is ok. */
    }
    else {
      ldc = 0;
      c   = NULL;
    }
  }

  np_rows = nprow_a;
  np_cols = npcol_a;
  my_row  = my_row_a;
  my_col  = my_col_a;

  if( nprow_a != nprow_b || nprow_a != nprow_c ||
      npcol_a != npcol_b || npcol_a != npcol_c ||
      my_row_a != my_row_b || my_row_a != my_row_c ||
      my_col_a != my_col_b || my_col_a != my_col_c ) {
    status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
    ga_summa_dealloc( ma_handles, n_handles );
    return( status );
  }

  for( i=0; i<nprow_a*npcol_a; i++ )
    if( proclist_a[i] != proclist_b[i]  ||  proclist_a[i] != proclist_c[i] ) {
      status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
      ga_summa_dealloc( ma_handles, n_handles );
      return( status );
    }

  if( np_rows * np_cols > num_nodes )
    ga_error("ga_summa_c_-bug:  np_rows*np_cols > ga_nnodes_()", 0);

  if( my_col < 0 ) {

    /* Extra nodes wait till others finish matrix multi. before returning. */

    if( (status = ga_summa_ok( proclist_a, SUCCESSFUL_EXIT ) ) != SUCCESSFUL_EXIT ) {
      ga_summa_dealloc( ma_handles, n_handles );
      return( status );
    }
    ga_sync_();
    ga_sync_();
    ga_summa_dealloc( ma_handles, n_handles );
    ga_sync_();
    return( SUCCESSFUL_EXIT );
  }

  /* Compute size & allocate memory for work buffers */

  if( ! transa ) {
    if( transb ) {

      n_work1 = nb * ms_c[my_row_c];
      n_work2 = nb * ns_b[my_col_b];

      n_work2 = MAX( n_work2, n_work1 );
    }
    else {
      n_work1 = nb * ms_a[my_row_a];
      n_work2 = nb * ns_b[my_col_b];
    }
  }
  else {
    if( ! transb ) {

      n_work1 = nb * ms_a[my_row_a];
      n_work2 = nb * ns_c[my_col_c];

      n_work1 = MAX( n_work2, n_work1 );
    }
    else {

      n_work1 = nb * ms_b[my_row_b];
      n_work2 = nb * ns_a[my_col_a];
    }
  }

  work1 = ga_summa_alloc( n_work1, "ga_summa_work1", ma_handles+n_handles );
  n_handles++;

  if( work1 == NULL ) {
    status = ga_summa_ok( proclist_a, INSUFFICIENT_MEMORY );
    ga_summa_dealloc( ma_handles, n_handles-1 );
    return( status );
  }

  work2 = ga_summa_alloc( n_work2, "ga_summa_work2", ma_handles+n_handles );
  n_handles++;

  if( work2 == NULL ) {
    status = ga_summa_ok( proclist_a, INSUFFICIENT_MEMORY );
    ga_summa_dealloc( ma_handles, n_handles-1 );
    return( status );
  }


  /* --------------------- */
  /* Matrix multiplication */
  /* --------------------- */

  if( ! transa  ) {
    if( ! transb ) {

      /* A*B */

      for( i=0; i<nprow_a; i++ )
        if( ms_a[i] != ms_c[i] ) {
          status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
          ga_summa_dealloc( ma_handles, n_handles );
          return( status );
        }

      for( i=0; i<npcol_b; i++ )
        if( ns_b[i] != ns_c[i] ) {
          status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
          ga_summa_dealloc( ma_handles, n_handles );
          return( status );
        }

      if( (status = ga_summa_ok( proclist_a, SUCCESSFUL_EXIT ) ) != SUCCESSFUL_EXIT ) {
        ga_summa_dealloc( ma_handles, n_handles );
        return( status );
      }

      ga_sync_();

#ifdef SUMMA_TIMING
      times[3] = TCGTIME_();
#endif

      summa_ab( m, n, k, nb, d_alpha, a, lda, b, ldb,
                d_beta, c, ldc, ms_a, ns_a,  ms_b, ns_b, ms_c, ns_c,
                my_row, my_col, np_rows, np_cols, proclist_a, work1, work2 );

    }
    else {

      /* A*B' or indirect A'*B' */

      for( i=0; i<nprow_a; i++ )
        if( ms_a[i] != ms_c[i] ) {
          status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
          ga_summa_dealloc( ma_handles, n_handles );
          return( status );
        }

      for( i=0; i<npcol_b; i++ )
        if( ns_a[i] != ns_b[i] ) {
          status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
          ga_summa_dealloc( ma_handles, n_handles );
          return( status );
        }

      if( (status = ga_summa_ok( proclist_a, SUCCESSFUL_EXIT ) ) != SUCCESSFUL_EXIT ) {
        ga_summa_dealloc( ma_handles, n_handles );
        return( status );
      }

      /* Everything is ok, now copy g_a to a, if needed. */

      if( copy_a  && my_row_a > -1 && my_col_a > -1 )
        ga_summa_layout2( 0, my_col_a, ns_a, my_row_a, ms_a, a, lda, g_a, dummy );

      ga_sync_();

#ifdef SUMMA_TIMING
      times[3] = TCGTIME_();
#endif

      if( ! transab )
        summa_abt( m, n, k, nb, d_alpha, a, lda, b, ldb,
                   d_beta, c, ldc, ms_a, ns_a,  ms_b, ns_b, ms_c, ns_c,
                   my_row, my_col, np_rows, np_cols, proclist_a, work1, work2 );
      else
        summa_abt2( m, n, k, nb, d_alpha, a, lda, b, ldb,
                    d_beta, c, ldc, ms_a, ns_a,  ms_b, ns_b, ms_c, ns_c,
                    my_row, my_col, np_rows, np_cols, proclist_a, work1, work2 );
    }
  }
  else {
    if( ! transb  ) {

      /* A'*B or indirect A'*B' */

      for( i=0; i<nprow_a; i++ )
        if( ms_a[i] != ms_b[i] ) {
          status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
          ga_summa_dealloc( ma_handles, n_handles );
          return( status );
        }

      for( i=0; i<npcol_b; i++ )
        if( ns_b[i] != ns_c[i] ) {
          status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
          ga_summa_dealloc( ma_handles, n_handles );
          return( status );
        }

      if( (status = ga_summa_ok( proclist_a, SUCCESSFUL_EXIT ) ) != SUCCESSFUL_EXIT ) {
        ga_summa_dealloc( ma_handles, n_handles );
        return( status );
      }

      /* Everything is ok, now copy g_b to b, if needed. */

      if( copy_b  && my_row_b > -1 && my_col_b > -1 )
         ga_summa_layout2( 0, my_col_b, ns_b, my_row_b, ms_b, b, ldb, g_b, dummy );

      ga_sync_();

#ifdef SUMMA_TIMING
      times[3] = TCGTIME_();
#endif

      if( ! transab )
        summa_atb( m, n, k, nb, d_alpha, a, lda, b, ldb,
                   d_beta, c, ldc, ms_a, ns_a,  ms_b, ns_b, ms_c, ns_c,
                   my_row, my_col, np_rows, np_cols, proclist_a, work1, work2 );
      else
        summa_atb2( m, n, k, nb, d_alpha, a, lda, b, ldb,
                    d_beta, c, ldc, ms_a, ns_a,  ms_b, ns_b, ms_c, ns_c,
                    my_row, my_col, np_rows, np_cols, proclist_a, work1, work2 );
    }
    else {

      /* A'*B' */

      for( i=0; i<nprow_b; i++ )
        if( ms_b[i] != ms_c[i] ) {
          status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
          ga_summa_dealloc( ma_handles, n_handles );
          return( status );
        }

      for( i=0; i<npcol_a; i++ )
        if( ns_a[i] != ns_c[i] ) {
          status = ga_summa_ok( proclist_a, GA_ARRAYS_WRONG_SHAPE );
          ga_summa_dealloc( ma_handles, n_handles );
          return( status );
        }

      if( (status = ga_summa_ok( proclist_a, SUCCESSFUL_EXIT ) ) != SUCCESSFUL_EXIT ) {
        ga_summa_dealloc( ma_handles, n_handles );
        return( status );
      }

      /* Everything is ok, now copy g_c to c. */

      if( my_row_c > -1 && my_col_c > -1 )
        ga_summa_layout2( 0, my_col_c, ns_c, my_row_c, ms_c, c, ldc, g_c, dummy );

      ga_sync_();

#ifdef SUMMA_TIMING
      times[3] = TCGTIME_();
#endif

      summa_ab2( n, m, k, nb, d_alpha, b, ldb, a, lda, 
                 d_beta, c, ldc, ms_b, ns_b,  ms_a, ns_a, ms_c, ns_c,
                 my_row, my_col, np_rows, np_cols, proclist_a, work1, work2 );

      /* Copy SUMMA results to GA c array. */
      ga_summa_to_ga( my_col_c, ns_c, my_row_c, ms_c, c, ldc, g_c );

    }
  }

  /* Synchronize processors to make sure they end together */
  ga_sync_();

#ifdef SUMMA_TIMING
  times[4] = TCGTIME_();

  if( my_row == 0 && my_col == 0 )
    fprintf( stderr, "                       %d-by-%d nb = %d   %f  \n",
             np_rows, np_cols,  nb, times[4]-times[3] );
#endif

  ga_summa_dealloc( ma_handles, n_handles );

  /* Synchronize ALL processors coming out of here */
  ga_sync_();

  return( SUCCESSFUL_EXIT );
}

Integer ga_summa_ok( proclist, flag )
  Integer flag, *proclist;
{
  Integer type, value, n;
  char    op[6];

  extern char *strcpy();
 
  type  = 1;
  value = flag;
  n     = 1;
  strcpy( op, "max" );

  ga_igop( type, &value, n, op );

  return( value );

}
void ga_summa_dealloc( ma_handles, n_handles )
  Integer ma_handles[], n_handles;
{
  Integer i;

  for( i=n_handles-1; i>= 0; i-- )
    if( ! MA_pop_stack( ma_handles[i] ) )
      ga_error("ga_summa_:  pop stack failed for ga_summa workspace!", i);

}
void ga_to_summa( g_a, m, n, a, lda, nrow, ncol, my_row, my_col, mms, nns, plist, ma_handle )

Integer          g_a, m, n;
Integer          *lda, *nrow, *ncol, *my_row, *my_col, **mms, **nns, **plist, *ma_handle;
DoublePrecision  **a;

{
  static Integer   I_ONE = 1;

  Integer          me, num_nodes, i, j, jj, np, dim1, dim2, type, nprow, npcol,
                   totrow, totcol, ilo, ihi, jlo, jhi, index, ld, iproc;
  Integer          map_handle;

  Integer          *p, *map, *proclist, *ms, *ns;

  char             *ma_pointer;

  /* Initialize all outputs to allow graceful error return. */

  *a         = NULL;
  *lda       = 0;
  *nrow      = 0;
  *ncol      = 0;
  *my_row    = -1;
  *my_col    = -1;
  *mms       = NULL;
  *nns       = NULL;
  *plist     = NULL;
  *ma_handle = -1;

  me        = ga_nodeid_(); 
  num_nodes = ga_nnodes_(); 


  /* check n,m and data type of g_a */


  ga_inquire_( &g_a, &type, &dim1, &dim2 );
  if( dim1 != m || dim2 != n || type != MT_F_DBL )
     ga_error("ga_to_summa: ga type doesn't agree with given n,m, data type",0);


  /* Get an array in which to store the processor list, ms, and ns. */

  ma_pointer = NULL;
  if( MA_push_stack(MT_F_INT, 2*num_nodes+2, "proclist", ma_handle ) ) {
     if( ! MA_get_pointer( *ma_handle, &ma_pointer  ) )
       ga_error("ga_summa_: MA_get_pointer, failed for proclist.", 0);
  }
  else
     ga_error("ga_summa_: MA_push_stack, failed for proclist.", 0);

  proclist = (Integer *) ma_pointer;

  /* Get a work array to pass to ga_locate region. */

  if( MA_push_stack(MT_F_INT, 5*num_nodes, "map", &map_handle ) )
     MA_get_pointer( map_handle, &ma_pointer );
  else
     ga_error("ga_summa_: MA_push_stack, failed for map.", 0);

  map = (Integer *) ma_pointer;

  /* Find which processor owns which block of g_a. */

  if( ! ga_locate_region_( &g_a, &I_ONE, &m, &I_ONE, &n, map, &np ) )
     ga_error("ga_to_summa-bug: error calling ga_locate_region", 0);

  /* Processor ids in map should be 0,1,...,np-1, check this. */

  if( np < 1 || np > num_nodes )
     ga_error("ga_to_summa-bug: ga_locate_region return bad numb. of proc.", 0);

  if( map[2] == 1 && map[3] == n ) {
    nprow = np;
    npcol = 1;
  }
  else if( np > 1 ) {
    nprow = map[9];
    npcol = np / nprow;
  }
  else {
    nprow = 1;
    npcol = np;
  }

  if( nprow < 1 || nprow *npcol != np ) 
     ga_error("ga_to_summa-bug: nprow < 1 or nprow*npcol != np", 0);

  p = map+4;
  for( j=0; j<nprow; j++ ) {
    iproc = j;
    for( i=0; i<npcol; i++, iproc+=nprow, p+=5 )
      if( *p != iproc )
         ga_error("ga_to_summa-bug: ga_locate_region's proclist is not 0,1,...", 0);
  }


  /* Get list of "real, i.e., tcgmsg" node ids. */

  ga_list_nodeid_( proclist, &num_nodes );

  if( proclist[me] != NODEID_() )
    ga_error("ga_to_summa-bug: proclist is wrong!", 0);


  /* Fill in ms, ns */

  ns = proclist + num_nodes;
  ms = ns + npcol;

  iproc  = 0;
  ihi    = 0;
  totcol = 0;
  for( i=2; i<5*np; i+=5 ) {
    ilo = map[i];
    if( ilo == ihi + 1 ) {
      ns[iproc] = map[i+1] - ilo + 1;
      totcol += ns[iproc];
      iproc++;
      ihi = map[i+1];
    }
    else if( ilo == 1 )
       break;
    else
       ga_error("ga_to_summa-bug: column ilos out of order ", 0);
       
  }

/*
  if( me == 0 ) {
          fprintf( stderr, " iproc,nproc, %d %d  %d %d\n", iproc, npcol, totcol, n);
          for( i=0; i<iproc; i++)
            fprintf( stderr, " ns[%d]= %d \n", i, ns[i]);
          for( i=0; i<5*np; i++)
            fprintf( stderr, " map[%d]= %d \n", i, map[i]);
   }
  ga_sync_();
  exit(-1);
*/

  if( iproc != npcol  || totcol != n )
       ga_error("ga_to_summa-bug: iproc != npcol or block sizes do not sum to n", 0);

  iproc  = 0;
  ihi    = 0;
  totrow = 0;
  for( i=0; i<5*np; i+=5*npcol ) {
    ilo = map[i];
    if( ilo == ihi + 1 ) {
      ms[iproc] = map[i+1] - ilo + 1;
      totrow += ms[iproc];
      iproc++;
      ihi = map[i+1];
    }
    else if( ilo == 1 )
       break;
    else
       ga_error("ga_to_summa: bug, row ilos out of order ", 0);
  }

  if( iproc != nprow  || totrow != m )
       ga_error("ga_to_summa-bug: iproc != nprow or block sizes do not sum to m", 0);


  /* Get address of start of g_a and lda */

  if( me < np ) {
    ga_distribution_( &g_a, &me, &ilo, &ihi, &jlo, &jhi );

    i = ( me / nprow +  ( me%nprow ) * npcol ) * 5;
    if( map[i] != ilo || map[i+1] != ihi || map[i+2] != jlo || map[i+3] != jhi )
       ga_error("ga_to_summa-bug: ilo,ihi,etc from ga_distribution and ga_locate_region conflict", 0);

    ga_access_( &g_a, &ilo, &ihi, &jlo, &jhi, &index, &ld );

    if( ld < ihi-ilo+1 )
       ga_error("ga_to_summa-bug: ld from ga_access is too small. ", 0);

    *a = DBL_MB + index - 1;
    *lda   = ld;
  }

  *nrow  = nprow;
  *ncol  = npcol;
  *plist = proclist;
  *mms   = ms;
  *nns   = ns;

  if( me < np ) {
    *my_col = me / nprow;
    *my_row = me - *my_col * nprow;
  }

  /* Release workspace for map, but not proclist. */

  if( ! MA_pop_stack( map_handle ) )
    ga_error("ga_to_summa:  pop stack failed for map!", 0);

}
