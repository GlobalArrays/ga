#ifndef _macdecls_h
#define _macdecls_h

/* 
 * $Id: macdecls.h,v 1.9 2000-07-04 11:27:03 d3g001 Exp $
 */

/* 
 * Public header file for a portable dynamic memory allocator.
 *
 * This file may be included by internal and external C files.
 */

#include "macommon.h"
#include "matypes.h"

/**
 ** constants
 **/

/* datatypes */
#define MT_CHAR		MT_C_CHAR	/* char */
#define MT_INT		MT_C_INT	/* int */
#define MT_LONGINT	MT_C_LONGINT	/* long int */
#define MT_FLOAT	MT_C_FLOAT	/* float */
#define MT_DBL		MT_C_DBL	/* double */
#define MT_LDBL		MT_C_LDBL	/* long double */
#define MT_SCPL		MT_C_SCPL	/* single precision complex */
#define MT_DCPL		MT_C_DCPL	/* double precision complex */
#define MT_LDCPL	MT_C_LDCPL	/* long double precision complex */

#define MT_C_FIRST	MT_CHAR		/* first type */
#define MT_C_LAST	MT_LDCPL	/* last type */

/**
 ** function types
 **/

extern Boolean MA_alloc_get();
extern Boolean MA_allocate_heap();
extern Boolean MA_chop_stack();
extern Boolean MA_free_heap();
extern Boolean MA_free_heap_piece();
extern Boolean MA_get_index();
extern Pointer MA_get_mbase();
extern Boolean MA_get_next_memhandle();
extern Boolean MA_get_numalign();
extern Boolean MA_get_pointer();
extern Boolean MA_init();
extern Boolean MA_initialized();
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
extern Boolean MA_set_numalign();
extern Integer MA_sizeof();
extern Integer MA_sizeof_overhead();
extern void MA_summarize_allocated_blocks();
extern void MA_trace();
extern Boolean MA_verify_allocator_stuff();

/**
 ** variables
 **/

/* base arrays for the C datatypes */
extern char			ma_cb_char[];	/* MT_C_CHAR */
extern int			ma_cb_int[];	/* MT_C_INT */
extern long			ma_cb_long[];	/* MT_C_LONGINT */
extern float			ma_cb_float[];	/* MT_C_FLOAT */
extern double			ma_cb_dbl[];	/* MT_C_DBL */
extern MA_LongDouble		ma_cb_ldbl[];	/* MT_C_LDBL */
extern MA_SingleComplex		ma_cb_scpl[];	/* MT_C_SCPL */
extern MA_DoubleComplex		ma_cb_dcpl[];	/* MT_C_DCPL */
extern MA_LongDoubleComplex	ma_cb_ldcpl[];	/* MT_C_LDCPL */

#endif /* _macdecls_h */
