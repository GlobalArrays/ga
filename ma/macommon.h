#ifndef _macommon_h
#define _macommon_h

/* 
 * $Id: macommon.h,v 1.3 1994-10-07 19:41:27 d3g681 Exp $
 */

/* 
 * Private header file containing common symbolic definitions.
 *
 * This file should only be included directly by internal C and FORTRAN
 * header files (e.g., ma[cf]decls.h).  It may be included indirectly
 * by external C and FORTRAN files that include the appropriate header
 * files (e.g., ma[cf]decls.h).
 */

/**
 ** constants
 **/

/* values for Boolean */
#define MA_FALSE	0
#define MA_TRUE		1

/* symbolic default value for size parameters in MA_init */
#define MA_DEFAULT_SPACE	(-1)

/* max length (including trailing \0) of client-assigned block name */
#define MA_NAMESIZE	32

/* added to MT values to keep them from being small integers */
#define MT_BASE		1000

/* internal C datatypes */
#define MT_C_CHAR	(MT_BASE + 0)	/* char */
#define MT_C_INT	(MT_BASE + 1)	/* int */
#define MT_C_LONGINT	(MT_BASE + 2)	/* long int */
#define MT_C_FLOAT	(MT_BASE + 3)	/* float */
#define MT_C_DBL	(MT_BASE + 4)	/* double */
#define MT_C_LDBL	(MT_BASE + 5)	/* long double */
#define MT_C_SCPL	(MT_BASE + 6)	/* single precision complex */
#define MT_C_DCPL	(MT_BASE + 7)	/* double precision complex */
#define MT_C_LDCPL	(MT_BASE + 8)	/* long double precision complex */

/* internal FORTRAN datatypes */
#define MT_F_BYTE	(MT_BASE + 9)	/* byte */
#define MT_F_INT	(MT_BASE + 10)	/* integer */
#define MT_F_LOG	(MT_BASE + 11)	/* logical */
#define MT_F_REAL	(MT_BASE + 12)	/* real */
#define MT_F_DBL	(MT_BASE + 13)	/* double precision */
#define MT_F_SCPL	(MT_BASE + 14)	/* single precision complex */
#define MT_F_DCPL	(MT_BASE + 15)	/* double precision complex */

/* internal datatype constants */
#define MT_FIRST	MT_C_CHAR			/* first type */
#define MT_LAST		MT_F_DCPL			/* last type */
#define MT_NUMTYPES	(MT_LAST - MT_FIRST + 1)	/* # of types */

/* map foo_ to FOO on certain platforms */
#ifdef _CRAY
#define ma_set_sizes_				MA_SET_SIZES
#define f2c_alloc_get_				F2C_ALLOC_GET
#define f2c_allocate_heap_			F2C_ALLOCATE_HEAP
#define f2c_chop_stack_				F2C_CHOP_STACK
#define f2c_free_heap_				F2C_FREE_HEAP
#define f2c_get_index_				F2C_GET_INDEX
#define f2c_get_next_memhandle_			F2C_GET_NEXT_MEMHANDLE
#define f2c_inform_base_			F2C_INFORM_BASE
#define f2c_inform_base_fcd_			F2C_INFORM_BASE_FCD
#define f2c_init_				F2C_INIT
#define f2c_init_memhandle_iterator_		F2C_INIT_MEMHANDLE_ITERATOR
#define f2c_inquire_avail_			F2C_INQUIRE_AVAIL
#define f2c_inquire_heap_			F2C_INQUIRE_HEAP
#define f2c_inquire_stack_			F2C_INQUIRE_STACK
#define f2c_pop_stack_				F2C_POP_STACK
#define f2c_print_stats_			F2C_PRINT_STATS
#define f2c_push_get_				F2C_PUSH_GET
#define f2c_push_stack_				F2C_PUSH_STACK
#define f2c_set_auto_verify_			F2C_SET_AUTO_VERIFY
#define f2c_set_error_print_			F2C_SET_ERROR_PRINT
#define f2c_set_hard_fail_			F2C_SET_HARD_FAIL
#define f2c_sizeof_				F2C_SIZEOF
#define f2c_sizeof_overhead_			F2C_SIZEOF_OVERHEAD
#define f2c_summarize_allocated_blocks_		F2C_SUMMARIZE_ALLOCATED_BLOCKS
#define f2c_verify_allocator_stuff_		F2C_VERIFY_ALLOCATOR_STUFF
#endif /* _CRAY */

#endif /* _macommon_h */
