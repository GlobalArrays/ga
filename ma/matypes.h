#ifndef _matypes_h
#define _matypes_h

/* 
 * $Id: matypes.h,v 1.3 1994-10-07 19:41:30 d3g681 Exp $
 */

/* 
 * Private header file containing C type definitions.
 *
 * This file should only be included directly by internal C
 * header files (e.g., macdecls.h).  It may be included indirectly
 * by external C files that include the appropriate header
 * file (e.g., macdecls.h).
 */

/**
 ** types
 **/

/* sizeof(Integer) must equal sizeof(FORTRAN integer) */
#ifdef LongInteger
    typedef long Integer;
#else /* LongInteger */
    typedef int Integer;
#endif /* LongInteger */

typedef Integer Boolean;	/* MA_TRUE or MA_FALSE */
typedef char * Pointer;		/* generic pointer */

/* not all C compilers support long double */
#ifdef UseIntrinsicLongDouble
    typedef long double MA_LongDouble;
#else /* UseIntrinsicLongDouble */
    typedef struct {double dummy[2];} MA_LongDouble;
#endif /* UseIntrinsicLongDouble */

/* no C compilers support complex types */
typedef struct {float dummy[2];} MA_SingleComplex;
typedef struct {double dummy[2];} MA_DoubleComplex;
typedef struct {double dummy[4];} MA_LongDoubleComplex;

#endif /* _matypes_h */
