#ifndef _matypes_h
#define _matypes_h

/* 
 * $Id: matypes.h,v 1.4 1996-10-02 23:59:37 d3h325 Exp $
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
/* JN: changed LongInteger into EXT_INT to avoid conflicts with GA and PEIGS */
#ifdef EXT_INT 
    typedef long Integer;
#else 
    typedef int Integer;
#endif 

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
