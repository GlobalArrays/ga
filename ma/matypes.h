#ifndef _matypes_h
#define _matypes_h

/* 
 * $Id: matypes.h,v 1.1.1.1 1994-03-29 06:44:34 d3g681 Exp $
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
#else
typedef int Integer;
#endif

typedef Integer Boolean;	/* MA_TRUE or MA_FALSE */
typedef char * Pointer;		/* generic pointer */

#endif /* _matypes_h */
