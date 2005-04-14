#ifndef _memcpy_h
#define _memcpy_h

/* 
 * $Id: memcpy.h,v 1.6 2005-04-14 19:05:21 manoj Exp $
 */

/* 
 * Private header file containing symbolic constants, type declarations,
 * and macro definitions for OS memory routines, to provide a level of
 * abstraction between them and routines that use them.
 *
 * This file should only be included by internal C files.
 */

#if !defined(MACX) && !defined(__FreeBSD__)
#include <malloc.h>
#endif

/**
 ** constants
 **/

/* ensure that NULL is defined */
#ifndef NULL
#define NULL 0
#endif

/**
 ** macros
 **/

/* allocate bytes */
#define bytealloc(nbytes)	malloc((unsigned long)(nbytes))

/* deallocate bytes */
#define bytefree(pointer)	(void)free((char *)(pointer))

#ifdef WIN32
#  define NO_BCOPY
#  include <string.h>
#endif

/* copy bytes */
#ifdef NO_BCOPY
#ifndef WIN32
extern void *memcpy();
#endif /* WIN32 */
#define bytecopy(from,to,nbytes)	\
	((void)memcpy((char *)(to), (char *)(from), (int)(nbytes)))
#else /* NO_BCOPY */
extern void bcopy();
#define bytecopy(from,to,nbytes)	\
	(bcopy((char *)(from), (char *)(to), (int)(nbytes)))
#endif /* NO_BCOPY */

#endif /* _memcpy_h */
