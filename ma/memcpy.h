#ifndef _memcpy_h
#define _memcpy_h

/* 
 * $Id: memcpy.h,v 1.4 2002-04-02 01:36:22 edo Exp $
 */

/* 
 * Private header file containing symbolic constants, type declarations,
 * and macro definitions for OS memory routines, to provide a level of
 * abstraction between them and routines that use them.
 *
 * This file should only be included by internal C files.
 */

#include <malloc.h>

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
