#ifndef _memcpy_h
#define _memcpy_h

/* 
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
#define bytealloc(nbytes)	malloc((unsigned)(nbytes))

/* deallocate bytes */
#define bytefree(pointer)	(void)free((char *)(pointer))

/* copy bytes */
#ifdef NO_BCOPY
extern void *memcpy();
#define bytecopy(from,to,nbytes)	\
	((void)memcpy((char *)(to), (char *)(from), (int)(nbytes)))
#else
extern void bcopy();
#define bytecopy(from,to,nbytes)	\
	(bcopy((char *)(from), (char *)(to), (int)(nbytes)))
#endif

#endif /* _memcpy_h */
