#ifndef _ma_h
#define _ma_h

/* 
 * $Id: ma.h,v 1.3 1994-10-07 19:41:25 d3g681 Exp $
 */

/* 
 * Private header file containing symbolic constants and type declarations
 * for internal C routines.
 *
 * This file should only be included by internal C files.
 */

#include "macdecls.h"

/**
 ** function types
 **/

extern Boolean ma_inform_base();
extern void ma_summarize_allocated_blocks();

#endif /* _ma_h */
