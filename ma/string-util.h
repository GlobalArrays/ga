#ifndef _string_util_h
#define _string_util_h

/*
 * $Id: string-util.h,v 1.3 2000-07-04 05:54:56 d3g001 Exp $
 */

/* 
 * Private header file for string utilities.
 *
 * This file should only be included by internal C files.
 */

/**
 ** constants
 **/

/* str_match return values */
#define SM_NONE (-1)
#define SM_MANY (-2)

/**
 ** function types
 **/

extern unsigned int str_len();
extern int str_match();

#endif /* _string_util_h */
