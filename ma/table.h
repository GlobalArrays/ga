#ifndef _table_h
#define _table_h

/* 
 * $Id: table.h,v 1.1.1.1 1994-03-29 06:44:34 d3g681 Exp $
 */

/* 
 * Private header file containing symbolic constants and type declarations
 * for the table module.
 *
 * This file should only be included by internal C files.
 */

#include "matypes.h"

/**
 ** constants
 **/

/* invalid handle */
#define TABLE_HANDLE_NONE (Integer)(-1)

/**
 ** types
 **/

/* type of data in each table entry */
typedef char * TableData;

/**
 ** function types
 **/

extern Integer table_allocate();
extern void table_deallocate();
extern TableData table_lookup();
extern Integer table_lookup_assoc();
extern Boolean table_verify();

#endif /* _table_h */
