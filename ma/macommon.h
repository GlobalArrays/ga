#ifndef _macommon_h
#define _macommon_h

/* 
 * $Id: macommon.h,v 1.1.1.1 1994-03-29 06:44:34 d3g681 Exp $
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

#endif /* _macommon_h */
