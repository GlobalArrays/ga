#ifndef _WINUTIL_H
#define _WINUTIL_H

/*
 * $Id: winutil.h,v 1.2 2000-07-04 05:54:56 d3g001 Exp $
 */

#define bzero(A,N) memset((A), 0, (N))

#include <windows.h>
#define sleep(x) Sleep(1000*(x))

#endif /* _WINUTIL_H */
