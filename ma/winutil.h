#ifndef _WINUTIL_H
#define _WINUTIL_H

#define bzero(A,N) memset((A), 0, (N))

#include <windows.h>
#define sleep(x) Sleep(1000*(x))

#endif
