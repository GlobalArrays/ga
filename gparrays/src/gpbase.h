#ifndef __GPBASE_H__
#define __GPBASE_H__

#include "gacommon.h"

#if SIZEOF_VOIDP == SIZEOF_INT
#   define GP_POINTER_TYPE C_INT
#elif SIZEOF_VOIDP == SIZEOF_LONG
#   define GP_POINTER_TYPE C_LONG
#else
#   error sizeof(void*) is not sizeof(int) nor sizeof(long)
#endif

#endif /* __GPBASE_H__ */
