/* comex definition file. This file is used to get around some issues with GPU
 * compilers */
#ifndef _COMEX_DEFS_H
#define _COMEX_DEFS_H

#define COMEX_GROUP_WORLD 0
#define COMEX_GROUP_DEVICE_WORLD 1
#define COMEX_GROUP_NULL -1
 
#define COMEX_SUCCESS 0
#define COMEX_FAILURE 1

#define COMEX_SWAP 10
#define COMEX_SWAP_LONG 11
#define COMEX_FETCH_AND_ADD 12
#define COMEX_FETCH_AND_ADD_LONG 13

#define COMEX_ACC_OFF 36
#define COMEX_ACC_INT (COMEX_ACC_OFF + 1)
#define COMEX_ACC_DBL (COMEX_ACC_OFF + 2)
#define COMEX_ACC_FLT (COMEX_ACC_OFF + 3)
#define COMEX_ACC_CPL (COMEX_ACC_OFF + 4)
#define COMEX_ACC_DCP (COMEX_ACC_OFF + 5)
#define COMEX_ACC_LNG (COMEX_ACC_OFF + 6)

#define COMEX_MAX_STRIDE_LEVEL 8

#endif //_COMEX_DEFS_H
