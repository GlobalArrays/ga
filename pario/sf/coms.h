/*************** interface to a parallel communication system ********/
#define SFSYNC 29000

#ifndef TSCMSG 
#   include "global.h"
#   define SYNC      GA_sync
#   define NPROC     GA_nnodes
#   define ME        GA_nodeid
#   define ERROR     GA_error
#else
#   include "sndrcv.h"
    long sync_flag = SFSYNC;
#   define SYNC()    SYNCH_(&sync_flag)
#   define NPROC     NNODES_
#   define ME        NODEID_
#   define ERROR  Error
#endif

