/*************** interface to a parallel communication system ********/
#define SFSYNC 29000

#ifndef TSCMSG 
#   include "ga.h"
#   define SYNC      GA_Sync
#   define NPROC     GA_Nnodes
#   define ME        GA_Nodeid
#   define ERROR     GA_Error
#else
#   include "sndrcv.h"
    long sync_flag = SFSYNC;
#   define SYNC()    SYNCH_(&sync_flag)
#   define NPROC     NNODES_
#   define ME        NODEID_
#   define ERROR  Error
#endif

