/*************** interface to a parallel communication system ********/
#define SFSYNC 29000

#ifndef TSCMSG 
#   ifdef CRAY
#      define SYNC      GA_SYNC
#      define NPROC     GA_NNODES
#      define ME        GA_NODEID
#   else
#      define SYNC      ga_sync_
#      define NPROC     ga_nnodes_
#      define ME        ga_nodeid_
#   endif
#   define ERROR  ga_error
    extern void    SYNC();
#else
    long sync_flag = SFSYNC;
#   ifdef CRAY
#      define SYNC()    SYNCH(&sync_flag)
#      define NPROC     NNODES
#      define ME        NODEID
       extern void      SYNCH();
#   else
#      define SYNC()    synch_(&sync_flag)
#      define NPROC     nnodes_
#      define ME        nodeid_
       extern void      synch_();
#   endif
#   define ERROR  Error
#endif

extern Integer NPROC(), ME();
extern void    ERROR();


