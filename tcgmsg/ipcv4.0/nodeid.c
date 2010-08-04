#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "sndrcv.h"
#include "sndrcvP.h"

/**
 * Return logical node no. of current process.
 */
Integer NODEID_()
{
    return SR_proc_id;
}
