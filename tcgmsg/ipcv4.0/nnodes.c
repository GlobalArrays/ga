#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "sndrcv.h"
#include "sndrcvP.h"

/**
 * Return total no. of processes.
 */
Integer NNODES_()
{
    return SR_n_proc;
}
