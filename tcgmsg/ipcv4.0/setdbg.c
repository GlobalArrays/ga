#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include "typesf2c.h"
#include "sndrcv.h"
#include "sndrcvP.h"

/**
 * Set global debug flag for this process to value.
 */
void SETDBG_(Integer *value)
{
    SR_debug = *value;
}
