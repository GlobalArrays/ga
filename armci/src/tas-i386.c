#include "tas-i386.h"

int gcc_testandset(int *spinlock)
{
    int rc=testandset(spinlock);
    return rc;
}

