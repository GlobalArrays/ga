/* 
 * DISCLAIMER
 *
 * This material was prepared as an account of work sponsored by an
 * agency of the United States Government.  Neither the United States
 * Government nor the United States Department of Energy, nor Battelle,
 * nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
 * COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
 * SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
 * INFRINGE PRIVATELY OWNED RIGHTS.
 *
 *
 * ACKNOWLEDGMENT
 *
 * This software and its documentation were produced with United States
 * Government support under Contract Number DE-AC06-76RLO-1830 awarded by
 * the United States Department of Energy.  The United States Government
 * retains a paid-up non-exclusive, irrevocable worldwide license to
 * reproduce, prepare derivative works, perform publicly and display
 * publicly by or for the US Government, including the right to
 * distribute to other US Government contractors.
 */
#if HAVE_CONFIG_H
#   include "config.h"
#endif
 
/*#define PERMUTE_PIDS */
/*#define USE_GATSCAT_NEW 1 */

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif
#if HAVE_STDINT_H
#   include <stdint.h>
#endif
#if HAVE_MATH_H
#   include <math.h>
#endif
#if HAVE_ASSERT_H
#   include <assert.h>
#endif
#if HAVE_STDDEF_H
#include <stddef.h>
#endif

#include "global.h"
#include "globalp.h"
#include "base.h"
#include "armci.h"
#include "macdecls.h"
#include "ga-papi.h"
#include "ga-wapi.h"

#ifndef DISABLE_UNSAFE_GA_FENCE
char *fence_array; /* RACE */
static int GA_fence_set=0; /* RACE */
#endif // DISABLE_UNSAFE_GA_FENCE

/**
 *  Wait until all requests initiated by calling process are completed
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_fence = pnga_fence
#endif

void pnga_fence(void)
{
#ifndef DISABLE_UNSAFE_GA_FENCE
    if(GA_fence_set<1) {
        pnga_error("ga_fence: fence not initialized",0);
    }

    /* Why does this not set the fence array to zero? */
    GA_fence_set--;

    for(int proc=0; proc<GAnproc; proc++) {
        if(fence_array[proc]) {
            ARMCI_Fence(proc);
        }
    }
    bzero(fence_array,(int)GAnproc);
#else
    ARMCI_AllFence();
#endif // DISABLE_UNSAFE_GA_FENCE
}

/**
 *  Initialize tracing of request completion
 */
#if HAVE_SYS_WEAK_ALIAS_PRAGMA
#   pragma weak wnga_init_fence = pnga_init_fence
#endif

void pnga_init_fence(void)
{
#ifndef DISABLE_UNSAFE_GA_FENCE
    /* Why is this not setting it to 1? */
    GA_fence_set++;
#else
    /* NO-OP */
#endif // DISABLE_UNSAFE_GA_FENCE
}

void gai_init_onesided(void)
{
#ifndef DISABLE_UNSAFE_GA_FENCE
    fence_array = calloc((size_t)GAnproc,1);
    if(!fence_array)
        pnga_error("ga_init:calloc failed",0);
#else
    /* NO-OP */
#endif // DISABLE_UNSAFE_GA_FENCE
}

void gai_finalize_onesided(void)
{
#ifndef DISABLE_UNSAFE_GA_FENCE
    free(fence_array);
    fence_array = NULL;
#else
    /* NO-OP */
#endif // DISABLE_UNSAFE_GA_FENCE
}

void gai_fence_reset(void)
{
#ifndef DISABLE_UNSAFE_GA_FENCE
       if (GA_fence_set) {
           bzero(fence_array,(int)GAnproc);
       }
       GA_fence_set=0;
#else
    /* NO-OP */
#endif // DISABLE_UNSAFE_GA_FENCE
}

void gai_fence_set(int proc)
{
#ifndef DISABLE_UNSAFE_GA_FENCE
    if(GA_fence_set)
      fence_array[proc]=1;
#else
    /* NO-OP */
#endif // DISABLE_UNSAFE_GA_FENCE
}


