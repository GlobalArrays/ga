/*
 * USC.H  (Public header file for the Microsecond Clock package)
 *     This header file has to be included by an application using
 * the USC function calls.
 *
 * Written by:  Arun Nanda    (07/17/91)
 *
 * The following machine-specific libraries need to be linked in
 * with the application using the UST functions:
 *
 *      MULTIMAX      -lpp
 *      BALANCE       -lseq
 *      SYMMETRY      -lseq
 */

#ifndef _USC_DEFS_	/* takes care of multiple inclusion of this file */
#define _USC_DEFS_

/* ---------------------
 Global declarations
--------------------- */

typedef unsigned long usc_time_t;

#ifndef VOID
#    if defined(BALANCE)
#        define VOID int
#    else
#        define VOID void
#    endif
#endif

/* --------------------------------
 Machine dependent declarations
-------------------------------- */

#if defined(MULTIMAX)

     extern unsigned *usc_multimax_timer;

#elif (defined(BALANCE) || defined(SYMMETRY))

#ifndef GETUSCLK
#    include <usclkc.h>
#endif

#else

    extern usc_time_t usc_MD_reference_time;

#endif


extern usc_time_t usc_MD_rollover_val;

/* -----------------------
 user interface macros
----------------------- */

#if defined(MULTIMAX)

#    define usc_clock() ((usc_time_t) *usc_multimax_timer)
#    define usc_rollover_val()  (usc_MD_rollover_val)

#elif (defined(BALANCE) || defined(SYMMETRY))

#    define usc_clock() ((usc_time_t) getusclk())
#    define usc_rollover_val()  (usc_MD_rollover_val)

#elif (defined(BFLY2) || defined(BFLY2_TCMP) || defined (IPSC860_NODE) \
    || defined(IPSC860_NODE_PGI) || defined(DELTA))

#    define usc_clock() usc_MD_clock()
#    define usc_rollover_val()  (usc_MD_rollover_val)

#else

#    define usc_clock() usc_MD_clock()
#    define usc_rollover_val()  (usc_MD_rollover_val * 1000000 - 1)

#endif

/*#    define usc_clock() 0*/
/*#    define usc_rollover_val() 0*/


/* ----------------------
  function prototypes
---------------------- */

VOID usc_init();
usc_time_t usc_MD_clock();

#endif
