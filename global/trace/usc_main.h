/*
 * USC_MAIN.H  (Private header file for the Microsecond Clock package)
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


#include "usc.h"


#if defined(MULTIMAX)

#    include <parallel.h>
#    define usc_MD_timer_size  (sizeof(unsigned)*8)
     unsigned *usc_multimax_timer;

#elif (defined(BALANCE) || defined(SYMMETRY))

#    define usc_MD_timer_size  (sizeof(usclk_t)*8)


#elif (defined(BFLY2) || defined(BFLY2_TCMP))

#    define usc_MD_timer_size  (sizeof(unsigned long)*8)

#elif (defined(DELTA)||defined(PARAGON))

#    if (defined (DELTA))
#        include <mesh.h>
#    else
#        include <nx.h>
#    endif
#    define usc_MD_timer_size ((sizeof(long)*8)+3)
#    define usc_MD_ticks_per_usec (HWHZ/1000000)

#else

#	include <sys/time.h>
	usc_time_t usc_MD_reference_time = 0;

#endif


usc_time_t usc_MD_rollover_val = 0;

