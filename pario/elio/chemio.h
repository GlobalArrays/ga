/*
Header file for interface with ELIO based codes:
   Defines:
      Read/Write permission macros
      Asynch I/O status flags
      PRINT_AND_ABORT if not already defined
*/


#if !defined(CHEMIO_H)
#define CHEMIO_H


#define   ELIO_RW  -1
#define   ELIO_W   -2
#define   ELIO_R   -3
 

#define   ELIO_DONE    -1
#define   ELIO_PENDING  1
#define   ELIO_ERROR    2

#define   ELIO_OK       10
#define   ELIO_FAIL     11

#endif

/*  Pablo profiler definitions */

#  define PABLO_elio_write	710000
#  define PABLO_elio_awrite	710001
#  define PABLO_elio_read	710002
#  define PABLO_elio_aread	710003
#  define PABLO_elio_wait	710004
#  define PABLO_elio_probe	710005
#  define PABLO_elio_stat	710006
#  define PABLO_elio_open	710007
#  define PABLO_elio_gopen	710008
#  define PABLO_elio_close	710009
#  define PABLO_elio_set_cb	710010
#  define PABLO_elio_delete	710011
#  define PABLO_elio_init	710012

#  define PABLO_eaf_writec	720000
#  define PABLO_eaf_awritec	720001
#  define PABLO_eaf_readc	720002
#  define PABLO_eaf_areadc	720003
#  define PABLO_eaf_waitc	720004
#  define PABLO_eaf_probec	720005
#  define PABLO_eaf_openpc	720006
#  define PABLO_eaf_opensc	720007
#  define PABLO_eaf_closec	720008
#  define PABLO_eaf_init	720009

#if defined(PABLO)
#  define PABLO_init		initIOTrace()
#  define PABLO_start(_id)	startTimeEvent( _id )
#  define PABLO_end(_id)	endTimeEvent( _id )
#  define PABLO_terminate	{endIOTrace(); endTracing();}
#else
#  define PABLO_init
#  define PABLO_start(_id)
#  define PABLO_end(_id)
#  define PABLO_terminate
#endif

