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
