/******************************************************************************
Source File:    eaf_c2f.h

Description:    Fortran bindings header file for EAF
                Mainly stolen from Brian Toonen

Author:         Jace A Mogill

Date Created:   16 May 1996

Modifications:

CVS: $Source: /tmp/hpctools/ga/pario/eaf/eaf_c2f.h,v $
CVS: $Date: 1996-07-17 15:28:24 $
CVS: $Revision: 1.1 $
CVS: $State: Exp $
******************************************************************************/


/*** Macro concatenation (useful for creating function names) ***/
#define UTIL_CAT(_str1, _str2) _str1 ## _str2
#define CAT(_str1, _str2) UTIL_CAT(_str1, _str2)
#define CAT3(_str1, _str2, _str3) CAT(CAT(_str1, _str2), _str3)



/******************************************************************************
                          Name Space Transformations
******************************************************************************/
#if defined(EAF_FC_PREFIX) && defined(EAF_FC_POSTFIX)
#  define FORT_NAME(_func) CAT3(EAF_FC_PREFIX, _func, EAF_FC_POSTFIX)
#elif  defined(EAF_FC_PREFIX)
#  define FORT_NAME(_func) CAT(EAF_FC_PREFIX, _func)
#elif  defined(EAF_FC_POSTFIX)
#  define FORT_NAME(_func) CAT(_func, EAF_FC_POSTFIX)
#else
#  define FORT_NAME(_func) _func
#endif

#if defined(EAF_FC_LOWERCASE)
#  define EAF_Close FORT_NAME(eaf_close)
#  define EAF_OpenPersist FORT_NAME(eaf_openpersist)
#  define EAF_OpenScratch FORT_NAME(eaf_openscratch)
#  define EAF_Read FORT_NAME(eaf_read)
#  define EAF_ARead FORT_NAME(eaf_aread)
#  define EAF_Write FORT_NAME(eaf_write)
#  define EAF_AWrite FORT_NAME(eaf_awrite)
#  define EAF_Probe FORT_NAME (eaf_probe)
#  define EAF_Wait FORT_NAME (eaf_wait)
#elif defined(EAF_FC_UPPERCASE)
#  define EAF_Close FORT_NAME(EAF_CLOSE)
#  define EAF_OpenPersist FORT_NAME(EAF_OPENPERSIST)
#  define EAF_OpenScratch FORT_NAME(EAF_OPENSCRATCH)
#  define EAF_Read FORT_NAME(EAF_READ)
#  define EAF_ARead FORT_NAME(EAF_AREAD)
#  define EAF_Write FORT_NAME(EAF_WRITE)
#  define EAF_AWrite FORT_NAME(EAF_AWRITE)
#  define EAF_Probe FORT_NAME (EAF_PROBE)
#  define EAF_Wait FORT_NAME (EAF_WAIT)
#else
#  define EAF_Close FORT_NAME(EAF_Close)
#  define EAF_OpenPersist FORT_NAME(EAF_OpenPersist)
#  define EAF_OpenScratch FORT_NAME(EAF_OpenScratch)
#  define EAF_Read FORT_NAME(EAF_Read)
#  define EAF_ARead FORT_NAME(EAF_ARead)
#  define EAF_Write FORT_NAME(EAF_Write)
#  define EAF_AWrite FORT_NAME(EAF_AWrite)
#  define EAF_Probe FORT_NAME (EAF_Probe)
#  define EAF_Wait FORT_NAME (EAF_Wait)
#endif


/******************************************************************************
                               Type definitions
******************************************************************************/
#if defined(SUNOS)   || defined(AIX) || defined(PARAGON) || \
    defined(HP_HPUX) || defined(LINUX)
typedef char                    eaf_fort_char_t;
typedef int                     eaf_fort_int_t;
typedef unsigned                eaf_fort_uint_t;
typedef float                   eaf_fort_real_t;
typedef double                  eaf_fort_double_t;

typedef int                     eaf_fort_strlen_t;

#elif defined(SGITFP)

typedef char                    eaf_fort_char_t;
typedef long int                eaf_fort_int_t;
typedef long unsigned           eaf_fort_uint_t;
typedef float                   eaf_fort_real_t;
typedef double                  eaf_fort_double_t;

typedef long int                eaf_fort_strlen_t;

#elif defined(T3D)

typedef _fcd                    eaf_fort_char_t;
typedef long int                eaf_fort_int_t;
typedef long unsigned           eaf_fort_uint_t;
typedef float                   eaf_fort_real_t;
typedef double                  eaf_fort_double_t;

typedef unsigned long           eaf_fort_strlen_t;

#endif

/* .  .  .  .  .  .  .  .  .  Common type definitions  .  .  .  .  .  .  .  */
typedef Fd_t*                    eaf_fort_fd_t;        /* Changed from uint */
typedef eaf_fort_int_t           eaf_fort_mode_t;
typedef eaf_fort_int_t           eaf_fort_oflags_t;
typedef Size_t                   eaf_fort_size_t;      /* Now using ELIO size */
typedef eaf_fort_int_t           eaf_fort_status_t;
typedef eaf_fort_int_t           eaf_fort_whence_t;
typedef io_request_t             eaf_fort_req_t;       /* Added for asynch */

#include "c2f.h"

