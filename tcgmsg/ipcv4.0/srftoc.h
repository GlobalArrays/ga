/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/srftoc.h,v 1.5 1995-02-24 02:17:52 d3h325 Exp $ */

#ifndef SRFTOC_H_
#define SRFTOC_H_
/* 
  This header file provides definitions for c for the names of the
  c message passing routines accessible from FORTRAN. It need not
  be included directly in user c code, assuming that sndrcv.h has already.

  It is needed as the FORTRAN naming convention varies between machines
  and it is the FORTRAN interface that is portable, not the c interface.
  However by coding with the macro defnition names c portability is
  ensured.

  On most system V machines (at least Cray and Ardent) FORTRAN uppercases
  the names of all FORTRAN externals. On most BSD machines (at least
  Sun, Alliant, Encore, Balance) FORTRAN appends an underbar (_).
  Here the uppercase c routine name with an underbar is used as a
  macro name appropriately defined for each machine. Also the BSD naming
  is assumed by default.
  Note that pbegin and pfilecopy are only called from c.
*/

#if (defined(CRAY) || defined(ARDENT))

#define NICEFTN_     NICEFTN
#define NODEID_      NODEID
#define PROBE_       PROBE
#define NNODES_      NNODES
#define MTIME_       MTIME
#define TCGTIME_     TCGTIME
#define SND_         SND
#define RCV_         RCV
#define BRDCST_      BRDCST
#define SYNCH_       SYNCH
#define PBEGINF_     PBEGINF
#define PBGINF_      PBGINF
#define PEND_        PEND
#define SETDBG_      SETDBG
#define NXTVAL_      NXTVAL
#define PBFTOC_      PBFTOC
#define PARERR_      PARERR
#define LLOG_        LLOG
#define STATS_       STATS
#define WAITCOM_     WAITCOM
#define MITOD_       MITOD
#define MDTOI_       MDTOI
#define MDTOB_       MDTOB
#define MITOB_       MITOB
#define DRAND48_     DRAND48
#define SRAND48_     SRAND48
#define PFCOPY_      PFCOPY
#define dgop_        DGOP
#define igop_        IGOP

#else

#if (defined(AIX) || defined(NEXT) || defined(HPUX)) && !defined(EXTNAME)
#define NICEFTN_     niceftn
#define NODEID_      nodeid
#define NNODES_      nnodes
#define MTIME_       mtime
#define TCGTIME_     tcgtime
#define SND_         snd
#define RCV_         rcv
#define BRDCST_      brdcst
#define SYNCH_       synch
#define PBEGINF_     pbeginf
#define PBGINF_      pbginf
#define PEND_        pend
#define SETDBG_      setdbg
#define NXTVAL_      nxtval
#define PBFTOC_      pbftoc
#define PARERR_      parerr
#define LLOG_        llog
#define STATS_       stats
#define WAITCOM_     waitcom
#define MITOD_       mitod
#define MDTOI_       mdtoi
#define MDTOB_       mdtob
#define MITOB_       mitob
#define DRAND48_     drand48
#define SRAND48_     srand48
#define PFCOPY_      pfcopy
#define dgop_        dgop
#define igop_        igop
#define DGOP_        dgop
#define IGOP_        igop
#else
#define NICEFTN_     niceftn_
#define NODEID_      nodeid_
#define PROBE_       probe_
#define NNODES_      nnodes_
#define MTIME_       mtime_
#define TCGTIME_     tcgtime_
#define SND_         snd_
#define RCV_         rcv_
#define BRDCST_      brdcst_
#define SYNCH_       synch_
#define PBEGINF_     pbeginf_
#define PBGINF_      pbginf_
#define PEND_        pend_
#define SETDBG_      setdbg_
#define NXTVAL_      nxtval_
#define PBFTOC_      pbftoc_
#define PARERR_      parerr_
#define LLOG_        llog_
#define STATS_       stats_
#define WAITCOM_     waitcom_
#define MITOD_       mitod_
#define MDTOI_       mdtoi_
#define MDTOB_       mdtob_
#define MITOB_       mitob_
#define DRAND48_     drand48_
#define SRAND48_     srand48_
#define PFCOPY_      pfcopy_
#if defined(SP1)
#define DGOP_        dgop_
#define IGOP_        igop_
#endif
#endif

#endif

#endif
