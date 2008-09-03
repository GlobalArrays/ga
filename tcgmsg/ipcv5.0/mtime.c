/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv5.0/mtime.c,v 1.5 2002-03-12 18:59:31 d3h325 Exp $ */

#include <stdio.h>
#include "srftoc.h"

long MTIME_()
/*
  return wall clock time in centiseconds
*/
{
double TCGTIME_();
  return (long) (TCGTIME_()*100.0);
}

#if defined(LAPI) && defined(AIX)
#  define LAPI_AIX
#endif

#if !(defined(KSR) || defined(ALLIANT) || defined(CRAY_T3D) || defined(LAPI_AIX)) 

#include <sys/types.h>
#include <sys/time.h>

static unsigned firstsec=0;     /* Reference for timer */
static unsigned firstusec=0;    /* Reference for timer */

void MtimeReset()               /* Sets timer reference */
{
  struct timeval tp;
#ifdef __INTERIX
  char tzp[10];
#else
  struct timezone tzp;
#endif
  
  (void) gettimeofday(&tp,&tzp);

  firstsec = tp.tv_sec;
  firstusec = tp.tv_usec;
}

double TCGTIME_()
/*
  Return wall clock time in seconds as accurately as possible
*/
{
  static int firstcall=1;
  double low, high;

  struct timeval tp;
#ifdef __INTERIX
  char tzp[10];
#else
  struct timezone tzp;
#endif

  if (firstcall) {
    MtimeReset();
    firstcall = 0;
  }

  (void) gettimeofday(&tp,&tzp);

  low = (double) (tp.tv_usec>>1) - (double) (firstusec>>1);
  high = (double) (tp.tv_sec - firstsec);

  return high + 1.0e-6*(low+low);
}

#endif

#ifdef KSR
static double firsttime = 0;

static double KSRTime()
{
  long time;
#pragma setregval (time, i12)
 
  /* Read timer */
  asm("finop; movb8_8 %x_all_timer,%i12");
  asm("finop; cxnop");
  asm("finop; cxnop");
  
  return(time * 4.0e-7);
}

double TCGTIME_()
/*
  Return wall clock time in seconds as accurately as possible
*/
{
  static int firstcall = 1;
  
  if (firstcall) {
    firstcall = 0;
    MtimeReset();
  }

  return KSRTime() - firsttime;
}

void MtimeReset()               /* Sets timer reference */
{
  firsttime = KSRTime();
}

#endif

#ifdef ALLIANT

#include <sys/time.h>

struct hrcval firsttime;

void MtimeReset()
{
  hrcstamp(&firsttime);
}

double TCGTIME_()
{
  double low, high;
  struct hrcval current;
  static int firstcall = 1;

  if (firstcall) {
    firstcall = 0;
    MtimeReset();
  }

  hrcstamp(&current);

  /* Lose a bit but does this avoid the roll problem ? */

  low = (double) (current.hv_low>>1) - (double) (firsttime.hv_low>>1);
    
  high = (double) (current.hv_high - firsttime.hv_high);

  return (high*4294967296e-6+ 2.0*low) * 0.997e-5;
}

#endif

#ifdef CRAY_T3D
/* JN: gettimeofday broken on this machine-- we use fortran timef() instead */ 
double TCGTIME_(){
double TIMEF();
       /* initialization not needed since initial call to TIMEF returns 0.0 */
       return (1.e-3*TIMEF());
}
#endif


#ifdef LAPI_AIX
#include <sys/time.h>
#include <sys/systemcfg.h>

static int firstsec=0;    /* Reference for timer */
static int firstnsec=0;    

void MtimeReset()               /* Sets timer reference */
{
  timebasestruct_t t;
  read_real_time(&t, TIMEBASE_SZ);
  time_base_to_time(&t, TIMEBASE_SZ);

  firstsec = t.tb_high;
  firstnsec = t.tb_low;
}



double TCGTIME_()
/*
  Return wall clock time in seconds as accurately as possible
*/
{
  static int firstcall=1;
  timebasestruct_t t;
  int low, high;
  int secs, nsecs;

  if (firstcall) {
    MtimeReset();
    firstcall = 0;
  }

  
  read_real_time(&t, TIMEBASE_SZ);
  time_base_to_time(&t, TIMEBASE_SZ);

  secs = t.tb_high - firstsec;
  nsecs = t.tb_low - firstnsec;

  /* If there was a carry from low-order to high-order during
     the measurement, we have to undo it */
  if(nsecs < 0){
     secs--;
     nsecs+= 1000000000;
  }
  return (double)(secs + 1.0e-9*nsecs);
}

#endif
