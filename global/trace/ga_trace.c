/*$Id: ga_trace.c,v 1.3 1995-02-02 23:14:37 d3g681 Exp $*/
 /***********************************************************************\
 * Tracing and Timing functions for the GA routines:                     *
 *   trace_init       - initialization                                   *
 *   trace_stime      - starts timing                                    *
 *   trace_etime      - ends timing                                      *
 *   trace_genrec     - generates a trace record for the calling routine *
 *   trace_end        - ends tracing & writes report to a file 'proc'    *
 * Note: the usc timer from the ALOG package is used                     *
 * Jarek Nieplocha, 10.14.93                                             *
 \***********************************************************************/

#include <macdecls.h>
#include <stdio.h>
#include "usc.h"


static usc_time_t *tlog, tt0, tt1;

static unsigned  long current, MAX_EVENTS=0; 
  
static unsigned long *indlog, ihandle, thandle; 

usc_time_t usc_MD_clock();

#define min(a,b) ((a)<(b) ? (a) : (b))



void trace_init_(n)
long *n; /* max number of events to be traced */
{
long index,err;

  if(*n<=0){
    printf("trace_init>>  invalid max number of events: %d\n",*n);
    return;
  }
  current = 0;
  err = 0;

/*  MA_initialize(MT_INT,10000,10000); */ 

  MAX_EVENTS = *n;
  if(!MA_push_get(MT_LONGINT, *n*2, "timeLog", &thandle, &index)){
                 printf("trace_init>> failed to allocate memory 1\n");
                 err ++;
  }
  MA_get_pointer(thandle, &tlog);
  if(!tlog){
                 printf("trace_init>> null pointer: 1\n");
                 err ++;
  }
  if(!MA_push_get(MT_LONGINT, *n*6, "indexLog", &ihandle, &index)){
                 printf("trace_init>> failed to allocate memory 2\n");
                 err ++;
  }
  MA_get_pointer(ihandle, &indlog);
  if(!indlog) { 
                 printf("trace_init>> null pointer: 2\n");
                 err ++;
  }
  if(err) MAX_EVENTS = 0;
  usc_init();
}

double tcgtime_();

void  trace_stime_()
{
#ifdef KSR
double t = tcgtime_();
tt0 = (unsigned long)1e6*t;
#else
tt0 =  usc_MD_clock();
#endif

} 


void  trace_etime_()
{
#ifdef KSR
double t = tcgtime_();
tt1 = (unsigned long)1e6*t;
#else
tt1 =  usc_MD_clock();
#endif
}


void trace_genrec_(ga, ilo, ihi, jlo, jhi, op)
long *ga, *ilo, *ihi, *jlo, *jhi, *op;
{
   if(current >=  MAX_EVENTS)return;

   tlog[current*2]     = tt0;
   tlog[current*2+1]   = tt1;
   indlog[current*6]   = *ga;
   indlog[current*6+1] = *ilo;
   indlog[current*6+2] = *ihi;
   indlog[current*6+3] = *jlo;
   indlog[current*6+4] = *jhi;
   indlog[current*6+5] = *op;
   current++;
}



void trace_end_(proc)
long *proc; /* processor number */
{
FILE *fout;
char fname[10];
int i,k;

  sprintf(fname,"%03d",*proc);
  fout=fopen(fname,"w");

  for(i=0;i<min(current,MAX_EVENTS);i++){
     fprintf(fout,"%d ",*proc);
     for(k=i*6;k<6*(i+1);k++)fprintf(fout,"%d ",indlog[k]);
     fprintf(fout,"%ld %ld\n",(unsigned long)tlog[i*2],(unsigned long)tlog[i*2+1]);
  }

  MA_pop_stack(ihandle);
  MA_pop_stack(thandle);

  fclose(fout);
}
