/* $Id: armci_profile.c,v 1.4 2004-07-20 02:26:10 manoj Exp $ */

/**
 * Set an environment variable as follows to enable ARMCI profiling
 *    export ARMCI_PROFILE=YES (bash)
 *    setenv ARMCI_PROFILE YES (csh/tcsh)
 * 
 * Profiler can profile the following ARMCI Calls:
 *    ARMCI_Get,ARMCI_Put,ARMCI_Acc,ARMCI_NbGet,ARMCI_NbPut,ARMCI_NbAcc,
 *    ARMCI_GetS,ARMCI_PutS,ARMCI_AccS,ARMCI_NbGetS,ARMCI_NbPutS,ARMCI_NbAccS,
 *    ARMCI_GetV,ARMCI_PutV,ARMCI_AccV,ARMCI_NbGetV,ARMCI_NbPutV,ARMCI_NbAccV,
 *    ARMCI_Wait, armci_wait_notify
 *      (NOTE: As armci_notify is same as ARMCI_Put, it is not profiled.)
 *   
 * 
 * Note #1: Right now, only process 0's profile is printed.
 * Each and every process saves its profile in the correspoding data struture.
 * However profiler prints process 0's profile when armci_profile_terminate()
 * is called. Do the corresponding changes in armci_profile_terminate() to 
 * print the profile of other processes.
 *
 * Note #2: By default profiler prints msg ranges 0 to 21. Example: range 10
 * corresponds to message ranges from 1024 bytes to 2047 bytes.
 * Message ranges are in the power of 2. for ex:
 * ------------------------------------
 *  MSG_RANGE (r)        BYTES (2^r to 2^(r+1)-1)
 * ------------------------------------
 *      0                    0-1 
 *      1                    2-3
 *      2                    4-7
 *     ...                   ...
 *      10                1024-2047 bytes
 *     ...                   ...
 *      20                1MB - (2MB-1)
 *      21                  >= 2MB
 * -------------------------------------
 * To increase the message range, set ARMCI_MAX_MSG_RANGE accordingly.
 *
 * Note #3: If Stride information needs to be printed, set ARMCI_PRINT_STRIDE.
 * Stride information is printed in armci_profile_terminate() for a various 
 * selective message ranges and event types.Modify it according to your needs.
 *
 * Note #4: There is no profiling support for non-blocking operations yet!!
 */


#ifdef ARMCI_PROFILE

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "armci.h"
#include "armcip.h"
#include "armci_profile.h" 

#ifndef MPI
#  include "sndrcv.h"
#   define MP_TIMER TCGTIME_
#else
#  include "mpi.h"
#   define MP_TIMER MPI_Wtime
#endif


#define ARMCI_PRINT_STRIDE 1
#define ARMCI_MAX_MSG_RANGE 22 /* 0 to 21 */

#if ARMCI_PRINT_STRIDE
#define STRIDE_COUNT 1000

  typedef struct armci_stride {
    int stride_levels;
    int proc;
    int count[ARMCI_MAX_DIM];
    double time;
  }armci_stride_t;

  typedef struct giov {
    int ptr_array_len;
    int bytes;
  }giov_t;

  typedef struct armci_vector {
    int vec_len;
    int proc;
    giov_t *giov;
    double time;
  }armci_vector_t;

#endif

#define ARMCI_EVENTS 23 /* contiguous, strided and vector get/put/acc calls */
enum events {CONTIG_GET, CONTIG_PUT, CONTIG_ACC, /* Contiguous get/put/acc */
	     STR_GET, STR_PUT, STR_ACC,          /* strided Get/put/acc */
	     VEC_GET, VEC_PUT, VEC_ACC,          /* vector Get/put/acc */
	     CONTIG_NBGET, CONTIG_NBPUT, CONTIG_NBACC, /* non-blocking */
             STR_NBGET, STR_NBPUT, STR_NBACC,
             VEC_NBGET, VEC_NBPUT, VEC_NBACC,
	     BARRIER, WAIT, NOTIFY, FENCE, ALLFENCE /* misc */
};

char *event_name[ARMCI_EVENTS]={
  "GET", "PUT", "ACC", 
  "STRIDED GET", "STRIDED PUT", "STRIDED ACC",
  "VECTOR GET", "VECTOR PUT", "VECTOR ACC",
  "NBGET", "NBPUT", "NBACC",
  "STRIDED NBGET", "STRIDED NBPUT", "STRIDED NBACC",
  "VECTOR NBGET", "VECTOR NBPUT", "VECTOR NBACC",
  "BARRIER","ARMCI_WAIT","ARMCI_NOTIFY_WAIT",
  "FENCE", "ALLFENCE"
};

typedef struct armci_profile {
  int count;          /* number of times called */
  double time;  /* total execution time for "count" calls */
#if ARMCI_PRINT_STRIDE
  armci_stride_t *stride;
  armci_vector_t *vector;
#endif
}armci_profile_t;

/* profile get/put/acc for various message ranges (i.e ARMCI_MAX_MSG_RANGE) */
static armci_profile_t ARMCI_PROF[ARMCI_EVENTS][ARMCI_MAX_MSG_RANGE];

/* Current event */
struct event_info {
  int event_type;
  int range;
  int is_set;
  double start_time;
} gCURRENT_EVNT; 

void armci_profile_init() {
    int i,j;
    if(armci_me==0) {printf("\nProfiling ARMCI - ON\n");fflush(stdout);}

    for(i=0; i<ARMCI_EVENTS; i++)
       for(j=0; j<ARMCI_MAX_MSG_RANGE; j++) {
	  ARMCI_PROF[i][j].count = 0; ARMCI_PROF[i][j].time = 0.0; 
       }

#if ARMCI_PRINT_STRIDE
    for(i=0; i<ARMCI_EVENTS; i++) {
       if(i==STR_GET || i==STR_PUT || i==STR_ACC ||
	  i==STR_NBGET || i==STR_NBPUT || i==STR_NBACC)
	  for(j=0; j<ARMCI_MAX_MSG_RANGE; j++) {
	     ARMCI_PROF[i][j].stride = (armci_stride_t*)malloc(STRIDE_COUNT*sizeof(armci_stride_t));
	     ARMCI_PROF[i][j].vector = NULL;
	     if( ARMCI_PROF[i][j].stride == NULL)
		armci_die("armci_profile_init(): malloc failed", armci_me);
	  }
       if(i==VEC_GET || i==VEC_PUT || i==VEC_ACC ||
	  i==VEC_NBGET || i==VEC_NBPUT || i==VEC_NBACC)
	  for(j=0; j<ARMCI_MAX_MSG_RANGE; j++) {
	     ARMCI_PROF[i][j].vector = (armci_vector_t*)malloc(STRIDE_COUNT*sizeof(armci_vector_t));
	     ARMCI_PROF[i][j].stride = NULL;
	     if( ARMCI_PROF[i][j].vector == NULL)
		armci_die("armci_profile_init(): malloc failed", armci_me);
	  }
    }
#endif
}

static void armci_profile_set_event(int event_type, int range) {
    gCURRENT_EVNT.event_type = event_type;
    gCURRENT_EVNT.range      = range;
    gCURRENT_EVNT.is_set     = 1;
    gCURRENT_EVNT.start_time = MP_TIMER();
}

void armci_profile_start_strided(int count[], int stride_levels, int proc, 
				 int comm_type) {
    int i, bytes=1, non_contig=0, event_type=-1, range;

    /* find the message range */
    for(i=0; i<= stride_levels; i++)  bytes *= count[i];
    if(bytes<=0) range=0;
    else range = (int) (log((double)bytes)/log(2.0));
    if(range>=ARMCI_MAX_MSG_RANGE-1) range = ARMCI_MAX_MSG_RANGE-1;
 
    /* check contiguous or non-contiguous */
    if(stride_levels>0) non_contig=1; /* i.e. non-contiguous */
    if(stride_levels >= ARMCI_MAX_DIM) 
       armci_die("ARMCI_PROFILE: stride_levels >= ARMCI_MAX_DIM. Increase ARMCI_MAX_DIM.", armci_me);
    switch(comm_type) {
       case ARMCI_PROFILE_PUT:
	  if(non_contig) event_type = STR_PUT;
	  else event_type = CONTIG_PUT;
	  break;
       case ARMCI_PROFILE_GET: 
	  if(non_contig) event_type = STR_GET;
	  else event_type = CONTIG_GET;
	  break;
       case ARMCI_PROFILE_ACC: 
	  if(non_contig) event_type = STR_ACC;
	  else event_type = CONTIG_ACC;
	  break;
       case ARMCI_PROFILE_NBPUT:
	  if(non_contig) event_type = STR_NBPUT;
	  else event_type = CONTIG_NBPUT;
	  break;
       case ARMCI_PROFILE_NBGET: 
	  if(non_contig) event_type = STR_NBGET;
	  else event_type = CONTIG_NBGET;
	  break;
       case ARMCI_PROFILE_NBACC: 
	  if(non_contig) event_type = STR_NBACC;
	  else event_type = CONTIG_NBACC;
	  break;
       default: armci_die("ARMCI_PROFILE: Invalid communication type", armci_me);
    }

    /* set the curent event for timer */
    armci_profile_set_event(event_type, range);
    
    /* profile update: i.e. update event count */
    ARMCI_PROF[event_type][range].count++;
    
#if ARMCI_PRINT_STRIDE 
    if(non_contig) {
       int idx = ARMCI_PROF[event_type][range].count-1;
       if(idx<STRIDE_COUNT) {
	  ARMCI_PROF[event_type][range].stride[idx].stride_levels = stride_levels;
	  ARMCI_PROF[event_type][range].stride[idx].proc = proc;
	  for(i=0;i<=stride_levels;i++) {
	     ARMCI_PROF[event_type][range].stride[idx].count[i] = count[i];
	  }
       }
    }
#endif
}

void armci_profile_stop_strided() {
    int event_type = gCURRENT_EVNT.event_type;
    int range = gCURRENT_EVNT.range;
    double time = MP_TIMER() - gCURRENT_EVNT.start_time;

    if(gCURRENT_EVNT.is_set) { /* Yep, there is an event set */
       ARMCI_PROF[event_type][range].time += time;
       gCURRENT_EVNT.is_set = 0; /* clear the event */
    }
    else
       armci_die("ARMCI_PROFILE: No event set. Probably armci_profile_stop_strided() is called before armci_profile_start_strided()", armci_me);

#if ARMCI_PRINT_STRIDE
    /* record the time of each strided data transfer */
    if(event_type==STR_GET || event_type==STR_PUT || event_type==STR_ACC) {  
       int idx = ARMCI_PROF[event_type][range].count-1;
       if(idx<STRIDE_COUNT) ARMCI_PROF[event_type][range].stride[idx].time = time;
    }
#endif
}

void armci_profile_start_vector(armci_giov_t darr[], int len, int proc, 
				int comm_type) {

    int i, bytes=0, event_type=-1, range;

    /* find the message range */
    for(i=0; i<len; i++) bytes += darr[i].bytes;
    if(bytes<=0) range=0;
    else range = (int) (log((double)bytes)/log(2.0));
    if(range>=ARMCI_MAX_MSG_RANGE-1) range = ARMCI_MAX_MSG_RANGE-1;
    
    switch(comm_type) {
       case ARMCI_PROFILE_PUT:
	  event_type = VEC_PUT;
	  break;
       case ARMCI_PROFILE_GET: 
	  event_type = VEC_GET;
	  break;
       case ARMCI_PROFILE_ACC: 
	  event_type = VEC_ACC;
	  break;
       case ARMCI_PROFILE_NBPUT:
	  event_type = VEC_NBPUT;
	  break;
       case ARMCI_PROFILE_NBGET: 
	  event_type = VEC_NBGET;
	  break;
       case ARMCI_PROFILE_NBACC: 
	  event_type = VEC_NBACC;
	  break;
       default: armci_die("ARMCI_PROFILE: Invalid comm type", armci_me);
    }
       
    /* set the curent event for timer */
    armci_profile_set_event(event_type, range);
    
    /* profile update: i.e. update event count */
    ARMCI_PROF[event_type][range].count++;
       
#if ARMCI_PRINT_STRIDE 
       {
	  int idx = ARMCI_PROF[event_type][range].count-1;
	  if(idx<STRIDE_COUNT) {
	     ARMCI_PROF[event_type][range].vector[idx].vec_len = len;
	     ARMCI_PROF[event_type][range].vector[idx].proc = proc;
	     ARMCI_PROF[event_type][range].vector[idx].giov = 
	       (giov_t*)malloc(len*sizeof(giov_t));
	     for(i=0;i<len;i++) {
		ARMCI_PROF[event_type][range].vector[idx].giov[i].ptr_array_len = 
		  darr[i].ptr_array_len;
		ARMCI_PROF[event_type][range].vector[idx].giov[i].bytes = 
		  darr[i].bytes;
	     }
	  }
       }
#endif
}

void armci_profile_stop_vector() {
    int event_type = gCURRENT_EVNT.event_type;
    int range = gCURRENT_EVNT.range;
    double time = MP_TIMER() - gCURRENT_EVNT.start_time;
    
    if(gCURRENT_EVNT.is_set) { /* Yep, there is an event set */
       ARMCI_PROF[event_type][range].time += time;
       gCURRENT_EVNT.is_set = 0; /* clear the event */
    }
    else
       armci_die("ARMCI_PROFILE: No event set. Probably armci_profile_stop_vector() is called before armci_profile_start_vector()", armci_me);

#if ARMCI_PRINT_STRIDE
    {  /* record the time of each vector data transfer */
       int idx = ARMCI_PROF[event_type][range].count-1;
       if(idx<STRIDE_COUNT)  
	  ARMCI_PROF[event_type][range].vector[idx].time = time;
    }
#endif
}

void armci_profile_start(int comm_type) {
    int event_type=-1, range;

    /* message range is zero for events registered using this call */
    range=0;
 
    switch(comm_type) {
       case ARMCI_PROFILE_BARRIER:
	  event_type = BARRIER;
	  break;
       case ARMCI_PROFILE_WAIT: 
	  event_type = WAIT;
	  break;
       case ARMCI_PROFILE_NOTIFY_WAIT: 
	  event_type = NOTIFY;
	  break;
       case ARMCI_PROFILE_FENCE: 
	  event_type = FENCE;
	  break;
       case ARMCI_PROFILE_ALLFENCE: 
	  event_type = ALLFENCE;
	  break;
       default: armci_die("ARMCI_PROFILE: Invalid communication type", armci_me);
    }

    /* set the curent event for timer */
    armci_profile_set_event(event_type, range);
    
    /* profile update: i.e. update event count */
    ARMCI_PROF[event_type][range].count++;
}

void armci_profile_stop() {
    int event_type = gCURRENT_EVNT.event_type;
    int range = gCURRENT_EVNT.range;
    double time = MP_TIMER() - gCURRENT_EVNT.start_time;
    
    if(gCURRENT_EVNT.is_set) { /* Yep, there is an event set */
       ARMCI_PROF[event_type][range].time += time;
       gCURRENT_EVNT.is_set = 0; /* clear the event */
    }
    else
       armci_die("ARMCI_PROFILE: No event set. Probably armci_profile_stop() is called before armci_profile_start()", armci_me);
}

#define ARMCI_HDR0(fp) fprintf(fp, "\n\n************** TOTAL DATA TRANSFERS **************\n\n");
#define ARMCI_HDR1(fp) fprintf(fp, "\n\n************ CONTIGUOUS DATA TRANSFER ************\n\n");
#define ARMCI_HDR2(fp) fprintf(fp, "\n\n********** NON-CONTIGUOUS DATA TRANSFER **********\n\n"); 
#define ARMCI_HDR3(fp) fprintf(fp, "#gets\t #puts\t #accs\t get_time   put_time   acc_time   RANGE(bytes)\n\n");
#define ARMCI_HDR4(fp) fprintf(fp, "SL#\tndim\t proc\t time      stride_info\n\n");
#define ARMCI_HDR5(fp) fprintf(fp, "SL#\tnvec\t proc\t time\t    [ #arrays\t bytes\t]\n");
#define ARMCI_HDR6(fp) fprintf(fp, "\n\n****** NON-BLOCKING CONTIGUOUS DATA TRANSFER *****\n\n");
#define ARMCI_HDR7(fp) fprintf(fp, "\n\n*** NON-BLOCKING NON-CONTIGUOUS DATA TRANSFER ****\n\n"); 
#define ARMCI_HDR8(fp) fprintf(fp, "#gets\t #puts\t #accs\t get_time   put_time   acc_time   RANGE(bytes)\n\n");
#define ARMCI_HDR9(fp) fprintf(fp, "\n\n******************* ARMCI MISC *******************\n\n");

/* print profile of all get/put/acc calls for every message range */
static void armci_print_all(FILE *fp) {
    int i, nget, nput, nacc, nrange=ARMCI_MAX_MSG_RANGE;
    double gtime, ptime, atime;
 
    ARMCI_HDR0(fp); ARMCI_HDR3(fp);
    for(i=0; i< nrange; i++) {

       nget =(ARMCI_PROF[CONTIG_GET][i].count + 
	      ARMCI_PROF[STR_GET][i].count + ARMCI_PROF[VEC_GET][i].count +
	      ARMCI_PROF[CONTIG_NBGET][i].count + 
	      ARMCI_PROF[STR_NBGET][i].count + ARMCI_PROF[VEC_NBGET][i].count);
       nput =(ARMCI_PROF[CONTIG_PUT][i].count + 
	      ARMCI_PROF[STR_PUT][i].count + ARMCI_PROF[VEC_PUT][i].count +
	      ARMCI_PROF[CONTIG_NBPUT][i].count +
              ARMCI_PROF[STR_NBPUT][i].count + ARMCI_PROF[VEC_NBPUT][i].count);
       nacc =(ARMCI_PROF[CONTIG_ACC][i].count + 
	      ARMCI_PROF[STR_ACC][i].count + ARMCI_PROF[VEC_ACC][i].count +
	      ARMCI_PROF[CONTIG_NBACC][i].count +
              ARMCI_PROF[STR_NBACC][i].count + ARMCI_PROF[VEC_NBACC][i].count);

       gtime = (ARMCI_PROF[CONTIG_GET][i].time + 
		ARMCI_PROF[STR_GET][i].time + ARMCI_PROF[VEC_GET][i].time +
		ARMCI_PROF[CONTIG_NBGET][i].time +
		ARMCI_PROF[STR_NBGET][i].time + ARMCI_PROF[VEC_NBGET][i].time);
       ptime = (ARMCI_PROF[CONTIG_PUT][i].time + 
		ARMCI_PROF[STR_PUT][i].time + ARMCI_PROF[VEC_PUT][i].time +
		ARMCI_PROF[CONTIG_NBPUT][i].time +
		ARMCI_PROF[STR_NBPUT][i].time + ARMCI_PROF[VEC_NBPUT][i].time);
       atime = (ARMCI_PROF[CONTIG_ACC][i].time + 
		ARMCI_PROF[STR_ACC][i].time+ARMCI_PROF[VEC_ACC][i].time +
		ARMCI_PROF[CONTIG_NBACC][i].time +
                ARMCI_PROF[STR_NBACC][i].time+ARMCI_PROF[VEC_NBACC][i].time);
       
       fprintf(fp, "%d\t %d\t %d\t %.2e   %.2e   %.2e  ",
               nget, nput, nacc,  gtime, ptime, atime);
       if (i< nrange-1) fprintf(fp, "(%d-%d)\n", 1<<i, (1<<(i+1))-1);
       else fprintf(fp, "(>=%d)\n", 1<<(ARMCI_MAX_MSG_RANGE-1));
    }
}

/* print profile of contiguous get/put/acc calls for every message range */
static void armci_print_contig(FILE *fp) {
    int i, nrange=ARMCI_MAX_MSG_RANGE; 
    ARMCI_HDR1(fp); ARMCI_HDR3(fp);
    for(i=0; i< nrange; i++) {
       fprintf(fp, "%d\t %d\t %d\t %.2e   %.2e   %.2e  ",
	       ARMCI_PROF[CONTIG_GET][i].count,
	       ARMCI_PROF[CONTIG_PUT][i].count,
	       ARMCI_PROF[CONTIG_ACC][i].count, 
	       ARMCI_PROF[CONTIG_GET][i].time,
	       ARMCI_PROF[CONTIG_PUT][i].time,
	       ARMCI_PROF[CONTIG_ACC][i].time);
       if(i< nrange-1) fprintf(fp, "(%d-%d)\n", 1<<i, (1<<(i+1))-1);
       else fprintf(fp, "(>=%d)\n", 1<<(ARMCI_MAX_MSG_RANGE-1));
    }
}

/* This prints the number of non-contiguous get/put/acc/ calls for every 
   message range */
static void armci_print_noncontig(FILE *fp) {
    int i, nget, nput, nacc, nrange=ARMCI_MAX_MSG_RANGE;
    double gtime, ptime, atime;

    ARMCI_HDR2(fp); ARMCI_HDR3(fp);
    for(i=0; i< nrange; i++) {
       nget = ARMCI_PROF[STR_GET][i].count + ARMCI_PROF[VEC_GET][i].count;
       nput = ARMCI_PROF[STR_PUT][i].count + ARMCI_PROF[VEC_PUT][i].count;
       nacc = ARMCI_PROF[STR_ACC][i].count + ARMCI_PROF[VEC_ACC][i].count;
       gtime=ARMCI_PROF[STR_GET][i].time+ARMCI_PROF[VEC_GET][i].time;
       ptime=ARMCI_PROF[STR_PUT][i].time+ARMCI_PROF[VEC_PUT][i].time;
       atime=ARMCI_PROF[STR_ACC][i].time+ARMCI_PROF[VEC_ACC][i].time;
       
       fprintf(fp, "%d\t %d\t %d\t %.2e   %.2e   %.2e  ",
	       nget, nput, nacc,  gtime, ptime, atime);
       if (i< nrange-1) fprintf(fp, "(%d-%d)\n", 1<<i, (1<<(i+1))-1);
       else fprintf(fp, "(>=%d)\n", 1<<(ARMCI_MAX_MSG_RANGE-1));
    }
}

/* print profile of non-blocking contiguous get/put/acc calls for every 
   message range */
static void armci_print_nbcontig(FILE *fp) {
    int i, nrange=ARMCI_MAX_MSG_RANGE; 
    ARMCI_HDR6(fp); ARMCI_HDR8(fp);
    for(i=0; i< nrange; i++) {
       fprintf(fp, "%d\t %d\t %d\t %.2e   %.2e   %.2e  ",
	       ARMCI_PROF[CONTIG_NBGET][i].count,
	       ARMCI_PROF[CONTIG_NBPUT][i].count,
	       ARMCI_PROF[CONTIG_NBACC][i].count, 
	       ARMCI_PROF[CONTIG_NBGET][i].time,
	       ARMCI_PROF[CONTIG_NBPUT][i].time,
	       ARMCI_PROF[CONTIG_NBACC][i].time);
       if(i< nrange-1) fprintf(fp, "(%d-%d)\n", 1<<i, (1<<(i+1))-1);
       else fprintf(fp, "(>=%d)\n", 1<<(ARMCI_MAX_MSG_RANGE-1));
    }
}

/* This prints the number of non-blocking non-contiguous get/put/acc/ calls 
   for every message range */
static void armci_print_nbnoncontig(FILE *fp) {
    int i, nget, nput, nacc, nrange=ARMCI_MAX_MSG_RANGE;
    double gtime, ptime, atime;

    ARMCI_HDR7(fp); ARMCI_HDR8(fp);
    for(i=0; i< nrange; i++) {
       nget = ARMCI_PROF[STR_NBGET][i].count + ARMCI_PROF[VEC_NBGET][i].count;
       nput = ARMCI_PROF[STR_NBPUT][i].count + ARMCI_PROF[VEC_NBPUT][i].count;
       nacc = ARMCI_PROF[STR_NBACC][i].count + ARMCI_PROF[VEC_NBACC][i].count;
       gtime = (ARMCI_PROF[STR_NBGET][i].time + 
		ARMCI_PROF[VEC_NBGET][i].time);
       ptime = (ARMCI_PROF[STR_NBPUT][i].time + 
		ARMCI_PROF[VEC_NBPUT][i].time);
       atime = (ARMCI_PROF[STR_NBACC][i].time + 
		ARMCI_PROF[VEC_NBACC][i].time);

       fprintf(fp, "%d\t %d\t %d\t %.2e   %.2e   %.2e  ",
	       nget, nput, nacc,  gtime, ptime, atime);
       if (i< nrange-1) fprintf(fp, "(%d-%d)\n", 1<<i, (1<<(i+1))-1);
       else fprintf(fp, "(>=%d)\n", 1<<(ARMCI_MAX_MSG_RANGE-1));
    }
}

/* Profile of armci_notify_wait(), ARMCI_Wait() and ARMCI_Barrier() */
static void armci_print_misc(FILE *fp) {
    ARMCI_HDR9(fp);
    fprintf(fp, "#calls\t time\t   EVENT\n\n");
    fprintf(fp, "%d\t %.2e  ARMCI_Wait()\n", 
	    ARMCI_PROF[WAIT][0].count, ARMCI_PROF[WAIT][0].time);
    fprintf(fp, "%d\t %.2e  armci_notify_wait()\n", 
	    ARMCI_PROF[NOTIFY][0].count, ARMCI_PROF[NOTIFY][0].time);
#if 0
    fprintf(fp, "%d\t %.2e  ARMCI_Barrier()\n", 
	    ARMCI_PROF[BARRIER][0].count, ARMCI_PROF[BARRIER][0].time);
    fprintf(fp, "%d\t %.2e  ARMCI_Fence()\n", 
	    ARMCI_PROF[FENCE][0].count, ARMCI_PROF[FENCE][0].time);
    fprintf(fp, "%d\t %.2e  ARMCI_Allfence()\n", 
	    ARMCI_PROF[ALLFENCE][0].count, ARMCI_PROF[ALLFENCE][0].time);
#endif
}

#if ARMCI_PRINT_STRIDE 
static void armci_print_warning_msg(FILE *fp, int range, int str_count) {
    fprintf(fp, "WARNING: In your program, total number of data transfers\n");
    fprintf(fp, "for message range[%d - %d] is %d. This exceeds\n", 1<<range, 1<<(range+1), str_count);
    fprintf(fp, "the maximum # of data transfers [%d] that can be profiled.\n", STRIDE_COUNT); 
    fprintf(fp, "Therefore profile of only first %d data \n", STRIDE_COUNT);
    fprintf(fp, "transfers are shown below. To increase the count, set\n");
    fprintf(fp, "STRIDE_COUNT > %d (in armci_profile.c)\n", str_count);
}

static void armci_print_stridedinfo(FILE *fp, int event, int range) {
    int i, j, stride_levels, str_count;
    double time=0.0;
    
    str_count = ARMCI_PROF[event][range].count;
    if(str_count <=0) return;    
    if(str_count > STRIDE_COUNT) { 
       armci_print_warning_msg(fp, range, str_count);
       str_count = STRIDE_COUNT;
    }

    fprintf(fp, "\n\nSTRIDE INFORMATION FOR MSG_RANGE %d-%d for EVENT: %s\n", 
	    1<<range, 1<<(range+1), event_name[event]);
    ARMCI_HDR4(fp);

    for(i=0; i< str_count; i++) {
       time += ARMCI_PROF[event][range].stride[i].time;
       stride_levels  = ARMCI_PROF[event][range].stride[i].stride_levels;
       fprintf(fp, "%d\t%d\t %d\t %.2e  (",i, stride_levels,
	       ARMCI_PROF[event][range].stride[i].proc,
	       ARMCI_PROF[event][range].stride[i].time);
       for(j=0;j<=stride_levels;j++) {
	  fprintf(fp, "%d", ARMCI_PROF[event][range].stride[i].count[j]);
	  if(j!=stride_levels) fprintf(fp, "x");
       }
       fprintf(fp, ")\n");
    }
    /*This o/p is just for verification*/
    fprintf(fp, "**** STRIDE_COUNT = %d ; TOTAL TIME = %.2e\n",
	    str_count, time);
}

static void armci_print_vectorinfo(FILE *fp, int event, int range) {
    int i, j, vec_len, str_count;
    double time=0.0;
    
    str_count = ARMCI_PROF[event][range].count;
    if(str_count <=0) return; 
    if(str_count > STRIDE_COUNT) { 
       armci_print_warning_msg(fp, range, str_count);
       str_count = STRIDE_COUNT;
    }
    
    fprintf(fp, "\n\nVECTOR INFORMATION FOR MSG_RANGE %d-%d for EVENT: %s\n", 
	    1<<range, 1<<(range+1), event_name[event]);
    ARMCI_HDR5(fp);

    for(i=0; i< str_count; i++) {
       time += ARMCI_PROF[event][range].vector[i].time;
       vec_len  = ARMCI_PROF[event][range].vector[i].vec_len;
       fprintf(fp, "%d\t%d\t %d\t %.2e   [  ",i, vec_len,
	       ARMCI_PROF[event][range].vector[i].proc,
	       ARMCI_PROF[event][range].vector[i].time);
       for(j=0;j<vec_len;j++) {
	  fprintf(fp, "%-9d %d\t]\n", 
		  ARMCI_PROF[event][range].vector[i].giov[j].ptr_array_len,
		  ARMCI_PROF[event][range].vector[i].giov[j].bytes);
	  if(j!=vec_len-1) fprintf(fp, "\t\t\t\t    [  ");
       }
    }
    /*This o/p is just for verification*/
    fprintf(fp, "**** STRIDE_COUNT = %d ; TOTAL TIME = %.2e\n",
	    str_count, time);
}
#endif /* end of ARMCI_PRINT_STRIDE */

void armci_profile_terminate() {
    FILE *fp = stdout;
    char file_name[50];
    sprintf(file_name, "armci_profile.%d", armci_me);
    fp = fopen(file_name, "w");

    armci_print_all(fp);         /* total get/put/acc calls */
    armci_print_contig(fp);      /* contiguous calls */
    armci_print_noncontig(fp);   /* non-contiguous calls */
    armci_print_nbcontig(fp);    /* non-blocking contiguous calls */
    armci_print_nbnoncontig(fp); /* non-blocking non-contiguous calls */
    
    /* miscellaneous (barrier, armci_wait, notify_wait) */
    armci_print_misc(fp);

#if ARMCI_PRINT_STRIDE
    {
       /**
	* printing stride info for non-contiguous get (STR_GET) for message
	* range #6. 2^6 - 2^(6+1) bytes (i.e. 64-128 bytes)
	*    Ex: armci_print_stridedinfo(STR_GET,6);
 	*/
#define ARMCI_PRINT_EVENTS 6
       int i,j,str_event[ARMCI_PRINT_EVENTS]={ STR_GET, STR_PUT, STR_ACC,
					       STR_NBGET,STR_NBPUT,STR_NBACC};
       int vec_event[ARMCI_PRINT_EVENTS] = { VEC_GET, VEC_PUT, VEC_ACC,
					     VEC_NBGET, VEC_NBPUT, VEC_NBACC};

       fprintf(fp,"\n\n***************************************************\n");
       fprintf(fp,    " STRIDE INFORMATION for all strided data transfers\n");
       fprintf(fp,    "***************************************************\n");
       for(i=0; i<ARMCI_MAX_MSG_RANGE; i++)
	  for(j=0; j<ARMCI_PRINT_EVENTS; j++)
	     armci_print_stridedinfo(fp,str_event[j], i);

       fprintf(fp,"\n\n**************************************************\n");
       fprintf(fp,    " VECTOR INFORMATION for all vector data transfers\n");
       fprintf(fp,    "**************************************************\n");
       for(i=0; i<ARMCI_MAX_MSG_RANGE; i++)
	  for(j=0; j<ARMCI_PRINT_EVENTS; j++)
	     armci_print_vectorinfo(fp,vec_event[j], i);
    }
#endif
    fclose(fp);
}

#endif /* end of ARMCI_PROFILE */

