/* $Id: armci_profile.c,v 1.2 2004-07-14 02:32:16 manoj Exp $ */
/**
 * Note #1: Right now, only process 0's profile is printed.
 * Each and every process saves its profile in the correspoding data struture.
 * However profiler prints process 0's profile when armci_profile_terminate()
 * is called. Do the corresponding changes in armci_profile_terminate() to 
 * print the profile of other processes.
 *
 * Note #2: By default profiles prints message ranges #s 21. Example: range 10
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
#define ARMCI_MAX_MSG_RANGE 21

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

#define ARMCI_EVENTS 9 /*  get, put, acc, Non-Contiguous get, put, acc*/
enum events {CONTIG_GET,    /* Contiguous get */
	     CONTIG_PUT, 
	     CONTIG_ACC, 
	     STR_GET, /* strided (Non contiguous) Get */
	     STR_PUT,
	     STR_ACC,
	     VEC_GET, /* vector (Non contiguous) Get */
	     VEC_PUT,
	     VEC_ACC
};

char *event_name[ARMCI_EVENTS] = {"GET", "PUT", "ACC", "NON CONTIGUOUS GET",
                               "NON CONTIGUOUS PUT", "NON CONTIGUOUS ACC"};

typedef struct armci_profile {
  int count;          /* number of times called */
  double exectime;  /* total execution time for "count" calls */
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
	  ARMCI_PROF[i][j].count = 0; ARMCI_PROF[i][j].exectime = 0.0; 
       }

#if ARMCI_PRINT_STRIDE
    for(i=0; i<ARMCI_EVENTS; i++) {
       if(i<=CONTIG_ACC) continue;
       else if(i==STR_GET || i==STR_PUT || i==STR_ACC)
	  for(j=0; j<ARMCI_MAX_MSG_RANGE; j++) {
	     ARMCI_PROF[i][j].stride = (armci_stride_t*)malloc(STRIDE_COUNT*sizeof(armci_stride_t));
	     ARMCI_PROF[i][j].vector = NULL;
	     if( ARMCI_PROF[i][j].stride == NULL)
		armci_die("armci_profile_init(): malloc failed", armci_me);
	  }
       else if(i==VEC_GET || i==VEC_PUT || i==VEC_ACC)
	  for(j=0; j<ARMCI_MAX_MSG_RANGE; j++) {
	     ARMCI_PROF[i][j].vector = (armci_vector_t*)malloc(STRIDE_COUNT*sizeof(armci_vector_t));
	     ARMCI_PROF[i][j].stride = NULL;
	     if( ARMCI_PROF[i][j].vector == NULL)
		armci_die("armci_profile_init(): malloc failed", armci_me);
	  }
       else armci_die("armci_profile_init(): Invalid EVENT", armci_me);
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
    if(range>=ARMCI_MAX_MSG_RANGE) range = ARMCI_MAX_MSG_RANGE;
 
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
       ARMCI_PROF[event_type][range].exectime += time;
       gCURRENT_EVNT.is_set = 0; /* clear the event */
    }
    else
       armci_die("ARMCI_PROFILE: No event set. Probably armci_profile_stop() is called before armci_profile_start()", armci_me);

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
    if(range>=ARMCI_MAX_MSG_RANGE) range = ARMCI_MAX_MSG_RANGE;
    
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
       ARMCI_PROF[event_type][range].exectime += time;
       gCURRENT_EVNT.is_set = 0; /* clear the event */
    }
    else
       armci_die("ARMCI_PROFILE: No event set. Probably armci_profile_stop() is called before armci_profile_start()", armci_me);

#if ARMCI_PRINT_STRIDE
    {  /* record the time of each vector data transfer */
       int idx = ARMCI_PROF[event_type][range].count-1;
       if(idx<STRIDE_COUNT)  
	  ARMCI_PROF[event_type][range].vector[idx].time = time;
    }
#endif
}

#define ARMCI_HDR1(fp) fprintf(fp, "\n\n************ CONTIGUOUS DATA TRANSFER ************\n\n");
#define ARMCI_HDR2(fp) fprintf(fp, "\n\n********** NON-CONTIGUOUS DATA TRANSFER **********\n\n"); 
#define ARMCI_HDR3(fp) fprintf(fp, "#Gets\t #puts\t #accs\t get_time   put_time   acc_time   RANGE\n\n");
#define ARMCI_HDR4(fp) fprintf(fp, "SL#\tndim\t proc\t time      stride_info\n\n");
#define ARMCI_HDR5(fp) fprintf(fp, "SL#\tnvec\t proc\t time\t    [ #arrays\t bytes\t]\n");

/* print profile of contiguous get/put/acc calls for every message range */
void armci_print_contig(FILE *fp) {
    int i, nrange=ARMCI_MAX_MSG_RANGE-1; 
    ARMCI_HDR1(fp); ARMCI_HDR3(fp);
    for(i=0; i< nrange; i++) {
       fprintf(fp, "%d\t %d\t %d\t %.2e   %.2e   %.2e  ",
	       ARMCI_PROF[CONTIG_GET][i].count,
	       ARMCI_PROF[CONTIG_PUT][i].count,
	       ARMCI_PROF[CONTIG_ACC][i].count, 
	       ARMCI_PROF[CONTIG_GET][i].exectime,
	       ARMCI_PROF[CONTIG_PUT][i].exectime,
	       ARMCI_PROF[CONTIG_ACC][i].exectime);
       if(i< nrange-1) fprintf(fp, "(%d-%d)\n", 1<<i, 1<<(i+1));
       else fprintf(fp, "(>%d)\n", 1<<ARMCI_MAX_MSG_RANGE);
    }
}

/* This prints the number of non-contiguous get/put/acc/ calls for every 
   message range */
void armci_print_noncontig(FILE *fp) {
    int i, nget, nput, nacc, nrange=ARMCI_MAX_MSG_RANGE-1;
    double gtime, ptime, atime;

    ARMCI_HDR2(fp); ARMCI_HDR3(fp);
    for(i=0; i< nrange; i++) {
       nget = ARMCI_PROF[STR_GET][i].count + ARMCI_PROF[VEC_GET][i].count;
       nput = ARMCI_PROF[STR_PUT][i].count + ARMCI_PROF[VEC_PUT][i].count;
       nacc = ARMCI_PROF[STR_ACC][i].count + ARMCI_PROF[VEC_ACC][i].count;
       gtime=ARMCI_PROF[STR_GET][i].exectime+ARMCI_PROF[VEC_GET][i].exectime;
       ptime=ARMCI_PROF[STR_PUT][i].exectime+ARMCI_PROF[VEC_PUT][i].exectime;
       atime=ARMCI_PROF[STR_ACC][i].exectime+ARMCI_PROF[VEC_ACC][i].exectime;
       
       fprintf(fp, "%d\t %d\t %d\t %.2e   %.2e   %.2e  ",
	       nget, nput, nacc,  gtime, ptime, atime);
       if (i< nrange-1) fprintf(fp, "(%d-%d)\n", 1<<i, 1<<(i+1));
       else fprintf(fp, "(>%d)\n", 1<<ARMCI_MAX_MSG_RANGE);
    }
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

void armci_print_stridedinfo(FILE *fp, int event, int range) {
    int i, j, stride_levels, str_count;
    double time=0.0;
    fprintf(fp, "\n\nSTRIDE INFORMATION FOR MSG_RANGE %d-%d for EVENT: %s\n", 
	    1<<range, 1<<(range+1), event_name[event]);
    ARMCI_HDR4(fp);
    
    str_count = ARMCI_PROF[event][range].count;
    if(str_count > STRIDE_COUNT) { 
       armci_print_warning_msg(fp, range, str_count);
       str_count = STRIDE_COUNT;
    }
    
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

void armci_print_vectorinfo(FILE *fp, int event, int range) {
    int i, j, vec_len, str_count;
    double time=0.0;
    fprintf(fp, "\n\nVECTOR INFORMATION FOR MSG_RANGE %d-%d for EVENT: %s\n", 
	    1<<range, 1<<(range+1), event_name[event]);
    ARMCI_HDR5(fp);
    
    str_count = ARMCI_PROF[event][range].count;
    if(str_count > STRIDE_COUNT) { 
       armci_print_warning_msg(fp, range, str_count);
       str_count = STRIDE_COUNT;
    }
    
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
	  if(j!=vec_len-1) fprintf(fp, "\t\t\t\t  [ ");
       }
    }
    /*This o/p is just for verification*/
    fprintf(fp, "**** STRIDE_COUNT = %d ; TOTAL TIME = %.2e\n",
	    str_count, time);
}
#endif /* end of ARMCI_PRINT_STRIDE */

void armci_profile_terminate() {
    FILE *fp = stdout;
    if(armci_me == 0) { /* process 0's profile only */

       /* contiguous calls */
       armci_print_contig(fp);

       /* non-contiguous calls */
       armci_print_noncontig(fp);
       
#if ARMCI_PRINT_STRIDE
       {
	  int msg_range, event_type;
	  /**
	   * printing stride info for non-contiguous get (STR_GET) for message
	   * range #6. 2^6 - 2^(6+1) bytes (i.e. 64-128 bytes)
	   */
	  msg_range  = 6; event_type = STR_GET;
	  armci_print_stridedinfo(fp,event_type,msg_range);
	  msg_range  = 6; event_type = VEC_GET;
	  armci_print_vectorinfo(fp,event_type,msg_range);
	  /*armci_print_stridedinfo(STR_GET,19);*/ /* (524288-1MB) */
       }
#endif
    }
}

#endif /* end of ARMCI_PROFILE */

