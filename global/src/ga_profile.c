#ifdef GA_PROFILE

#include <stdio.h>
#include <math.h>
#include "typesf2c.h"
#include "ga_profile.h" 

#ifndef MPI
#  include "sndrcv.h"
#   define MP_TIMER TCGTIME_
#else
#  include "mpi.h"
#   define MP_TIMER MPI_Wtime
#endif


/** Message ranges are in the power of 2. for ex:
 *   0-2       ->    range#1
 *   2-4       ->    range#2
 *   4-8       ->    range#3
 *   8-16      ->    range#4
 *   ...
 *   512-1024  ->    range#10
 */
#define GA_MAX_MSG_RANGE 21

#define GA_EVENTS 6 /*  get, put, acc, Non-Contiguous get, put, acc*/
enum events {GET,    /* Contiguous get */
	     PUT, 
	     ACC, 
	     NC_GET, /* Non contiguous Get */
	     NC_PUT,
	     NC_ACC
};

typedef struct ga_profile {
  int count;          /* number of times called */
  double exectime;  /* total execution time for "count" calls */
}ga_profile_t;

/* profile get/put/acc for various message ranges (i.e GA_MAX_MSG_RANGE) */
static ga_profile_t GA_PROF[GA_EVENTS][GA_MAX_MSG_RANGE]; 

/* Current event */
struct event_info {
  int event_type;
  int range;
  int is_set;
  double start_time;
} gCURRENT_EVNT; 

void ga_profile_init() {
    int i,j;
    if(ga_nodeid_()==0) {printf("\nProfiling Get/Put ON\n");fflush(stdout);}
    for(i=0; i<GA_EVENTS; i++)
       for(j=0; j<GA_MAX_MSG_RANGE; j++) {
	  GA_PROF[i][j].count = 0;  GA_PROF[i][j].exectime = 0.0; 
       }
}

static void ga_profile_set_event(int event_type, int range) {
    gCURRENT_EVNT.event_type = event_type;
    gCURRENT_EVNT.range      = range;
    gCURRENT_EVNT.is_set     = 1;
    gCURRENT_EVNT.start_time = MP_TIMER();
}
void ga_profile_start(long bytes, int ndim, Integer *lo, Integer *hi,
                      int comm_type) {
    int i, count=0, non_contig=0, event_type, range;

    /* find the message range */
    range = (int) (log((double)bytes)/log(2.0));
    if(range>=GA_MAX_MSG_RANGE) range = GA_MAX_MSG_RANGE;
 
    /* check contiguous or non-contiguous */
    for(i=0; i<ndim; i++) if(hi[0]-lo[0]) count++;
    if(count>1) non_contig=1; /* i.e. non-contiguous */
 
    switch(comm_type) {
       case GA_PROFILE_PUT:
	  if(non_contig) event_type = NC_PUT;
	  else event_type = PUT;
	  break;
       case GA_PROFILE_GET: 
	  if(non_contig) event_type = NC_GET;
	  else event_type = GET;
	  break;
       case GA_PROFILE_ACC: 
	  if(non_contig) event_type = NC_ACC;
	  else event_type = ACC;
	  break;
       default: ga_error("GA_PROFILE: Invalid communication type", 0L);
    }

    /* set the curent event for timer */
    ga_profile_set_event(event_type, range);
    
    /* profile update: i.e. update event count */
    GA_PROF[event_type][range].count++;
}

void ga_profile_stop() {
    int event_type = gCURRENT_EVNT.event_type;
    int range = gCURRENT_EVNT.range;

    if(gCURRENT_EVNT.is_set) { /* Yep, there is an event set */
       GA_PROF[event_type][range].exectime += (MP_TIMER() - 
					       gCURRENT_EVNT.start_time);
       gCURRENT_EVNT.is_set = 0; /* clear the event */
    }
    else
       ga_error("GA_PROFILE: No event set. Probably ga_profile_stop() is called before ga_profile_start()", 0L);
}

#define GA_HDR1() printf("\n\n************ CONTIGUOUS DATA TRANSFER ************\n\n");
#define GA_HDR2() printf("\n\n********** NON-CONTIGUOUS DATA TRANSFER **********\n\n"); 
#define GA_HDR3() printf("RANK\t #Gets\t #puts\t #accs\t RANGE\n\n");
#define GA_HDR4()  printf("RANK\t get_time\t put_time\t acc_time\t RANGE\n\n");

void ga_profile_terminate() {
    int i; 
    if(ga_nodeid_() == 0) { /* process 0's profile only */

       GA_HDR1(); GA_HDR3();
       for(i=0; i< GA_MAX_MSG_RANGE-1; i++)
          printf("%d\t %d\t %d\t %d\t (%d-%d)\n", ga_nodeid_(), 
		 GA_PROF[GET][i].count, GA_PROF[PUT][i].count, 
		 GA_PROF[ACC][i].count, 1<<i, 1<<(i+1));
       printf("%d\t %d\t %d\t %d\t (>%d)\n", ga_nodeid_(), 
	      GA_PROF[GET][i].count, GA_PROF[PUT][i].count, 
	      GA_PROF[ACC][i].count, 1<<GA_MAX_MSG_RANGE);


       GA_HDR1(); GA_HDR4();
       for(i=0; i< GA_MAX_MSG_RANGE-1; i++)
          printf("%d\t %.2e\t %.2e\t %.2e\t (%d-%d)\n", ga_nodeid_(), 
		 GA_PROF[GET][i].exectime, GA_PROF[PUT][i].exectime, 
		 GA_PROF[ACC][i].exectime, 1<<i, 1<<(i+1));
       printf("%d\t %.2e\t %.2e\t %.2e\t (>%d)\n", ga_nodeid_(), 
	      GA_PROF[GET][i].exectime, GA_PROF[PUT][i].exectime, 
	      GA_PROF[ACC][i].exectime, 1<<GA_MAX_MSG_RANGE);


       GA_HDR2(); GA_HDR3();
       for(i=0; i< GA_MAX_MSG_RANGE-1; i++)
	  printf("%d\t %d\t %d\t %d\t (%d-%d)\n", ga_nodeid_(), 
		 GA_PROF[NC_GET][i].count, GA_PROF[NC_PUT][i].count, 
		 GA_PROF[NC_ACC][i].count, 1<<i, 1<<(i+1));
       printf("%d\t %d\t %d\t %d\t (>%d)\n",ga_nodeid_(), 
	      GA_PROF[NC_GET][i].count, GA_PROF[NC_PUT][i].count, 
	      GA_PROF[NC_ACC][i].count, 1<<GA_MAX_MSG_RANGE);

       
       GA_HDR2(); GA_HDR4();
       for(i=0; i< GA_MAX_MSG_RANGE-1; i++)
          printf("%d\t %.2e\t %.2e\t %.2e\t (%d-%d)\n", ga_nodeid_(), 
		 GA_PROF[NC_GET][i].exectime, GA_PROF[NC_PUT][i].exectime, 
		 GA_PROF[NC_ACC][i].exectime, 1<<i, 1<<(i+1));
       printf("%d\t %.2e\t %.2e\t %.2e\t (>%d)\n", ga_nodeid_(), 
	      GA_PROF[NC_GET][i].exectime, GA_PROF[NC_PUT][i].exectime, 
	      GA_PROF[NC_ACC][i].exectime, 1<<GA_MAX_MSG_RANGE);
    }
}

#endif /* end of GA_PROFILE */

