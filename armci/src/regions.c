/* $Id: regions.c,v 1.1 2003-03-27 02:08:56 d3h325 Exp $ interface to keep track of memory regions accross the cluster */
/* 
 * armci_region_init - allocates list of regions, initialization
 * armci_region_register_shm - registers shared memory on the current node
 * armci_region_register_loc - registers local memory
 * armci_region_clus_record  - stores info on pinned region on a given cluster node
 * armci_region_clus_found   - returns 1/0 if specified memory is registered
 * armci_region_loc_found    - same for local memory
 * armci_region_loc_both_found - returns 1 if local and remote are found, otherwise 0
 *
 */

#include "armcip.h"
#include <stdlib.h>
#include <stdio.h>

#define MAX_REGIONS 8

typedef struct {
  void *start;
  void *end;
} armci_region_t;

typedef struct {
        long n;
        armci_region_t list[MAX_REGIONS];
} armci_reglist_t;

static int allow_pin=0;
static armci_reglist_t  *clus_regions;            /* cluster shared/remote memory */
static armci_reglist_t loc_regions_arr;           /* local memory */
static void *needs_pin_shmptr=NULL, *needs_pin_ptr=NULL;
static int  needs_pin_shmsize=0, needs_pin_size=0;

extern int armci_pin_contig1(void *ptr, int bytes);
static void **exch_list=(void**)0;
static char exch_loc[MAX_REGIONS];
static char exch_rem[MAX_REGIONS];


static int armci_region_record(void *start, void *end, armci_reglist_t *reg)
{
     int cur=reg->n;
#ifdef DEBUG_
     int i;
     for(i=0; i<reg->n; i++)
        if(reg->start >= start && reg->end < start) 
           armci_die("armci_region_record: already recorded",i);
#endif
     if(reg->n >= MAX_REGIONS) return 0;
     (reg->list+cur)->start = start; 
     (reg->list+cur)->end   = end;
     reg->n++;
     return 1;
}

static void armci_region_register(void *start, long size, armci_reglist_t *reg)
{
     if(reg->n >= MAX_REGIONS) return;

     if(!armci_pin_contig1(start, (int) size)){
        printf("%d pin failed %p bytes=%ld\n",armci_me,start,size);
        fflush(stdout); return; }
     (void)armci_region_record(start,((char*)start)+size,reg);
}


     

void armci_region_register_shm(void *start, long size)
{
     if(allow_pin)armci_region_register(start, size, clus_regions+armci_clus_me);     
     else{
         needs_pin_shmptr = start;
         needs_pin_shmsize= size;
     }
#if 0
     if(allow_pin){
        printf("%d:%d registering shm %p bytes=%ld\n",armci_me,allow_pin,start,size);
        fflush(stdout);
     }
#endif
}


void armci_region_register_loc(void *start, long size)
{
     if(allow_pin)armci_region_register(start, size, &loc_regions_arr);
     else{
         needs_pin_ptr = start;
         needs_pin_size= size;
     }
#if 0
     if(allow_pin){
        printf("%d:%d registered local %p bytes=%ld\n",armci_me,allow_pin,start,size);
        fflush(stdout);
     }
#endif
}


void armci_region_clus_record(int node, void *start, long size)
{
     if(node > armci_nclus || node <0 ) 
               armci_die("armci_region_remote: bad node ",node);

     (void)armci_region_record(start,((char*)start)+size,clus_regions+node);
}


void armci_region_init()
{ 
     allow_pin =1; 
     clus_regions = (armci_reglist_t*)calloc(armci_nclus, sizeof(armci_reglist_t));
     if(!clus_regions) armci_die("armci_region_init: calloc failed",armci_nclus);
     exch_list = (void**)calloc(2*armci_nclus, sizeof(void*));
     if(!exch_list)  armci_die("armci_region_init: calloc 2 failed",armci_nclus);
     bzero(exch_loc,sizeof(exch_loc));
     bzero(exch_rem,sizeof(exch_rem));

#if 0
     printf("%d: initialized regions\n",armci_me); fflush(stdout);
#endif
     if(needs_pin_ptr) armci_region_register_loc(needs_pin_ptr,needs_pin_size); 
     if(needs_pin_shmptr) armci_region_register_shm(needs_pin_shmptr,needs_pin_shmsize); 
} 
 


int armci_region_clus_found(int node, void *start, int size)
{
    armci_reglist_t *reg=clus_regions+node;
    int i,found=-1;
    if(!allow_pin) return 0;
    if(node > armci_nclus || node <0 ) 
               armci_die("armci_region_clus_found: bad node ",node);
    for(i=0; i<reg->n; i++)
        if((reg->list+i)->start <= start && (reg->list+i)->end > start){found=i; break;}
    
    return(found);
}

int armci_region_loc_found(void *start, int size)
{
     armci_reglist_t *reg = &loc_regions_arr;
     int i,found=-1;
     if(!allow_pin) return 0;
     for(i=0; i<reg->n; i++)
        if((reg->list+i)->start <= start && (reg->list+i)->end > start){found=i; break;}
#if 0
     if(found){printf("%d: found loc %d n=%ld (%p,%p) %p\n",armci_me,found,reg->n,
            (reg->list)->start,(reg->list)->end, start); fflush(stdout);
     }
#endif

     return(found);
}


int armci_region_both_found(void *loc, void *rem, int size, int node)
{
     armci_reglist_t *reg = &loc_regions_arr;
     int i,found=0;
     if(!allow_pin) return 0;

     /* first scan for local */
     for(i=0; i<reg->n; i++)
        if((reg->list+i)->start <= loc && (reg->list+i)->end > loc){found=1; break;}

     if(!found){ /* might be local shared */
         reg=clus_regions+armci_clus_me;
         for(i=0; i<reg->n; i++)
           if((reg->list+i)->start <= loc && (reg->list+i)->end > loc){found=1; break;}
     }
     if(!found) return 0;

     /* now check remote shared */
     reg=clus_regions+node;
     for(i=0; i<reg->n; i++)
         if((reg->list+i)->start <= rem && (reg->list+i)->end > rem){found=2;break;}

#if 0
     if(found==2){printf("%d: found both %d\n",armci_me,node); fflush(stdout); }
#endif
     if(found==2) return 1;
     else return 0;
}
     


void armci_region_exchange(void *start, long size)
{
     int found=0, i;
     armci_region_t *reg=0;

     if(!allow_pin)return;

     found=armci_region_clus_found(armci_clus_me, start,size);
     if(found>-1){
        if(!exch_rem[found]){
           reg = (clus_regions+armci_clus_me)->list+found; 
           exch_rem[found]=1;
        }
     }else{
        found= armci_region_loc_found(start,size);
        if(found>-1){
             if(!exch_loc[found]){
               reg =  (&loc_regions_arr)->list+found;
               exch_loc[found]=1;
             }
        }
     }

     bzero(exch_list,armci_nclus*sizeof(armci_region_t));
     if( reg && (armci_me == armci_master)){  
        exch_list[2*armci_clus_me] = reg->start;
        exch_list[2*armci_clus_me+1] = reg->end;
     }

     /* exchange info on new regions with other nodes */
     armci_exchange_address(exch_list,2*armci_nclus);
     for(i=0; i<armci_nclus; i++){
         armci_reglist_t *r=clus_regions+i;
         if(i==armci_clus_me) continue;
         if(exch_list[2*i]){
#if 0
            printf("%d recording clus=%d mem %p-%p\n",armci_me,i,exch_list[2*i],
                   exch_list[2*i+1]);
            fflush(stdout);
#endif
            armci_region_record(exch_list[2*i],exch_list[2*i+1], r);
         }
     }
}

