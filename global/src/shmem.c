/*$Id: shmem.c,v 1.5 1995-02-02 23:13:54 d3g681 Exp $*/
/* System V shared memory allocation and managment for GAs:
 *
 * Interface:
 * ~~~~~~~~~
 *  char *Create_Shared_Region(long *idlist, long *size, long *offset)
 *       . to be called by just one process. 
 *       . calls shmalloc, a modified by Robert Harrison version of malloc-like
 *         memory allocator from K&R. shmalloc in turn calls allocate() that
 *         does shmget() and shmat(). 
 *       . idlist might be just a pointer to integer or a true array in the
 *         MULTIPLE_REGIONS versions (calling routine has to take cere of it) 
 *  char *Attach_Shared_Region(long *idlist, long *size, long *offset)
 *       . called by any other process to attach to existing shmem region or
 *         if attached just calculate the address based on the offset
 *       . has to be called after shmem region was created
 *  void  Free_Shmem_Ptr(long id, long size, char* addr)
 *       . called ONLY by the process that created shmem region (id) to return
 *         pointer to shmalloc (shmem is not destroyed)
 *  long  Delete_All_Regions()
 *       . destroys all shared memory regions
 *       . can be called by any process assuming that all processes attached
 *         to alllocated shmem regions 
 *       . needs to by called by cleanup procedure(s)
 *
 * Jarek Nieplocha, 06.13.94
 * 
 */

#ifdef SHMEM


#define DEBUG 0
#define STAMP 0

extern void ga_error();

#ifdef SUN
#define MULTIPLE_REGIONS
#endif

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/param.h>
#include <errno.h>
#include <stdio.h>

#if !defined(SGI) && !defined(KSR) && !defined(DECOSF)
extern char *shmat();
#endif


/* Limits for the largest shmem segment are in Kilobytes to avoid passing
 * Gigavalues to shmalloc
 * the limit for the KSR is lower than SHMMAX in sys/param.h because
 * shmat would fail -- SHMMAX cannot be trusted (a bug)
 */
#define _SHMMAX need to know the limits for this machine

#ifdef SUN
#  undef _SHMMAX
#  define _SHMMAX (1024)  /* memory in KB */
#elif defined(SGI) || defined(AIX)
#  undef _SHMMAX
#  define _SHMMAX ((unsigned long)228*1024)
#elif defined(KSR)
#  undef _SHMMAX
#  define _SHMMAX ((unsigned long)512*1024)
#elif defined(DECOSF)
#  undef _SHMMAX
#  define _SHMMAX ((unsigned long)4*1024)
#elif defined(SHMAX)
#  undef _SHMMAX
#  define _SHMMAX SHMMAX
#endif

#define MAX_REGIONS 100
#define SHM_MAX  (MAX_REGIONS*_SHMMAX)
#define SHM_MIN  (_SHMMAX)
#define SHM_UNIT (1024)


static struct shm_region_list{
   char     *addr;
   long     id;
   int      attached;
}region_list[MAX_REGIONS];
static long alloc_regions=0;
static long occup_blocks=0;

/* Terminology
 *   region - actual piece of shared memory allocated from OS
 *   block  - a part of allocated shmem that is given to the requesting process
 */



#if defined(MULTIPLE_REGIONS)
/********************************* MULTIPLE_REGIONS *******************/
/* allocate contiguous shmem -- glue pieces together -- works on SUN 
 * SUN max shmem segment is only 1MB so we might need several to    
 * satisfy a request
 */


/* SHM_OP is an operator to calculate shmem address to attach 
 * might be + or - depending on the system 
 */
#define SHM_OP -


static long prev_alloc_regions=0;

/*\
 *   assembles the list of shmem id for the block 
\*/
find_regions(addrp,   idlist, first)
    char *addrp;
    long *idlist, *first;
{
long reg, nreg, freg=-1, min_reg, max_reg;

       /* find the region where addrp belongs */
       for(reg = 0; reg < alloc_regions-1; reg++){
          if(region_list[reg].addr < region_list[reg+1].addr){
             min_reg = reg; max_reg = reg+1;
          }else{
             min_reg = reg+1; max_reg = reg;
          }
          if(region_list[min_reg].addr <= addrp  && 
             region_list[max_reg].addr > addrp){
             freg = min_reg;
             break;
          }
       }
       /* if not found yet, it must be the last region */
       if(freg < 0) freg=alloc_regions-1;

       if( alloc_regions == prev_alloc_regions){
           /* no new regions were allocated this time - just get the id */
           idlist[0] = 1;
           idlist[1] = region_list[freg].id;
       }else{
           /* get ids of the allocated regions */
           idlist[0] = alloc_regions - prev_alloc_regions;
           if(idlist[0] < 0)ga_error(" find_regions error ",0);
           for(reg =prev_alloc_regions,nreg=1; reg <alloc_regions;reg++,nreg++){
               idlist[nreg] = region_list[reg].id;
           }
           prev_alloc_regions = alloc_regions;
       }
       *first = freg;
       return 1;
}


/*\
\*/
char *Create_Shared_Region(idlist, size, offset)
     long *size, *idlist, *offset;
{
char *temp,  *shmalloc();
void shmalloc_request();
long reg;
  
  if(alloc_regions>=MAX_REGIONS)
       ga_error("Create_Shared_Region: to many regions ",0L);

  /*initialization */
  if(!alloc_regions){
      for(reg=0;reg<MAX_REGIONS;reg++){
        region_list[reg].addr=(char*)0;
        region_list[reg].attached=0;
        region_list[reg].id=0;
      }
      shmalloc_request((unsigned)SHM_MIN, (unsigned)SHM_MAX);
  }

  temp = shmalloc((unsigned long)*size);
  if(temp == (char*)0 )
     ga_error("Create_Shared_Region: shmalloc failed ",0L);
    
  if(!find_regions(temp, idlist,&reg))
     ga_error("Create_Shared_Region: allocation inconsistent ",0L);

  /* stamp at the beginning address to be tested by other processes  */
  if(STAMP) *(int*)temp = alloc_regions-1;

/*  *offset = (long) (temp - region_list[reg].addr);*/
  *offset = (long) (temp - region_list[0].addr);
  occup_blocks ++;

  if(DEBUG) fprintf(stderr, ">Create_Shared_Region: reg=%d id= %d  off=%d  addr=%d addr+off=%d s=%d stamp=%d num ids=%d\n",reg,region_list[reg].id, *offset, region_list[reg].addr, temp, *size, *(int*)temp,idlist[0]);

  return temp;
}


char *Attach_Shared_Region(idlist, size, offset)
     long *idlist, *offset, size;
{
int ir, reg,  found, first;
static char *temp;
char *pref_addr;
long ga_nodeid_();

  if(alloc_regions>=MAX_REGIONS)
       ga_error("Attach_Shared_Region: to many regions ",0L);

  /* first time needs to initialize region_list structure */
  if(!alloc_regions){
      for(reg=0;reg<MAX_REGIONS;reg++){
        region_list[reg].addr=(char*)0;
        region_list[reg].attached=0;
        region_list[reg].id=0;
      }
  }

 /* 
  * Now, process the idlist list:
  *    . for every shemem ID make sure that it is attached
  *    . calulate shmem address by adding offset to the address for 1st region
  *    . idlist[0] has the number of shmem regions to process
  *    . idlist is assumed to be ordered -- first region comes first etc.
  */
  for (ir = 0; ir< idlist[0]; ir++){
      /* search region_list for the current shmem id */
      for(found =0, reg=0; reg < MAX_REGIONS;reg++)
         if(found=(region_list[reg].id == idlist[1+ir])) break;

      if(!found){
         /* shmem id is not on the list */ 
         reg = alloc_regions;
         region_list[reg].id =idlist[1+ir];
      }

      /* attach if not attached yet */
      if(!region_list[reg].attached){
        /* make sure the next shmem region will be adjacent to previous one */
         if(alloc_regions)
           pref_addr= region_list[alloc_regions-1].addr SHM_OP (SHM_MIN*SHM_UNIT);
         else
            pref_addr = (char*)0;   /* first time let the OS choose address */

         if ( (int) (temp = (char*)shmat((int)idlist[1+ir], pref_addr, 0))==-1){
           fprintf(stderr, "shmat err: id= %d off=%d \n",idlist[1+ir],*offset);
           ga_error("Attach_Shared_Region:failed to attach",(long)idlist[1+ir]);
         }

         region_list[reg].addr = temp; 
         region_list[reg].attached = 1;
         alloc_regions++;

         if(DEBUG) fprintf(stderr, "-Attach_Shared_Region: id=%d addr=%d \n",
                           idlist[1+ir], temp);
      }
      /* now we have this region attached and ready to go */

      if(!ir)first = reg;  /* store the first region */
  }

  reg = first; /* first region on the list */ 

  if(DEBUG) fprintf(stderr, ">Attach_Shared_Region: reg=%d id= %d  off=%d  addr=%d addr+off=%d \n",reg,region_list[reg].id, *offset, region_list[reg].addr, region_list[reg].addr+ *offset);

  if(STAMP)
  /* check stamp to make sure that we are attached in the right place */
  if(*((int*)(region_list[reg].addr+ *offset))!= alloc_regions-1){
      fprintf(stderr, "attach: region=%d  ",alloc_regions);
      ga_error("Attach_Shared_Region: wrong stamp value !", 
                *((int*)(region_list[reg].addr+ *offset)));
  }
  occup_blocks++;
/*  return (region_list[reg].addr + *offset);*/
  return (region_list[0].addr + *offset);
}


/*\ allocates shmem, to be called by shmalloc that is called by process that
 *  creates shmem region
\*/
char *allocate(size)
     long size;
{
#define min(a,b) ((a)>(b)? (b): (a))
char *temp, *ftemp, *pref_addr, *valloc();
int id, newreg, i;
long sz;

    newreg = (size+(SHM_UNIT*SHM_MIN)-1)/(SHM_UNIT*SHM_MIN);
    if( (alloc_regions + newreg)> MAX_REGIONS)
       ga_error("allocate: to many regions already allocated ",(long)newreg);

    prev_alloc_regions = alloc_regions; 

    /* allocate shmem in as many segments as neccesary */
    for(i =0; i< newreg; i++){ 
       sz =(i==newreg-1)? size - i*SHM_MIN*SHM_UNIT: min(size,SHM_UNIT*SHM_MIN);

       if ( (int)(id = shmget(IPC_PRIVATE, (int) sz,
                     (int) (IPC_CREAT | 00600))) < 0 ){
          fprintf(stderr,"id=%d size=%d MAX=%d\n",id,  (int) sz, SHM_MIN);
          alloc_regions++;
          ga_error("allocate: failed to create shared region ",(long)id);
       }

       /* make sure the next shmem region will be adjacent to previous one */
       if(alloc_regions)
         pref_addr =region_list[alloc_regions-1].addr SHM_OP (SHM_MIN*SHM_UNIT);
       else
         pref_addr = (char*)0;   /* first time let the OS choose address */

       if(DEBUG)printf("  calling shmat: id=%d adr=%d sz=%d\n",id,pref_addr,sz);
       if ( (int)(temp = (char*)shmat((int) id, pref_addr, 0)) == -1){
          perror((char*)0);
          ga_error("allocate: failed to attach to shared region",  0L);
       }

       region_list[alloc_regions].addr = temp;
       region_list[alloc_regions].id = id;

       if(DEBUG) fprintf(stderr,"  allocate:attach: id=%d addr=%d \n",id, temp);
       alloc_regions++;
       if(i==0)ftemp = temp;
    }
    return (min(ftemp,temp));
}
    
/********************************* MULTIPLE_REGIONS *******************/
#else /* Now, the machines where shm segments are not glued together */ 


int last_allocated=-1;
/*\
\*/
char *Create_Shared_Region(id, size, offset)
     long *size, *id, *offset;
{
char *temp,  *shmalloc();
void shmalloc_request();
int  reg, nreg;
  
  if(alloc_regions>=MAX_REGIONS)
       ga_error("Create_Shared_Region: to many regions ",0L);

  /*initialization */
  if(!alloc_regions){
      for(reg=0;reg<MAX_REGIONS;reg++){
        region_list[reg].addr=(char*)0;
        region_list[reg].attached=0;
        region_list[reg].id=0;
      }
      fprintf(stderr,"allocation unit: %dK, max shmem: %dK\n",SHM_MIN,SHM_MAX);
      shmalloc_request((unsigned)SHM_MIN, (unsigned)SHM_MAX);
  }


    temp = shmalloc((unsigned)*size);
    if(temp == (char*)0 )
       ga_error("Create_Shared_Region: shmalloc failed ",0L);
    
    /* find the region */
    if(last_allocated!=-1){
       reg=last_allocated;
       last_allocated = -1;
    } else{
       for(reg=0,nreg=0;nreg<alloc_regions; nreg++){
          if(region_list[nreg].addr > temp )break;
          reg = nreg;
       }
    }

    if(STAMP) *((int*)temp) = alloc_regions-1;

    *offset = (long) (temp - region_list[reg].addr);
    *id = region_list[reg].id;
    occup_blocks++;
  
  if(DEBUG) fprintf(stderr,"Create_Shared_Region: reg=%d id= %d  off=%d  addr=%d    addr+off=%d s=%d\n",reg,*id, *offset, region_list[reg].addr, temp, *size);

    return temp;
}


char *Attach_Shared_Region(id, size, offset)
     long *id, *offset, size;
{
int reg, found;
static char *temp;
long ga_nodeid_();

  if(alloc_regions>=MAX_REGIONS)
       ga_error("Attach_Shared_Region: to many regions ",0L);

  /* first time needs to initialize region_list structure */
  if(!alloc_regions){
      for(reg=0;reg<MAX_REGIONS;reg++){
        region_list[reg].addr=(char*)0;
        region_list[reg].attached=0;
        region_list[reg].id=0;
      }
  }

  /* search region_list for the current shmem id */
  for(found = 0, reg=0; reg < MAX_REGIONS;reg++)
     if(found=(region_list[reg].id == *id))break;

  if(!found){
     reg = alloc_regions;
     region_list[reg].id =*id;
     alloc_regions++;
  }

  /* attach if not attached yet */
  if(!region_list[reg].attached){
   if ( (long) (temp = shmat((int) *id, (char *)NULL, 0)) == -1L){
       fprintf(stderr, " err: id= %d  off=%d \n",*id, *offset);
       ga_error("Attach_Shared_Region: failed to attach ",(long)id);
    }
    region_list[reg].addr = temp; 
    region_list[reg].attached = 1;
  }

  if(DEBUG) fprintf(stderr, "attach: reg=%d id= %d  off=%d  addr=%d addr+off=%d   f=%d\n",reg,*id,*offset, region_list[reg].addr,region_list[reg].addr+ *offset,  found);

  if(STAMP)
  /* check stamp to make sure that we are attached in the right place */
  if(*((int*)(region_list[reg].addr+ *offset))!= alloc_regions-1)
      ga_error("Attach_Shared_Region: wrong stamp value !", 
                *((int*)(region_list[reg].addr+ *offset)));
  occup_blocks++;
  return (region_list[reg].addr+ *offset);
}


/*\ allocates shmem, to be called by shmalloc that is called by process that
 *  creates shmem region
\*/
char *allocate(size)
     long size;
{
char * temp;
long id;

    if( alloc_regions >= MAX_REGIONS)
       ga_error("Create_Shared_Region: to many regions already allocated ",0L);

    last_allocated = alloc_regions;
    if ( (id = (long)shmget(IPC_PRIVATE, (int) size,
                     (int) (IPC_CREAT | 00600))) < 0L ){
       perror((char*)0);
       fprintf(stderr,"id=%d size=%d\n",id, (int) size);
       ga_error("allocate: failed to create shared region ",(long)id);
    }

    if ( (long)(temp = shmat((int) id, (char *) NULL, 0)) == -1L){
       perror((char*)0);
       ga_error("allocate: failed to attach to shared region",  temp);
    }

    if(DEBUG)fprintf(stderr,"allocate: id= %d addr=%d size=%d\n",id,temp,size);
    region_list[alloc_regions].addr = temp;
    region_list[alloc_regions].id = id;
    alloc_regions++;
    return (temp);
}
    
#endif

/* common code for the two versions */


/*\ only process that created shared region returns the pointer to shmalloc 
\*/
void Free_Shmem_Ptr( id, size, addr)
     long id, size;
     char *addr;
{
  void shfree();
  shfree(addr);
}


long Delete_All_Regions()
{
int reg;
long code=0;

  for(reg = 0; reg < MAX_REGIONS; reg++){
    if(region_list[reg].addr != (char*)0){
      code += shmctl((int)region_list[reg].id,IPC_RMID,(struct shmid_ds *)NULL);
      region_list[reg].addr = (char*)0;
      region_list[reg].attached = 0;
    }
  }
  return (code);
}


/* the rest is obsolete */

long Detach_Shared_Region( id, size, addr)
     long id, size;
     char *addr;
{
int reg;
long code;
void shfree();

  shfree(addr);
  occup_blocks--;
  code = 0;

  if(!occup_blocks)
    for(reg=0;reg<MAX_REGIONS;reg++){
       if(region_list[reg].attached) code += shmdt(region_list[reg].addr);
  }
  return (code);
}



long Delete_Shared_Region(id)
     long id;
{
long code;

 code = 0;
 occup_blocks--;

 /* don't actually detach until all blocks are gone */
/* if(!occup_blocks)code = Delete_All_Regions(); */
 return (code);
}

#else
 what are doing here ?
#endif
