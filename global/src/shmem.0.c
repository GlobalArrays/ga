
extern void ga_error();

#if defined(SYSV) && defined(SUN) 

/* allocate contiguous shmem -- glue pieces together -- works on SUN 
 * SUN max shmem segment is only 1MB so we might need several to    
 * satisfy a request
 */


/* SHM_OP is an operator to calculate shmem address to attach 
 * might be + or - depending on the system 
 */
#define SHM_OP -


#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/param.h>
#include <errno.h>
#include <stdio.h>


#define _SHMMAX (1024)  /* memory in KB */
#define SHM_UNIT (1024)

#define MAX_REGIONS 128
#define SHM_MAX  (MAX_REGIONS*_SHMMAX)
#define SHM_MIN  (_SHMMAX)


static struct shm_region_list{
   char     *addr;
   long     id;
   int      attached;
}region_list[MAX_REGIONS];

static long alloc_regions=0;
static long prev_alloc_regions=0;
static long occup_blocks=0;


/* Terminology
 *   region - actual piece of shared memory allocated from OS
 *   block  - a part of region that is given to the requesting process
 */



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
          if(region_list[min_reg].addr <= addrp  && region_list[max_reg].addr > addrp){
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

    *offset = (long) (temp - region_list[reg].addr);

    /*debug only */
    *(int*)temp = occup_blocks;

    occup_blocks ++;
/*  
  fprintf(stderr, "create: reg=%d id= %d  off=%d  addr=%d addr+off=%d s=%d stamp=%d ids=%d\n",reg,region_list[reg].id, *offset, region_list[reg].addr, temp, *size, *(int*)temp,idlist[0]);
*/
    return temp;
}


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



long Delete_All_Regions()
{
int reg;
long code=0;

  for(reg = 0; reg < MAX_REGIONS; reg++){
    if(region_list[reg].addr != (char*)0){
      code += shmctl((int)region_list[reg].id,IPC_RMID,(struct shmid_ds *)NULL);
      region_list[reg].addr = (char*)0;
    }
  }
  return (code);
}



long Delete_Shared_Region(id)
     long id;
{
int reg;
long code;

 code = 0;
 occup_blocks--;

 /* don't actually detach until all blocks are gone */
/* if(!occup_blocks)code = Delete_All_Regions(); */

 return (code);
}



char *Attach_Shared_Region(idlist, size, offset)
     long *idlist, *offset, size;
{
int ir, reg,  found;
static char *temp, *s_addr;
char *pref_addr;
long ga_nodeid_();

  if(alloc_regions>=MAX_REGIONS)
       ga_error("Attach_Shared_Region: to many regions ",0L);

  if(!alloc_regions){
      for(reg=0;reg<MAX_REGIONS;reg++){
        region_list[reg].addr=(char*)0;
        region_list[reg].attached=0;
        region_list[reg].id=0;
      }
  }
  for (ir = 0; ir< idlist[0]; ir++){
      found = 0;
      for(reg=0;reg<MAX_REGIONS;reg++)
         if(found=(region_list[reg].id == idlist[1+ir])){
	     /* we also found base address to add to offset */
	     s_addr = region_list[reg].addr;
	     break;
         }

      if(!found){
         reg = alloc_regions;
         region_list[reg].id =idlist[1+ir];
      }

/*
      fprintf(stderr, "in attach: found=%d attached=%d id=%d   \n", found, region_list[reg].attached, region_list[reg].id );
*/
      if(!region_list[reg].attached){

       /* make sure the next shmem region will be adjacent to previous one */
         if(alloc_regions)
            pref_addr = region_list[alloc_regions-1].addr SHM_OP SHM_MIN;
         else
            pref_addr = (char*)0;

/*
*/
         printf(" Attach_Shared_Region: calling shmat: %d    %d\n",idlist[1+ir], pref_addr);          
         if ( (int) (temp = (char*)shmat((int)idlist[1+ir], pref_addr, 0))==-1){
           fprintf(stderr, " err: id= %d  off=%d \n",idlist[1+ir], *offset);
           ga_error("Attach_Shared_Region:failed to attach",(long)idlist[1+ir]);
         }
         region_list[reg].addr = temp; 
         region_list[reg].attached = 1;
         alloc_regions++;
/*
         fprintf(stderr, "attach: id=%d addr=%d val=%d\n", idlist[1+ir], temp, *((int*)temp));
*/
      }
     /* first region provides base address to add to offset */
      if(!ir)s_addr = temp;

/*
  fprintf(stderr, "attach: reg=%d id= %d  off=%d  addr=%d addr+off=%d f=%d stamp=%d\n",reg,region_list[reg].id,  *offset, region_list[reg].addr, region_list[reg].addr+ *offset, found, *((int*)(s_addr+ *offset)));
*/

  }
  occup_blocks++;
  return (s_addr+ *offset);
}


char *allocate(size)
     long size;
{
char *temp, *ftemp, *pref_addr, *valloc();
int id, newreg, i;
long sz;
#define min(a,b) ((a)>(b)? (b): (a))

    newreg = (size+(SHM_UNIT*SHM_MIN)-1)/(SHM_UNIT*SHM_MIN);
    if( (alloc_regions + newreg)> MAX_REGIONS)
       ga_error("allocate: to many regions allocated ",(long)newreg);

    prev_alloc_regions = alloc_regions; 
    for(i =0; i< newreg; i++){ 
       sz = (i== newreg-1) ? size - i*SHM_MIN : min(size, SHM_MIN);

       if ( (int)(id = shmget(IPC_PRIVATE, (int) sz,
                     (int) (IPC_CREAT | 00600))) < 0 ){
          fprintf(stderr,"id=%d size=%d MAX=%d\n",id,  (int) sz, SHM_MIN);
          alloc_regions++;
          ga_error("allocate: failed to create shared region ",(long)id);
       }

       /* make sure the next shmem region will be adjacent to previous one */
       if(alloc_regions)
          pref_addr = region_list[alloc_regions-1].addr SHM_OP SHM_MIN;
       else
          pref_addr = (char*)0;
       /*
       printf("calling shmat: %d     %d\n",id, pref_addr);          
       */
       if ( (int)(temp = (char*)shmat((int) id, pref_addr, 0)) == -1){
          perror((char*)0);
          ga_error("allocate: failed to attach to shared region",  0L);
       }

       /*
       fprintf(stderr, "allocate:  id= %d   addr=%d size=%d \n",id,temp, sz);
       */
       region_list[alloc_regions].addr = temp;
       region_list[alloc_regions].id = id;
       *(int*)temp = 1+i;
       /*
         fprintf(stderr, "allocate:attach: id=%d addr=%d val=%d\n", id, temp, *((int*)temp));
       */
       alloc_regions++;
       if(i==0)ftemp = temp;
    }
    return (MIN(ftemp,temp));
}
    
#else

#ifdef SYSV

#include <sys/types.h>
#include <sys/ipc.h>
#include <sys/shm.h>
#include <sys/param.h>
#include <errno.h>
#include <stdio.h>


#if !defined(SGI) && !defined(KSR)
extern char *shmat();
#endif

/* limits for the largest shmem segment are in Kilobytes to avoid passing
 * gigabyte values to shmalloc
 *
 * the limit for the KSR is lower than SHMMAX in sys/param.h because
 * shmat would fail -- SHMMAX cannot be trusted
 */

#if defined(SGI) || defined(IBM)
#define _SHMMAX ((unsigned long)228*1024)
#else
#ifdef KSR
#define _SHMMAX ((unsigned long)512*1024)
#else
#define _SHMMAX ((unsigned long)128*1024) /*let's hope ... */ 
#endif
#endif


#define MAX_REGIONS 100 
#define SHM_MAX  (MAX_REGIONS*_SHMMAX)
#define SHM_MIN  (_SHMMAX)      /* unit of shmem allocation */


static struct shm_region_list{
   char     *addr;
   long     id;
   int      attached;
}region_list[MAX_REGIONS];

static int alloc_regions=0;
static int first_allocated=-1;
static int occup_blocks=0;


/* Terminology
 *   region - actual piece of shared memory allocated from OS
 *   block  - a part of region that is given to the requesting process
 */


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
    if(first_allocated!=-1){
       reg=first_allocated;
       first_allocated = -1;
    } else{
       for(reg=0,nreg=0;nreg<alloc_regions; nreg++){
          if(region_list[nreg].addr > temp )break;
          reg = nreg;
       }
    }

    *offset = (long) (temp - region_list[reg].addr);
    *id = region_list[reg].id;
    occup_blocks++;
  
/*
  fprintf(stderr, "create: reg=%d id= %d  off=%d  addr=%d addr+off=%d s=%d\n",reg,*id, *offset, region_list[reg].addr, temp, *size);
*/
    return temp;
}




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

long Delete_All_Regions()
{
int reg;
long code=0;

  for(reg = 0; reg < MAX_REGIONS; reg++){
    if(region_list[reg].addr != (char*)0){
      code += shmctl((int)region_list[reg].id,IPC_RMID,(struct shmid_ds *)NULL);
      region_list[reg].addr = (char*)0;
    }
  }
  return (code);
}



long Delete_Shared_Region(id)
     long id;
{
int reg;
long code;

 code = 0;
 occup_blocks--;

 /* don't actually detach until all blocks are gone */
 /*if(!occup_blocks)code = Delete_All_Regions();*/

 return (code);
}


char *Attach_Shared_Region(id, size, offset)
     long *id, *offset, size;
{
int reg, b, found;
static char *temp;
long ga_nodeid_();

  if(alloc_regions>=MAX_REGIONS)
       ga_error("Attach_Shared_Region: to many regions ",0L);

  if(!alloc_regions){
      for(reg=0;reg<MAX_REGIONS;reg++){
        region_list[reg].addr=(char*)0;
        region_list[reg].attached=0;
        region_list[reg].id=0;
      }
  }
  found = 0;
  for(reg=0;reg<MAX_REGIONS;reg++)
     if(found=(region_list[reg].id == *id))break;

  if(!found){
     reg = alloc_regions;
     region_list[reg].id =*id;
     alloc_regions++;
  }

  if(!region_list[reg].attached){
   if ( (int) (temp = shmat((int) *id, (char *)NULL, 0)) == -1){
       fprintf(stderr, " err: id= %d  off=%d \n",*id, *offset);
       ga_error("Attach_Shared_Region: failed to attach ",(long)id);
    }
    region_list[reg].addr = temp; 
    region_list[reg].attached = 1;
  }

/*
  fprintf(stderr, "attach: reg=%d id= %d  off=%d  addr=%d addr+off=%d f=%d\n",reg,*id,  *offset, region_list[reg].addr, region_list[reg].addr+ *offset, found);
*/

  occup_blocks++;
  return (region_list[reg].addr+ *offset);
}


char *allocate(size)
     long size;
{
char * temp;
int id;

    if( alloc_regions >= MAX_REGIONS)
       ga_error("Create_Shared_Region: to many regions allocated ",0L);

    first_allocated = alloc_regions;
    if ( (int)(id = shmget(IPC_PRIVATE, (int) size,
                     (int) (IPC_CREAT | 00600))) < 0 ){
       perror((char*)0);
       fprintf(stderr,"id=%d size=%d\n",id, (int) size);
       ga_error("allocate: failed to create shared region ",(long)id);
    }

    if ( (int)(temp = shmat((int) id, (char *) NULL, 0)) == -1){
       perror((char*)0);
       ga_error("allocate: failed to attach to shared region",  temp);
    }


  fprintf(stderr, "allocate:  id= %d   addr=%d size=%d \n",id,temp, size);
    region_list[alloc_regions].addr = temp;
    region_list[alloc_regions].id = id;
    alloc_regions++;
    return (temp);
}
    
    

#endif
#endif
