/* $Id: base.c,v 1.9 2001-10-29 19:55:12 d3h325 Exp $ */
/* 
 * module: base.c
 * author: Jarek Nieplocha
 * description: implements GA primitive operations --
 *              create (regular& irregular) and duplicate, destroy
 * 
 * DISCLAIMER
 *
 * This material was prepared as an account of work sponsored by an
 * agency of the United States Government.  Neither the United States
 * Government nor the United States Department of Energy, nor Battelle,
 * nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR
 * ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY,
 * COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT,
 * SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT
 * INFRINGE PRIVATELY OWNED RIGHTS.
 *
 *
 * ACKNOWLEDGMENT
 *
 * This software and its documentation were produced with United States
 * Government support under Contract Number DE-AC06-76RLO-1830 awarded by
 * the United States Department of Energy.  The United States Government
 * retains a paid-up non-exclusive, irrevocable worldwide license to
 * reproduce, prepare derivative works, perform publicly and display
 * publicly by or for the US Government, including the right to
 * distribute to other US Government contractors.
 */

 
/*#define PERMUTE_PIDS */

#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include "globalp.h"
#include "message.h"
#include "base.h"
#include "macdecls.h"

#define DEBUG 0
#define USE_MALLOC 1
#define INVALID_MA_HANDLE -1 
#define NEAR_INT(x) (x)< 0.0 ? ceil( (x) - 0.5) : floor((x) + 0.5)

#define MAX_PTR MAX_NPROC
#define MAPLEN  (MIN(GAnproc, MAX_NPROC) +MAXDIM)
#define FLEN        80              /* length of Fortran strings */

global_array_t GA[MAX_ARRAYS];
static int GAinitialized = 0;
int _max_global_array = MAX_ARRAYS;
Integer *GA_proclist;
int* GA_Proc_list = NULL;
int* GA_inv_Proc_list=NULL;

/* MA addressing */
DoubleComplex   *DCPL_MB;           /* double precision complex base address */
DoublePrecision *DBL_MB;            /* double precision base address */
Integer         *INT_MB;            /* integer base address */
float           *FLT_MB;            /* float base address */
int **GA_Update_Flags;

/*uncomment line below to verify consistency of MA in every sync */
/*#define CHECK_MA yes */

/* uncomment line below to verify if MA base address is alligned wrt datatype*/ 
#if !(defined(LINUX) || defined(CRAY))
#define CHECK_MA_ALGN 1
#endif

/*uncomment line below to initialize arrays in ga_create/duplicate */
/*#define GA_CREATE_INDEF yes */

typedef struct {
long id;
long type;
long size;
long dummy;
} getmem_t;

/* set total limit (bytes) for memory usage per processor to "unlimited" */
static Integer GA_total_memory = -1;
static Integer GA_memory_limited = 0;
struct ga_stat_t GAstat = {0,0,0,0,0,0,0,0,0,0,0};
struct ga_bytes_t GAbytes ={0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.};
long   *GAstat_arr;
static Integer GA_memory_limit=0;
Integer GAme, GAnproc;
static Integer MPme;
Integer mapALL[MAX_NPROC+1];

char *GA_name_stack[NAME_STACK_LEN];  /* stack for storing names of GA ops */
int  GA_stack_size=0;

/*************************************************************************/

/*\ This macro computes index (place in ordered set) for the element
 *  identified by _subscript in ndim- dimensional array of dimensions _dim[]
 *  assume that first subscript component changes first
\*/
#define ga_ComputeIndexM(_index, _ndim, _subscript, _dims)                     \
{                                                                              \
  Integer  _i, _factor=1;                                                      \
  for(_i=0,*(_index)=0; _i<_ndim; _i++){                                       \
      *(_index) += _subscript[_i]*_factor;                                     \
      if(_i<_ndim-1)_factor *= _dims[_i];                                      \
  }                                                                            \
}


/*\ updates subscript corresponding to next element in a patch <lo[]:hi[]>
\*/
#define ga_UpdateSubscriptM(_ndim, _subscript, _lo, _hi, _dims)\
{                                                                              \
  Integer  _i;                                                                 \
  for(_i=0; _i<_ndim; _i++){                                                   \
       if(_subscript[_i] < _hi[_i]) { _subscript[_i]++; break;}                \
       _subscript[_i] = _lo[_i];                                               \
  }                                                                            \
}


/*\ Initialize n-dimensional loop by counting elements and setting subscript=lo
\*/
#define ga_InitLoopM(_elems, _ndim, _subscript, _lo, _hi, _dims)\
{                                                                              \
  Integer  _i;                                                                 \
  *_elems = 1;                                                                 \
  for(_i=0; _i<_ndim; _i++){                                                   \
       *_elems *= _hi[_i]-_lo[_i] +1;                                          \
       _subscript[_i] = _lo[_i];                                               \
  }                                                                            \
}


Integer GAsizeof(type)    
        Integer type;
{
  switch (type) {
     case MT_F_DBL  : return (sizeof(DoublePrecision));
     case MT_F_INT  : return (sizeof(Integer));
     case MT_F_DCPL : return (sizeof(DoubleComplex));
     case MT_F_REAL : return (sizeof(float));
          default   : return 0; 
  }
}


/*\ Register process list
 *  process list can be used to:
 *   1. permute process ids w.r.t. message-passing ids (set PERMUTE_PIDS), or
 *   2. change logical mapping of array blocks to processes
\*/
void ga_register_proclist_(Integer *list, Integer* np)
{
int i;

      GA_PUSH_NAME("ga_register_proclist");
      if( *np <0 || *np >GAnproc) ga_error("invalid number of processors",*np);
      if( *np <GAnproc) ga_error("Invalid number of processors",*np);

      GA_Proc_list = (int*)malloc((size_t)GAnproc * sizeof(int)*2);
      GA_inv_Proc_list = GA_Proc_list + *np;
      if(!GA_Proc_list) ga_error("could not allocate proclist",*np);

      for(i=0;i< (int)*np; i++){
          int p  = (int)list[i];
          if(p<0 || p>= GAnproc) ga_error("invalid list entry",p);
          GA_Proc_list[i] = p; 
          GA_inv_Proc_list[p]=i;
      }

      GA_POP_NAME;
}


void GA_Register_proclist(int *list, int np)
{
      int i;
      GA_PUSH_NAME("ga_register_proclist");
      if( np <0 || np >GAnproc) ga_error("invalid number of processors",np);
      if( np <GAnproc) ga_error("Invalid number of processors",np);

      GA_Proc_list = (int*)malloc((size_t)GAnproc * sizeof(int)*2);
      GA_inv_Proc_list = GA_Proc_list + np;
      if(!GA_Proc_list) ga_error("could not allocate proclist",np);

      for(i=0; i< np; i++){
          int p  = list[i];
          if(p<0 || p>= GAnproc) ga_error("invalid list entry",p);
          GA_Proc_list[i] = p;
          GA_inv_Proc_list[p]=i;
      }
      GA_POP_NAME;
}



/*\ FINAL CLEANUP of shmem when terminating
\*/
void ga_clean_resources()
{
    ARMCI_Cleanup();
}


/*\ CHECK GA HANDLE and if it's wrong TERMINATE
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
void FATR  ga_check_handle_(g_a, fstring)
     Integer *g_a;
     _fcd fstring;
#else
void FATR  ga_check_handle_(g_a, fstring,slen)
     Integer *g_a;
     int  slen;
     char *fstring;
#endif
{
char  buf[FLEN];

    if( GA_OFFSET + (*g_a) < 0 || GA_OFFSET + (*g_a) >= _max_global_array){
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(fstring), _fcdlen(fstring), buf, FLEN);
#else
      f2cstring(fstring, slen, buf, FLEN);
#endif
      fprintf(stderr, " ga_check_handle: %s ", buf);
      ga_error(" invalid global array handle ", (*g_a));
    }
    if( ! (GA[GA_OFFSET + (*g_a)].actv) ){
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(fstring), _fcdlen(fstring), buf, FLEN);
#else
      f2cstring(fstring, slen, buf, FLEN);
#endif
      ga_error(" global array is not active ", (*g_a));
    }
}



/*\ CHECK GA HANDLE and if it's wrong TERMINATE
 *  C version
\*/
void ga_check_handle(Integer *g_a,char * string)
{
  ga_check_handleM(g_a,string);
}


/*\ Initialize MA-like addressing:
 *  get addressees for the base arrays for double, complex and int types
\*/
static int ma_address_init=0;
void gai_ma_address_init()
{
#ifdef CHECK_MA_ALGN
Integer  off_dbl, off_int, off_dcpl, off_flt;
#endif
     ma_address_init=1;
     INT_MB = (Integer*)MA_get_mbase(MT_F_INT);
     DBL_MB = (DoublePrecision*)MA_get_mbase(MT_F_DBL);
     DCPL_MB= (DoubleComplex*)MA_get_mbase(MT_F_DCPL);
     FLT_MB = (float*)MA_get_mbase(MT_F_REAL);  

#   ifdef CHECK_MA_ALGN
        off_dbl = 0 != ((long)DBL_MB)%sizeof(DoublePrecision);
        off_int = 0 != ((long)INT_MB)%sizeof(Integer);
        off_dcpl= 0 != ((long)DCPL_MB)%sizeof(DoublePrecision);
        off_flt = 0 != ((long)FLT_MB)%sizeof(float);  

        if(off_dbl)
           ga_error("GA initialize: MA DBL_MB not alligned", (Integer)DBL_MB);

        if(off_int)
           ga_error("GA initialize: INT_MB not alligned", (Integer)INT_MB);

        if(off_dcpl)
          ga_error("GA initialize: DCPL_MB not alligned", (Integer)DCPL_MB);

        if(off_flt)
           ga_error("GA initialize: FLT_MB not alligned", (Integer)FLT_MB);   
#   endif

    if(DEBUG)
       printf("%d INT_MB=%ld(%lx) DBL_MB=%ld(%lx) DCPL_MB=%ld(%lx) FLT_MB=%ld(%lx)\n",
          (int)GAme, INT_MB,INT_MB, DBL_MB,DBL_MB, DCPL_MB,DCPL_MB, FLT_MB,FLT_MB);
}



/*\ INITIALIZE GLOBAL ARRAY STRUCTURES
 *
 *  either ga_initialize_ltd or ga_initialize must be the first 
 *         GA routine called (except ga_uses_ma)
\*/
void FATR  ga_initialize_()
{
Integer  i;
int bytes;

    if(GAinitialized) return;

    /* zero in pointers in GA array */
    for(i=0;i<MAX_ARRAYS; i++) {
       GA[i].ptr  = (char**)0;
       GA[i].mapc = (int*)0;
    }
    GAnproc = (Integer)armci_msg_nproc();

#ifdef PERMUTE_PIDS
    ga_sync_();
    ga_hook_();
    if(GA_Proc_list) GAme = (Integer)GA_Proc_list[ga_msg_nodeid_()];
    else
#endif
    GAme = (Integer)armci_msg_me();
    if(GAme<0 || GAme>20000) 
       ga_error("ga_init:message-passing initialization problem: my ID=",GAme);

    MPme= (Integer)armci_msg_me();

    if(GA_Proc_list)
      fprintf(stderr,"permutation applied %d now becomes %d\n",(int)MPme,(int)GAme);

    if(GAnproc > MAX_NPROC && MPme==0){
      fprintf(stderr,"Current GA setup is for up to %d processors\n",MAX_NPROC);
      fprintf(stderr,"Please change MAX_NPROC in config.h & recompile\n");
      ga_error("terminating...",0);
    }

    GA_proclist = (Integer*)malloc((size_t)GAnproc*sizeof(Integer)); 
    if(!GA_proclist) ga_error("ga_init:malloc failed (proclist)",0);
    gai_init_onesided();

    /* set activity status for all arrays to inactive */
    for(i=0;i<_max_global_array;i++)GA[i].actv=0;

    ARMCI_Init(); /* initialize GA run-time library */

    /* assure that GA will not alocate more shared memory than specified */
    if(ARMCI_Uses_shm())
       if(GA_memory_limited) ARMCI_Set_shm_limit(GA_total_memory);

    /* Allocate memory for update flags */
    bytes = 2*MAXDIM*sizeof(int);
    GA_Update_Flags = (int**)malloc(GAnproc*sizeof(int*));
    if (ARMCI_Malloc((void**)GA_Update_Flags, bytes))
      ga_error("ga_init:Failed to initialize memory for update flags",GAme);

    GAinitialized = 1;

}


/*\ IS MA USED FOR ALLOCATION OF GA MEMORY ?
\*/ 
logical FATR ga_uses_ma_()
{
#ifdef AVOID_MA_STORAGE
   return FALSE;
#else
   if(!GAinitialized) return FALSE;
   
   if(ARMCI_Uses_shm()) return FALSE;
   else return TRUE;
#endif
}


/*\ IS MEMORY LIMIT SET ?
\*/
logical FATR ga_memory_limited_()
{
   if(GA_memory_limited) return TRUE;
   else                  return FALSE;
}



/*\ RETURNS AMOUNT OF MEMORY on each processor IN ACTIVE GLOBAL ARRAYS 
\*/
Integer  FATR ga_inquire_memory_()
{
Integer i, sum=0;
    for(i=0; i<_max_global_array; i++) 
        if(GA[i].actv) sum += GA[i].size; 
    return(sum);
}


/*\ RETURNS AMOUNT OF GA MEMORY AVAILABLE on calling processor 
\*/
Integer FATR ga_memory_avail_()
{
   if(!ga_uses_ma_()) return(GA_total_memory);
   else{
      Integer ma_limit = MA_inquire_avail(MT_F_BYTE);

      if ( GA_memory_limited ) return( MIN(GA_total_memory, ma_limit) );
      else return( ma_limit );
   }
}



/*\ (re)set limit on GA memory usage
\*/
void FATR ga_set_memory_limit_(Integer *mem_limit)
{
     if(GA_memory_limited){

         /* if we had the limit set we need to adjust the amount available */
         if (*mem_limit>=0)
             /* adjust the current value by diff between old and new limit */
             GA_total_memory += (*mem_limit - GA_memory_limit);     
         else{

             /* negative values reset limit to "unlimited" */
             GA_memory_limited =  0;     
             GA_total_memory= -1;     
         }

     }else{
         
          GA_total_memory = GA_memory_limit  = *mem_limit;
          if(*mem_limit >= 0) GA_memory_limited = 1;
     }
}


/*\ INITIALIZE GLOBAL ARRAY STRUCTURES and SET LIMIT ON GA MEMORY USAGE
 *  
 *  the byte limit is per processor (even for shared memory)
 *  either ga_initialize_ltd or ga_initialize must be the first 
 *         GA routine called (except ga_uses_ma)
 *  ga_initialize_ltd is another version of ga_initialize 
 *         without memory control
 *  mem_limit < 0 means "memory unlimited"
\*/
void FATR  ga_initialize_ltd_(Integer *mem_limit)
{

  GA_total_memory =GA_memory_limit  = *mem_limit; 
  if(*mem_limit >= 0) GA_memory_limited = 1; 
  ga_initialize_();
}


  

#define gam_checktype(_type)\
       if(_type != MT_F_DBL  && _type != MT_F_INT &&  \
          _type != MT_F_DCPL && _type != MT_F_REAL)\
         ga_error("ttype not yet supported ",  _type)

#define gam_checkdim(ndim, dims)\
{\
int _d;\
    if(ndim<1||ndim>MAXDIM) ga_error("unsupported number of dimensions",ndim);\
    for(_d=0; _d<ndim; _d++)\
         if(dims[_d]<1)ga_error("wrong dimension specified",dims[_d]);\
}


/*\ print subscript of ndim dimensional array with two strings before and after
\*/
void gai_print_subscript(char *pre,int ndim, Integer subscript[], char* post)
{
        int i;

        printf("%s [",pre);
        for(i=0;i<ndim;i++){
                printf("%ld",subscript[i]);
                if(i==ndim-1)printf("] %s",post);
                else printf(",");
        }
}

void gai_init_struct(handle)
Integer handle;
{
     if(!GA[handle].ptr){
        int len = (int)MIN(GAnproc, MAX_PTR);
        GA[handle].ptr = (char**)malloc(len*sizeof(char**));
     }
     if(!GA[handle].mapc){
        int len = (int)MAPLEN;
        GA[handle].mapc = (int*)malloc(len*sizeof(int*));
     }
     if(!GA[handle].ptr)ga_error("malloc failed: ptr:",0);
     if(!GA[handle].mapc)ga_error("malloc failed: mapc:",0);
}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY WITH GHOST CELLS
 *   -- IRREGULAR DISTRIBUTION
 *  This is the master routine. All other creation routines are derived
 *  from this one.
\*/
logical nga_create_ghosts_irreg(
        Integer type,     /* MA type */ 
        Integer ndim,     /* number of dimensions */
        Integer dims[],   /* array of dimensions */
        Integer width[],  /* width of boundary cells for each dimension */
        char *array_name, /* array name */
        Integer map[],    /* decomposition map array */ 
        Integer nblock[], /* number of blocks for each dimension in map */
        Integer *g_a)     /* array handle (output) */
{

Integer  hi[MAXDIM];
Integer  mem_size, nelem;
Integer  i, ga_handle, status, maplen=0;

      ga_sync_();
      GA_PUSH_NAME("nga_create_ghosts_irreg");

      if(!GAinitialized) ga_error("GA not initialized ", 0);
      if(!ma_address_init) gai_ma_address_init();

      gam_checktype(type);
      gam_checkdim(ndim, dims);
      for(i=0; i< ndim; i++)
         if(nblock[i]>dims[i]) 
            ga_error("number of blocks must be <= corresponding dimension",i);
      for(i=0; i< ndim; i++)
         if(width[i]<0) 
            ga_error("Boundary widths must be >= 0",i);
      for(i=0; i< ndim; i++)
         if(width[i]>dims[i]) 
            ga_error("Boundary widths must be <= corresponding dimension",i);

      GAstat.numcre ++;

      /*** Get next free global array handle ***/
      ga_handle =-1; i=0;
      do{
          if(!GA[i].actv) ga_handle=i;
          i++;
      }while(i<_max_global_array && ga_handle==-1);
      if( ga_handle == -1)
          ga_error(" too many arrays ", (Integer)_max_global_array);
      *g_a = (Integer)ga_handle - GA_OFFSET;

      /*** fill in Global Info Record for g_a ***/
      gai_init_struct(ga_handle);
      GA[ga_handle].type = (int)type;
      GA[ga_handle].actv = 1;
      strcpy(GA[ga_handle].name, array_name);
      GA[ga_handle].ndim    = (int) ndim;

      GA[ga_handle].ghosts = 0;
      for( i = 0; i< ndim; i++){
         GA[ga_handle].dims[i] = (int)dims[i];
         GA[ga_handle].nblock[i] = (int)nblock[i];
         GA[ga_handle].scale[i] = (double)nblock[i]/(double)dims[i];
         GA[ga_handle].width[i] = (int)width[i];
         if (width[i] > 0) GA[ga_handle].ghosts = 1;
         maplen += nblock[i];
      } 
      for(i = 0; i< maplen; i++)GA[ga_handle].mapc[i] = (int)map[i];
      GA[ga_handle].mapc[maplen] = -1;
      GA[ga_handle].elemsize = GAsizeofM(type);
      /*** determine which portion of the array I am supposed to hold ***/
      nga_distribution_(g_a, &GAme, GA[ga_handle].lo, hi);
      for( i = 0, nelem=1; i< ndim; i++){
           GA[ga_handle].chunk[i] = (int)(hi[i]-GA[ga_handle].lo[i]+1);
           nelem *= (int)(hi[i]-GA[ga_handle].lo[i]+1+2*width[i]);
      }
      mem_size = nelem * GA[ga_handle].elemsize;
      GA[ga_handle].id = INVALID_MA_HANDLE;
      GA[ga_handle].size = mem_size;
      /* if requested, enforce limits on memory consumption */
      if(GA_memory_limited) GA_total_memory -= mem_size;
      /* check if everybody has enough memory left */
      if(GA_memory_limited){
         status = (GA_total_memory >= 0) ? 1 : 0;
         ga_igop(GA_TYPE_GSM, &status, 1, "*");
      }else status = 1;
/*      fprintf(stderr,"%d, elems=%d size=%d status=%d\n",GAme,nelem,mem_size,status);*/
/*      ga_sync_();*/
      if(status){
          status = !gai_getmem(array_name, GA[ga_handle].ptr,mem_size,
                                 (int)type, &GA[ga_handle].id);
      }else{
          GA[ga_handle].ptr[GAme]=NULL;
      }
/*      fprintf(stderr,"Memory on %d is located at %u\n",
              GAme,GA[ga_handle].ptr[GAme]); */
      ga_sync_();

      if(status){
         GAstat.curmem += GA[ga_handle].size;
         GAstat.maxmem  = MAX(GAstat.maxmem, GAstat.curmem);
         status = TRUE;
      }else{
         ga_destroy_(g_a);
         status = FALSE;
      }

      GA_POP_NAME;
      return status;
}


/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY
 *  Allow machine to choose location of array boundaries on individual
 *  processors.
\*/
logical nga_create(Integer type,
                   Integer ndim,
                   Integer dims[],
                   char* array_name,
                   Integer chunk[],
                   Integer *g_a)
{
Integer pe[MAXDIM], *pmap[MAXDIM], *map;
Integer d, i, width[MAXDIM];
Integer blk[MAXDIM];
logical status;
/*
extern void ddb(Integer ndims, Integer dims[], Integer npes,
Integer blk[], Integer pedims[]);
*/
extern void ddb_h2(Integer ndims, Integer dims[], Integer npes,double thr,
            Integer bias, Integer blk[], Integer pedims[]);

      GA_PUSH_NAME("nga_create");
      if(!GAinitialized) ga_error("GA not initialized ", 0);
      gam_checktype(type);
      gam_checkdim(ndim, dims);

      if(chunk && chunk[0]!=0) /* for either NULL or chunk[0]=0 compute all */
          for(d=0; d< ndim; d++) blk[d]=MIN(chunk[d],dims[d]);
      else
          for(d=0; d< ndim; d++) blk[d]=-1;

      /* eliminate dimensions =1 from ddb analysis */
      for(d=0; d<ndim; d++)if(dims[d]==1)blk[d]=1;
      
      /* for load balancing overwrite block size if needed */ 
/*      for(d=0; d<ndim; d++)if(blk[d]*GAnproc < dims[d])blk[d]=-1;*/

      if(GAme==0 && DEBUG )for(d=0;d<ndim;d++) fprintf(stderr,"b[%ld]=%ld\n",d,blk[d]);
      ga_sync_();

/*      ddb(ndim, dims, GAnproc, blk, pe);*/
      ddb_h2(ndim, dims, GAnproc, 0.0, (Integer)0, blk, pe);

      for(d=0, map=mapALL; d< ndim; d++){
        Integer nblock;
        Integer pcut; /* # procs that get full blk[] elements; the rest gets less*/
        int p;

        pmap[d] = map;
        width[d] = 0;

        /* RJH ... don't leave some nodes without data if possible
         but respect the users block size */
           
        if (chunk && chunk[d] > 1) {
          Integer ddim = (dims[d]-1)/MIN(chunk[d],dims[d]) + 1;
          pcut = (ddim -(blk[d]-1)*pe[d]) ;
        }
        else {
          pcut = (dims[d]-(blk[d]-1)*pe[d]) ;
        }

        for (nblock=i=p=0; (p<pe[d]) && (i<dims[d]); p++, nblock++) {
          Integer b = blk[d];
          if (p >= pcut)
            b = b-1;
          map[nblock] = i+1;
          if (chunk && chunk[d]>1) b *= MIN(chunk[d],dims[d]);
          i += b;
        }

        pe[d] = MIN(pe[d],nblock);
        map +=  pe[d]; 
      }

      if(GAme==0&& DEBUG){
         gai_print_subscript("pe ",(int)ndim, pe,"\n");
         gai_print_subscript("blocks ",(int)ndim, blk,"\n");
         printf("decomposition map\n");
         for(d=0; d< ndim; d++){
           printf("dim=%ld: ",d); 
           for (i=0;i<pe[d];i++)printf("%ld ",pmap[d][i]);
           printf("\n"); 
         }
         fflush(stdout);
      }
      status = nga_create_ghosts_irreg(type, ndim, dims, width, array_name,
               mapALL, pe, g_a);
      GA_POP_NAME;
      return status;
}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY WITH GHOST CELLS
 *  Allow machine to choose location of array boundaries on individual
 *  processors.
\*/
logical nga_create_ghosts(Integer type,
                   Integer ndim,
                   Integer dims[],
                   Integer width[],
                   char* array_name,
                   Integer chunk[],
                   Integer *g_a)
{
Integer pe[MAXDIM], *pmap[MAXDIM], *map;
Integer d, i;
Integer blk[MAXDIM];
logical status;
/*
extern void ddb(Integer ndims, Integer dims[], Integer npes,
Integer blk[], Integer pedims[]);
*/
extern void ddb_h2(Integer ndims, Integer dims[], Integer npes,double thr,
            Integer bias, Integer blk[], Integer pedims[]);

      GA_PUSH_NAME("nga_create_ghosts");
      if(!GAinitialized) ga_error("GA not initialized ", 0);
      gam_checktype(type);
      gam_checkdim(ndim, dims);

      if(chunk && chunk[0]!=0) /* for either NULL or chunk[0]=0 compute all */
          for(d=0; d< ndim; d++) blk[d]=MIN(chunk[d],dims[d]);
      else
          for(d=0; d< ndim; d++) blk[d]=-1;

      /* eliminate dimensions =1 from ddb analysis */
      for(d=0; d<ndim; d++)if(dims[d]==1)blk[d]=1;
      
      /* for load balancing overwrite block size if needed */ 
/*      for(d=0; d<ndim; d++)if(blk[d]*GAnproc < dims[d])blk[d]=-1;*/

      if(GAme==0 && DEBUG )for(d=0;d<ndim;d++) fprintf(stderr,"b[%ld]=%ld\n",d,blk[d]);
      ga_sync_();

/*      ddb(ndim, dims, GAnproc, blk, pe);*/
      ddb_h2(ndim, dims, GAnproc, 0.0, (Integer)0, blk, pe);

      for(d=0, map=mapALL; d< ndim; d++){
        Integer nblock;
        Integer pcut; /* # procs that get full blk[] elements; the rest gets less*/
        int p;

        pmap[d] = map;

        /* RJH ... don't leave some nodes without data if possible
         but respect the users block size */
           
        if (chunk && chunk[d] > 1) {
          Integer ddim = (dims[d]-1)/MIN(chunk[d],dims[d]) + 1;
          pcut = (ddim -(blk[d]-1)*pe[d]) ;
        }
        else {
          pcut = (dims[d]-(blk[d]-1)*pe[d]) ;
        }

        for (nblock=i=p=0; (p<pe[d]) && (i<dims[d]); p++, nblock++) {
          Integer b = blk[d];
          if (p >= pcut)
            b = b-1;
          map[nblock] = i+1;
          if (chunk && chunk[d]>1) b *= MIN(chunk[d],dims[d]);
          i += b;
        }

        pe[d] = MIN(pe[d],nblock);
        map +=  pe[d]; 
      }

      if(GAme==0&& DEBUG){
         gai_print_subscript("pe ",(int)ndim, pe,"\n");
         gai_print_subscript("blocks ",(int)ndim, blk,"\n");
         printf("decomposition map\n");
         for(d=0; d< ndim; d++){
           printf("dim=%ld: ",d); 
           for (i=0;i<pe[d];i++)printf("%ld ",pmap[d][i]);
           printf("\n"); 
         }
         fflush(stdout);
      }
      status = nga_create_ghosts_irreg(type, ndim, dims, width, array_name,
               mapALL, pe, g_a);
      GA_POP_NAME;
      return status;
}

/*\ CREATE A 2-DIMENSIONAL GLOBAL ARRAY
 *  Allow machine to choose location of array boundaries on individual
 *  processors.
\*/
logical ga_create(type, dim1, dim2, array_name, chunk1, chunk2, g_a)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a;
     char *array_name;
{
Integer ndim=2, dims[2], chunk[2];
logical status;
#ifdef  OLD_DEFAULT_BLK
#define BLK_THR 1
#else
#define BLK_THR 0
#endif

    dims[0]=*dim1;
    dims[1]=*dim2;

    /*block size of 1 is troublesome, old ga treated it as "use default" */
    /* for backward compatibility we use old convention */
    chunk[0] = (*chunk1 ==BLK_THR)? -1: *chunk1;
    chunk[1] = (*chunk2 ==BLK_THR)? -1: *chunk2;

    status = nga_create(*type, ndim,  dims, array_name, chunk, g_a);

    return status;
}

/*\ CREATE A GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
 *  User can specify location of array boundaries on individual
 *  processors.
\*/
logical nga_create_irreg(
        Integer type,     /* MA type */ 
        Integer ndim,     /* number of dimensions */
        Integer dims[],   /* array of dimensions */
        char *array_name, /* array name */
        Integer map[],    /* decomposition map array */ 
        Integer nblock[], /* number of blocks for each dimension in map */
        Integer *g_a)     /* array handle (output) */
{

Integer  d,width[MAXDIM];
logical status;

      for (d=0; d<ndim; d++) width[d] = 0;
      status = nga_create_ghosts_irreg(type, ndim, dims, width,
          array_name, map, nblock, g_a);
      return status;
}

/*\ CREATE A 2-DIMENSIONAL GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
 *  User can specify location of array boundaries on individual
 *  processors.
\*/
logical ga_create_irreg(type, dim1, dim2, array_name, map1, nblock1, map2,
                        nblock2, g_a)
      Integer *type, *dim1, *dim2, *map1, *nblock1, *map2, *nblock2, *g_a;
      char *array_name;
     /*
      * array_name    - a unique character string [input]
      * type          - MA type [input]
      * dim1/2        - array(dim1,dim2) as in FORTRAN [input]
      * nblock1       - no. of blocks first dimension is divided into [input]
      * nblock2       - no. of blocks second dimension is divided into [input]
      * map1          - no. ilo in each block [input]
      * map2          - no. jlo in each block [input]
      * g_a           - Integer handle for future references [output]
      */
{
Integer  ndim, dims[MAXDIM], width[MAXDIM], nblock[MAXDIM], *map;
Integer  i;
logical status;


      if(*type != MT_F_DBL  && *type != MT_F_INT &&  
         *type != MT_F_DCPL && *type != MT_F_REAL)
         ga_error("ga_create_irreg: type not yet supported ",  *type);
      else if( *dim1 <= 0 )
         ga_error("ga_create_irreg: array dimension1 invalid ",  *dim1);
      else if( *dim2 <= 0)
         ga_error("ga_create_irreg: array dimension2 invalid ",  *dim2);
      else if(*nblock1 <= 0)
         ga_error("ga_create_irreg: nblock1 <=0  ",  *nblock1);
      else if(*nblock2 <= 0)
         ga_error("ga_create_irreg: nblock2 <=0  ",  *nblock2);
      else if(*nblock1 * *nblock2 > GAnproc)
         ga_error("ga_create_irreg: too many blocks ",*nblock1 * *nblock2);

      if(GAme==0&& DEBUG){
        fprintf(stderr," array:%d map1:\n", (int)*g_a);
        for (i=0;i<*nblock1;i++)fprintf(stderr," %ld |",map1[i]);
        fprintf(stderr," \n array:%d map2:\n",(int) *g_a);
        for (i=0;i<*nblock2;i++)fprintf(stderr," %ld |",map2[i]);
        fprintf(stderr,"\n\n");
      }
      ndim = 2;
      dims[0] = *dim1;
      dims[1] = *dim2;
      width[0] = 0;
      width[1] = 0;
      nblock[0] = *nblock1;
      nblock[1] = *nblock2;
      map = mapALL;
      for(i=0;i< *nblock1; i++) map[i] = (int)map1[i];
      for(i=0;i< *nblock2; i++) map[i+ *nblock1] = (int)map2[i];
      status = nga_create_ghosts_irreg(*type, ndim, dims, width,
          array_name, mapALL, nblock, g_a);
      return status;

}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY WITH GHOST CELLS
 * -- IRREGULAR DISTRIBUTION
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_ghosts_irreg_(Integer *type, Integer *ndim,
    Integer *dims, Integer width[],
    _fcd array_name, Integer map[], Integer block[], Integer *g_a)
#else
logical FATR nga_create_ghosts_irreg_(Integer *type, Integer *ndim,
    Integer *dims, Integer width[], char* array_name, Integer map[],
    Integer block[], Integer *g_a, int slen)
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return (nga_create_ghosts_irreg(*type, *ndim,  dims, width, buf, map,
        block, g_a));
}

/*\ CREATE A 2-DIMENSIONAL GLOBAL ARRAY
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR ga_create_(type, dim1, dim2, array_name, chunk1, chunk2, g_a)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a;
     _fcd array_name;
#else
logical ga_create_(type, dim1, dim2, array_name, chunk1, chunk2, g_a, slen)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a;
     char* array_name;
     int slen;
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return(ga_create(type, dim1, dim2, buf, chunk1, chunk2, g_a));
}


/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_(Integer *type, Integer *ndim, Integer *dims,
                   _fcd array_name, Integer *chunk, Integer *g_a)
#else
logical FATR nga_create_(Integer *type, Integer *ndim, Integer *dims,
                   char* array_name, Integer *chunk, Integer *g_a, int slen)
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return (nga_create(*type, *ndim,  dims, buf, chunk, g_a));
}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY WITH GHOST CELLS
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_ghosts_(Integer *type, Integer *ndim, Integer *dims,
                   Integer *width, _fcd array_name, Integer *chunk, Integer *g_a)
#else
logical FATR nga_create_ghosts_(Integer *type, Integer *ndim, Integer *dims,
                   Integer *width, char* array_name, Integer *chunk, Integer *g_a,
                   int slen)
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return (nga_create_ghosts(*type, *ndim,  dims, width, buf, chunk, g_a));
}

/*\ CREATE A 2-DIMENSIONAL GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR ga_create_irreg_(type, dim1, dim2, array_name, map1, nblock1,
                         map2, nblock2, g_a)
     Integer *type, *dim1, *dim2, *map1, *map2, *nblock1, *nblock2, *g_a;
     _fcd array_name;
#else
logical FATR ga_create_irreg_(type, dim1, dim2, array_name, map1, nblock1,
                         map2, nblock2, g_a, slen)
     Integer *type, *dim1, *dim2, *map1, *map2, *nblock1, *nblock2, *g_a;
     char *array_name;
     int slen;
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif
  return( ga_create_irreg(type, dim1, dim2, buf, map1, nblock1,
                         map2, nblock2, g_a));
}

/*\ CREATE AN N-DIMENSIONAL GLOBAL ARRAY -- IRREGULAR DISTRIBUTION
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR nga_create_irreg_(Integer *type, Integer *ndim, Integer *dims,
                 _fcd array_name, Integer map[], Integer block[], Integer *g_a)
#else
logical FATR nga_create_irreg_(Integer *type, Integer *ndim, Integer *dims,
                 char* array_name, Integer map[], Integer block[],
                 Integer *g_a, int slen)
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return (nga_create_irreg(*type, *ndim,  dims, buf, map, block, g_a));
}

#ifdef PERMUTE_PIDS
char* ptr_array[MAX_NPROC];
#endif

/*\ get memory alligned w.r.t. MA base
 *  required on Linux as g77 ignores natural data alignment in common blocks
\*/ 
int gai_get_shmem(char **ptr_arr, Integer bytes, int type, long *adj)
{
int status=0;
#ifndef _CHECK_MA_ALGN
char *base;
long diff, item_size;  
Integer *adjust;
int i;

    /* need to enforce proper, natural allignment (on size boundary)  */
    switch (type){
      case MT_F_DBL: base =  (char *) DBL_MB; break;
      case MT_F_INT: base =  (char *) INT_MB; break;
      case MT_F_DCPL: base =  (char *) DCPL_MB; break;
      case MT_F_REAL: base =  (char *) FLT_MB; break;  
      default:        base = (char*)0;
    }

    item_size = GAsizeofM(type);
    bytes += item_size; 

#endif

    *adj = 0;
#ifdef PERMUTE_PIDS
    if(GA_Proc_list){
      bzero(ptr_array,GAnproc*sizeof(char*));
      status = ARMCI_Malloc((void**)ptr_array, bytes);
      for(i=0;i<GAnproc;i++)ptr_arr[i] = ptr_array[GA_inv_Proc_list[i]]; 
    }else
#endif

    status = ARMCI_Malloc((void**)ptr_arr, (int)bytes);
    if(status) return status;

#ifndef _CHECK_MA_ALGN

    /* adjust all addresses if they are not alligned on corresponding nodes*/

    /* we need storage for GAnproc*sizeof(Integer) -- _ga_map is bigger */
    adjust = (Integer*)_ga_map;

    diff = (ABS( base - (char *) ptr_arr[GAme])) % item_size; 
    for(i=0;i<GAnproc;i++)adjust[i]=0;
    adjust[GAme] = (diff > 0) ? item_size - diff : 0;
    *adj = adjust[GAme];

    ga_igop(GA_TYPE_GSM, adjust, GAnproc, "+");
    
    for(i=0;i<GAnproc;i++){
       ptr_arr[i] = adjust[i] + (char*)ptr_arr[i];
    }

#endif
    return status;
}


int gai_getmem(char* name, char **ptr_arr, Integer bytes, int type, long *id)
{
Integer handle = INVALID_MA_HANDLE, index;
Integer nelem, item_size = GAsizeofM(type);
char *ptr = (char*)0;

#ifdef AVOID_MA_STORAGE
   return gai_get_shmem(ptr_arr, bytes, type, id);
#else
   if(ARMCI_Uses_shm()) return gai_get_shmem(ptr_arr, bytes, type, id);
   else{

     nelem = bytes/item_size + 1;
     if(bytes)
        if(MA_alloc_get(type, nelem, name, &handle, &index))
                MA_get_pointer(handle, &ptr);
     *id   = (long)handle;

     /*
            printf("bytes=%d ptr=%ld index=%d\n",bytes, ptr,index);
            fflush(stdout);
     */

     bzero(ptr_arr,(int)GAnproc*sizeof(char*));
     ptr_arr[GAme] = ptr;
     armci_exchange_address((void**)ptr_arr,(int)GAnproc);
     if(bytes && !ptr) return 1; 
     else return 0;
   }
#endif
}


/*\ externalized version of gai_getmem to facilitate two-step array creation
\*/
void *GA_Getmem(int type, int nelem)
{
char **ptr_arr=(char**)0;
int  rc,i;
long id;
int bytes = nelem *  GAsizeofM(type);
int extra=sizeof(getmem_t)+GAnproc*sizeof(char*);
char *myptr;
Integer status;

     if(GA_memory_limited){
         GA_total_memory -= bytes+extra;
         status = (GA_total_memory >= 0) ? 1 : 0;
         ga_igop(GA_TYPE_GSM, &status, 1, "*");
         if(!status)GA_total_memory +=bytes+extra;
     }else status = 1;

     ptr_arr=(char**)_ga_map; /* need memory GAnproc*sizeof(char**) */
     rc= gai_getmem("ga_getmem", ptr_arr,(Integer)bytes+extra, type, &id);
     if(rc)ga_error("ga_getmem: failed to allocate memory",bytes+extra);

     myptr = ptr_arr[GAme];  

     /* make sure that remote memory addresses point to user memory */
     for(i=0; i<GAnproc; i++)ptr_arr[i] += extra;

#ifndef AVOID_MA_STORAGE
     if(ARMCI_Uses_shm()) 
#endif
        id += extra; /* id is used to store offset */

     /* stuff the type and id info at the beginning */
     ((getmem_t*)myptr)->id = id;
     ((getmem_t*)myptr)->type = type;
     ((getmem_t*)myptr)->size = bytes+extra;

     /* add ptr info */
     memcpy(myptr+sizeof(getmem_t),ptr_arr,(size_t)GAnproc*sizeof(char**));

     return (void*)(myptr+extra);
}


void GA_Freemem(void *ptr)
{
int extra = sizeof(getmem_t)+GAnproc*sizeof(char*); 
getmem_t *info = (getmem_t *)((char*)ptr - extra);
char **ptr_arr = (char**)(info+1);

#ifndef AVOID_MA_STORAGE
    if(ARMCI_Uses_shm()){
#endif
      /* make sure that we free original (before address alignment) pointer */
      ARMCI_Free(ptr_arr[GAme] - info->id);
#ifndef AVOID_MA_STORAGE
    }else{
      if(info->id != INVALID_MA_HANDLE) MA_free_heap(info->id);
    }
#endif

    if(GA_memory_limited) GA_total_memory += info->size;
}

/*\ RETURN COORDINATES OF A GA PATCH ASSOCIATED WITH PROCESSOR proc
\*/
void FATR nga_distribution_(Integer *g_a, Integer *proc, Integer *lo, Integer *
hi)
{
Integer ga_handle;

   ga_check_handleM(g_a, "nga_distribution");
   ga_handle = (GA_OFFSET + *g_a);
   ga_ownsM(ga_handle, *proc, lo, hi);
}




/*\ RETURN COORDINATES OF A GA PATCH ASSOCIATED WITH PROCESSOR proc
\*/
void FATR nga_distribution_no_handle_(Integer *ndim, Integer *dims,
        Integer *nblock, Integer *mapc, Integer *proc, Integer *lo,
        Integer * hi)
{
   ga_ownsM_no_handle(*ndim, dims, nblock, mapc, *proc, lo, hi);
}


/*\ Check to see if array has ghost cells.
\*/
logical FATR ga_has_ghosts_(Integer* g_a)
{
      int h_a = (int)*g_a + GA_OFFSET;
      return GA[h_a].ghosts;
}

Integer FATR ga_ndim_(Integer *g_a)
{
      ga_check_handleM(g_a,"ga_ndim");       
      return GA[*g_a +GA_OFFSET].ndim;
}
 


/*\ DUPLICATE A GLOBAL ARRAY
 *  -- new array g_b will have properties of g_a
\*/
logical ga_duplicate(Integer *g_a, Integer *g_b, char* array_name)
     /*
      * array_name    - a character string [input]
      * g_a           - Integer handle for reference array [input]
      * g_b           - Integer handle for new array [output]
      */
{
char     **save_ptr;
Integer  mem_size, mem_size_proc;
Integer  i, ga_handle, status;
int      *save_mapc;

      ga_sync_();

      GAstat.numcre ++; 

      ga_check_handleM(g_a,"ga_duplicate");       

      /* find a free global_array handle for g_b */
      ga_handle =-1; i=0;
      do{
        if(!GA[i].actv) ga_handle=i;
        i++;
      }while(i<_max_global_array && ga_handle==-1);
      if( ga_handle == -1)
          ga_error("ga_duplicate: too many arrays", (Integer)_max_global_array);
      *g_b = (Integer)ga_handle - GA_OFFSET;

      gai_init_struct(ga_handle);

      /*** copy content of the data structure ***/
      save_ptr = GA[ga_handle].ptr;
      save_mapc = GA[ga_handle].mapc;
      GA[ga_handle] = GA[GA_OFFSET + *g_a];
      strcpy(GA[ga_handle].name, array_name);
      GA[ga_handle].ptr = save_ptr;
      GA[ga_handle].mapc = save_mapc;
      for(i=0;i<MAPLEN; i++)GA[ga_handle].mapc[i] = GA[GA_OFFSET+ *g_a].mapc[i];

      /*** Memory Allocation & Initialization of GA Addressing Space ***/
      mem_size = mem_size_proc = GA[ga_handle].size; 
      GA[ga_handle].id = INVALID_MA_HANDLE;
      /* if requested, enforce limits on memory consumption */
      if(GA_memory_limited) GA_total_memory -= mem_size_proc;

      /* check if everybody has enough memory left */
      if(GA_memory_limited){
         status = (GA_total_memory >= 0) ? 1 : 0;
         ga_igop(GA_TYPE_GSM, &status, 1, "*");
      }else status = 1;

      if(status)
          status = !gai_getmem(array_name, GA[ga_handle].ptr,mem_size,
                               (int)GA[ga_handle].type, &GA[ga_handle].id);
      else{
          GA[ga_handle].ptr[GAme]=NULL;
      }

      ga_sync_();

#     ifdef GA_CREATE_INDEF
      if(status){
         Integer one = 1; 
         Integer dim1 =GA[ga_handle].dims[1], dim2=GA[ga_handle].dims[2];
         if(GAme==0)fprintf(stderr,"duplicate:initializing GA array%ld\n",*g_b);
         if(GA[ga_handle].type == MT_F_DBL) {
             DoublePrecision bad = DBL_MAX;
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);
         } else if (GA[ga_handle].type == MT_F_INT) {
             Integer bad = (Integer) INT_MAX;
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);
         } else if (GA[ga_handle].type == MT_F_DCPL) {
             DoubleComplex bad = {DBL_MAX, DBL_MAX};
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);
         } else if (GA[ga_handle].type == MT_F_REAL) {
             float bad = FLT_MAX;
             ga_fill_patch_(g_b, &one, &dim1, &one, &dim2,  &bad);   
         } else {
             ga_error("ga_duplicate: type not supported ",GA[ga_handle].type);
         }
      }
#     endif

      if(status){
         GAstat.curmem += GA[ga_handle].size;
         GAstat.maxmem  = MAX(GAstat.maxmem, GAstat.curmem);
         return(TRUE);
      }else{ 
         ga_destroy_(g_b);
         return(FALSE);
      }
}

/*\ DUPLICATE A GLOBAL ARRAY -- memory comes from user
 *  -- new array g_b will have properties of g_a
\*/
int GA_Assemble_duplicate(int g_a, char* array_name, void* ptr)
{
char     **save_ptr;
int      i, ga_handle;
int      *save_mapc;
int extra = sizeof(getmem_t)+GAnproc*sizeof(char*);
getmem_t *info = (getmem_t *)((char*)ptr - extra);
char **ptr_arr = (char**)(info+1);
int g_b;


      ga_sync_();

      GAstat.numcre ++;

      ga_check_handleM(&g_a,"ga_assemble_duplicate");

      /* find a free global_array handle for g_b */
      ga_handle =-1; i=0;
      do{
        if(!GA[i].actv) ga_handle=i;
        i++;
      }while(i<_max_global_array && ga_handle==-1);
      if( ga_handle == -1)
          ga_error("ga_assemble_duplicate: too many arrays ", 
                                           (Integer)_max_global_array);
      g_b = ga_handle - GA_OFFSET;

      gai_init_struct(ga_handle);

      /*** copy content of the data structure ***/
      save_ptr = GA[ga_handle].ptr;
      save_mapc = GA[ga_handle].mapc;
      GA[ga_handle] = GA[GA_OFFSET + g_a];
      strcpy(GA[ga_handle].name, array_name);
      GA[ga_handle].ptr = save_ptr;
      GA[ga_handle].mapc = save_mapc;
      for(i=0;i<MAPLEN; i++)GA[ga_handle].mapc[i] = GA[GA_OFFSET+ g_a].mapc[i];

      /* get ptrs and datatype from user memory */
      gam_checktype(info->type);
      GA[ga_handle].type = info->type;
      GA[ga_handle].size = info->size;
      GA[ga_handle].id = info->id;
      memcpy(GA[ga_handle].ptr,ptr_arr,(size_t)GAnproc*sizeof(char**));

      GAstat.curmem += GA[ga_handle].size;
      GAstat.maxmem  = MAX(GAstat.maxmem, GAstat.curmem);

      ga_sync_();

      return(g_b);
}


/*\ DUPLICATE A GLOBAL ARRAY
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
logical FATR ga_duplicate_(g_a, g_b, array_name)
     Integer *g_a, *g_b;
     _fcd array_name;
#else
logical FATR ga_duplicate_(g_a, g_b, array_name, slen)
     Integer *g_a, *g_b;
     char  *array_name;
     int   slen;
#endif
{
char buf[FNAM];
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  return(ga_duplicate(g_a, g_b, buf));
}



/*\ DESTROY A GLOBAL ARRAY
\*/
logical FATR ga_destroy_(Integer *g_a)
{
Integer ga_handle = GA_OFFSET + *g_a;

    ga_sync_();
    GAstat.numdes ++; /*regardless of array status we count this call */

    /* fails if handle is out of range or array not active */
    if(ga_handle < 0 || ga_handle >= _max_global_array) return FALSE;
    if(GA[ga_handle].actv==0) return FALSE;       
    GA[ga_handle].actv = 0;     
    if(GA[ga_handle].ptr[GAme]==NULL) return TRUE;
 
#ifndef AVOID_MA_STORAGE
    if(ARMCI_Uses_shm()){
#endif
      /* make sure that we free original (before address allignment) pointer */
      ARMCI_Free(GA[ga_handle].ptr[GAme] - GA[ga_handle].id);
#ifndef AVOID_MA_STORAGE
    }else{
      if(GA[ga_handle].id != INVALID_MA_HANDLE) MA_free_heap(GA[ga_handle].id);
    }
#endif

    if(GA_memory_limited) GA_total_memory += GA[ga_handle].size;
    GAstat.curmem -= GA[ga_handle].size;

    return(TRUE);
}

    
     
/*\ TERMINATE GLOBAL ARRAY STRUCTURES
 *
 *  all GA arrays are destroyed & shared memory is dealocated
 *  GA routines (except for ga_initialize) should not be called thereafter 
\*/
void FATR  ga_terminate_() 
{
Integer i, handle;
extern double t_dgop, n_dgop, s_dgop;


    if(!GAinitialized) return;
    for (i=0;i<_max_global_array;i++){
          handle = i - GA_OFFSET ;
          if(GA[i].actv) ga_destroy_(&handle);
          if(GA[i].ptr) free(GA[i].ptr);
          if(GA[i].mapc) free(GA[i].mapc);
    }
    ga_sync_();

    GA_total_memory = -1; /* restore "unlimited" memory usage status */
    GA_memory_limited = 0;
    free(_ga_map);
    free(GA_proclist);

    ARMCI_Finalize();
    GAinitialized = 0;
    ga_sync_();
}   

    
/*\ IS ARRAY ACTIVE/INACTIVE
\*/ 
Integer FATR ga_verify_handle_(g_a)
     Integer *g_a;
{
  return (Integer)
    ((*g_a + GA_OFFSET>= 0) && (*g_a + GA_OFFSET< _max_global_array) && 
             GA[GA_OFFSET + (*g_a)].actv);
}
 


/*\fill array with value
\*/
void FATR ga_fill_(Integer *g_a, void* val)
{
int i,elems,handle=GA_OFFSET + (int)*g_a;
char *ptr;

   GA_PUSH_NAME("ga_fill");
   ga_sync_();

   ga_check_handleM(g_a, "ga_fill");
   gam_checktype(GA[handle].type);
   elems = (int)GA[handle].size/GA[handle].elemsize;
   ptr = GA[handle].ptr[GAme];

   switch (GA[handle].type){
   case MT_F_DCPL: 
        for(i=0; i<elems;i++)((DoubleComplex*)ptr)[i]=*(DoubleComplex*)val;
        break;
   case MT_F_DBL:  
        for(i=0; i<elems;i++)((DoublePrecision*)ptr)[i]=*(DoublePrecision*)val;
        break;
   case MT_F_INT:  
        for(i=0; i<elems;i++)((Integer*)ptr)[i]=*(Integer*)val;
        break;
   case MT_F_REAL:
        for(i=0; i<elems;i++)((float*)ptr)[i]=*(float*)val;
        break;     
   default:
        ga_error("type not supported",GA[handle].type);
   }
   ga_sync_();
   GA_POP_NAME;
}




/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *  Fortran version
\*/ 
void FATR  ga_inquire_(Integer* g_a, Integer* type, Integer* dim1,Integer* dim2)
{
Integer ndim = ga_ndim_(g_a);

   if(ndim != 2)
      ga_error("ga_inquire: 2D API cannot be used for array dimension",ndim);

   *type       = (Integer)ga_type_c2f(GA[GA_OFFSET + *g_a].type);
   *dim1       = GA[GA_OFFSET + *g_a].dims[0];
   *dim2       = GA[GA_OFFSET + *g_a].dims[1];
}


/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *  C version
\*/
void ga_inquire(Integer* g_a, Integer* type, Integer* dim1, Integer* dim2)
{
Integer ndim = ga_ndim_(g_a);

   if(ndim != 2)
      ga_error("ga_inquire: 2D API cannot be used for array dimension",ndim);

   *type       = GA[GA_OFFSET + *g_a].type;
   *dim1       = GA[GA_OFFSET + *g_a].dims[0];
   *dim2       = GA[GA_OFFSET + *g_a].dims[1];
}


/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *  Fortran version
\*/
void FATR nga_inquire_(Integer *g_a, Integer *type, Integer *ndim,Integer *dims)
{
Integer handle = GA_OFFSET + *g_a,i;
   ga_check_handleM(g_a, "nga_inquire");
   *type       = (Integer)ga_type_c2f(GA[handle].type);
   *ndim       = GA[handle].ndim;
   for(i=0;i<*ndim;i++)dims[i]=GA[handle].dims[i];
}

/*\ INQUIRE POPERTIES OF A GLOBAL ARRAY
 *  C version
\*/
void nga_inquire(Integer *g_a, Integer *type, Integer *ndim,Integer *dims)
{
Integer handle = GA_OFFSET + *g_a,i;
   ga_check_handleM(g_a, "nga_inquire");
   *type       = GA[handle].type;
   *ndim       = GA[handle].ndim;
   for(i=0;i<*ndim;i++)dims[i]=GA[handle].dims[i];
}



/*\ INQUIRE NAME OF A GLOBAL ARRAY
 *  Fortran version
\*/
#if defined(CRAY) || defined(WIN32)
void FATR  ga_inquire_name_(g_a, array_name)
      Integer *g_a;
      _fcd    array_name;
{
   c2fstring(GA[GA_OFFSET+ *g_a].name,_fcdtocp(array_name),_fcdlen(array_name));
}
#else
void FATR  ga_inquire_name_(g_a, array_name, len)
      Integer *g_a;
      char    *array_name;
      int     len;
{
   c2fstring(GA[GA_OFFSET + *g_a].name, array_name, len);
}
#endif




/*\ INQUIRE NAME OF A GLOBAL ARRAY
 *  C version
\*/
void ga_inquire_name(g_a, array_name)
      Integer *g_a;
      char    **array_name;
{ 
   ga_check_handleM(g_a, "ga_inquire_name");
   *array_name = GA[GA_OFFSET + *g_a].name;
}





/*\ RETURN COORDINATES OF ARRAY BLOCK HELD BY A PROCESSOR
\*/
void FATR nga_proc_topology_(Integer* g_a, Integer* proc, Integer* subscript)
{
Integer d, index, ndim, ga_handle = GA_OFFSET + *g_a;

   ga_check_handleM(g_a, "nga_proc_topology");
   ndim = GA[ga_handle].ndim;

   index = GA_Proc_list ? GA_Proc_list[*proc]: *proc;

   for(d=0; d<ndim; d++){
       subscript[d] = index% GA[ga_handle].nblock[d];
       index  /= GA[ga_handle].nblock[d];  
   }
}




#define findblock(map_ij,n,scale,elem, block)\
{\
int candidate, found, b, *map= (map_ij);\
\
    candidate = (int)(scale*(elem));\
    found = 0;\
    if(map[candidate] <= (elem)){ /* search downward */\
         b= candidate;\
         while(b<(n)-1){ \
            found = (map[b+1]>(elem));\
            if(found)break;\
            b++;\
         } \
    }else{ /* search upward */\
         b= candidate-1;\
         while(b>=0){\
            found = (map[b]<=(elem));\
            if(found)break;\
            b--;\
         }\
    }\
    if(!found)b=(n)-1;\
    *(block) = b;\
}



/*\ LOCATE THE OWNER OF SPECIFIED ELEMENT OF A GLOBAL ARRAY
\*/
logical FATR nga_locate_(Integer *g_a, Integer* subscript, Integer* owner)
{
Integer d, proc, dpos, ndim, ga_handle = GA_OFFSET + *g_a, proc_s[MAXDIM];

   ga_check_handleM(g_a, "nga_locate");
   ndim = GA[ga_handle].ndim;

   for(d=0, *owner=-1; d< ndim; d++) 
       if(subscript[d]< 1 || subscript[d]>GA[ga_handle].dims[d]) return FALSE;

   for(d = 0, dpos = 0; d< ndim; d++){
       findblock(GA[ga_handle].mapc + dpos, GA[ga_handle].nblock[d],
                 GA[ga_handle].scale[d], (int)subscript[d], &proc_s[d]);
       dpos += GA[ga_handle].nblock[d];
   }

   ga_ComputeIndexM(&proc, ndim, proc_s, GA[ga_handle].nblock); 

   *owner = GA_Proc_list ? GA_Proc_list[proc]: proc;

   return TRUE;
}



/*\ LOCATE PROCESSORS/OWNERS OF THE SPECIFIED PATCH OF A GLOBAL ARRAY
\*/
logical FATR nga_locate_region_( Integer *g_a,
                                 Integer *lo,
                                 Integer *hi,
                                 Integer *map,
                                 Integer *proclist,
                                 Integer *np)
/*    g_a      [input]  global array handle
      lo       [input]  lower indices of patch in global array
      hi       [input]  upper indices of patch in global array
      map      [output] list of lower and upper indices for portion of
                        patch that exists on each processor containing a
                        portion of the patch. The map is constructed so
                        that for a D dimensional global array, the first
                        D elements are the lower indices on the first
                        processor in proclist, the next D elements are
                        the upper indices of the first processor in
                        proclist, the next D elements are the lower
                        indices for the second processor in proclist, and
                        so on.
      proclist [output] list of processors containing some portion of the
                        patch
      np       [output] total number of processors containing a portion
                        of the patch
*/
{
int  procT[MAXDIM], procB[MAXDIM], proc_subscript[MAXDIM];
Integer  proc, owner, i, ga_handle;
Integer  d, dpos, ndim, elems;

   ga_check_handleM(g_a, "nga_locate_region");

   ga_handle = GA_OFFSET + *g_a;

   for(d = 0; d< GA[ga_handle].ndim; d++)
       if((lo[d]<1 || hi[d]>GA[ga_handle].dims[d]) ||(lo[d]>hi[d]))return FALSE;

   ndim = GA[ga_handle].ndim;

   /* find "processor coordinates" for the top left corner and store them
    * in ProcT */
   for(d = 0, dpos = 0; d< GA[ga_handle].ndim; d++){
       findblock(GA[ga_handle].mapc + dpos, GA[ga_handle].nblock[d], 
                 GA[ga_handle].scale[d], (int)lo[d], &procT[d]);
       dpos += GA[ga_handle].nblock[d];
   }

   /* find "processor coordinates" for the right bottom corner and store
    * them in procB */
   for(d = 0, dpos = 0; d< GA[ga_handle].ndim; d++){
       findblock(GA[ga_handle].mapc + dpos, GA[ga_handle].nblock[d], 
                 GA[ga_handle].scale[d], (int)hi[d], &procB[d]);
       dpos += GA[ga_handle].nblock[d];
   }

   *np = 0;

   /* Find total number of processors containing data and return the
    * result in elems. Also find the lowest "processor coordinates" of the
    * processor block containing data and return these in proc_subscript.
   */
   ga_InitLoopM(&elems, ndim, proc_subscript, procT,procB,GA[ga_handle].nblock);

   for(i= 0; i< elems; i++){ 
      Integer _lo[MAXDIM], _hi[MAXDIM];
      Integer  offset;

      /* convert i to owner processor id using the current values in
       proc_subscript */
      ga_ComputeIndexM(&proc, ndim, proc_subscript, GA[ga_handle].nblock); 
      owner = GA_Proc_list ? GA_Proc_list[proc]: proc;
      /* get range of global array indices that are owned by owner */
      ga_ownsM(ga_handle, owner, _lo, _hi);

      offset = *np *(ndim*2); /* location in map to put patch range */

      for(d = 0; d< ndim; d++)
              map[d + offset ] = lo[d] < _lo[d] ? _lo[d] : lo[d];
      for(d = 0; d< ndim; d++)
              map[ndim + d + offset ] = hi[d] > _hi[d] ? _hi[d] : hi[d];

      proclist[i] = owner;
      /* Update to proc_subscript so that it corresponds to the next
       * processor in the block of processors containing the patch */
      ga_UpdateSubscriptM(ndim,proc_subscript,procT,procB,GA[ga_handle].nblock);
      (*np)++;
   }
   return(TRUE);
}
    

/*\ returns in nblock array the number of blocks each dimension is divided to
\*/
void GA_Nblock(int g_a, int *nblock)
{
int ga_handle = GA_OFFSET + g_a;
int i, n;

     ga_check_handleM(&g_a, "GA_Nblock");

     n = GA[ga_handle].ndim;

#ifdef USE_FAPI 
     for(i=0; i<n; i++) nblock[i] = GA[ga_handle].nblock[i]);
#else
     for(i=0; i<n; i++) nblock[n-i-1] = GA[ga_handle].nblock[i];
#endif
     
}
     

void FATR ga_nblock_(Integer *g_a, Integer *nblock)
{
Integer ga_handle = GA_OFFSET + *g_a;
int i, n;

     ga_check_handleM(g_a, "ga_nblock");

     n = GA[ga_handle].ndim;

     for(i=0; i<n; i++) nblock[i] = (Integer)GA[ga_handle].nblock[i];
}


Integer FATR ga_nodeid_()
{
  return ((Integer)GAme);
}


Integer FATR ga_nnodes_()
{
  return ((Integer)GAnproc);
}



/*\ COMPARE DISTRIBUTIONS of two global arrays
\*/
logical FATR ga_compare_distr_(Integer *g_a, Integer *g_b)
{
int h_a =(int)*g_a + GA_OFFSET;
int h_b =(int)*g_b + GA_OFFSET;
int i;

   GA_PUSH_NAME("ga_compare_distr");
   ga_check_handleM(g_a, "distribution a");
   ga_check_handleM(g_b, "distribution b");
   
   GA_POP_NAME;

   if(GA[h_a].ndim != GA[h_b].ndim) return FALSE; 

   for(i=0; i <GA[h_a].ndim; i++)
       if(GA[h_a].dims[i] != GA[h_b].dims[i]) return FALSE;

   for(i=0; i <MAPLEN; i++){
      if(GA[h_a].mapc[i] != GA[h_b].mapc[i]) return FALSE;
      if(GA[h_a].mapc[i] == -1) break;
   }
   return TRUE;
}


static int num_mutexes=0;
static int chunk_mutex;

logical FATR ga_create_mutexes_(Integer *num)
{
int myshare;

   if (*num <= 0 || *num > MAX_MUTEXES) return(FALSE);
   if(num_mutexes) ga_error("mutexes already created",num_mutexes);

   num_mutexes= (int)*num;

   if(GAnproc == 1) return(TRUE);

   chunk_mutex = (int)((*num + GAnproc-1)/GAnproc);
   if(GAme * chunk_mutex >= *num)myshare =0;
   else myshare=chunk_mutex;

   /* need work here to use permutation */
   if(ARMCI_Create_mutexes(myshare)) return FALSE;
   return TRUE;
}


void FATR ga_lock_(Integer *mutex)
{
int m,p;

   if(GAnproc == 1) return;
   if(num_mutexes< *mutex)ga_error("invalid mutex",*mutex);

   p = num_mutexes/chunk_mutex -1;
   m = num_mutexes%chunk_mutex;

#ifdef PERMUTE_PIDS
    if(GA_Proc_list) p = GA_inv_Proc_list[p];
#endif

   ARMCI_Lock(m,p);
}


void FATR ga_unlock_(Integer *mutex)
{
int m,p;

   if(GAnproc == 1) return;
   if(num_mutexes< *mutex)ga_error("invalid mutex",*mutex);
   
   p = num_mutexes/chunk_mutex -1;
   m = num_mutexes%chunk_mutex;

#ifdef PERMUTE_PIDS
    if(GA_Proc_list) p = GA_inv_Proc_list[p];
#endif

   ARMCI_Unlock(m,p);
}              
   

logical FATR ga_destroy_mutexes_()
{
   if(num_mutexes<1) ga_error("mutexes destroyed",0);
   num_mutexes= 0;
   if(GAnproc == 1) return TRUE;
   if(ARMCI_Destroy_mutexes()) return FALSE;
   return TRUE;
}


/*\ return list of message-passing process ids for GA process ids
\*/
void FATR ga_list_nodeid_(list, num_procs)
     Integer *list, *num_procs;
{
Integer proc;
   for( proc = 0; proc < *num_procs; proc++)

#ifdef PERMUTE_PIDS
       if(GA_Proc_list) list[proc] = GA_inv_Proc_list[proc]; 
       else
#endif
       list[proc]=proc;
}


/*************************************************************************/

logical FATR ga_locate_region_(g_a, ilo, ihi, jlo, jhi, mapl, np )
        Integer *g_a, *ilo, *jlo, *ihi, *jhi, mapl[][5], *np;
{
   logical status;
   Integer lo[2], hi[2], p;
   lo[0]=*ilo; lo[1]=*jlo;
   hi[0]=*ihi; hi[1]=*jhi;

   status = nga_locate_region_(g_a,lo,hi,_ga_map, GA_proclist, np);

   /* need to swap elements (ilo,jlo,ihi,jhi) -> (ilo,ihi,jlo,jhi) */
   for(p = 0; p< *np; p++){
     mapl[p][0] = _ga_map[4*p];
     mapl[p][1] = _ga_map[4*p + 2];
     mapl[p][2] = _ga_map[4*p + 1];
     mapl[p][3] = _ga_map[4*p + 3];
     mapl[p][4] = GA_proclist[p];
   } 

   return status;
}



/*\ LOCATE THE OWNER OF THE (i,j) ELEMENT OF A GLOBAL ARRAY
\*/
logical FATR ga_locate_(g_a, i, j, owner)
        Integer *g_a, *i, *j, *owner;
{
Integer subscript[2];
  
    subscript[0]=*i; subscript[1]=*j;

    return nga_locate_(g_a, subscript, owner);
}


/*\ RETURN COORDINATES OF A 2-D GA PATCH ASSOCIATED WITH PROCESSOR proc
\*/
void FATR  ga_distribution_(g_a, proc, ilo, ihi, jlo, jhi)
   Integer *g_a, *ilo, *ihi, *jlo, *jhi, *proc;
{
Integer lo[2], hi[2];
Integer ndim = ga_ndim_(g_a);

   if(ndim != 2)
      ga_error("ga_distribution:2D API cannot be used for dimension",ndim);

   nga_distribution_(g_a, proc, lo, hi);
   *ilo = lo[0]; *ihi=hi[0];
   *jlo = lo[1]; *jhi=hi[1]; 
}


/*\ RETURN COORDINATES OF ARRAY BLOCK HELD BY A PROCESSOR
\*/
void FATR ga_proc_topology_(g_a, proc, pr, pc)
   Integer *g_a, *proc, *pr, *pc;
{
Integer subscript[2];
   nga_proc_topology_(g_a, proc,subscript);
   *pr = subscript[0]; 
   *pc = subscript[1]; 
}


/*\ returns true/false depending on validity of the handle
\*/
logical FATR ga_valid_handle_(Integer *g_a)
{
   if(GA_OFFSET+ (*g_a) < 0 || GA_OFFSET+(*g_a) >= _max_global_array ||
      ! (GA[GA_OFFSET+(*g_a)].actv) ) return FALSE;
   else return TRUE;
}

int gai_getval(int *ptr) { return *ptr;}
