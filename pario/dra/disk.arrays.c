/*$Id: disk.arrays.c,v 1.53 2002-08-16 14:48:30 d3g293 Exp $*/

/************************** DISK ARRAYS **************************************\
|*         Jarek Nieplocha, Fri May 12 11:26:38 PDT 1995                     *|
\*****************************************************************************/

/* DISCLAIMER
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

#include "global.h"
#include "drap.h"
#include "dra.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "macdecls.h"

/************************** constants ****************************************/

/*  buffer size --- adjust to be a multiplicity of the
    striping factor in a parallel filesystem */
#ifdef SP
#define DRA_DBL_BUF_SIZE 131072
#else
#define DRA_DBL_BUF_SIZE 100000 
#endif

#define DRA_FAIL  (Integer)1
#define COLUMN    1
#define ROW       0
#define ON        1
#define OFF       0

#define ILO       0
#define IHI       1
#define JLO       2
#define JHI       3

#define DRA_OP_WRITE  777
#define DRA_OP_READ   888

#define MAX_REQ   5

/* message type/tag used by DRA */
#define  GA_TYPE_GSM 32760 - 6
#define  GA_TYPE_GOP 32760 - 7

/* alignment factor for the internal buffer */
#if defined(CRAY)
#   define ALIGN 512
#else
#   define ALIGN 16
#endif

#define INFINITE_NUM_PROCS  8094

#define CLIENT_TO_SERVER 2

#ifdef PARAGON
#  define DRA_NUM_IOPROCS  64
#  define DRA_NUM_FILE_MGR INFINITE_NUM_PROCS
#elif defined(CRAY_T3D)
#  define DRA_NUM_IOPROCS 16 
#elif defined(CRAY_YMP)
#  define DRA_NUM_IOPROCS 4 
#elif defined(SP1)|| defined(SP) || defined(LAPI)
#     define DRA_NUM_IOPROCS 8 
#elif defined(KSR)
#  define DRA_NUM_IOPROCS 8
#else
#  define DRA_NUM_IOPROCS 1
#endif

#ifndef DRA_NUM_FILE_MGR
#  define DRA_NUM_FILE_MGR DRA_NUM_IOPROCS
#endif

#define DRA_BUF_SIZE     (DRA_DBL_BUF_SIZE*sizeof(DoublePrecision)) 
#define DRA_INT_BUF_SIZE (DRA_BUF_SIZE/sizeof(Integer))
/***************************** Global Data ***********************************/

#ifdef STATBUF
DoublePrecision _dra_dbl_buffer[DRA_DBL_BUF_SIZE];        /* DRA data buffer */
char*           _dra_buffer = (char*)_dra_dbl_buffer;
#else
char*           _dra_buffer;
Integer         _idx_buffer, _handle_buffer;
#endif

disk_array_t *DRA;          /* array of struct for basic info about DRA arrays*/
Integer _max_disk_array;    /* max number of disk arrays open at a time      */
logical dra_debug_flag;     /* globally defined debug parameter */


request_t     Requests[MAX_REQ];
int num_pending_requests=0;
Integer _dra_turn=0;
int     Dra_num_serv=DRA_NUM_IOPROCS;
 
/****************************** Macros ***************************************/

#define dai_sizeofM(_type) MA_sizeof(_type, 1, MT_C_CHAR)

#define dai_check_typeM(_type)  if (_type != MT_F_DBL && _type != MT_F_INT \
     && _type != MT_INT && _type != MT_DBL\
     && _type != MT_REAL && _type != MT_F_DCPL && _type != MT_F_REAL)\
                                  dai_error("invalid type ",_type)  
#define dai_check_handleM(_handle, msg)                                    \
{\
        if((_handle+DRA_OFFSET)>=_max_disk_array || (_handle+DRA_OFFSET)<0) \
        {fprintf(stderr,"%s, %ld --",msg, (long)_max_disk_array);\
        dai_error("invalid DRA handle",_handle);}                           \
        if( DRA[(_handle+DRA_OFFSET)].actv == 0)                            \
        {fprintf(stderr,"%s:",msg);\
        dai_error("disk array not active",_handle);}                       \
}
        
#define dai_check_rangeM(_lo, _hi, _dim, _err_msg)                         \
        if(_lo < (Integer)1   || _lo > _dim ||_hi < _lo || _hi > _dim)     \
        dai_error(_err_msg, _dim)
 
#define ga_get_sectM(sect, _buf, _ld)\
   ga_get_(&sect.handle, &sect.lo[0], &sect.hi[0], &sect.lo[1], &sect.hi[1], _buf, &_ld)

#define ga_put_sectM(sect, _buf, _ld)\
   ga_put_(&sect.handle, &sect.lo[0], &sect.hi[0], &sect.lo[1], &sect.hi[1], _buf, &_ld)

#define fill_sectionM(sect, _hndl, _ilo, _ihi, _jlo, _jhi) \
{ \
        sect.handle = _hndl;\
        sect.ndim   = 2; \
        sect.lo[0]    = _ilo;\
        sect.hi[0]    = _ihi;\
        sect.lo[1]    = _jlo;\
        sect.hi[1]    = _jhi;\
}

#define sect_to_blockM(ds_a, CR)\
{\
      Integer   hndl = (ds_a).handle+DRA_OFFSET;\
      Integer   br   = ((ds_a).lo[0]-1)/DRA[hndl].chunk[0];\
      Integer   bc   = ((ds_a).lo[1]-1)/DRA[hndl].chunk[1];\
      Integer   R    = (DRA[hndl].dims[0] + DRA[hndl].chunk[0] -1)/DRA[hndl].chunk[0];\
               *(CR) = bc * R + br;\
}

#define block_to_sectM(ds_a, CR)\
{\
      Integer   hndl = (ds_a)->handle+DRA_OFFSET;\
      Integer   R    = (DRA[hndl].dims[0] + DRA[hndl].chunk[0]-1)/DRA[hndl].chunk[0];\
      Integer   br = (CR)%R;\
      Integer   bc = ((CR) - br)/R;\
      (ds_a)->  lo[0]= br * DRA[hndl].chunk[0] +1;\
      (ds_a)->  lo[1]= bc * DRA[hndl].chunk[1] +1;\
      (ds_a)->  hi[0]= (ds_a)->lo[0] + DRA[hndl].chunk[0];\
      (ds_a)->  hi[1]= (ds_a)->lo[1] + DRA[hndl].chunk[1];\
      if( (ds_a)->hi[0] > DRA[hndl].dims[0]) (ds_a)->hi[0] = DRA[hndl].dims[0];\
      if( (ds_a)->hi[1] > DRA[hndl].dims[1]) (ds_a)->hi[1] = DRA[hndl].dims[1];\
}
      
#define INDEPFILES(x) (DRA[(x)+DRA_OFFSET].indep)

char dummy_fname[DRA_MAX_FNAME];
/*****************************************************************************/


/*#define DEBUG 1*/
/*#define CLEAR_BUF 1*/


/*\ determines if write operation to a disk array is allowed
\*/
int dai_write_allowed(Integer d_a)
{
Integer handle = d_a+DRA_OFFSET;
        if(DRA[handle].mode == DRA_W || DRA[handle].mode == DRA_RW) return 1;
        else return 0;
}


/*\ determines if read operation from a disk array is allowed
\*/
int dai_read_allowed(Integer d_a)
{
Integer handle = d_a+DRA_OFFSET;
        if(DRA[handle].mode == DRA_R || DRA[handle].mode == DRA_RW) return 1;
        else return 0;
}


/*\  number of processes that could perform I/O
\*/
Integer dai_io_procs(Integer d_a)
{
Integer num;

        /* this one of many possibilities -- depends on the system */
/*
#ifdef _CRAYMPP
        num = DRA_NUM_IOPROCS;
#else
        num = (INDEPFILES(d_a)) ? INFINITE_NUM_PROCS: DRA_NUM_IOPROCS; 
#endif
*/
        num = ga_cluster_nnodes_();

        return( MIN( ga_nnodes_(), num));
}


/*\  rank of calling process in group of processes that could perform I/O
 *   a negative value means that this process doesn't do I/O
\*/
Integer dai_io_nodeid(Integer d_a)
{
Integer me = ga_nodeid_();
Integer nodeid = ga_cluster_nodeid_();
Integer zero = 0;

       /* again, one of many possibilities: 
        * if proc id beyond I/O procs number, negate it
        */
        if(me == ga_cluster_procid_(&nodeid, &zero)) me = nodeid;
        else me = -1;
        /*if (me >= dai_io_procs(d_a)) me = -me;*/
        return (me);
}


/*\ determines if I/O process participates in file management (create/delete)
\*/
Integer dai_io_manage(Integer d_a)
{
        Integer me = dai_io_nodeid(d_a);

        if(me >= 0 )
          return (1);
        else
          return (0);
}


/*\ select one master process for each file associated with d_a
\*/
Integer dai_file_master(Integer d_a)
{
       if(dai_io_nodeid(d_a)<0)return 0;

       /* for indep files each I/O process has its own file
        * for shared file 0 is the master
        */
     
       if(INDEPFILES(d_a) || dai_io_nodeid(d_a) == 0 ) return 1;
       else return 0;

}



/*\  registers callback function associated with completion of asynch. I/O
\*/
void dai_callback(int op, int transp, section_t gs_a, section_t ds_a, 
                  section_t ds_chunk, Integer ld[], Integer req)
{
  Integer i;
        if(Requests[req].callback==ON) dai_error("DRA: callback not cleared",0);
        Requests[req].callback = ON;
        Requests[req].args.op = op;
        Requests[req].args.transp = transp;
        Requests[req].args.gs_a = gs_a;
        Requests[req].args.ds_a = ds_a;
        Requests[req].args.ds_chunk = ds_chunk;
        Requests[req].args.ld[0] = ld[0];
        for (i=1; i<gs_a.ndim-1; i++) Requests[req].args.ld[i] = ld[i];
}




/*\ INITIALIZE DISK ARRAY DATA STRUCTURES
\*/
Integer FATR dra_init_(
        Integer *max_arrays,              /* input */
        DoublePrecision *max_array_size,  /* input */
        DoublePrecision *tot_disk_space,  /* input */
        DoublePrecision *max_memory)      /* input */
{
#define DEF_MAX_ARRAYS 16
#define MAX_ARRAYS 1024
int i;
        ga_sync_();

        if(*max_arrays<-1 || *max_arrays> MAX_ARRAYS)
           dai_error("dra_init: incorrect max number of arrays",*max_arrays);
        _max_disk_array = (*max_arrays==-1) ? DEF_MAX_ARRAYS: *max_arrays;

        Dra_num_serv = drai_get_num_serv();

        DRA = (disk_array_t*)malloc(sizeof(disk_array_t)* (int)*max_arrays);
        if(!DRA) dai_error("dra_init: memory alocation failed\n",0);
        for(i=0; i<_max_disk_array ; i++)DRA[i].actv=0;

        for(i=0; i<MAX_REQ; i++)Requests[i].num_pending=0;

        dra_debug_flag = FALSE;
#ifndef STATBUF
        {
            /* check if we have enough MA memory for DRA buffer on every node */
            Integer avail = MA_inquire_avail(MT_C_DBL);
            long diff;
            ga_igop(GA_TYPE_GOP, &avail, (Integer)1, "min");
            if(avail < (ALIGN -1 + DRA_DBL_BUF_SIZE) && ga_nodeid_() == 0)
              dai_error("Not enough memory available from MA for DRA",avail);

            /* get buffer memory */
            if(MA_alloc_get(MT_C_DBL, DRA_DBL_BUF_SIZE+ALIGN-1, "DRA buf", 
              &_handle_buffer, &_idx_buffer))
                MA_get_pointer(_handle_buffer, &_dra_buffer); 
            else
                dai_error("dra_init: ma_alloc_get failed",DRA_DBL_BUF_SIZE); 

            /* align buffer address */
            diff = ((long)_dra_buffer) % (sizeof(DoublePrecision)*ALIGN);
            if(diff) _dra_buffer += (sizeof(DoublePrecision)*ALIGN - diff);
        }
#endif
 
        ga_sync_();

        return(ELIO_OK);
}



/*\ correct chunk size to fit the buffer and introduce allignment
\*/
void dai_correct_chunking(Integer* a, Integer* b, Integer prod, double ratio)
/*
    a:     Initial guess for larger chunk [input/output]
    b:     Initial guess for smaller chunk [input/output]
    prod:  Total number of elements that will fit in buffer [input]
    ratio: Ratio of size of current patch formed by a and b to size of
           I/O buffer
*/
{
#define EPS_SEARCH 100

Integer b0, bt, eps0, eps1, trial;
double  da = (double)*a, db = (double)*b;
double  ch_ratio = da/db;
   
        db = sqrt(prod /ch_ratio); 
        trial = b0 = (Integer)db; eps0 = prod%b0;
        trial = MAX(trial,EPS_SEARCH); /******/
        /* search in the neighborhood for a better solution */
        for (bt = trial+ EPS_SEARCH; bt> MIN(1,trial-EPS_SEARCH); bt--){
            eps1 =  prod%bt;
            if(eps1 < eps0){
              /* better solution found */
              b0 = bt; eps0 = eps1;
            }
        } 
        *a = prod/b0;
        *b = b0;
}    
   

   
/*\ compute chunk parameters for layout of arrays on the disk
 *   ---- a very simple algorithm to be refined later ----
\*/
void dai_chunking(Integer elem_size, Integer block1, Integer block2, 
                  Integer dim1, Integer dim2, Integer *chunk1, Integer *chunk2)
/*   elem_size:     Size of stored data [input]
     block1:        Estimated size of request in dimension 1 [input]
     block2:        Estimated size of request in dimension 2 [input]
     dim1:          Size of DRA in dimension 1 [input]
     dim2:          Size of DRA in dimension 2 [input]
     chunk1:        Data block size in dimension 1? [output]
     chunk2:        Data block size in dimension 2? [output]
*/
{
Integer patch_size;
  
        *chunk1 = *chunk2 =0; 
        if(block1 <= 0 && block2 <= 0){

          *chunk1 = dim1;
          *chunk2 = dim2;

        }else if(block1 <= 0){
          *chunk2 = block2;
          *chunk1 = MAX(1, DRA_BUF_SIZE/(elem_size* (*chunk2)));
        }else if(block2 <= 0){
          *chunk1 = block1;
          *chunk2 = MAX(1, DRA_BUF_SIZE/(elem_size* (*chunk1)));
        }else{
          *chunk1 = block1;
          *chunk2 = block2;
        }

        /* need to correct chunk size to fit chunk1 x chunk2 request in buffer*/
        patch_size = (*chunk1)* (*chunk2)*elem_size;
          
        if (patch_size > DRA_BUF_SIZE){
             
           if( *chunk1 == 1) *chunk2  = DRA_BUF_SIZE/elem_size;
           else if( *chunk2 == 1) *chunk1  = DRA_BUF_SIZE/elem_size;
           else {
             double  ratio = ((double)patch_size)/((double)DRA_BUF_SIZE); 
             /* smaller chunk to be scaled first */
             if(*chunk1 < *chunk2){
               dai_correct_chunking(chunk2,chunk1,DRA_BUF_SIZE/elem_size,ratio);
             }else{
               dai_correct_chunking(chunk1,chunk2,DRA_BUF_SIZE/elem_size,ratio);
             }
           }
        }
#       ifdef DEBUG
          printf("\n%d:CREATE chunk=(%d,%d) elem_size=%d req=(%d,%d) buf=%d\n",
                ga_nodeid_(),*chunk1, *chunk2, elem_size, block1, block2,
                DRA_DBL_BUF_SIZE); 
          fflush(stdout);
#       endif
}



/*\ get a new handle for disk array 
\*/
Integer dai_get_handle(void)
{
Integer dra_handle =-1, candidate = 0;

        do{
            if(!DRA[candidate].actv){ 
               dra_handle=candidate;
               DRA[candidate].actv =1;
            }
            candidate++;
        }while(candidate < _max_disk_array && dra_handle == -1);
        return(dra_handle);
}      
     


/*\ release handle -- makes array inactive
\*/
void dai_release_handle(Integer *handle)
{
     DRA[*handle+DRA_OFFSET].actv =0;
     *handle = 0;
}



/*\ find offset in file for (ilo,ihi) element
\*/
void dai_file_location(section_t ds_a, Off_t* offset)
{
Integer row_blocks, handle=ds_a.handle+DRA_OFFSET, offelem, cur_ld, part_chunk1;

        if((ds_a.lo[0]-1)%DRA[handle].chunk[0])
            dai_error("dai_file_location: not alligned ??",ds_a.lo[0]);

        row_blocks  = (ds_a.lo[0]-1)/DRA[handle].chunk[0];/* # row blocks from top*/
        part_chunk1 = DRA[handle].dims[0]%DRA[handle].chunk[0];/*dim1 in part block*/
        cur_ld      = (row_blocks == DRA[handle].dims[0] / DRA[handle].chunk[0]) ? 
                       part_chunk1: DRA[handle].chunk[0];

        /* compute offset (in elements) */

        if(INDEPFILES(ds_a.handle)) {

           Integer   CR, R; 
           Integer   i, num_part_block = 0;
           Integer   ioprocs=dai_io_procs(ds_a.handle); 
           Integer   iome = dai_io_nodeid(ds_a.handle);
           
           sect_to_blockM(ds_a, &CR); 

           R    = (DRA[handle].dims[0] + DRA[handle].chunk[0]-1)/DRA[handle].chunk[0];
           for(i = R -1; i< CR; i+=R) if(i%ioprocs == iome)num_part_block++;

           if(!part_chunk1) part_chunk1=DRA[handle].chunk[0];
           offelem = ((CR/ioprocs - num_part_block)*DRA[handle].chunk[0] +
                     num_part_block * part_chunk1 ) * DRA[handle].chunk[1];

           /* add offset within block */
           offelem += ((ds_a.lo[1]-1) %DRA[handle].chunk[1])*cur_ld; 
        } else {

           offelem = row_blocks  * DRA[handle].dims[1] * DRA[handle].chunk[0];
           offelem += (ds_a.lo[1]-1)*cur_ld;

        }

        *offset = offelem* dai_sizeofM(DRA[handle].type); 
}
                  

                        
/*\ write aligned block of data from memory buffer to d_a
\*/
void dai_put(
        section_t    ds_a,
        Void         *buf,
        Integer      ld,
        io_request_t *id)
{
Integer handle = ds_a.handle + DRA_OFFSET, elem;
Off_t   offset;
Size_t  bytes;

        /* find location in a file where data should be written */
        dai_file_location(ds_a, &offset);
        
        if((ds_a.hi[0] - ds_a.lo[0] + 1) != ld) dai_error("dai_put: bad ld",ld); 

        /* since everything is aligned, write data to disk */
        elem = (ds_a.hi[0] - ds_a.lo[0] + 1) * (ds_a.hi[1] - ds_a.lo[1] + 1);
        bytes= (Size_t) elem * dai_sizeofM(DRA[handle].type);
        if( ELIO_OK != elio_awrite(DRA[handle].fd, offset, buf, bytes, id ))
                       dai_error("dai_put failed", ds_a.handle);
}



/*\ write zero at EOF
\*/
void dai_zero_eof(Integer d_a)
{
Integer handle = d_a+DRA_OFFSET, nelem;
Off_t offset;
Size_t  bytes;

        if(DRA[handle].type == MT_F_DBL) *(DoublePrecision*)_dra_buffer = 0.;
        if(DRA[handle].type == MT_F_INT) *(Integer*)_dra_buffer = 0;
        if(DRA[handle].type == MT_F_REAL) *(float*)_dra_buffer = 0;

        if(INDEPFILES(d_a)) {

          Integer   CR, i, nblocks; 
          section_t ds_a;
          /* number of processors that do io */
          Integer   ioprocs=dai_io_procs(d_a); 
          /* node id of current process (if it does io) */
          Integer   iome = dai_io_nodeid(d_a);

          /* total number of blocks in the disk resident array */
          nblocks = ((DRA[handle].dims[0]
                  + DRA[handle].chunk[0]-1)/DRA[handle].chunk[0])
                  * ((DRA[handle].dims[1]
                  + DRA[handle].chunk[1]-1)/DRA[handle].chunk[1]);
          fill_sectionM(ds_a, d_a, 0, 0, 0, 0); 

          /* search for the last block for each I/O processor */
          for(i = 0; i <ioprocs; i++){
             CR = nblocks - 1 -i;
             if(CR % ioprocs == iome) break;
          }
          if(CR<0) return; /* no blocks owned */

          block_to_sectM(&ds_a, CR); /* convert block number to section */
          dai_file_location(ds_a, &offset);
          nelem = (ds_a.hi[0] - ds_a.lo[0] +1)*(ds_a.hi[1] - ds_a.lo[1] +1) -1; 
          offset += ((Off_t)nelem) * dai_sizeofM(DRA[handle].type);

#         ifdef DEBUG
            printf("me=%d zeroing EOF (%d) at %ld bytes \n",iome,CR,offset);
#         endif
        } else {

          nelem = DRA[handle].dims[0]*DRA[handle].dims[1] - 1;
          offset = ((Off_t)nelem) * dai_sizeofM(DRA[handle].type);
        }

        bytes = dai_sizeofM(DRA[handle].type);
        if(bytes != elio_write(DRA[handle].fd, offset, _dra_buffer, bytes))
                     dai_error("dai_zero_eof: write error ",0);
}



/*\ read aligned block of data from d_a to memory buffer
\*/
void dai_get(section_t ds_a, Void *buf, Integer ld, io_request_t *id)
{
Integer handle = ds_a.handle + DRA_OFFSET, elem, rc;
Off_t   offset;
Size_t  bytes;
void    dai_clear_buffer();

        /* find location in a file where data should be read from */
        dai_file_location(ds_a, &offset);

#       ifdef CLEAR_BUF
          dai_clear_buffer();
#       endif

        if((ds_a.hi[0] - ds_a.lo[0] + 1) != ld) dai_error("dai_get: bad ld",ld); 
        /* since everything is aligned, read data from disk */
        elem = (ds_a.hi[0] - ds_a.lo[0] + 1) * (ds_a.hi[1] - ds_a.lo[1] + 1);
        bytes= (Size_t) elem * dai_sizeofM(DRA[handle].type);
        rc= elio_aread(DRA[handle].fd, offset, buf, bytes, id );
        if(rc !=  ELIO_OK) dai_error("dai_get failed", rc);
}



void dai_assign_request_handle(Integer* request)
{
 int      i;
 
        *request = -1;
        for(i=0;i<MAX_REQ;i++)if(Requests[i].num_pending==0){
                *request = i;
                break;
        }
           
        if(*request ==-1) 
            dai_error("DRA: number of pending I/O requests exceeded",MAX_REQ);
           
        Requests[*request].na=0;
        Requests[*request].nu=0;
        Requests[*request].id = ELIO_DONE;
        Requests[*request].callback = OFF;
        Requests[i].num_pending = ON;
}



/*\ CREATE A DISK ARRAY
\*/
Integer dra_create(
        Integer *type,                     /*input*/
        Integer *dim1,                     /*input*/
        Integer *dim2,                     /*input*/
        char    *name,                     /*input*/
        char    *filename,                 /*input*/
        Integer *mode,                     /*input*/
        Integer *reqdim1,                  /*input: dim1 of typical request*/
        Integer *reqdim2,                  /*input: dim2 of typical request*/
        Integer *d_a)                      /*output:DRA handle*/
{
Integer handle, elem_size;

        ga_sync_();

        /* if we have an error here, it is fatal */       
        dai_check_typeM(*type);    
        if( *dim1 <= 0 )
              dai_error("dra_create: disk array dimension1 invalid ",  *dim1);
        else if( *dim2 <= 0)
              dai_error("dra_create: disk array dimension2 invalid ",  *dim2);
        if(strlen(filename)>DRA_MAX_FNAME)
              dai_error("dra_create: filename too long", DRA_MAX_FNAME);

       /*** Get next free DRA handle ***/
       if( (handle = dai_get_handle()) == -1)
           dai_error("dai_create: too many disk arrays ", _max_disk_array);
       *d_a = handle - DRA_OFFSET;

       /* determine disk array decomposition */ 
        elem_size = dai_sizeofM(*type);
        dai_chunking( elem_size, *reqdim1, *reqdim2, *dim1, *dim2, 
                    &DRA[handle].chunk[0], &DRA[handle].chunk[1]);

       /* determine layout -- by row or column */
        DRA[handle].layout = COLUMN;

       /* complete initialization */
        DRA[handle].dims[0] = *dim1;
        DRA[handle].dims[1] = *dim2;
        DRA[handle].ndim = 2;
        DRA[handle].type = ga_type_f2c((int)*type);
        DRA[handle].mode = (int)*mode;
        strncpy (DRA[handle].fname, filename,  DRA_MAX_FNAME);
        strncpy(DRA[handle].name, name, DRA_MAX_NAME );

        dai_write_param(DRA[handle].fname, *d_a);      /* create param file */
        DRA[handle].indep = dai_file_config(filename); /*check file configuration*/

        /* create file */
        if(dai_io_manage(*d_a)){ 

           if (INDEPFILES(*d_a)) {

             sprintf(dummy_fname,"%s.%ld",DRA[handle].fname,(long)dai_io_nodeid(*d_a));
             DRA[handle].fd = elio_open(dummy_fname,(int)*mode, ELIO_PRIVATE);

           }else{

              /* collective open supported only on Paragon */
#             ifdef PARAGON
                 DRA[handle].fd = elio_gopen(DRA[handle].fname,(int)*mode); 
#             else
                 DRA[handle].fd = elio_open(DRA[handle].fname,(int)*mode, ELIO_SHARED); 
#             endif
           }

           if(DRA[handle].fd==NULL)dai_error("dra_create:failed to open file",0);
           if(DRA[handle].fd->fd==-1)dai_error("dra_create:failed to open file",0);
        }

        /*
         *  Need to zero the last element in the array on the disk so that
         *  we never read beyond EOF.
         *
         *  For multiple component files will stamp every one of them.
         *
         */
        ga_sync_();

        if(dai_file_master(*d_a) && dai_write_allowed(*d_a)) dai_zero_eof(*d_a);
/*        if(dai_io_nodeid(*d_a)==0)printf("chunking: %d x %d\n",DRA[handle].chunk1,
                                                          DRA[handle].chunk2);
*/

        ga_sync_();

        return(ELIO_OK);
}
     
 

/*\ OPEN AN ARRAY THAT EXISTS ON THE DISK
\*/
Integer dra_open(
        char *filename,                  /* input  */
        Integer *mode,                   /*input*/
        Integer *d_a)                    /* output */
{
Integer handle;

        ga_sync_();

       /*** Get next free DRA handle ***/
        if( (handle = dai_get_handle()) == -1)
             dai_error("dra_open: too many disk arrays ", _max_disk_array);
        *d_a = handle - DRA_OFFSET;

        DRA[handle].mode = (int)*mode;
        strncpy (DRA[handle].fname, filename,  DRA_MAX_FNAME);

        if(dai_read_param(DRA[handle].fname, *d_a))return((Integer)-1);

        DRA[handle].indep = dai_file_config(filename); /*check file configuration*/

        if(dai_io_manage(*d_a)){ 

           if (INDEPFILES(*d_a)) {

             sprintf(dummy_fname,"%s.%ld",DRA[handle].fname,(long)dai_io_nodeid(*d_a));
             DRA[handle].fd = elio_open(dummy_fname,(int)*mode, ELIO_PRIVATE);

           }else{

              /* collective open supported only on Paragon */
#             ifdef PARAGON
                 DRA[handle].fd = elio_gopen(DRA[handle].fname,(int)*mode);
#             else
                 DRA[handle].fd = elio_open(DRA[handle].fname,(int)*mode, ELIO_SHARED);
#             endif
           }

           if(DRA[handle].fd ==NULL)dai_error("dra_open failed",ga_nodeid_());
           if(DRA[handle].fd->fd ==-1)dai_error("dra_open failed",ga_nodeid_());  
        }


#       ifdef DEBUG
             printf("\n%d:OPEN chunking=(%d,%d) type=%d buf=%d\n",
                   ga_nodeid_(),DRA[handle].chunk[0], DRA[handle].chunk[1], 
                   DRA[handle].type, DRA_DBL_BUF_SIZE);
             fflush(stdout);
#       endif

        ga_sync_();

        return(ELIO_OK);
}



/*\ CLOSE AN ARRAY AND SAVE IT ON THE DISK
\*/
Integer FATR dra_close_(Integer* d_a) /* input:DRA handle*/ 
{
Integer handle = *d_a+DRA_OFFSET;
int rc;

        ga_sync_();

        dai_check_handleM(*d_a, "dra_close");
        if(dai_io_manage(*d_a)) if(ELIO_OK != (rc=elio_close(DRA[handle].fd)))
                            dai_error("dra_close: close failed",rc);
        dai_release_handle(d_a); 

        ga_sync_();

        return(ELIO_OK);
}



/*\ decompose [ilo:ihi, jlo:jhi] into aligned and unaligned DRA subsections
\*/ 
void dai_decomp_section(
        section_t ds_a,
        Integer aligned[][2*MAXDIM], 
        int *na,
        Integer cover[][2*MAXDIM],
        Integer unaligned[][2*MAXDIM], 
        int *nu) 
{
Integer a=0, u=0, handle = ds_a.handle+DRA_OFFSET, off, chunk_units, algn_flag;
              
       /* 
        * section [ilo:ihi, jlo:jhi] is decomposed into a number of
        * 'aligned' and 'unaligned' (on chunk1/chunk2 boundary) subsections
        * depending on the layout of the 2D array on the disk;
        *
        * 'cover' subsections correspond to 'unaligned' subsections and
        * extend them to aligned on chunk1/chunk2 boundaries;
        *
        * disk I/O will be actually performed on 'aligned' and
        * 'cover' instead of 'unaligned' subsections
        */
        
        aligned[a][ ILO ] = ds_a.lo[0]; aligned[a][ IHI ] = ds_a.hi[0];
        aligned[a][ JLO ] = ds_a.lo[1]; aligned[a][ JHI ] = ds_a.hi[1];

        switch   (DRA[handle].layout){
        case COLUMN : /* need to check row alignment only */
                 
                 algn_flag = ON; /* has at least one aligned subsection */

                 /* top of section */
                 off = (ds_a.lo[0] -1) % DRA[handle].chunk[0]; 
                 if(off){ 

                        if(MAX_UNLG<= u) 
                           dai_error("dai_decomp_sect:insufficient nu",u);

                        chunk_units = (ds_a.lo[0] -1) / DRA[handle].chunk[0];
                        
                        cover[u][ ILO ] = chunk_units*DRA[handle].chunk[0] + 1;
                        cover[u][ IHI ] = MIN(cover[u][ ILO ] + DRA[handle].chunk[0]-1,
                                          DRA[handle].dims[0]);

                        unaligned[u][ ILO ] = ds_a.lo[0];
                        unaligned[u][ IHI ] = MIN(ds_a.hi[0],cover[u][ IHI ]);
                        unaligned[u][ JLO ] = cover[u][ JLO ] = ds_a.lo[1];
                        unaligned[u][ JHI ] = cover[u][ JHI ] = ds_a.hi[1];

                        if(cover[u][ IHI ] < ds_a.hi[0]){
                           /* cover subsection ends above ihi */
                           if(MAX_ALGN<=a)
                              dai_error("dai_decomp_sect: na too small",a);
                           aligned[a][ ILO ] = cover[u][ IHI ]+1; 
                        }else{
                           /* cover subsection includes ihi */
                           algn_flag = OFF;
                        }
                        u++;
                 }

                 /* bottom of section */
                 off = ds_a.hi[0] % DRA[handle].chunk[0]; 
                 if(off && (ds_a.hi[0] != DRA[handle].dims[0]) && (algn_flag == ON)){

                        if(MAX_UNLG<=u) 
                           dai_error("dai_decomp_sect:insufficient nu",u); 
                        chunk_units = ds_a.hi[0] / DRA[handle].chunk[0];
 
                        cover[u][ ILO ] = chunk_units*DRA[handle].chunk[0] + 1;
                        cover[u][ IHI ] = MIN(cover[u][ ILO ] + DRA[handle].chunk[0]-1,
                                          DRA[handle].dims[0]);

                        unaligned[u][ ILO ] = cover[u][ ILO ];
                        unaligned[u][ IHI ] = ds_a.hi[0];
                        unaligned[u][ JLO ] = cover[u][ JLO ] = ds_a.lo[1];
                        unaligned[u][ JHI ] = cover[u][ JHI ] = ds_a.hi[1];

                        aligned[a][ IHI ] = MAX(1,unaligned[u][ ILO ]-1);
                        algn_flag=(DRA[handle].chunk[0] == DRA[handle].dims[0])?OFF:ON;
                        u++;
                 }
                 *nu = (int)u;
                 if(aligned[0][ IHI ]-aligned[0][ ILO ] < 0) algn_flag= OFF;
                 *na = (algn_flag== OFF)? 0: 1;
                 break;

      case ROW : /* we need to check column alignment only */

        default: dai_error("dai_decomp_sect: ROW layout not yet implemented",
                           DRA[handle].layout);
        }
}
       

/*\ WRITE g_a TO d_a
\*/
Integer FATR dra_write_(
        Integer *g_a,                      /*input:GA handle*/
        Integer *d_a,                      /*input:DRA handle*/
        Integer *request)                  /*output: handle to async oper. */
{
Integer gdim1, gdim2, gtype, handle=*d_a+DRA_OFFSET;
logical transp = FALSE;
Integer ilo, ihi, jlo, jhi;

        ga_sync_();

        /* usual argument/type/range checking stuff */

        dai_check_handleM(*d_a,"dra_write");
        if( !dai_write_allowed(*d_a))
             dai_error("dra_write: write not allowed to this array",*d_a);

        ga_inquire_internal_(g_a, &gtype, &gdim1, &gdim2);
        if(DRA[handle].type != (int)gtype)dai_error("dra_write: type mismatch",gtype);
        if(DRA[handle].dims[0] != gdim1)dai_error("dra_write: dim1 mismatch",gdim1);
        if(DRA[handle].dims[1] != gdim2)dai_error("dra_write: dim2 mismatch",gdim2);

        /* right now, naive implementation just calls dra_write_section */
        ilo = 1; ihi = DRA[handle].dims[0]; 
        jlo = 1; jhi = DRA[handle].dims[1]; 
        return(dra_write_section_(&transp, g_a, &ilo, &ihi, &jlo, &jhi,
                                          d_a, &ilo, &ihi, &jlo, &jhi,request));
}


/*     given current (i,j) compute (ni, nj) - next loop index
 *     o - outermost loop, i- innermost loop
 *     iinc increment for i
 *     oinc increment for o
 */
int dai_next2d(Integer* i, Integer imin, Integer imax, Integer iinc, 
               Integer* o, Integer omin, Integer omax, Integer oinc)
{
    int retval;
    if (*o == 0  || *i == 0) {
       /* to handle initial out-of range indices */
        *o = omin;
        *i = imin;
    } else {
        *i = *i + iinc;
    }
    if (*i > imax) {
        *i = imin;
        *o += oinc;
    }
    retval = (*o <= omax);
    return retval;
}


/*\ compute next chunk of array to process
\*/
int dai_next_chunk(Integer req, Integer* list, section_t* ds_chunk)
{
Integer   handle = ds_chunk->handle+DRA_OFFSET;
int       retval;

    if(INDEPFILES(ds_chunk->handle))
      if(ds_chunk->lo[1] && DRA[handle].chunk[1]>1) 
         ds_chunk->lo[1] -= (ds_chunk->lo[1] -1) % DRA[handle].chunk[1];
    
    retval = dai_next2d(&ds_chunk->lo[0], list[ ILO ], list[ IHI ],
                                        DRA[handle].chunk[0],
                        &ds_chunk->lo[1], list[ JLO ], list[ JHI ],
                                        DRA[handle].chunk[1]);
    if(!retval) return(retval);

    ds_chunk->hi[0] = MIN(list[ IHI ], ds_chunk->lo[0] + DRA[handle].chunk[0] -1);
    ds_chunk->hi[1] = MIN(list[ JHI ], ds_chunk->lo[1] + DRA[handle].chunk[1] -1);

    if(INDEPFILES(ds_chunk->handle)) { 
         Integer jhi_temp =  ds_chunk->lo[1] + DRA[handle].chunk[1] -1;
         jhi_temp -= jhi_temp % DRA[handle].chunk[1];
         ds_chunk->hi[1] = MIN(ds_chunk->hi[1], jhi_temp); 

         /*this line was absent from older version on bonnie that worked */
         if(ds_chunk->lo[1] < list[ JLO ]) ds_chunk->lo[1] = list[ JLO ]; 
    }

    return 1;
}

#define nsect_to_blockM(ds_a, CR) \
{ \
  Integer hndl = (ds_a).handle+DRA_OFFSET;\
  Integer _i, _ndim = DRA[hndl].ndim; \
  Integer _R, _b; \
  *(CR) = 0; \
  _R = 0; \
  for (_i=_ndim-1; _i >= 0; _i--) { \
    _b = ((ds_a).lo[_i]-1)/DRA[hndl].chunk[_i]; \
    _R = (DRA[hndl].dims[_i]+DRA[hndl].chunk[_i]-1)/DRA[hndl].chunk[_i];\
    *(CR) = *(CR) * _R + _b; \
  } \
}

int dai_myturn(section_t ds_chunk)
{
Integer   ioprocs = dai_io_procs(ds_chunk.handle); 
Integer   iome    = dai_io_nodeid(ds_chunk.handle);
    
    if(INDEPFILES(ds_chunk.handle)){

      /* compute cardinal number for the current chunk */
      nsect_to_blockM(ds_chunk, &_dra_turn);

    }else{
      _dra_turn++;
    }

    return ((_dra_turn%ioprocs) == iome);
}


#define LOAD 1
#define STORE 2
#define TRANS 1
#define NOTRANS 0

/******* print routine for debugging purposes only (double) */
void dai_print_buf(buf, ld, rows, cols)
Integer ld, rows,cols;
double *buf;  /*<<<<<*/
{
   int i,j;
   printf("\n ld=%ld rows=%ld cols=%ld\n",ld,rows,cols);
 
   for (i=0; i<rows; i++){
   for (j=0; j<cols; j++)
   printf("%f ", buf[j*ld+i]);
   printf("\n");
   }
}

void dra_set_mode_(Integer* val)
{
}


#define dai_dest_indices_1d_M(index, id, jd, ilod, jlod, ldd) \
{ \
    Integer _index_;\
    _index_ = (is)-(ilos);\
    *(id) = (_index_)%(ldd) + (ilod);\
    *(jd) = (_index_)/(ldd) + (jlod);\
}
#define dai_dest_indicesM(is, js, ilos, jlos, lds, id, jd, ilod, jlod, ldd)\
{ \
    Integer _index_;\
    _index_ = (lds)*((js)-(jlos)) + (is)-(ilos);\
    *(id) = (_index_)%(ldd) + (ilod);\
    *(jd) = (_index_)/(ldd) + (jlod);\
}


void ga_move_1d(int op, section_t gs_a, section_t ds_a,
                section_t ds_chunk, void* buffer, Integer ldb)
{
     Integer index, ldd = gs_a.hi[0] - gs_a.lo[0] + 1, one=1;
     Integer atype, cols, rows, elemsize, ilo, ihi, jlo, jhi;
     Integer istart, iend, jstart, jend;
     void  (FATR *f)(Integer*,Integer*,Integer*,Integer*,Integer*,void*,Integer*); 
     char *buf = (char*)buffer;

     if(op==LOAD) f = ga_get_;
     else f = ga_put_;

     ga_inquire_(&gs_a.handle, &atype, &rows, &cols);     
     elemsize = MA_sizeof(atype, 1, MT_C_CHAR);

     /* find where in global array the first dra chunk element in buffer goes*/
     index = ds_chunk.lo[0] - ds_a.lo[0];
     istart = index%ldd + gs_a.lo[0]; 
     jstart = index/ldd + gs_a.lo[1];
     
     /* find where in global array the last dra chunk element in buffer goes*/
     index = ds_chunk.hi[0] - ds_a.lo[0];
     iend = index%ldd + gs_a.lo[0]; 
     jend = index/ldd + gs_a.lo[1];
     
     /* we have up to 3 rectangle chunks corresponding to gs_chunk 
       .|' incomplete first column, full complete middle column, and
           incomplete last column */
     if(istart != gs_a.lo[0] || jstart==jend ){
        ilo = istart; 
        ihi = gs_a.hi[0]; 
        if(jstart==jend) ihi=iend;
        f(&gs_a.handle, &ilo, &ihi, &jstart, &jstart, buf, &one); 
        buf += elemsize*(ihi -ilo+1);
        if(jstart==jend) return;
        jstart++;
     }

     if(iend != gs_a.hi[0]) jend--;

     if(jstart <= jend) { 
        f(&gs_a.handle, &gs_a.lo[0], &gs_a.hi[0], &jstart, &jend, buf, &ldd);
        buf += elemsize*ldd*(jend-jstart+1); 
     } 

     if(iend != gs_a.hi[0]){
        jend++; /* Since decremented above */  
        f(&gs_a.handle, &gs_a.lo[0], &iend, &jend, &jend, buf, &one);
     }
}


void ga_move(int op, int trans, section_t gs_a, section_t ds_a, 
             section_t ds_chunk, void* buffer, Integer ldb)
{
    if(!trans && (gs_a.lo[0]- gs_a.hi[0] ==  ds_a.lo[0]- ds_a.hi[0]) ){
        /*** straight copy possible if there's no reshaping or transpose ***/

        /* determine gs_chunk corresponding to ds_chunk */
        section_t gs_chunk = gs_a;
        dai_dest_indicesM(ds_chunk.lo[0], ds_chunk.lo[1], ds_a.lo[0], ds_a.lo[1], 
                ds_a.hi[0]-ds_a.lo[0]+1, &gs_chunk.lo[0], &gs_chunk.lo[1], 
                gs_a.lo[0], gs_a.lo[1],   gs_a.hi[0] - gs_a.lo[0] + 1);
        dai_dest_indicesM(ds_chunk.hi[0], ds_chunk.hi[1], ds_a.lo[0], ds_a.lo[1], 
                ds_a.hi[0]-ds_a.lo[0]+1, &gs_chunk.hi[0], &gs_chunk.hi[1],
                gs_a.lo[0], gs_a.lo[1],  gs_a.hi[0] - gs_a.lo[0] + 1);

        /* move data */
        if(op==LOAD) ga_get_sectM(gs_chunk, buffer, ldb);
        else         ga_put_sectM(gs_chunk, buffer, ldb);
    
#ifdef MOVE1D_ENABLED
    }else if(!trans && (ds_a.lo[1]==ds_a.hi[1]) ){

        /* for a 1-dim section (column) some optimization possible */
        ga_move_1d(op, gs_a, ds_a, ds_chunk, buffer, ldb);        
#endif
    }else{
        /** due to generality of this transformation scatter/gather is required **/

         Integer ihandle, jhandle, vhandle, iindex, jindex, vindex;
         Integer pindex, phandle;
         int type = DRA[ds_a.handle+DRA_OFFSET].type;
         Integer i, j, ii, jj, base,nelem;  
         char    *base_addr;

#        define ITERATOR_2D(i,j, base, ds_chunk)\
                for(j = ds_chunk.lo[1], base=0, jj=0; j<= ds_chunk.hi[1]; j++,jj++)\
                  for(i = ds_chunk.lo[0], ii=0; i<= ds_chunk.hi[0]; i++,ii++,base++)

#        define COPY_SCATTER(ADDR_BASE, TYPE, ds_chunk)\
		ITERATOR_2D(i,j, base, ds_chunk) \
		ADDR_BASE[base+vindex] = ((TYPE*)buffer)[ldb*jj + ii]

#        define COPY_GATHER(ADDR_BASE, TYPE, ds_chunk)\
                for(i=0; i< nelem; i++){\
                   Integer ldc = ds_chunk.hi[0] - ds_chunk.lo[0]+1;\
                   base = INT_MB[pindex+i]; jj = base/ldc; ii = base%ldc;\
                   ((TYPE*)buffer)[ldb*jj + ii] = ADDR_BASE[i+vindex];\
                }

#        define COPY_TYPE(OPERATION, MATYPE, ds_chunk)\
         switch(MATYPE){\
         case MT_F_DBL:  COPY_ ## OPERATION(DBL_MB,DoublePrecision,ds_chunk);break;\
         case MT_F_INT:  COPY_ ## OPERATION(INT_MB, Integer, ds_chunk); break;\
         case MT_F_DCPL: COPY_ ## OPERATION(DCPL_MB, DoubleComplex, ds_chunk);break;\
         case MT_F_REAL: COPY_ ## OPERATION(FLT_MB, float, ds_chunk);\
         }

         if(ga_nodeid_()==0) printf("DRA warning: using scatter/gather\n");

         nelem = (ds_chunk.hi[0]-ds_chunk.lo[0]+1)
               * (ds_chunk.hi[1]-ds_chunk.lo[1]+1);
         if(!MA_push_get(MT_F_INT, nelem, "i_", &ihandle, &iindex))
                         dai_error("DRA move: MA failed-i ", 0L);
         if(!MA_push_get(MT_F_INT, nelem, "j_", &jhandle, &jindex))
                         dai_error("DRA move: MA failed-j ", 0L);
         if(!MA_push_get(type, nelem, "v_", &vhandle, &vindex))
                         dai_error("DRA move: MA failed-v ", 0L);

         /* set the address of base for each datatype */
         switch(type){
              case  MT_F_DBL:  base_addr = (char*) (DBL_MB+vindex); break;
              case  MT_F_INT:  base_addr = (char*) (INT_MB+vindex); break;
              case  MT_F_DCPL: base_addr = (char*) (DCPL_MB+vindex);break;
              case  MT_F_REAL: base_addr = (char*) (FLT_MB+vindex);
         }
    
         if(trans==TRANS) 
           ITERATOR_2D(i,j, base, ds_chunk) {
              dai_dest_indicesM(j, i, ds_a.lo[0], ds_a.lo[1], ds_a.hi[0]-ds_a.lo[0]+1, 
                                INT_MB+base+iindex, INT_MB+base+jindex,
                                gs_a.lo[0], gs_a.lo[1],  gs_a.hi[0] - gs_a.lo[0] + 1);
           }
         else
           ITERATOR_2D(i,j, base, ds_chunk) {
              dai_dest_indicesM(i, j, ds_a.lo[0], ds_a.lo[1], ds_a.hi[0]-ds_a.lo[0]+1, 
                                INT_MB+base+iindex, INT_MB+base+jindex,
                                gs_a.lo[0], gs_a.lo[1],  gs_a.hi[0] - gs_a.lo[0] + 1);
           }

        /* move data */
         if(op==LOAD){

           if(!MA_push_get(MT_F_INT, nelem, "pindex", &phandle, &pindex))
                         dai_error("DRA move: MA failed-p ", 0L);
           for(i=0; i< nelem; i++) INT_MB[pindex+i] = i; 
           ga_sort_permut_(&gs_a.handle, INT_MB+pindex, INT_MB+iindex, INT_MB+jindex, &nelem);
           ga_gather_(&gs_a.handle, base_addr, INT_MB+iindex, INT_MB+jindex, &nelem);
           COPY_TYPE(GATHER, type, ds_chunk);
           MA_pop_stack(phandle);

         }else{ 

           COPY_TYPE(SCATTER, type, ds_chunk);
           ga_scatter_(&gs_a.handle, base_addr, INT_MB+iindex, INT_MB+jindex, &nelem);
         }

         MA_pop_stack(vhandle);
         MA_pop_stack(jhandle);
         MA_pop_stack(ihandle);
    }
}

#define nga_get_sectM(sect, _buf, _ld)\
   nga_get_(&sect.handle, sect.lo, sect.hi, _buf, _ld)

#define nga_put_sectM(sect, _buf, _ld)\
   nga_put_(&sect.handle, sect.lo, sect.hi, _buf, _ld)

#define ndai_dest_indicesM(ds_chunk, ds_a, gs_chunk, gs_a)   \
{\
  Integer _i; \
  Integer _ndim = ds_a.ndim; \
  for (_i=0; _i<_ndim; _i++) { \
    gs_chunk.lo[_i] = gs_a.lo[_i] + ds_chunk.lo[_i]- ds_a.lo[_i]; \
    gs_chunk.hi[_i] = gs_a.lo[_i] + ds_chunk.hi[_i]- ds_a.lo[_i]; \
  } \
}

#define ndai_trnsp_dest_indicesM(ds_chunk, ds_a, gs_chunk, gs_a)   \
{\
  Integer _i; \
  Integer _ndim = ds_a.ndim; \
  for (_i=0; _i<_ndim; _i++) { \
    gs_chunk.lo[_ndim-1-_i] = gs_a.lo[_ndim-1-_i] \
                            + ds_chunk.lo[_i]- ds_a.lo[_i]; \
    gs_chunk.hi[_ndim-1-_i] = gs_a.lo[_ndim-1-_i] \
                            + ds_chunk.hi[_i]- ds_a.lo[_i]; \
  } \
}

/*#define ndai_dest_indicesM(ds_chunk, ds_a, gs_chunk, gs_a)   \
{ \
  Integer _lds[MAXDIM], _ldd[MAXDIM], _i;\
  Integer _ndim = ds_a.ndim;\
  Integer _index_;\
  for (_i=0; _i<_ndim; _i++) {\
    _lds[_i] = ds_a.hi[_i]-ds_a.lo[_i] + 1;\
    _ldd[_i] = gs_chunk.hi[_i]-gs_chunk.lo[_i] + 1;\
  }\
  _index_ = ds_chunk.lo[_ndim-1] - ds_a.lo[_ndim-1];\
  for (_i=_ndim-2; _i>=0; _i--) {\
    _index_ = _lds[_i]*_index_;\
    _index_ += ds_chunk.lo[_i] - ds_a.lo[_i];\
  }\
  for (_i=0; _i<_ndim; _i++) {\
    if (_i < _ndim-1) {\
      gs_chunk.lo[_i] = _index_%_ldd[_i];\
      _index_ = (_index_ - gs_chunk.lo[_i])/_ldd[_i];\
      gs_chunk.lo[_i] += gs_a.lo[_i];\
    } else {\
      gs_chunk.lo[_i] = _index_ + gs_a.lo[_i];\
    }\
  }\
  _index_ = ds_chunk.hi[_ndim-1] - ds_a.lo[_ndim-1];\
  for (_i=_ndim-2; _i>=0; _i--) {\
    _index_ = _lds[_i]*_index_;\
    _index_ += ds_chunk.hi[_i] - ds_a.lo[_i];\
  }\
  for (_i=0; _i<_ndim; _i++) {\
    if (_i < _ndim-1) {\
      gs_chunk.hi[_i] = _index_%_ldd[_i];\
      _index_ = (_index_ - gs_chunk.hi[_i])/_ldd[_i];\
      gs_chunk.hi[_i] += gs_a.lo[_i];\
    } else {\
      gs_chunk.hi[_i] = _index_ + gs_a.lo[_i];\
    }\
  }\
}*/

void nga_move(int op,             /*[input] flag for read or write */
              int trans,          /*[input] flag for transpose */
              section_t gs_a,     /*[input] complete section of global array */
              section_t ds_a,     /*[input] complete section of DRA */
              section_t ds_chunk, /*[input] actual DRA chunk */
              void* buffer,       /*[input] pointer to io buffer containing
                                            DRA cover section */
              Integer ldb[])
{
  Integer ndim = gs_a.ndim, i;
  logical consistent = TRUE;
  if (!trans) {
    for (i=0; i<ndim-1; i++) 
      if (gs_a.lo[i]-gs_a.hi[i] != ds_a.lo[i]-ds_a.hi[i]) consistent = FALSE;
  } else {
    for (i=0; i<ndim-1; i++) 
      if (gs_a.lo[ndim-1-i]-gs_a.hi[ndim-1-i]
        != ds_a.lo[i]-ds_a.hi[i]) consistent = FALSE;
  }
  if (!trans && consistent){

    /*** straight copy possible if there's no reshaping or transpose ***/

    /* determine gs_chunk corresponding to ds_chunk */
    section_t gs_chunk = gs_a;
        /*dai_dest_indicesM(ds_chunk.lo[0], ds_chunk.lo[1], ds_a.lo[0], ds_a.lo[1], 
                ds_a.hi[0]-ds_a.lo[0]+1, &gs_chunk.lo[0], &gs_chunk.lo[1], 
                gs_a.lo[0], gs_a.lo[1],   gs_a.hi[0] - gs_a.lo[0] + 1);
        dai_dest_indicesM(ds_chunk.hi[0], ds_chunk.hi[1], ds_a.lo[0], ds_a.lo[1], 
                ds_a.hi[0]-ds_a.lo[0]+1, &gs_chunk.hi[0], &gs_chunk.hi[1],
                gs_a.lo[0], gs_a.lo[1],  gs_a.hi[0] - gs_a.lo[0] + 1);*/
    ndai_dest_indicesM(ds_chunk, ds_a, gs_chunk, gs_a);
    consistent = TRUE;
    for (i=0; i<ndim; i++) {
      if (gs_chunk.hi[i]<gs_chunk.lo[i] || gs_chunk.lo[i]<0) {
        consistent = FALSE;
      }
    }
    if (!consistent) {
      for(i=0; i<ndim; i++) {
        printf("gs_chunk[%d] %5d:%5d  ds_chunk[%d] %5d:%5d",
                (int)i,(int)gs_chunk.lo[i],(int)gs_chunk.hi[i],
                (int)i,(int)ds_chunk.lo[i],(int)ds_chunk.hi[i]);
        printf(" gs_a[%d] %5d:%5d  ds_a[%d] %5d:%5d\n",
                (int)i,(int)gs_a.lo[i],(int)gs_a.hi[i],
                (int)i,(int)ds_a.lo[i],(int)ds_a.hi[i]);
      }
    }
/*    printf("(nga_move) gs_chunk.ndim = %d\n",gs_chunk.ndim);
    for (i=0; i<ndim; i++) {
      printf("(nga_move) gs_chunk.lo[%d] = %d\n",i,gs_chunk.lo[i]);
      printf("(nga_move) gs_chunk.hi[%d] = %d\n",i,gs_chunk.hi[i]);
    }
    fflush(stdout); */
    /* move data */
    if (op==LOAD) {
      nga_get_sectM(gs_chunk, buffer, ldb);
    } else {
      nga_put_sectM(gs_chunk, buffer, ldb);
    }
    
  } else if (trans && consistent) {
    /* Only transpose is supported, not reshaping, so scatter/gather is not
       required */
    Integer vhandle, vindex, index[MAXDIM];
    Integer i, j, itmp, jtmp, nelem, ldt[MAXDIM], ldg[MAXDIM];
    Integer nelem1, nelem2, nelem3;
    int type = DRA[ds_a.handle+DRA_OFFSET].type;
    char    *base_addr;
    section_t gs_chunk = gs_a;

    /* create space to copy transpose of DRA section into */
    nelem = 1;
    for (i=0; i<ndim; i++) nelem *= (ds_chunk.hi[i] - ds_chunk.lo[i] + 1);
    nelem1 = 1;
    for (i=1; i<ndim; i++) nelem1 *= (ds_chunk.hi[i] - ds_chunk.lo[i] + 1);
    nelem2 = 1;
    for (i=0; i<ndim-1; i++) nelem2 *= ldb[i];
    if(!MA_push_get(type, nelem, "v_", &vhandle, &vindex))
          dai_error("DRA move: MA failed-v ", 0L);
    if(!MA_get_pointer(vhandle, &base_addr))
          dai_error("DRA move: MA get_pointer failed ", 0L);

    /* copy and transpose relevant numbers from IO buffer to temporary array */
    for (i=1; i<ndim; i++) ldt[ndim-1-i] = ds_chunk.hi[i] - ds_chunk.lo[i] + 1;
    if (op == LOAD) {
      /* transpose buffer with data from global array */
      ndai_trnsp_dest_indicesM(ds_chunk, ds_a, gs_chunk, gs_a);
      for (i=0; i<ndim; i++) ldg[i] = gs_chunk.hi[i] - gs_chunk.lo[i] + 1;
      /* copy data from global array to temporary buffer */
      nga_get_sectM(gs_chunk, base_addr, ldg); 
      for (i=0; i<nelem1; i++ ) {
        /* find indices of elements in MA buffer */
        if (ndim > 1) {
          itmp = i;
          index[1] = itmp%ldg[1];
          for (j=2; j<ndim; j++) {
            itmp = (itmp-index[j-1])/ldg[j-1];
            if (j != ndim-1) {
              index[j] = itmp%ldg[j];
            } else {
              index[j] = itmp;
            }
          }
          nelem3 = index[1];
          for (j=2; j<ndim; j++) {
            nelem3 *= ldb[ndim-1-j];
            nelem3 += index[j];
          }
        } else {
          nelem2 = 1;
          nelem3 = 0;
        }
        /* find corresponding indices of element from IO buffer */
        itmp = ldg[0]*i;
        jtmp = nelem3;
        for (j=0; j<ldg[0]; j++) {
          switch(ga_type_c2f(type)){
            case MT_F_DBL:
              ((DoublePrecision*)buffer)[jtmp]
                = ((DoublePrecision*)base_addr)[itmp];
              break;
            case MT_F_INT:
              ((Integer*)buffer)[jtmp]
                = ((Integer*)base_addr)[itmp];
              break;
            case MT_F_DCPL:
              ((DoublePrecision*)buffer)[2*jtmp]
                = ((DoublePrecision*)base_addr)[2*itmp];
              ((DoublePrecision*)buffer)[2*jtmp+1]
                = ((DoublePrecision*)base_addr)[2*itmp+1];
              break;
            case MT_F_REAL:
              ((float*)buffer)[jtmp]
                = ((float*)base_addr)[itmp];
              break;
          }
          itmp++;
          jtmp += nelem2;
        }
      }
    } else {
      /* get transposed indices */
      ndai_trnsp_dest_indicesM(ds_chunk, ds_a, gs_chunk, gs_a);
      for (i=0; i<ndim; i++) ldg[i] = gs_chunk.hi[i] - gs_chunk.lo[i] + 1;
      for (i=0; i<nelem1; i++ ) {
        /* find indices of elements in MA buffer */
        if (ndim > 1) {
          itmp = i;
          index[1] = itmp%ldg[1];
          for (j=2; j<ndim; j++) {
            itmp = (itmp-index[j-1])/ldg[j-1];
            if (j != ndim-1) {
              index[j] = itmp%ldg[j];
            } else {
              index[j] = itmp;
            }
          }
          nelem3 = index[1];
          for (j=2; j<ndim; j++) {
            nelem3 *= ldb[ndim-1-j];
            nelem3 += index[j];
          }
        } else {
          nelem2 = 1;
          nelem3 = 0;
        }
        /* find corresponding indices of element from IO buffer */
        itmp = ldg[0]*i;
        jtmp = nelem3;
        for (j=0; j<ldg[0]; j++) {
          switch(ga_type_c2f(type)){
            case MT_F_DBL:
              ((DoublePrecision*)base_addr)[itmp]
                = ((DoublePrecision*)buffer)[jtmp];
              break;
            case MT_F_INT:
              ((Integer*)base_addr)[itmp]
                = ((Integer*)buffer)[jtmp];
              break;
            case MT_F_DCPL:
              ((DoublePrecision*)base_addr)[2*itmp]
                = ((DoublePrecision*)buffer)[2*jtmp];
              ((DoublePrecision*)base_addr)[2*itmp+1]
                = ((DoublePrecision*)buffer)[2*jtmp+1];
              break;
            case MT_F_REAL:
              ((float*)base_addr)[itmp]
                = ((float*)buffer)[jtmp];
              break;
          }
          itmp++;
          jtmp += nelem2;
        }
      }
      nga_put_sectM(gs_chunk, base_addr, ldt); 
    }
    MA_pop_stack(vhandle);
  } else {
    dai_error("DRA move: Inconsistent dimensions found ", 0L);
  }
}


/*\  executes callback function associated with completion of asynch. I/O
\*/
void dai_exec_callback(request_t *request)
{
args_t   *arg;

        if(request->callback==OFF)return;
        request->callback = OFF;
        arg = &request->args;
        nga_move(arg->op, arg->transp, arg->gs_a, arg->ds_a, arg->ds_chunk,
               _dra_buffer, arg->ld);
}


/*\ wait until buffer space associated with request is avilable
\*/
void dai_wait(Integer req0)
{
Integer req;
 
        /* until more sophisticated buffer managment is implemented wait for
           all requests to complete */

        for(req=0; req<MAX_REQ; req++)
          if(Requests[req].num_pending)
             if(elio_wait(&Requests[req].id)==ELIO_OK)
                 dai_exec_callback(Requests + req);
             else
                 dai_error("dai_wait: DRA internal error",0);
}



/*\ Write or Read Unaligned Subsections to/from disk: 
 *  always read a aligned extension of a section from disk to local buffer then 
 *  for read :  copy requested data from buffer to global array;
 *  for write: overwrite part of buffer with data from g_a and write all to disk
 *
\*/
void dai_transfer_unlgn(int opcode, int transp, 
                        section_t ds_a, section_t gs_a, Integer req)
{
Integer   chunk_ld,  next, offset;
int   type = DRA[ds_a.handle+DRA_OFFSET].type;
section_t ds_chunk, ds_unlg;
char      *buffer; 

   ds_chunk =  ds_unlg = ds_a;

   for(next = 0; next < Requests[req].nu; next++){

      ds_chunk.lo[0] = ds_chunk.lo[1] = 0; /* init */
      while(dai_next_chunk(req, Requests[req].list_cover[next],&ds_chunk)){

          if(dai_myturn(ds_chunk)){

              dai_wait(req); /* needs free buffer to proceed */

             /*find corresponding to chunk of 'cover' unaligned sub-subsection*/
              ds_unlg.lo[0] = Requests[req].list_unlgn[next][ ILO ];
              ds_unlg.hi[0] = Requests[req].list_unlgn[next][ IHI ];
              ds_unlg.lo[1] = Requests[req].list_unlgn[next][ JLO ];
              ds_unlg.hi[1] = Requests[req].list_unlgn[next][ JHI ];

              if(!dai_section_intersect(ds_chunk, &ds_unlg))
                  dai_error("dai_transfer_unlgn: inconsistent cover",0);

             /* copy data from disk to DRA buffer */
              chunk_ld =  ds_chunk.hi[0] - ds_chunk.lo[0] + 1;
              dai_get(ds_chunk, _dra_buffer, chunk_ld, &Requests[req].id);
              elio_wait(&Requests[req].id); 

             /* determine location in the buffer where GA data should be */
              offset  = (ds_unlg.lo[1] - ds_chunk.lo[1])*chunk_ld + 
                         ds_unlg.lo[0] - ds_chunk.lo[0];
              buffer  = (char*)_dra_buffer;
              buffer += offset * dai_sizeofM(type);

              switch (opcode){
              case DRA_OP_WRITE: 
                 /* overwrite a part of buffer with data from g_a */  
                 ga_move(LOAD, transp, gs_a, ds_a, ds_unlg, buffer, chunk_ld);

                 /* write entire updated buffer back to disk */
                 dai_put(ds_chunk, _dra_buffer, chunk_ld, &Requests[req].id);
                 break;

              case DRA_OP_READ: 
                 /* copy requested data from buffer to g_a */
                 ga_move(STORE, transp, gs_a, ds_a, ds_unlg, buffer, chunk_ld);
                 break;

              default:
                 dai_error("dai_transfer_unlg: invalid opcode",(Integer)opcode);
              }

#             ifdef DEBUG
                fprintf(stderr,"%d transf unlg g[%d:%d,%d:%d]-d[%d:%d,%d:%d]\n",
                   dai_io_nodeid(), gs_chunk.lo[0], gs_chunk.hi[0],
                   gs_chunk.lo[1], gs_chunk.hi[1],
                   ds_unlg.lo[0], ds_unlg.hi[0],
                   ds_unlg.lo[1], ds_unlg.hi[1]);
#             endif
          }
      }
   }
}



/*\ write or read aligned subsections to disk 
\*/
void dai_transfer_algn(int opcode, int transp, 
                       section_t ds_a, section_t gs_a, Integer req)
{
Integer   next, chunk_ld[MAXDIM];
section_t ds_chunk = ds_a;

   for(next = 0; next < Requests[req].na; next++){

      ds_chunk.lo[0] = ds_chunk.lo[1] = 0; /* init */
      while(dai_next_chunk(req, Requests[req].list_algn[next], &ds_chunk)){

          if(dai_myturn(ds_chunk)){

              dai_wait(req); /* needs free buffer to proceed */

              chunk_ld[0] =  ds_chunk.hi[0] - ds_chunk.lo[0] + 1;

              switch (opcode){

              case DRA_OP_WRITE:
                 /* copy data from g_a to DRA buffer */
                 ga_move(LOAD, transp, gs_a, ds_a, ds_chunk, _dra_buffer,
                     chunk_ld[0]);

                 /* copy data from DRA buffer to disk */
                 dai_put(ds_chunk, _dra_buffer, chunk_ld[0], &Requests[req].id);
                 break;

              case DRA_OP_READ:
                 /* copy data from disk to DRA buffer */
                 dai_get(ds_chunk, _dra_buffer, chunk_ld[0], &Requests[req].id);
                 elio_wait(&Requests[req].id);

                 /* copy data from DRA buffer to g_a */
/*                 ga_move(STORE, transp, gs_a, ds_a, ds_chunk, _dra_buffer, chunk_ld);*/
                 dai_callback(STORE, transp, gs_a, ds_a, ds_chunk, chunk_ld, req);
                 break;

              default:

                 dai_error("dai_transfer_algn: invalid opcode",(Integer)opcode);
              }

#             ifdef DEBUG
                fprintf(stderr,"%d transf algn g[%d:%d,%d:%d]-d[%d:%d,%d:%d]\n",
                   dai_io_nodeid(), gs_chunk.lo[0], gs_chunk.hi[0],
                   gs_chunk.lo[1], gs_chunk.hi[1],
                   ds_chunk.lo[0], ds_chunk.hi[0],
                   ds_chunk.lo[1], ds_chunk.hi[1]);
#             endif
          }
      }
   }
}



/*\ WRITE SECTION g_a[gilo:gihi, gjlo:gjhi] TO d_a[dilo:dihi, djlo:djhi]
\*/
Integer FATR dra_write_section_(
        logical *transp,                   /*input:transpose operator*/
        Integer *g_a,                      /*input:GA handle*/ 
        Integer *gilo,                     /*input*/
        Integer *gihi,                     /*input*/
        Integer *gjlo,                     /*input*/
        Integer *gjhi,                     /*input*/
        Integer *d_a,                      /*input:DRA handle*/ 
        Integer *dilo,                     /*input*/
        Integer *dihi,                     /*input*/
        Integer *djlo,                     /*input*/
        Integer *djhi,                     /*input*/
        Integer *request)                  /*output: async. request id*/ 
{
Integer gdim1, gdim2, gtype, handle=*d_a+DRA_OFFSET;
section_t d_sect, g_sect;
  
   ga_sync_();

   /* usual argument/type/range checking stuff */

   dai_check_handleM(*d_a,"dra_write_sect");
   ga_inquire_internal_(g_a, &gtype, &gdim1, &gdim2);
   if(!dai_write_allowed(*d_a))dai_error("dra_write_sect: write not allowed",*d_a);
   if(DRA[handle].type != (int)gtype)dai_error("dra_write_sect: type mismatch",gtype);
   dai_check_rangeM(*gilo,*gihi, gdim1, "dra_write_sect: g_a dim1 error");
   dai_check_rangeM(*gjlo,*gjhi, gdim2, "dra_write_sect: g_a dim2 error");
   dai_check_rangeM(*dilo,*dihi,DRA[handle].dims[0],"dra_write_sect:d_a dim1 error");
   dai_check_rangeM(*djlo,*djhi,DRA[handle].dims[1],"dra_write_sect:d_a dim2 error");

   /* check if numbers of elements in g_a & d_a sections match */
   if ((*dihi - *dilo + 1) * (*djhi - *djlo + 1) !=
       (*gihi - *gilo + 1) * (*gjhi - *gjlo + 1))
       dai_error("dra_write_sect: d_a and g_a sections do not match ", 0L);

   dai_assign_request_handle(request);

   /* decompose d_a section into aligned and unaligned subsections
    * -- with respect to underlying array layout on the disk
    */

   Requests[*request].nu=MAX_ALGN;    
   Requests[*request].na=MAX_UNLG;

   fill_sectionM(d_sect, *d_a, *dilo, *dihi, *djlo, *djhi); 
   fill_sectionM(g_sect, *g_a, *gilo, *gihi, *gjlo, *gjhi); 

   dai_decomp_section(d_sect,
                     Requests[*request].list_algn, 
                    &Requests[*request].na,
                     Requests[*request].list_cover, 
                     Requests[*request].list_unlgn, 
                    &Requests[*request].nu);
   _dra_turn = 0;

   /* process unaligned subsections */
   dai_transfer_unlgn(DRA_OP_WRITE, (int)*transp, d_sect, g_sect, *request);
                  
   /* process aligned subsections */
   dai_transfer_algn (DRA_OP_WRITE, (int)*transp, d_sect, g_sect, *request);

   ga_sync_();

   return(ELIO_OK);
}



/*\ READ SECTION g_a[gilo:gihi, gjlo:gjhi] FROM d_a[dilo:dihi, djlo:djhi]
\*/
Integer FATR dra_read_section_(
        logical *transp,                   /*input:transpose operator*/
        Integer *g_a,                      /*input:GA handle*/ 
        Integer *gilo,                     /*input*/
        Integer *gihi,                     /*input*/
        Integer *gjlo,                     /*input*/
        Integer *gjhi,                     /*input*/
        Integer *d_a,                      /*input:DRA handle*/ 
        Integer *dilo,                     /*input*/
        Integer *dihi,                     /*input*/
        Integer *djlo,                     /*input*/
        Integer *djhi,                     /*input*/
        Integer *request)                  /*output: request id*/ 
{
Integer gdim1, gdim2, gtype, handle=*d_a+DRA_OFFSET;
section_t d_sect, g_sect;
 
   ga_sync_();

   /* usual argument/type/range checking stuff */
   dai_check_handleM(*d_a,"dra_read_sect");
   if(!dai_read_allowed(*d_a))dai_error("dra_read_sect: read not allowed",*d_a);
   ga_inquire_internal_(g_a, &gtype, &gdim1, &gdim2);
   if(DRA[handle].type != (int)gtype)dai_error("dra_read_sect: type mismatch",gtype);
   dai_check_rangeM(*gilo, *gihi, gdim1, "dra_read_sect: g_a dim1 error");
   dai_check_rangeM(*gjlo, *gjhi, gdim2, "dra_read_sect: g_a dim2 error");
   dai_check_rangeM(*dilo, *dihi,DRA[handle].dims[0],"dra_read_sect:d_a dim1 error");
   dai_check_rangeM(*djlo, *djhi,DRA[handle].dims[1],"dra_read_sect:d_a dim2 error");

   /* check if numbers of elements in g_a & d_a sections match */
   if ((*dihi - *dilo + 1) * (*djhi - *djlo + 1) !=
       (*gihi - *gilo + 1) * (*gjhi - *gjlo + 1))
       dai_error("dra_read_sect: d_a and g_a sections do not match ", 0L);
       
   dai_assign_request_handle(request);

   /* decompose d_a section into aligned and unaligned subsections
    * -- with respect to underlying array layout on the disk
    */

   Requests[*request].nu=MAX_ALGN;    
   Requests[*request].na=MAX_UNLG;

   fill_sectionM(d_sect, *d_a, *dilo, *dihi, *djlo, *djhi); 
   fill_sectionM(g_sect, *g_a, *gilo, *gihi, *gjlo, *gjhi); 

   dai_decomp_section(d_sect,
                     Requests[*request].list_algn, 
                    &Requests[*request].na,
                     Requests[*request].list_cover, 
                     Requests[*request].list_unlgn, 
                    &Requests[*request].nu);

   _dra_turn = 0;

   /* process unaligned subsections */
   dai_transfer_unlgn(DRA_OP_READ, (int)*transp,  d_sect, g_sect, *request);
           
   /* process aligned subsections */
   dai_transfer_algn (DRA_OP_READ, (int)*transp,  d_sect, g_sect, *request);

   return(ELIO_OK);
}



/*\ READ g_a FROM d_a
\*/
Integer FATR dra_read_(Integer* g_a, Integer* d_a, Integer* request)
{
Integer gdim1, gdim2, gtype, handle=*d_a+DRA_OFFSET;
logical transp = FALSE;
Integer ilo, ihi, jlo, jhi;

        ga_sync_();

        /* usual argument/type/range checking stuff */
        dai_check_handleM(*d_a,"dra_read");
        if(!dai_read_allowed(*d_a))dai_error("dra_read: read not allowed",*d_a);
        ga_inquire_internal_(g_a, &gtype, &gdim1, &gdim2);
        if(DRA[handle].type != (int)gtype)dai_error("dra_read: type mismatch",gtype);
        if(DRA[handle].dims[0] != gdim1)dai_error("dra_read: dim1 mismatch",gdim1);
        if(DRA[handle].dims[1] != gdim2)dai_error("dra_read: dim2 mismatch",gdim2);

        /* right now, naive implementation just calls dra_read_section */
        ilo = 1; ihi = DRA[handle].dims[0];
        jlo = 1; jhi = DRA[handle].dims[1];
        return(dra_read_section_(&transp, g_a, &ilo, &ihi, &jlo, &jhi,
                                         d_a, &ilo, &ihi, &jlo, &jhi, request));
}



/*\ WAIT FOR COMPLETION OF DRA OPERATION ASSOCIATED WITH request
\*/ 
Integer FATR dra_wait_(Integer* request)
{
        if(*request == DRA_REQ_INVALID) return(ELIO_OK);

        elio_wait(&Requests[*request].id);
        Requests[*request].num_pending=0;
        dai_exec_callback(Requests + *request);

        ga_sync_();

        return(ELIO_OK);
}


/*\ TEST FOR COMPLETION OF DRA OPERATION ASSOCIATED WITH request
\*/
Integer FATR dra_probe_(
        Integer *request,                  /*input*/
        Integer *status)                   /*output*/
{
Integer done,  type=GA_TYPE_GSM;
char *op="*";
int  stat;

        if(*request == DRA_REQ_INVALID){ *status = ELIO_DONE; return(ELIO_OK); }

        if(elio_probe(&Requests[*request].id, &stat)!=ELIO_OK)return(DRA_FAIL);
        *status = (Integer) stat;
        
        done = (*status==ELIO_DONE)? 1: 0;
        /* determine global status */
        ga_igop(type, &done, (Integer)1,op);
        
        if(done){
            *status = ELIO_DONE;
            Requests[*request].num_pending = 0;
            dai_exec_callback(Requests + *request);
        } else Requests[*request].num_pending = ELIO_PENDING;

        return(ELIO_OK);
}


/*\ Returns control to DRA for a VERY short time to improve progress
\*/
void dra_flick_()
{
Integer req;
int stat;

        if(!num_pending_requests)return; 

        for(req=0; req<MAX_REQ; req++)
          if(Requests[req].num_pending)
             if(elio_probe(&Requests[req].id, &stat)==ELIO_OK)
                if(stat == ELIO_DONE) dai_exec_callback(Requests + req); 
}
        


/*\ INQUIRE PARAMETERS OF EXISTING DISK ARRAY
\*/
Integer dra_inquire(
        Integer *d_a,                      /*input:DRA handle*/ 
        Integer *type,                     /*output*/
        Integer *dim1,                     /*output*/
        Integer *dim2,                     /*output*/
        char    *name,                     /*output*/
        char    *filename)                 /*output*/
{
Integer handle=*d_a+DRA_OFFSET;

        dai_check_handleM(*d_a,"dra_inquire");

        *type = (Integer)DRA[handle].type;
        *dim1 = DRA[handle].dims[0];
        *dim2 = DRA[handle].dims[1];
        strcpy(name, DRA[handle].name);
        strcpy(filename, DRA[handle].fname);
 
        return(ELIO_OK);
}


/*\ DELETE DISK ARRAY  -- relevant file(s) gone
\*/
Integer FATR dra_delete_(Integer* d_a)            /*input:DRA handle */
{
Integer handle = *d_a+DRA_OFFSET;

        ga_sync_();

        dai_check_handleM(*d_a,"dra_delete");
        dai_delete_param(DRA[handle].fname,*d_a);

        if(dai_file_master(*d_a))
          if(INDEPFILES(*d_a)){ 
             sprintf(dummy_fname,"%s.%ld",DRA[handle].fname,(long)dai_io_nodeid(*d_a));
             elio_delete(dummy_fname);
          }else {

             elio_delete(DRA[handle].fname);

          }

        dai_release_handle(d_a); 

        ga_sync_();
        return(ELIO_OK);
}


/*\ TERMINATE DRA DRATA STRUCTURES
\*/
Integer FATR dra_terminate_()
{
        free(DRA);
        MA_free_heap(_handle_buffer);
        ga_sync_();
        return(ELIO_OK);
}

void dai_clear_buffer()
{
int i;
     for (i=0;i<DRA_DBL_BUF_SIZE;i++) ((double*)_dra_buffer)[i]=0.;
}

/*\ routines for N-dimensional disk resident arrays
\*/

/*\ Simple sort using straight insertion
\*/
#define block_sortM(_ndim, _block_orig, _block_map) \
{\
  Integer _i,_j,_it,_bt; \
  Integer _block_tmp[MAXDIM]; \
  for (_i=0; _i < (_ndim); _i++) { \
    _block_map[_i] = _i; \
    _block_tmp[_i] = _block_orig[_i]; \
  } \
  for (_j=(_ndim)-2; _j >= 0; _j--) { \
    _i = _j + 1; \
    _bt = _block_tmp[_j]; \
    _it = _block_map[_j]; \
    while (_i < (_ndim) && _bt < _block_tmp[_i]) { \
      _block_tmp[_i-1] = _block_tmp[_i]; \
      _block_map[_i-1] = _block_map[_i]; \
      _i++; \
    }\
    _block_tmp[_i-1] = _bt; \
    _block_map[_i-1] = _it; \
  }\
}

/*\ compute chunk parameters for layout of arrays on the disk
 *   ---- a very simple algorithm to be refined later ----
\*/
void ndai_chunking(Integer elem_size, Integer ndim, Integer block_orig[], 
                  Integer dims[], Integer chunk[])
/*   elem_size:     Size of individual data element in bytes [input]
     ndim:          Dimension of DRA [input]
     block_orig[]:  Estimated size of request in each coordinate
                    direction. If size is unknown then use -1. [input]
     dims[]:        Size of DRA in each coordinate direction [input]
     chunk[]:       Size of data block size (in elements) in each
                    coordinate direction [output]
*/
{
  Integer patch_size;
  Integer i, j, tmp_patch, block[MAXDIM], block_map[MAXDIM];
  double ratio;
  logical full_buf, some_neg, overfull_buf;
  /* copy block_orig so that original guesses are not destroyed */
  for (i=0; i<ndim; i++) block[i] = block_orig[i];

  /* do some preliminary checks on block to make sure initial guesses
     are less than corresponding DRA dimensions */
  for (i=0; i<ndim; i++) {
    if (block[i] > dims[i]) block[i] = dims[i];
  }
  /* do additional adjustments to see if initial guesses are near some
     perfect factors of DRA dimensions */
  for (i=0; i<ndim; i++) {
    if (block[i] > 0 && block[i]<dims[i]) {
      if (dims[i]%block[i] != 0) {
        ratio = (double)dims[i]/(double)block[i];
        j = (int)(ratio+0.5);
        if (dims[i]%j ==0) block[i] = dims[i]/j;
      }
    }
  }

  /* initialize chunk array to zero and find out how big patch is based
     on specified block dimensions */
  patch_size = 1;
  some_neg = FALSE;
  full_buf = FALSE;
  overfull_buf = FALSE;
  for (i=0; i<ndim; i++) {
    if (block[i] > 0) patch_size *= block[i];
    else some_neg = TRUE;
  }
  if (patch_size*elem_size > DRA_BUF_SIZE) overfull_buf = TRUE;

  /* map dimension sizes from highest to lowest */
  block_sortM(ndim, dims, block_map);

  /* IO buffer is not full and there are some unspecied chunk dimensions.
     Set unspecified dimensions equal to block dimensions until buffer
     is filled. */
  if (!full_buf && !overfull_buf && some_neg) {
    for (i=ndim-1; i>=0; i--) {
      if (block[block_map[i]] < 0) {
        tmp_patch = patch_size * dims[block_map[i]];
        if (tmp_patch*elem_size < DRA_BUF_SIZE) {
          patch_size *= dims[block_map[i]];
          block[block_map[i]] = dims[block_map[i]];
        } else {
          block[block_map[i]] = DRA_BUF_SIZE/(patch_size*elem_size); 
          patch_size *= block[block_map[i]];
          full_buf = TRUE;
        }
      }
    }
  }

  /* copy block array to chunk array */
  for (i=0; i<ndim; i++) {
    if (block[i] > 0) chunk[i] = block[i];
    else chunk[i] = 1;
  }

  /* If patch overfills buffer, scale patch down until it fits */
  if (overfull_buf) {
    ratio = ((double)DRA_BUF_SIZE)/((double)(patch_size*elem_size));
    ratio = pow(ratio,1.0/((double)ndim));
    patch_size = 1;
    for (i=0; i<ndim; i++) {
      chunk[i] = (int)(((double)chunk[i])*ratio);
      if (chunk[i] < 1) chunk[i] = 1;
      patch_size *= chunk[i];
    }
  }

#ifdef DEBUG
  printf("Current patch at 2 is %d\n",(int)patch_size*elem_size);
#endif
  /* set remaining block sizes equal to 1 */
  for (i=0; i<ndim; i++) {
    if (chunk[i] == 0) chunk[i] = 1;
  }
  /* Patch size may be slightly larger than buffer. If so, nudge
     size down until patch is smaller than buffer. */
  if (elem_size*patch_size > DRA_BUF_SIZE) {
    /* map chunks from highest to lowest */
    block_sortM(ndim, chunk, block_map);
    for (i=0; i < ndim; i++) {
      while (chunk[block_map[i]] > 1 &&
             elem_size*patch_size > DRA_BUF_SIZE) {
        patch_size /= chunk[block_map[i]];
        chunk[block_map[i]]--;
        patch_size *= chunk[block_map[i]];
      }
    }
  }
}

#define nfill_sectionM(sect, _hndl, _ndim, _lo, _hi) \
{ \
  Integer _i; \
  sect.handle = _hndl; \
  sect.ndim = _ndim; \
  for (_i=0; _i<_ndim; _i++) { \
    sect.lo[_i]    = _lo[_i]; \
    sect.hi[_i]    = _hi[_i]; \
  } \
}

#define nblock_to_sectM(ds_a, _CR) \
{\
  Integer _i, _b[MAXDIM], _C = (_CR); \
  Integer _hndl = (ds_a)->handle+DRA_OFFSET; \
  Integer _R = (DRA[_hndl].dims[0]+DRA[_hndl].chunk[0]-1)/DRA[_hndl].chunk[0]; \
  (ds_a)->ndim = DRA[_hndl].ndim; \
  _b[0] = _C%_R; \
  for (_i=1; _i<DRA[_hndl].ndim; _i++) { \
    _C = (_C - _b[_i-1])/_R; \
    _R = (DRA[_hndl].dims[_i]+DRA[_hndl].chunk[_i]-1)/DRA[_hndl].chunk[_i]; \
    _b[_i] = (_C)%_R; \
  } \
  for (_i=0; _i<DRA[_hndl].ndim; _i++) { \
    (ds_a)->lo[_i] = _b[_i]*DRA[_hndl].chunk[_i] + 1; \
    (ds_a)->hi[_i] = (ds_a)->lo[_i] + DRA[_hndl].chunk[_i] - 1; \
    if ((ds_a)->hi[_i] > DRA[_hndl].dims[_i]) \
      (ds_a)->hi[_i] = DRA[_hndl].dims[_i]; \
  } \
}

#define nblock_to_indicesM(_index,_ndim,_block_dims,_CC) \
{ \
  Integer _i, _C=(_CC); \
  _index[0] = _C%_block_dims[0]; \
  for (_i=1; _i<(_ndim); _i++) { \
    _C = (_C - _index[_i-1])/_block_dims[_i-1]; \
    _index[_i] = _C%_block_dims[_i]; \
  } \
}

/*\ find offset in file for (lo,hi) element
\*/
void ndai_file_location(section_t ds_a, Off_t* offset)
{
Integer handle=ds_a.handle+DRA_OFFSET, ndim, i, j;
Integer blocks[MAXDIM], part_chunk[MAXDIM], cur_ld[MAXDIM];
long par_block[MAXDIM];
long offelem;

     
        ndim = DRA[handle].ndim;
        for (i=0; i<ndim-1; i++) {
          if((ds_a.lo[i]-1)%DRA[handle].chunk[i])
            dai_error("ndai_file_location: not alligned ??",ds_a.lo[i]);
        }

        for (i=0; i<ndim; i++) {
          /* number of blocks from edge */
          blocks[i] = (ds_a.lo[i]-1)/DRA[handle].chunk[i];
          /* size of incomplete chunk */
          part_chunk[i] = DRA[handle].dims[i]%DRA[handle].chunk[i];
          /* stride for this block of data in this direction */
          cur_ld[i] = (blocks[i] == DRA[handle].dims[i]/DRA[handle].chunk[i]) ?
                    part_chunk[i]: DRA[handle].chunk[i];
        }

        /* compute offset (in elements) */

        if (INDEPFILES(ds_a.handle)) {
          Integer   CR, block_dims[MAXDIM]; 
          Integer   index[MAXDIM];
          long      nelem;
          Integer   i, j;
          Integer   ioprocs = dai_io_procs(ds_a.handle); 
          Integer   iome = dai_io_nodeid(ds_a.handle);
           
           /* Find index of current block and find number of chunks in
              each dimension of DRA */
          nsect_to_blockM(ds_a, &CR); 
          for (i=0; i<ndim; i++) {
            block_dims[i] = (DRA[handle].dims[i]+DRA[handle].chunk[i]-1)
                          / DRA[handle].chunk[i];
          }
          if (iome >= 0) {
            offelem = 0;
            for (i=iome; i<CR; i+=ioprocs) {
              /* Copy i because macro destroys i */
              nblock_to_indicesM(index,ndim,block_dims,i);
              nelem = 1;
              for (j=0; j<ndim; j++) {
                if (index[j]<block_dims[j]-1) {
                  nelem *= (long)DRA[handle].chunk[j];
                } else {
                  if (part_chunk[j] != 0) {
                    nelem *= (long)part_chunk[j];
                  } else {
                    nelem *= (long)DRA[handle].chunk[j];
                  }
                }
              }
              offelem += nelem;
            }
            /* add fractional offset for current block */
            nelem = 1;
            nblock_to_indicesM(index,ndim,block_dims,CR);
            for (i=0; i<ndim-1; i++) {
              if (index[i]<block_dims[i]-1) {
                nelem *= (long)DRA[handle].chunk[i];
              } else {
                if (part_chunk[i] != 0) {
                  nelem *= (long)part_chunk[i];
                } else {
                  nelem *= (long)DRA[handle].chunk[i];
                }
              }
            }
            nelem *= (long)(ds_a.lo[ndim-1]-1)%DRA[handle].chunk[ndim-1];
            offelem += (long)nelem;
          }
        } else {
          /* Find offset by calculating the number of chunks that must be
           * traversed to get to the corner of block containing the lower
           * coordinate index ds_a.lo[]. Then move into the block along
           * the last dimension to the point ds_a.lo[ndim-1]. */
          for (i=0; i<ndim; i++) {
            par_block[i] = 1;
            for (j=0; j<ndim; j++) {
              if (j < i) {
                par_block[i] *= (long)cur_ld[j];
              } else if (j == i) {
                if (i == ndim-1) {
                  /* special case for last dimension, which may represent
                   * a fraction of a chunk */
                  par_block[i] *= (long)(ds_a.lo[i]-1);
                } else {
                  par_block[i] *= (long)(blocks[j]*DRA[handle].chunk[j]);
                }
              } else {
                par_block[i] *= (long)(DRA[handle].dims[j]);
              }
            }
          }
          offelem = 0;
          for (i=0; i<ndim; i++) offelem += (long)par_block[i];
        }

        *offset = (Off_t)offelem * dai_sizeofM(DRA[handle].type); 
}

/*\ write zero at EOF for NDRA
\*/
void ndai_zero_eof(Integer d_a)
{
Integer handle = d_a+DRA_OFFSET, nelem, i;
Integer zero[MAXDIM];
Off_t offset;
Size_t  bytes;

        if(DRA[handle].type == MT_F_DBL) *(DoublePrecision*)_dra_buffer = 0.;
        if(DRA[handle].type == MT_F_INT) *(Integer*)_dra_buffer = 0;
        if(DRA[handle].type == MT_F_REAL) *(float*)_dra_buffer = 0;

        if(INDEPFILES(d_a)) {

          Integer   CR, i, nblocks; 
          section_t ds_a;
          /* number of processors that do io */
          Integer   ioprocs=dai_io_procs(d_a); 
          /* node id of current process (if it does io) */
          Integer   iome = dai_io_nodeid(d_a);

          /* total number of blocks in the disk resident array */
          nblocks = 1;
          for (i=0; i<DRA[handle].ndim; i++) {
            nblocks *= (DRA[handle].dims[i]+DRA[handle].chunk[i]-1)
                     / DRA[handle].chunk[i];
            zero[i] = 0;
          }
          nfill_sectionM(ds_a, d_a, DRA[handle].ndim, zero, zero); 

          /* search for the last block for each I/O processor */
          for(i = 0; i <ioprocs; i++){
             CR = nblocks - 1 -i;
             if(CR % ioprocs == iome) break;
          }
          if(CR<0) return; /* no blocks owned */

          nblock_to_sectM(&ds_a, CR); /* convert block number to section */
          ndai_file_location(ds_a, &offset);
          nelem = 1;
          for (i=0; i<DRA[handle].ndim; i++) nelem *= (ds_a.hi[i] - ds_a.lo[i] + 1);
          nelem--;
          offset += ((Off_t)nelem) * dai_sizeofM(DRA[handle].type);

#         ifdef DEBUG
            printf("me=%d zeroing EOF (%d) at %ld bytes \n",iome,CR,offset);
#         endif
        } else {

          nelem = 1;
          for (i=0; i<DRA[handle].ndim; i++) nelem *= DRA[handle].dims[i];
          nelem--;
          offset = ((Off_t)nelem) * dai_sizeofM(DRA[handle].type);
        }

        bytes = dai_sizeofM(DRA[handle].type);
        if(bytes != elio_write(DRA[handle].fd, offset, _dra_buffer, bytes))
                     dai_error("ndai_zero_eof: write error ",0);
}


/*\ CREATE AN N-DIMENSIONAL DISK ARRAY
\*/
Integer ndra_create(
        Integer *type,                     /*input*/
        Integer *ndim,                     /*input: dimension of DRA*/
        Integer dims[],                    /*input: dimensions of DRA*/
        char    *name,                     /*input*/
        char    *filename,                 /*input*/
        Integer *mode,                     /*input*/
        Integer reqdims[],                 /*input: dimension of typical request*/
        Integer *d_a)                      /*output:DRA handle*/
{
Integer handle, elem_size, i;

        ga_sync_();

        /* if we have an error here, it is fatal */       
        dai_check_typeM(*type);    
        for (i=0; i<*ndim; i++) if (dims[i] <=0)
              dai_error("ndra_create: disk array dimension invalid ", dims[i]);
        if(strlen(filename)>DRA_MAX_FNAME)
              dai_error("ndra_create: filename too long", DRA_MAX_FNAME);

       /*** Get next free DRA handle ***/
       if( (handle = dai_get_handle()) == -1)
           dai_error("ndra_create: too many disk arrays ", _max_disk_array);
       *d_a = handle - DRA_OFFSET;

       /* determine disk array decomposition */ 
        elem_size = dai_sizeofM(*type);
        ndai_chunking( elem_size, *ndim, reqdims, dims, DRA[handle].chunk);

       /* determine layout -- by row or column */
        DRA[handle].layout = COLUMN;

       /* complete initialization */
        for (i=0; i<*ndim; i++) DRA[handle].dims[i] = dims[i];
        DRA[handle].ndim = *ndim;
        DRA[handle].type = ga_type_f2c((int)*type);
        DRA[handle].mode = (int)*mode;
        strncpy (DRA[handle].fname, filename,  DRA_MAX_FNAME);
        strncpy(DRA[handle].name, name, DRA_MAX_NAME );

        dai_write_param(DRA[handle].fname, *d_a);      /* create param file */
        DRA[handle].indep = dai_file_config(filename); /*check file configuration*/

        /* create file */
        if(dai_io_manage(*d_a)){ 

           if (INDEPFILES(*d_a)) {

             sprintf(dummy_fname,"%s.%ld",DRA[handle].fname,(long)dai_io_nodeid(*d_a));
             DRA[handle].fd = elio_open(dummy_fname,(int)*mode, ELIO_PRIVATE);

           }else{

              /* collective open supported only on Paragon */
#           ifdef PARAGON
              DRA[handle].fd = elio_gopen(DRA[handle].fname,(int)*mode); 
#           else
              DRA[handle].fd = elio_open(DRA[handle].fname,(int)*mode, ELIO_SHARED); 
#           endif
           }

           if(DRA[handle].fd==NULL)dai_error("ndra_create:failed to open file",0);
           if(DRA[handle].fd->fd==-1)dai_error("ndra_create:failed to open file",0);
        }

        /*
         *  Need to zero the last element in the array on the disk so that
         *  we never read beyond EOF.
         *
         *  For multiple component files will stamp every one of them.
         *
         */
        ga_sync_();

        if(dai_file_master(*d_a) && dai_write_allowed(*d_a)) ndai_zero_eof(*d_a);
/*        if(dai_io_nodeid(*d_a)==0)printf("chunking: %d x %d\n",DRA[handle].chunk1,
                                                          DRA[handle].chunk2);
*/

        ga_sync_();

        return(ELIO_OK);
}

/*\ write N-dimensional aligned block of data from memory buffer to d_a
\*/
void ndai_put(
        section_t    ds_a,  /*[input] section of DRA written to disk */
        Void         *buf,  /*[input] pointer to io buffer */
        Integer      ld[],  /*[input] array of strides */
        io_request_t *id)
{
  Integer handle = ds_a.handle + DRA_OFFSET, elem, i;
  Integer ndim = ds_a.ndim;
  Off_t   offset;
  Size_t  bytes;

  /* find location in a file where data should be written */
  ndai_file_location(ds_a, &offset);
  for (i=0; i<ndim-1; i++) if ((ds_a.hi[i]-ds_a.lo[i]+1) != ld[i])
       dai_error("ndai_put: bad ld",ld[i]); 

  /* since everything is aligned, write data to disk */
  elem = 1;
  for (i=0; i<ndim; i++) elem *= (ds_a.hi[i]-ds_a.lo[i]+1);
  bytes= (Size_t) elem * dai_sizeofM(DRA[handle].type);
  if( ELIO_OK != elio_awrite(DRA[handle].fd, offset, buf, bytes, id ))
                 dai_error("ndai_put failed", ds_a.handle);
}

/*\ read N-dimensional aligned block of data from d_a to memory buffer
\*/
void ndai_get(section_t ds_a, /*[input] section of DRA read from disk */
              Void *buf,      /*[input] pointer to io buffer */
              Integer ld[],   /*[input] array of strides */
              io_request_t *id)
{
  Integer handle = ds_a.handle + DRA_OFFSET, elem, rc;
  Integer ndim = DRA[handle].ndim, i;
  Off_t   offset;
  Size_t  bytes;
  void    dai_clear_buffer();

  /* find location in a file where data should be read from */
  ndai_file_location(ds_a, &offset);

# ifdef CLEAR_BUF
    dai_clear_buffer();
# endif

  for (i=0; i<ndim-1; i++) if ((ds_a.hi[i] - ds_a.lo[i] + 1) != ld[i])
    dai_error("ndai_get: bad ld",ld[i]); 
  /* since everything is aligned, read data from disk */
  elem = 1;
  for (i=0; i<ndim; i++) elem *= (ds_a.hi[i]-ds_a.lo[i]+1);
  bytes= (Size_t) elem * dai_sizeofM(DRA[handle].type);
  rc= elio_aread(DRA[handle].fd, offset, buf, bytes, id );
  if(rc !=  ELIO_OK) dai_error("ndai_get failed", rc);
}

#define ndai_check_rangeM(_lo, _hi, _ndim, _dims, _err_msg) \
{ \
  int _range_ok = 1, _i; \
  for (_i=0; _i < (_ndim); _i++) { \
    if (_lo[_i] < 1 || _lo[_i] > _dims[_i] || _hi[_i] < _lo[_i] \
                || _hi[_i] > _dims[_i] ) _range_ok = 0; \
  } \
  if(!_range_ok) dai_error(_err_msg, _dim); \
}

/*\ decompose section defined by lo and hi into aligned and unaligned DRA
 *  subsections
\*/ 
void ndai_decomp_section(
        section_t ds_a,
        Integer aligned[][2*MAXDIM],      /*[output]: Indices of aligned subsections.*/
        int *na,                          /*[output]: Number of aligned subsections.*/
        Integer cover[][2*MAXDIM],        /*[output]: Indices of cover subsections.*/
        Integer unaligned[][2*MAXDIM],    /*[output]: Indices of unaligned subsections.*/
        int *nu)                          /*[output]: Number of unaligned subsections.*/
{
  Integer a=0, u=0, handle = ds_a.handle+DRA_OFFSET, off, chunk_units, algn_flag;
  Integer i, j, idir, ndim = DRA[handle].ndim;
  Integer off_low[MAXDIM], off_hi[MAXDIM];
  Integer cover_lo[MAXDIM], cover_hi[MAXDIM];
  Integer check, chunk_lo, chunk_hi;

  /* section[lo,hi] is decomposed into 'aligned' and 'unaligned'
   * subsections.  The aligned subsections are aligned on
   * chunk[1]..chunk[ndim-1] boundaries. The unaligned subsections are
   * not completely covered by chunk[1]..chunk[ndim]-1 boundaries. These
   * are subsets of the 'cover' subsections which are aligned on chunk
   * boundaries and contain the unaligned subsections. Disk I/O will
   * actually be performed on 'aligned' and 'cover' subsections instead
   * of 'unaligned' subsections.
   *
   * The indexing of the aligned[][idir], cover[][idir], and
   * unaligned[][idir] arrays is idir = 0,1 corresponds to low and
   * high values in the 0 direction, idir = 2,3 corresponds to low and
   * high values in the 1 direction and so on up to a value of
   * idir = 2*ndim-1.
   *
   * The strategy for performing the decomposition is to first find the
   * coordinates corresponding to an aligned patch that completely covers
   * the originally requested section.
   * 
   * Begin by initializing some arrays. */

  for (i=0, j=0; i<ndim; i++) {
    aligned[a][j] = ds_a.lo[i];
    cover_lo[i] = ds_a.lo[i];
    off_low[i] = (ds_a.lo[i] - 1) % DRA[handle].chunk[i];
    j++;
    aligned[a][j] = ds_a.hi[i];
    cover_hi[i] = ds_a.hi[i];
    off_hi[i] = ds_a.hi[i] % DRA[handle].chunk[i];
    j++;
  }
  /* Find coordinates of aligned patch that completely covers the first
     ndim-1 dimensions of ds_a */
  for (i=0; i<ndim-1; i++) {
    if (off_low[i] !=0) {
      chunk_lo = (ds_a.lo[i] - 1) / DRA[handle].chunk[i];
      cover_lo[i] = chunk_lo * DRA[handle].chunk[i] + 1;
    }
    if (off_hi[i] !=0) {
      chunk_hi = ds_a.hi[i] / DRA[handle].chunk[i] + 1;
      cover_hi[i] = chunk_hi * DRA[handle].chunk[i];
      if (cover_hi[i] > DRA[handle].dims[i])
        cover_hi[i] = DRA[handle].dims[i];
    }
  }
  /* Find coordinates of aligned chunk (if there is one) */
  j = 0;
  check = 1;
  for (i=0; i<ndim-1; i++) {
    if (off_low[i] != 0) {
      chunk_lo = (ds_a.lo[i] - 1) / DRA[handle].chunk[i] + 1;
      aligned[a][j] = chunk_lo * DRA[handle].chunk[i] + 1;
    }
    j++;
    if (off_hi[i] !=0) {
      chunk_hi = ds_a.hi[i] / DRA[handle].chunk[i];
      aligned[a][j] = chunk_hi * DRA[handle].chunk[i];
    }
    if (aligned[a][j] < aligned[a][j-1]) check = 0;
    j++;
  }
  *na = (check == 1)  ?1 :0;

  /* evaluate cover sections and unaligned chunk dimensions. We
     break the evaluation of chunks into the following two
     cases:
     1) There is no aligned section
     2) There is an aligned section */

  if (*na == 0) {
    /* There is no aligned block. Just return with one cover
       section */
    for (i=0, j=0; i<ndim; i++) {
      cover[u][j] = cover_lo[i];
      unaligned[u][j] = ds_a.lo[i];
      j++;
      cover[u][j] = cover_hi[i];
      unaligned[u][j] = ds_a.hi[i];
      j++;
    }
    *nu = 1;
    return;
  }

  /* An aligned chunk exists so we must find cover sections that
     surround it. We scan over the coordinate directions idir
     and choose cover sections such that if the coordinate
     direction of the cover section is less than idir, then the
     cover section extends to the boundary of the aligned
     section. If the coordinate direction of the cover section
     is greater than idir then the cover extends beyond the
     dimensions of the aligned chunk (if there is a nonzero
     offset). This scheme guarantees that corner pieces of the
     sections are picked up once and only once. */

  for (idir=0; idir<ndim-1; idir++) {
    check = 1;
    /* cover over lower end of patch */
    if (off_low[idir] != 0) {
      for (i=0, j=0; i<ndim-1; i++) {
        if (i < idir) {
          if (off_low[i] != 0) {
            chunk_units = (ds_a.lo[i] - 1) / DRA[handle].chunk[i];
            cover[u][j] = chunk_units * DRA[handle].chunk[i] + 1;
          } else {
            cover[u][j] = ds_a.lo[i];
          }
          unaligned[u][j] = ds_a.lo[i];
          j++;
          if (off_hi[i] != 0) {
            chunk_units = ds_a.hi[i] / DRA[handle].chunk[i]+1;
            cover[u][j] = MIN(chunk_units * DRA[handle].chunk[i],
              DRA[handle].dims[i]);
          } else {
            cover[u][j] = ds_a.hi[i];
          }
          unaligned[u][j] = ds_a.hi[i];
          j++;
        } else if (i == idir) {
          chunk_units = (ds_a.lo[i] - 1) / DRA[handle].chunk[i];
          cover[u][j] = chunk_units * DRA[handle].chunk[i] + 1;
          unaligned[u][j] = ds_a.lo[i];
          j++;
          cover[u][j] = MIN(cover[u][j-1] + DRA[handle].chunk[i]-1,
            DRA[handle].dims[i]);
          unaligned[u][j] = MIN(ds_a.hi[i],cover[u][j]);
          j++;
        } else {
          if (off_low[i] != 0) {
            chunk_units = (ds_a.lo[i] - 1) / DRA[handle].chunk[i]+1;
            cover[u][j] = chunk_units * DRA[handle].chunk[i] + 1;
          } else {
            cover[u][j] = ds_a.lo[i];
          }
          unaligned[u][j] = ds_a.lo[i];
          j++;
          if (off_hi[i] != 0) {
            chunk_units = ds_a.hi[i] / DRA[handle].chunk[i];
            cover[u][j] = chunk_units * DRA[handle].chunk[i];
          } else {
            cover[u][j] = ds_a.hi[i];
          }
          unaligned[u][j] = ds_a.hi[i];
          j++;
        }
      }
      cover[u][j] = ds_a.lo[ndim-1];
      unaligned[u][j] = ds_a.lo[ndim-1];
      j++;
      cover[u][j] = ds_a.hi[ndim-1];
      unaligned[u][j] = ds_a.hi[ndim-1];
      u++;
      check = 1;
      aligned[a][2*idir] = cover[u-1][2*idir+1]+1;
    }
    /* check to see if there is only one unaligned section covering this
       dimension */
    if (check == 1) {
      if (cover[u-1][2*idir+1] >= ds_a.hi[idir]) check = 0;
    } else {
      check = 1;
    } 
    /* handle cover over upper end of patch */
    if (off_hi[idir] != 0 && check == 1) {
      for (i=0, j=0; i<ndim-1; i++) {
        if (i < idir) {
          if (off_low[i] != 0) {
            chunk_units = (ds_a.lo[i] - 1) / DRA[handle].chunk[i];
            cover[u][j] = chunk_units * DRA[handle].chunk[i] + 1;
          } else {
            cover[u][j] = ds_a.lo[i];
          }
          unaligned[u][j] = ds_a.lo[i];
          j++;
          if (off_hi[i] != 0) {
            chunk_units = ds_a.hi[i] / DRA[handle].chunk[i]+1;
            cover[u][j] = MIN(chunk_units * DRA[handle].chunk[i],
              DRA[handle].dims[i]);
          } else {
            cover[u][j] = ds_a.hi[i];
          }
          unaligned[u][j] = ds_a.hi[i];
          j++;
        } else if (i == idir) {
          chunk_units = ds_a.hi[i] / DRA[handle].chunk[i];
          cover[u][j] = chunk_units * DRA[handle].chunk[i] + 1;
          unaligned[u][j] = cover[u][j];
          aligned[a][2*i+1] = MIN(cover[u][j]-1,ds_a.hi[idir]);
          j++;
          cover[u][j] = MIN(cover[u][j-1] + DRA[handle].chunk[i]-1,
            DRA[handle].dims[i]);
          unaligned[u][j] = MIN(ds_a.hi[i],cover[u][j]);
          j++;
        } else {
          if (off_low[i] != 0) {
            chunk_units = (ds_a.lo[i] - 1) / DRA[handle].chunk[i]+1;
            cover[u][j] = chunk_units * DRA[handle].chunk[i] + 1;
          } else {
            cover[u][j] = ds_a.lo[i];
          }
          unaligned[u][j] = ds_a.lo[i];
          j++;
          if (off_hi[i] != 0) {
            chunk_units = ds_a.hi[i] / DRA[handle].chunk[i];
            cover[u][j] = chunk_units * DRA[handle].chunk[i];
          } else {
            cover[u][j] = ds_a.hi[i];
          }
          unaligned[u][j] = ds_a.hi[i];
          j++;
        }
      }
      cover[u][j] = ds_a.lo[ndim-1];
      unaligned[u][j] = ds_a.lo[ndim-1];
      j++;
      cover[u][j] = ds_a.hi[ndim-1];
      unaligned[u][j] = ds_a.hi[ndim-1];
      u++;
      aligned[a][2*idir+1] = cover[u-1][2*idir]-1;
    }
  }
  *nu = (int)u;
  return;
}

/*\ given the set of indices lo inside the patch cover, find the next
 *  set of indices assuming that the area represented by cover has been
 *  divided up into blocks whose size is given in inc
\*/
int ndai_next(Integer *lo, Integer *cover,
    Integer *inc, Integer ndim)
{
  /* first check to see if any of the low indices are out of range.
     If so then reset all low indices to minimum values. */
  int retval=1;
  Integer i;
  for (i = 0; i<ndim; i++) { 
    if (lo[i] == 0) retval = 0;
  }
  if (retval == 0) {
    for (i = 0; i<ndim; i++) { 
      lo[i] = cover[2*i];
    }
  }
  /* increment all indices in lo. If index exceeds value cover[2*i+1]
     for that index, then set index back to cover[2*i] and increment
     next index. */
  if (retval != 0) {
    for (i=0; i<ndim; i++) {
      lo[i] += inc[i];
      if (lo[i] > cover[2*i+1]) {
        if (i<ndim-1) lo[i] = cover[2*i];
      } else {
        break;
      }
    }
  }
  retval = (lo[ndim-1] <= cover[2*ndim-1]);
  return retval;
}


/*\ compute next chunk of array to process
\*/
int ndai_next_chunk(Integer req, Integer* list, section_t* ds_chunk)
{
Integer   handle = ds_chunk->handle+DRA_OFFSET;
int       retval, ndim = DRA[handle].ndim, i;

    /* If we are writing out to multiple files then we need to consider
       chunk boundaries along last dimension */
    if(INDEPFILES(ds_chunk->handle))
      if(ds_chunk->lo[ndim-1] && DRA[handle].chunk[ndim-1]>1) 
         ds_chunk->lo[ndim-1] -= (ds_chunk->lo[ndim-1] -1) %
           DRA[handle].chunk[ndim-1];
    
    /* ds_chunk->lo is getting set in this call. list contains the
       the lower and upper indices of the cover section. */
    retval = ndai_next(ds_chunk->lo, list, DRA[handle].chunk, ndim);
    /*
    printf("Request %d\n",req);
    for (i=0; i<ndim; i++) {
      printf("ds_chunk.lo[%d] = %d cover.lo[%d] = %d cover.hi[%d] = %d\n", i,
          ds_chunk->lo[i], i, list[2*i], i, list[2*i+1]);
    } */
    if(!retval) {
      return(retval);
    }

    for (i=0; i<ndim; i++) {
      ds_chunk->hi[i] = MIN(list[2*i+1],
          ds_chunk->lo[i]+DRA[handle].chunk[i]-1);
    }

    /* Again, if we are writing out to multiple files then we need to consider
       chunk boundaries along last dimension */
    if(INDEPFILES(ds_chunk->handle)) { 
         Integer nlo;
         Integer hi_temp =  ds_chunk->lo[ndim-1] +
           DRA[handle].chunk[ndim-1] -1;
         hi_temp -= hi_temp % DRA[handle].chunk[ndim-1];
         ds_chunk->hi[ndim-1] = MIN(ds_chunk->hi[ndim-1], hi_temp); 

         /*this line was absent from older version on bonnie that worked */
         nlo = 2*(ndim-1);
         if(ds_chunk->lo[ndim-1] < list[nlo]) ds_chunk->lo[ndim-1] = list[nlo]; 
    }
    /*
    for (i=0; i<ndim; i++) {
      printf("ds_chunk.hi[%d] = %d\n", i, ds_chunk->hi[i]);
    } */

    return 1;
}

/*\ Write or Read Unaligned Subsections to/from disk: 
 *  always read an aligned extension of a section from disk to local buffer then 
 *  for read :  copy requested data from buffer to global array;
 *  for write:  overwrite part of buffer with data from g_a and write
 *  complete buffer to disk
 *
\*/
void ndai_transfer_unlgn(int opcode,    /*[input]: signal for read or write */
                        int transp,     /*[input]: should data be transposed*/
                        section_t ds_a, /*[input]: section of DRA that is
                                         to be read from or written to*/
                        section_t gs_a, /*[input]: section of GA that is
                                          to be read from or written to*/
                        Integer req     /*[input]: request number*/
                        )
{
  Integer   chunk_ld[MAXDIM],  next, offset, i, j;
  int   type = DRA[ds_a.handle+DRA_OFFSET].type;
  Integer   ndim = DRA[ds_a.handle+DRA_OFFSET].ndim;
  section_t ds_chunk, ds_unlg;
  char      *buffer; 

  ds_chunk =  ds_unlg = ds_a;
  if (dra_debug_flag && 0) {
    for (i=0; i<ndim; i++) {
      printf("ndai_transfer_unlgn: ds_chunk.lo[%d] = %d\n",i,ds_chunk.lo[i]);
      printf("ndai_transfer_unlgn: ds_chunk.hi[%d] = %d\n",i,ds_chunk.hi[i]);
    }
    printf("ndai_transfer_unlgn: number of unaligned chunks = %d\n",
        Requests[req].nu);
    for (j=0; j<Requests[req].nu; j++) {
      for (i=0; i<ndim; i++) {
        printf("ndai_transfer_unlgn: list_cover[%d][%d] = %d\n",
            j,2*i,Requests[req].list_cover[j][2*i]);
        printf("ndai_transfer_unlgn: list_cover[%d][%d] = %d\n",
            j,2*i+1,Requests[req].list_cover[j][2*i+1]);
      }
    }
  }

  for(next = 0; next < Requests[req].nu; next++){

    for (i=0; i<ndim; i++) ds_chunk.lo[i] = 0;   /* initialize */
    while(ndai_next_chunk(req, Requests[req].list_cover[next],&ds_chunk)){
   /*   printf("Request %d\n",req);
      printf("ds_chunk.lo[0] = %d\n",ds_chunk.lo[0]);
      printf("ds_chunk.hi[0] = %d\n",ds_chunk.hi[0]);
      printf("ds_chunk.lo[1] = %d\n",ds_chunk.lo[1]);
      printf("ds_chunk.hi[1] = %d\n",ds_chunk.hi[1]); */

      if(dai_myturn(ds_chunk)){

        dai_wait(req); /* needs free buffer to proceed */

        /*find corresponding to chunk of 'cover' unaligned sub-subsection*/
        for (i=0; i<ndim; i++) {
          ds_unlg.lo[i] = Requests[req].list_unlgn[next][2*i];
          ds_unlg.hi[i] = Requests[req].list_unlgn[next][2*i+1];
        }

        if (dra_debug_flag && 0) {
          for (i=0; i<ndim; i++) {
            printf("ndai_transfer_unlgn: ds_chunk.lo[%d] = %d\n",i,ds_chunk.lo[i]);
            printf("ndai_transfer_unlgn: ds_chunk.hi[%d] = %d\n",i,ds_chunk.hi[i]);
          }
          for (i=0; i<ndim; i++) {
            printf("ndai_transfer_unlgn: ds_unlg.lo[%d] = %d\n",i,ds_unlg.lo[i]);
            printf("ndai_transfer_unlgn: ds_unlg.hi[%d] = %d\n",i,ds_unlg.hi[i]);
          }
        }
            
        if(!dai_section_intersect(ds_chunk, &ds_unlg))
            dai_error("ndai_transfer_unlgn: inconsistent cover",0);

        /* copy data from disk to DRA buffer */
        for (i=0; i<ndim-1; i++) chunk_ld[i] = ds_chunk.hi[i] - ds_chunk.lo[i] + 1;
        ndai_get(ds_chunk, _dra_buffer, chunk_ld, &Requests[req].id);
        elio_wait(&Requests[req].id); 

        /* determine location in the buffer where GA data should be */
        offset = ds_unlg.lo[ndim-1]-ds_chunk.lo[ndim-1];
        for (i=ndim-2; i>=0; i--)  {
          offset = offset*chunk_ld[i];
          offset += ds_unlg.lo[i] - ds_chunk.lo[i];
        }
        buffer  = (char*)_dra_buffer;
        buffer += offset * dai_sizeofM(type);

        switch (opcode){
          case DRA_OP_WRITE: 
            /* overwrite a part of buffer with data from g_a */  
          /*  printf("(unlgn) gs_a.lo[0] = %d\n",gs_a.lo[0]);
            printf("(unlgn) gs_a.hi[0] = %d\n",gs_a.hi[0]);
            printf("(unlgn) gs_a.lo[1] = %d\n",gs_a.lo[1]);
            printf("(unlgn) gs_a.hi[1] = %d\n",gs_a.hi[1]);
            printf("(unlgn) ds_a.lo[0] = %d\n",ds_a.lo[0]);
            printf("(unlgn) ds_a.hi[0] = %d\n",ds_a.hi[0]);
            printf("(unlgn) ds_a.lo[1] = %d\n",ds_a.lo[1]);
            printf("(unlgn) ds_a.hi[1] = %d\n",ds_a.hi[1]);
            printf("(unlgn) ds_chunk.lo[0] = %d\n",ds_chunk.lo[0]);
            printf("(unlgn) ds_chunk.hi[0] = %d\n",ds_chunk.hi[0]);
            printf("(unlgn) ds_chunk.lo[1] = %d\n",ds_chunk.lo[1]);
            printf("(unlgn) ds_chunk.hi[1] = %d\n",ds_chunk.hi[1]); */
            /*ga_move(LOAD, transp, gs_a, ds_a, ds_unlg, buffer,
                chunk_ld[0]);*/
            nga_move(LOAD, transp, gs_a, ds_a, ds_unlg, buffer, chunk_ld);

            /* write entire updated buffer back to disk */
            ndai_put(ds_chunk, _dra_buffer, chunk_ld, &Requests[req].id);
            break;

          case DRA_OP_READ: 
            /* copy requested data from buffer to g_a */
            nga_move(STORE, transp, gs_a, ds_a, ds_unlg, buffer, chunk_ld);
            break;

          default:
            dai_error("dai_transfer_unlg: invalid opcode",(Integer)opcode);
        }

#       ifdef DEBUG
          fprintf(stderr,"%d transf unlg g[%d:%d,%d:%d]-d[%d:%d,%d:%d]\n",
          dai_io_nodeid(), gs_chunk.lo[0], gs_chunk.hi[0],
          gs_chunk.lo[1], gs_chunk.hi[1],
          ds_unlg.lo[0], ds_unlg.hi[0],
          ds_unlg.lo[1], ds_unlg.hi[1]);
#       endif
      }
    }
  }
}



/*\ write or read aligned subsections to disk 
\*/
void ndai_transfer_algn(int opcode, int transp, 
                        section_t ds_a, section_t gs_a, Integer req)
{
  Integer  next, chunk_ld[MAXDIM], ndim = ds_a.ndim;
  Integer i;
  section_t ds_chunk = ds_a;

  for(next = 0; next < Requests[req].na; next++){

    for (i=0; i<ndim; i++) ds_chunk.lo[i] = 0; /*initialize */
    while(ndai_next_chunk(req, Requests[req].list_algn[next], &ds_chunk)){
      if (dra_debug_flag && 0) { 
        printf("ndai_transfer_algn: Request %d\n",req);
        for (i=0; i<ndim; i++) {
          printf("ndai_transfer_algn: ds_chunk.lo[%d] = %d\n",i,ds_chunk.lo[i]);
          printf("ndai_transfer_algn: ds_chunk.hi[%d] = %d\n",i,ds_chunk.hi[i]);
        }
      }

      if(dai_myturn(ds_chunk)){

        dai_wait(req); /* needs free buffer to proceed */

        for (i=0; i<ndim-1; i++) chunk_ld[i] = ds_chunk.hi[i] - ds_chunk.lo[i] + 1;

        switch (opcode){

          case DRA_OP_WRITE:
            /* copy data from g_a to DRA buffer */
/*            printf("(algn) gs_a.lo[0] = %d\n",gs_a.lo[0]);
            printf("(algn) gs_a.hi[0] = %d\n",gs_a.hi[0]);
            printf("(algn) gs_a.lo[1] = %d\n",gs_a.lo[1]);
            printf("(algn) gs_a.hi[1] = %d\n",gs_a.hi[1]);
            printf("(algn) ds_a.lo[0] = %d\n",ds_a.lo[0]);
            printf("(algn) ds_a.hi[0] = %d\n",ds_a.hi[0]);
            printf("(algn) ds_a.lo[1] = %d\n",ds_a.lo[1]);
            printf("(algn) ds_a.hi[1] = %d\n",ds_a.hi[1]);
            printf("(algn) ds_chunk.lo[0] = %d\n",ds_chunk.lo[0]);
            printf("(algn) ds_chunk.hi[0] = %d\n",ds_chunk.hi[0]);
            printf("(algn) ds_chunk.lo[1] = %d\n",ds_chunk.lo[1]);
            printf("(algn) ds_chunk.hi[1] = %d\n",ds_chunk.hi[1]); */
            nga_move(LOAD, transp, gs_a, ds_a, ds_chunk, _dra_buffer, chunk_ld);
            /* ga_move(LOAD, transp, gs_a, ds_a, ds_chunk, _dra_buffer,
                chunk_ld[0]); */

            /* copy data from DRA buffer to disk */
            ndai_put(ds_chunk, _dra_buffer, chunk_ld, &Requests[req].id);
            /* dai_put(ds_chunk, _dra_buffer, chunk_ld[0],
               &Requests[req].id);*/
            break;

          case DRA_OP_READ:
            /* copy data from disk to DRA buffer */
            ndai_get(ds_chunk, _dra_buffer, chunk_ld, &Requests[req].id);
            elio_wait(&Requests[req].id);

            /* copy data from DRA buffer to g_a */
/*          ga_move(STORE, transp, gs_a, ds_a, ds_chunk, _dra_buffer, chunk_ld);*/
            dai_callback(STORE, transp, gs_a, ds_a, ds_chunk,chunk_ld,req);
            break;

          default:
            dai_error("dai_transfer_algn: invalid opcode",(Integer)opcode);
        }

#       ifdef DEBUG
          fprintf(stderr,"%d transf algn g[%d:%d,%d:%d]-d[%d:%d,%d:%d]\n",
                  dai_io_nodeid(), gs_chunk.lo[0], gs_chunk.hi[0],
                  gs_chunk.lo[1], gs_chunk.hi[1],
                  ds_chunk.lo[0], ds_chunk.hi[0],
                  ds_chunk.lo[1], ds_chunk.hi[1]);
#       endif
      }
    }
  }
}

/*\ WRITE SECTION g_a[glo:ghi] TO d_a[dlo:dhi]
\*/
Integer FATR ndra_write_section_(
        logical *transp,                   /*input:transpose operator*/
        Integer *g_a,                      /*input:GA handle*/ 
        Integer glo[],                     /*input*/
        Integer ghi[],                     /*input*/
        Integer *d_a,                      /*input:DRA handle*/ 
        Integer dlo[],                    /*input*/
        Integer dhi[],                    /*input*/
        Integer *request)                  /*output: async. request id*/ 
{
Integer gdims[MAXDIM], gtype, handle=*d_a+DRA_OFFSET;
Integer i, gelem, delem, ndim;
section_t d_sect, g_sect;
  
   ga_sync_();

   /* usual argument/type/range checking stuff */

   dai_check_handleM(*d_a,"ndra_write_sect");
   nga_inquire_internal_(g_a, &gtype, &ndim, gdims);
   if(!dai_write_allowed(*d_a))dai_error("ndra_write_sect: write not allowed",*d_a);
   if(DRA[handle].type != (int)gtype)dai_error("ndra_write_sect: type mismatch",gtype);
   if(DRA[handle].ndim != ndim)dai_error("ndra_write_sect: dimension mismatch", ndim);
   for (i=0; i<ndim; i++) dai_check_rangeM(glo[i], ghi[i], gdims[i],
       "ndra_write_sect: g_a dim error");
   for (i=0; i<ndim; i++) dai_check_rangeM(dlo[i], dhi[i], DRA[handle].dims[i],
       "ndra_write_sect: d_a dim error");

   /* check if numbers of elements in g_a & d_a sections match */
   gelem = 1;
   delem = 1;
   for (i=0; i<ndim; i++) {
     gelem *= (ghi[i]-glo[i]+1);
     delem *= (dhi[i]-dlo[i]+1);
   }
   if (gelem != delem)
     dai_error("ndra_write_sect: d_a and g_a sections do not match ", 0L);

   dai_assign_request_handle(request);

   /* decompose d_a section into aligned and unaligned subsections
    * -- with respect to underlying array layout on the disk
    */

   Requests[*request].nu=MAX_ALGN;    
   Requests[*request].na=MAX_UNLG;

   nfill_sectionM(d_sect, *d_a, DRA[handle].ndim, dlo, dhi); 
   nfill_sectionM(g_sect, *g_a, ndim, glo, ghi); 

   ndai_decomp_section(d_sect,
                     Requests[*request].list_algn, 
                    &Requests[*request].na,
                     Requests[*request].list_cover, 
                     Requests[*request].list_unlgn, 
                    &Requests[*request].nu);
/*   printf("Request %d\n",*request);
   printf("list_algn[0][0] = %d\n",Requests[*request].list_algn[0][0]);
   printf("list_algn[0][1] = %d\n",Requests[*request].list_algn[0][1]);
   printf("list_algn[0][2] = %d\n",Requests[*request].list_algn[0][2]);
   printf("list_algn[0][3] = %d\n",Requests[*request].list_algn[0][3]);
   for (i=0; i<Requests[*request].nu; i++) {
   printf("list_cover[%d][0] = %d\n",(int)i,Requests[*request].list_cover[i][0]);
   printf("list_cover[%d][1] = %d\n",(int)i,Requests[*request].list_cover[i][1]);
   printf("list_cover[%d][2] = %d\n",(int)i,Requests[*request].list_cover[i][2]);
   printf("list_cover[%d][3] = %d\n",(int)i,Requests[*request].list_cover[i][3]);
   printf("list_unlgn[%d][0] = %d\n",(int)i,Requests[*request].list_unlgn[i][0]);
   printf("list_unlgn[%d][1] = %d\n",(int)i,Requests[*request].list_unlgn[i][1]);
   printf("list_unlgn[%d][2] = %d\n",(int)i,Requests[*request].list_unlgn[i][2]);
   printf("list_unlgn[%d][3] = %d\n",(int)i,Requests[*request].list_unlgn[i][3]);
   }
   printf("Number of aligned blocks = %d\n",Requests[*request].na);
   printf("Number of unaligned blocks = %d\n",Requests[*request].nu); */
   _dra_turn = 0;

   /* process unaligned subsections */
   ndai_transfer_unlgn(DRA_OP_WRITE, (int)*transp, d_sect, g_sect, *request);
                  
   /* process aligned subsections */
   ndai_transfer_algn (DRA_OP_WRITE, (int)*transp, d_sect, g_sect, *request);

   ga_sync_();

   return(ELIO_OK);
}

/*\ WRITE N-dimensional g_a TO d_a
\*/
Integer FATR ndra_write_(
        Integer *g_a,                      /*input:GA handle*/
        Integer *d_a,                      /*input:DRA handle*/
        Integer *request)                  /*output: handle to async oper. */
{
Integer gdims[MAXDIM], gtype, handle=*d_a+DRA_OFFSET;
logical transp = FALSE;
Integer lo[MAXDIM], hi[MAXDIM], ndim, i;

        ga_sync_();

        /* usual argument/type/range checking stuff */

        dai_check_handleM(*d_a,"ndra_write");
        if( !dai_write_allowed(*d_a))
             dai_error("ndra_write: write not allowed to this array",*d_a);

        nga_inquire_internal_(g_a, &gtype, &ndim, gdims);
        if(DRA[handle].type != (int)gtype)dai_error("ndra_write: type mismatch",gtype);
        if(DRA[handle].ndim != ndim)dai_error("ndra_write: dimension mismatch",ndim);
        for (i=0; i<ndim; i++) {
          if(DRA[handle].dims[i] != gdims[i])
            dai_error("ndra_write: dims mismatch",gdims[i]);
        }

        /* right now, naive implementation just calls ndra_write_section */
        for (i=0; i<ndim; i++) {
          lo[i] = 1;
          hi[i] = DRA[handle].dims[i];
        }
        
        return(ndra_write_section_(&transp, g_a, lo, hi, d_a, lo, hi, request));
}

/*\ READ SECTION g_a[glo:ghi] FROM d_a[dlo:dhi]
\*/
Integer FATR ndra_read_section_(
        logical *transp,                   /*input:transpose operator*/
        Integer *g_a,                      /*input:GA handle*/ 
        Integer glo[],                     /*input*/
        Integer ghi[],                     /*input*/
        Integer *d_a,                      /*input:DRA handle*/ 
        Integer dlo[],                     /*input*/
        Integer dhi[],                     /*input*/
        Integer *request)                  /*output: request id*/ 
{
Integer gdims[MAXDIM], gtype, handle=*d_a+DRA_OFFSET;
Integer i, gelem, delem, ndim, me;
section_t d_sect, g_sect;
 
   ga_sync_();
   me = ga_nodeid_();

   /* usual argument/type/range checking stuff */
   dai_check_handleM(*d_a,"ndra_read_sect");
   if(!dai_read_allowed(*d_a))dai_error("ndra_read_sect: read not allowed",*d_a);
   nga_inquire_internal_(g_a, &gtype, &ndim, gdims);
   if(DRA[handle].type != (int)gtype)dai_error("ndra_read_sect: type mismatch",gtype);
   if(DRA[handle].ndim != ndim)dai_error("ndra_read_sect: dimension mismatch", ndim);
   for (i=0; i<ndim; i++) dai_check_rangeM(glo[i], ghi[i], gdims[i],
       "ndra_write_sect: g_a dim error");
   for (i=0; i<ndim; i++) dai_check_rangeM(dlo[i], dhi[i], DRA[handle].dims[i],
       "ndra_write_sect: d_a dim error");

   /* check if numbers of elements in g_a & d_a sections match */
   gelem = 1;
   delem = 1;
   for (i=0; i<ndim; i++) {
     gelem *= (ghi[i] - glo[i] + 1);
     delem *= (dhi[i] - dlo[i] + 1);
   }
   if (gelem != delem)
       dai_error("ndra_read_sect: d_a and g_a sections do not match ", 0L);
       
   dai_assign_request_handle(request);

   /* decompose d_a section into aligned and unaligned subsections
    * -- with respect to underlying array layout on the disk
    */

   Requests[*request].nu=MAX_ALGN;    
   Requests[*request].na=MAX_UNLG;

   if (dra_debug_flag) {
     for (i=0; i<ndim; i++) {
      /* printf("ndra_read_section: d_sect.lo[%d] = %d\n",i,d_sect.lo[i]);
       printf("ndra_read_section: d_sect.hi[%d] = %d\n",i,d_sect.hi[i]); */
       printf("proc[%d] ndra_read_section: dlo[%d] = %d\n",me,i,dlo[i]);
       printf("proc[%d] ndra_read_section: dhi[%d] = %d\n",me,i,dhi[i]);
     }
     for (i=0; i<ndim; i++) {
       printf("proc[%d] ndra_read_section: glo[%d] = %d\n",me,i,glo[i]);
       printf("proc[%d] ndra_read_section: ghi[%d] = %d\n",me,i,ghi[i]);
     }
   }

   nfill_sectionM(d_sect, *d_a, DRA[handle].ndim, dlo, dhi); 
   nfill_sectionM(g_sect, *g_a, ndim, glo, ghi); 

   ndai_decomp_section(d_sect,
                     Requests[*request].list_algn, 
                    &Requests[*request].na,
                     Requests[*request].list_cover, 
                     Requests[*request].list_unlgn, 
                    &Requests[*request].nu);

   _dra_turn = 0;
   if (dra_debug_flag && 0) {
     printf("ndra_read_section: Number of aligned sections %d\n",
         Requests[*request].na);
     printf("ndra_read_section: Number of unaligned sections %d\n",
         Requests[*request].nu);
     for (i=0; i<2*ndim; i++) {
       printf("ndra_read_section: list_algn[%d] =  %d\n",
           i,Requests[*request].list_algn[0][i]);
     }
     for (i=0; i<2*ndim; i++) {
       printf("ndra_read_section: list_cover[%d] =  %d\n",
           i,Requests[*request].list_cover[0][i]);
     }
     for (i=0; i<2*ndim; i++) {
       printf("ndra_read_section: list_unlgn[%d] =  %d\n",i,
           Requests[*request].list_unlgn[0][i]);
     } 
   }

   /* process unaligned subsections */
   ndai_transfer_unlgn(DRA_OP_READ, (int)*transp,  d_sect, g_sect, *request);
           
   /* process aligned subsections */
   ndai_transfer_algn (DRA_OP_READ, (int)*transp,  d_sect, g_sect, *request);

   return(ELIO_OK);
}

/*\ READ N-dimensional g_a FROM d_a
\*/
Integer FATR ndra_read_(Integer* g_a, Integer* d_a, Integer* request)
{
Integer gdims[MAXDIM], gtype, handle=*d_a+DRA_OFFSET;
logical transp = FALSE;
Integer lo[MAXDIM], hi[MAXDIM], ndim, i;

        ga_sync_();

        /* usual argument/type/range checking stuff */
        dai_check_handleM(*d_a,"ndra_read");
        if(!dai_read_allowed(*d_a))dai_error("ndra_read: read not allowed",*d_a);
        nga_inquire_internal_(g_a, &gtype, &ndim, gdims);
        if(DRA[handle].type != (int)gtype)dai_error("ndra_read: type mismatch",gtype);
        if(DRA[handle].ndim != ndim)dai_error("ndra_read: dimension mismatch",ndim);
        for (i=0; i<ndim; i++) {
          if(DRA[handle].dims[i] != gdims[i])
            dai_error("ndra_read: dims mismatch",gdims[i]);
        }

        /* right now, naive implementation just calls ndra_read_section */
        for (i=0; i<ndim; i++) {
          lo[i] = 1;
          hi[i] = DRA[handle].dims[i];
        }
        return(ndra_read_section_(&transp, g_a, lo, hi, d_a, lo, hi, request));
}

/*\ INQUIRE PARAMETERS OF EXISTING N-DIMENSIONAL DISK ARRAY
\*/
Integer ndra_inquire(
        Integer *d_a,                      /*input:DRA handle*/ 
        Integer *type,                     /*output*/
        Integer *ndim,                     /*output*/
        Integer dims[],                    /*output*/
        char    *name,                     /*output*/
        char    *filename)                 /*output*/
{
Integer handle=*d_a+DRA_OFFSET;

        dai_check_handleM(*d_a,"dra_inquire");

        *type = (Integer)DRA[handle].type;
        *ndim = DRA[handle].ndim;
        dims = DRA[handle].dims;
        strcpy(name, DRA[handle].name);
        strcpy(filename, DRA[handle].fname);
 
        return(ELIO_OK);
}

/*\ PRINT OUT INTERNAL PARAMETERS OF DRA
\*/
void FATR dra_print_internals_(Integer *d_a)
{
  Integer i;
  Integer *dims, *chunks;
  Integer handle = *d_a + DRA_OFFSET;
  Integer ndim = DRA[handle].ndim;
  Integer me = ga_nodeid_();
  dims = DRA[handle].dims;
  chunks = DRA[handle].chunk;
  if (me == 0) {
    printf("Internal Data for DRA: %s\n",DRA[handle].name);
    printf("  DRA Metafile Name: %s\n",DRA[handle].fname);
    switch(ga_type_c2f(DRA[handle].type)){
      case MT_F_DBL:
        printf("  DRA data type is DOUBLE PRECISION\n");
        break;
      case MT_F_REAL:
        printf("  DRA data type is SINGLE PRECISION\n");
        break;
      case MT_F_INT:
        printf("  DRA data type is INTEGER\n");
        break;
      case MT_F_DCPL:
        printf("  DRA data type is DOUBLE COMPLEX\n");
        break;
      default:
        printf("  DRA data type is UNKNOWN\n");
        break;
    }
    switch(DRA[handle].mode) {
      case DRA_RW:
        printf("  DRA access permisions are READ/WRITE\n");
        break;
      case DRA_W:
        printf("  DRA access permisions are WRITE ONLY\n");
        break;
      case DRA_R:
        printf("  DRA access permisions are READ ONLY\n");
        break;
      default:
        printf("  DRA access permisions are UNKNOWN\n");
        break;
    }
    printf("  Dimension of DRA: %d\n",(int)ndim);
    printf("  Dimensions of DRA:\n");
    for (i=0; i<ndim; i++) {
      printf("    Dimension in direction [%d]: %d\n",(int)(i+1),
             (int)dims[i]);
    }
    printf("  Chunk dimensions of DRA:\n");
    for (i=0; i<ndim; i++) {
      printf("    Chunk dimension in direction [%d]: %d\n",(int)(i+1),
             (int)chunks[i]);
    }
    if (DRA[handle].actv) {
      printf("  DRA is currently active\n");
    } else {
      printf("  DRA is not currently active\n");
    }
    if (DRA[handle].indep) {
      printf("  DRA is using independent files\n");
    } else {
      printf("  DRA is using shared files\n");
    }
  }
}

/*\ SET DEBUG FLAG FOR DRA OPERATIONS TO TRUE OR FALSE
\*/
void FATR dra_set_debug_(logical *flag)
{
  if (*flag) {
    dra_debug_flag = TRUE;
  } else {
    dra_debug_flag = FALSE;
  }
}
