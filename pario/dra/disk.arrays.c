/************************** DISK ARRAYS **************************************\
|*         Jarek Nieplocha, Fri May 12 11:26:38 PDT 1995                     *|
\*****************************************************************************/

#include "global.h"
#include "drap.h"
#include "dra.h"
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include "macommon.h"

/************************** constants ****************************************/

#define DRA_DBL_BUF_SIZE 100000 /*  buffer size --- reduce for debugging */

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
#elif defined(SP1)|| defined(SP)
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
int             _idx_buffer, _handle_buffer;
#endif

disk_array_t *DRA;           /* array of struct for basic info about DRA arrays*/
Integer _max_disk_array;    /* max number of disk arrays open at a time      */


request_t     Requests[MAX_REQ];
int num_pending_requests=0;
Integer _dra_turn=0;
static int     Dra_num_serv=DRA_NUM_IOPROCS;
 
/****************************** Macros ***************************************/

#define dai_sizeofM(_type)    ((_type)==MT_F_DBL? sizeof(DoublePrecision):\
                               (_type)==MT_F_INT? sizeof(Integer): \
                                                  sizeof(DoubleComplex))

#define dai_check_typeM(_type)  if (_type != MT_F_DBL && _type != MT_F_INT)\
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
        if(_lo < 1   || _lo > _dim ||_hi < _lo || _hi > _dim)             \
        dai_error(_err_msg, _dim)
 

#define dai_dest_indicesM(is,js, ilos,jlos, lds, id,jd, ilod, jlod, ldd)   \
{ \
    Integer _index_;\
    _index_ = (lds)*((js)-(jlos)) + (is)-(ilos);\
    *(id) = (_index_)%(ldd) + (ilod);\
    *(jd) = (_index_)/(ldd) + (jlod);\
}

#define ga_get_sectM(sect, _buf, _ld)\
   ga_get_(&sect.handle, &sect.ilo, &sect.ihi, &sect.jlo, &sect.jhi, _buf, &_ld)

#define ga_put_sectM(sect, _buf, _ld)\
   ga_put_(&sect.handle, &sect.ilo, &sect.ihi, &sect.jlo, &sect.jhi, _buf, &_ld)

#define fill_sectionM(sect, _hndl, _ilo, _ihi, _jlo, _jhi) \
{ \
        sect.handle = _hndl;\
        sect.ilo    = _ilo;\
        sect.ihi    = _ihi;\
        sect.jlo    = _jlo;\
        sect.jhi    = _jhi;\
}

#define sect_to_blockM(ds_a, CR)\
{\
      Integer   hndl = (ds_a).handle+DRA_OFFSET;\
      Integer   br   = ((ds_a).ilo-1)/DRA[hndl].chunk1;\
      Integer   bc   = ((ds_a).jlo-1)/DRA[hndl].chunk2;\
      Integer   R    = (DRA[hndl].dim1 + DRA[hndl].chunk1 -1)/DRA[hndl].chunk1;\
               *(CR) = bc * R + br;\
}

#define block_to_sectM(ds_a, CR)\
{\
      Integer   hndl = (ds_a)->handle+DRA_OFFSET;\
      Integer   R    = (DRA[hndl].dim1 + DRA[hndl].chunk1 -1)/DRA[hndl].chunk1;\
      Integer   br = (CR)%R;\
      Integer   bc = ((CR) - br)/R;\
      (ds_a)->  ilo= br * DRA[hndl].chunk1 +1;\
      (ds_a)->  jlo= bc * DRA[hndl].chunk2 +1;\
      (ds_a)->  ihi= (ds_a)->ilo + DRA[hndl].chunk1;\
      (ds_a)->  jhi= (ds_a)->jlo + DRA[hndl].chunk2;\
      if( (ds_a)->ihi > DRA[hndl].dim1) (ds_a)->ihi = DRA[hndl].dim1;\
      if( (ds_a)->jhi > DRA[hndl].dim2) (ds_a)->jhi = DRA[hndl].dim2;\
}
      
#define INDEPFILES(x) (DRA[(x)+DRA_OFFSET].indep)

char dummy_fname[DRA_MAX_FNAME];
/*****************************************************************************/


/*#define DEBUG 1*/
/*#define CLEAR_BUF 1*/


/*\ determines if write operation to a disk array is allowed
\*/
Integer dai_write_allowed(Integer d_a)
{
Integer handle = d_a+DRA_OFFSET;
        if(DRA[handle].mode == DRA_W || DRA[handle].mode == DRA_RW) return 1;
        else return 0;
}


/*\ determines if read operation from a disk array is allowed
\*/
Integer dai_read_allowed(Integer d_a)
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
        num = (INDEPFILES(d_a)) ? INFINITE_NUM_PROCS: DRA_NUM_IOPROCS; 
        return( MIN( ga_nnodes_(), num));
}


/*\  rank of calling process in group of processes that could perform I/O
 *   a negative value means that this process doesn't do I/O
\*/
Integer dai_io_nodeid(Integer d_a)
{
Integer me = ga_nodeid_();

       /* again, one of many possibilities: 
        * if proc id beyond I/O procs number, negate it
        */
        if(me >= dai_io_procs(d_a)) me = -me;
        return (me);
}


/*\ determines if I/O process participates in file management (create/delete)
\*/
Integer dai_io_manage(d_a)
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
                  section_t ds_chunk, Integer ld, Integer req)
{
        if(Requests[req].callback==ON) dai_error("DRA: callback not cleared",0);
        Requests[req].callback = ON;
        Requests[req].args.op = op;
        Requests[req].args.transp = transp;
        Requests[req].args.ld = ld;
        Requests[req].args.gs_a = gs_a;
        Requests[req].args.ds_a = ds_a;
        Requests[req].args.ds_chunk = ds_chunk;
}




/*\ INITIALIZE DISK ARRAY DATA STRUCTURES
\*/
Integer dra_init_(max_arrays, max_array_size, tot_disk_space, max_memory)
        Integer *max_arrays;              /* input */
        DoublePrecision *max_array_size;  /* input */
        DoublePrecision *tot_disk_space;  /* input */
        DoublePrecision *max_memory;      /* input */
{
#define DEF_MAX_ARRAYS 16
#define MAX_ARRAYS 1024
int i;
        ga_sync_();

        if(*max_arrays<-1 || *max_arrays> MAX_ARRAYS)
           dai_error("dra_init: incorrect max number of arrays",*max_arrays);
        _max_disk_array = (*max_arrays==-1) ? DEF_MAX_ARRAYS: *max_arrays;

        Dra_num_serv = drai_get_num_serv();

        DRA = (disk_array_t*)malloc(sizeof(disk_array_t)**max_arrays);
        if(!DRA) dai_error("dra_init: memory alocation failed\n",0);
        for(i=0; i<_max_disk_array ; i++)DRA[i].actv=0;

        for(i=0; i<MAX_REQ; i++)Requests[i].num_pending=0;

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
{
Integer patch_size;
  
        *chunk1 = *chunk2 =0; 
        if(block1 <= 0 && block2 <= 0){

          *chunk1 = dim1;
          *chunk2 = dim2;

        }else if(block1 <= 0){
          *chunk2 = block2;
          *chunk1 = MAX(1, DRA_BUF_SIZE/(elem_size**chunk2));
        }else if(block2 <= 0){
          *chunk1 = block1;
          *chunk2 = MAX(1, DRA_BUF_SIZE/(elem_size**chunk1));
        }else{
          *chunk1 = block1;
          *chunk2 = block2;
        }

        /* need to correct chunk size to fit chunk1 x chunk2 request in buffer*/
        patch_size = *chunk1* *chunk2*elem_size;
          
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
void dai_file_location(section_t ds_a, off_t* offset)
{
Integer row_blocks, handle=ds_a.handle+DRA_OFFSET, offelem, cur_ld, part_chunk1;

        if((ds_a.ilo-1)%DRA[handle].chunk1)
            dai_error("dai_file_location: not alligned ??",ds_a.ilo);

        row_blocks  = (ds_a.ilo-1)/DRA[handle].chunk1;/* # row blocks from top*/
        part_chunk1 = DRA[handle].dim1%DRA[handle].chunk1;/*dim1 in part block*/
        cur_ld      = (row_blocks == DRA[handle].dim1 / DRA[handle].chunk1) ? 
                       part_chunk1: DRA[handle].chunk1;

        /* compute offset (in elements) */

        if(INDEPFILES(ds_a.handle)) {

           Integer   CR, R; 
           Integer   i, num_part_block = 0;
           Integer   ioprocs=dai_io_procs(ds_a.handle); 
           Integer   iome = dai_io_nodeid(ds_a.handle);
           
           sect_to_blockM(ds_a, &CR); 

           R    = (DRA[handle].dim1 + DRA[handle].chunk1 -1)/DRA[handle].chunk1;
           for(i = R -1; i< CR; i+=R) if(i%ioprocs == iome)num_part_block++;

           if(!part_chunk1) part_chunk1=DRA[handle].chunk1;
           offelem = ((CR/ioprocs - num_part_block)*DRA[handle].chunk1 +
                     num_part_block * part_chunk1 ) * DRA[handle].chunk2;

           /* add offset within block */
           offelem += ((ds_a.jlo-1) %DRA[handle].chunk2)*cur_ld; 
        } else {

           offelem = row_blocks  * DRA[handle].dim2 * DRA[handle].chunk1;
           offelem += (ds_a.jlo -1)*cur_ld;

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
off_t   offset;
Size_t  bytes;

        /* find location in a file where data should be written */
        dai_file_location(ds_a, &offset);
        
        if((ds_a.ihi - ds_a.ilo + 1) != ld) dai_error("dai_put: bad ld",ld); 

        /* since everything is aligned, write data to disk */
        elem = (ds_a.ihi - ds_a.ilo + 1) * (ds_a.jhi - ds_a.jlo + 1);
        bytes= (Size_t) elem * dai_sizeofM(DRA[handle].type);
        if( ELIO_OK != elio_awrite(DRA[handle].fd, offset, buf, bytes, id ))
                       dai_error("dai_put failed", ds_a.handle);
}



/*\ write zero at EOF
\*/
void dai_zero_eof(Integer d_a)
{
Integer handle = d_a+DRA_OFFSET, nelem;
off_t offset;
Size_t  bytes;

        if(DRA[handle].type == MT_F_DBL)*(DoublePrecision*)_dra_buffer = 0.;
        if(DRA[handle].type == MT_F_INT)*(Integer*)_dra_buffer = 0;

        if(INDEPFILES(d_a)) {

          Integer   CR, i, nblocks; 
          section_t ds_a;
          Integer   ioprocs=dai_io_procs(d_a); 
          Integer   iome = dai_io_nodeid(d_a);

          /* total number of blocks in the array */
          nblocks = ((DRA[handle].dim1 + DRA[handle].chunk1-1)/DRA[handle].chunk1)
                  * ((DRA[handle].dim2 + DRA[handle].chunk2-1)/DRA[handle].chunk2);
          fill_sectionM(ds_a, d_a, 0, 0, 0, 0); 

          /* search for the last block for each I/O processor */
          for(i = 0; i <ioprocs; i++){
             CR = nblocks - 1 -i;
             if(CR % ioprocs == iome) break;
          }
          if(CR<0) return; /* no blocks owned */

          block_to_sectM(&ds_a, CR); /* convert block number to section */
          dai_file_location(ds_a, &offset);
          nelem = (ds_a.ihi - ds_a.ilo +1)*(ds_a.jhi - ds_a.jlo +1) -1; 
          offset += nelem * dai_sizeofM(DRA[handle].type);

#         ifdef DEBUG
            printf("me=%d zeroing EOF (%d) at %ld bytes \n",iome,CR,offset);
#         endif
        } else {

          nelem = DRA[handle].dim1*DRA[handle].dim2 - 1;
          offset = nelem * dai_sizeofM(DRA[handle].type);
        }

        bytes = dai_sizeofM(DRA[handle].type);
        if(bytes != elio_write(DRA[handle].fd, offset, _dra_buffer, bytes))
                     dai_error("dai_zero_eof: write error ",0);
}



/*\ read aligned block of data from d_a to memory buffer
\*/
void dai_get(section_t ds_a, Void *buf, Integer ld, io_request_t *id)
{
Integer handle = ds_a.handle + DRA_OFFSET, elem;
off_t   offset;
Size_t  bytes;
void    dai_clear_buffer();

        /* find location in a file where data should be read from */
        dai_file_location(ds_a, &offset);

#       ifdef CLEAR_BUF
          dai_clear_buffer();
#       endif

        if((ds_a.ihi - ds_a.ilo + 1) != ld) dai_error("dai_get: bad ld",ld); 
        /* since everything is aligned, read data from disk */
        elem = (ds_a.ihi - ds_a.ilo + 1) * (ds_a.jhi - ds_a.jlo + 1);
        bytes= (Size_t) elem * dai_sizeofM(DRA[handle].type);
        if( ELIO_OK != elio_aread(DRA[handle].fd, offset, buf, bytes, id ))
                       dai_error("dai_get failed", ds_a.handle);
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
Integer dra_create(type, dim1, dim2, name, filename, mode, reqdim1, reqdim2, d_a)
        Integer *d_a;                      /*input:DRA handle*/
        Integer *type;                     /*input*/
        Integer *dim1;                     /*input*/
        Integer *dim2;                     /*input*/
        Integer *reqdim1;                  /*input: dim1 of typical request*/
        Integer *reqdim2;                  /*input: dim2 of typical request*/
        Integer *mode;                     /*input*/
        char    *name;                     /*input*/
        char    *filename;                 /*input*/
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
                    &DRA[handle].chunk1, &DRA[handle].chunk2);

       /* determine layout -- by row or column */
        DRA[handle].layout = COLUMN;

       /* complete initialization */
        DRA[handle].dim1 = *dim1;
        DRA[handle].dim2 = *dim2;
        DRA[handle].type = *type;
        DRA[handle].mode = *mode;
        strncpy (DRA[handle].fname, filename,  DRA_MAX_FNAME);
        strncpy(DRA[handle].name, name, DRA_MAX_NAME );

        dai_write_param(DRA[handle].fname, *d_a);      /* create param file */
        DRA[handle].indep = dai_file_config(filename); /*check file configuration*/

        /* create file */
        if(dai_io_manage(*d_a)){ 

           if (INDEPFILES(*d_a)) {

             sprintf(dummy_fname,"%s%d",DRA[handle].fname,dai_io_nodeid(*d_a));
             DRA[handle].fd = elio_open(dummy_fname,*mode, ELIO_PRIVATE);

           }else{

              /* collective open supported only on Paragon */
#             ifdef PARAGON
                 DRA[handle].fd = elio_gopen(DRA[handle].fname,*mode); 
#             else
                 DRA[handle].fd = elio_open(DRA[handle].fname,*mode, ELIO_SHARED); 
#             endif
           }

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
        if(dai_io_nodeid(*d_a)==0)printf("chunking: %d x %d\n",DRA[handle].chunk1,
                                                          DRA[handle].chunk2);

        ga_sync_();

        return(ELIO_OK);
}
     
 

/*\ OPEN AN ARRAY THAT EXISTS ON THE DISK
\*/
Integer dra_open(filename, mode, d_a)
        char *filename;                  /* input  */
        Integer *mode;                   /*input*/
        Integer *d_a;                    /* output */
{
Integer handle;

        ga_sync_();

       /*** Get next free DRA handle ***/
        if( (handle = dai_get_handle()) == -1)
             dai_error("dra_open: too many disk arrays ", _max_disk_array);
        *d_a = handle - DRA_OFFSET;

        DRA[handle].mode = *mode;
        strncpy (DRA[handle].fname, filename,  DRA_MAX_FNAME);

        dai_read_param(DRA[handle].fname, *d_a);
        DRA[handle].indep = dai_file_config(filename); /*check file configuration*/

        if(dai_io_manage(*d_a)){ 

           if (INDEPFILES(*d_a)) {

             sprintf(dummy_fname,"%s%d",DRA[handle].fname,dai_io_nodeid(*d_a));
             DRA[handle].fd = elio_open(dummy_fname,*mode, ELIO_PRIVATE);

           }else{

              /* collective open supported only on Paragon */
#             ifdef PARAGON
                 DRA[handle].fd = elio_gopen(DRA[handle].fname,*mode);
#             else
                 DRA[handle].fd = elio_open(DRA[handle].fname,*mode, ELIO_SHARED);
#             endif
           }

           if(DRA[handle].fd->fd ==-1) dai_error("dra_open failed",ga_nodeid_());  
        }


#       ifdef DEBUG
             printf("\n%d:OPEN chunking=(%d,%d) type=%d buf=%d\n",
                   ga_nodeid_(),DRA[handle].chunk1, DRA[handle].chunk2, 
                   DRA[handle].type, DRA_DBL_BUF_SIZE);
             fflush(stdout);
#       endif

        ga_sync_();

        return(ELIO_OK);
}



/*\ CLOSE AN ARRAY AND SAVE IT ON THE DISK
\*/
Integer dra_close_(Integer* d_a) /* input:DRA handle*/ 
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
/*ds_a, aligned, na, cover,unaligned,nu)*/
        section_t ds_a,
        Integer aligned[][4], 
        int *na,
        Integer cover[][4],
        Integer unaligned[][4], 
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
        
        aligned[a][ ILO ] = ds_a.ilo; aligned[a][ IHI ] = ds_a.ihi;
        aligned[a][ JLO ] = ds_a.jlo; aligned[a][ JHI ] = ds_a.jhi;

        switch   (DRA[handle].layout){
        case COLUMN : /* need to check row alignment only */
                 
                 algn_flag = ON; /* has at least one aligned subsection */

                 /* top of section */
                 off = (ds_a.ilo -1) % DRA[handle].chunk1; 
                 if(off){ 

                        if(MAX_UNLG<= u) 
                           dai_error("dai_decomp_sect:insufficient nu",u);

                        chunk_units = (ds_a.ilo -1) / DRA[handle].chunk1;
                        
                        cover[u][ ILO ] = chunk_units*DRA[handle].chunk1 + 1;
                        cover[u][ IHI ] = MIN(cover[u][ ILO ] + 
                                          DRA[handle].chunk1-1, DRA[handle].dim1);

                        unaligned[u][ ILO ] = ds_a.ilo;
                        unaligned[u][ IHI ] = MIN(ds_a.ihi,cover[u][ IHI ]);
                        unaligned[u][ JLO ] = cover[u][ JLO ] = ds_a.jlo;
                        unaligned[u][ JHI ] = cover[u][ JHI ] = ds_a.jhi;

                        if(cover[u][ IHI ] < ds_a.ihi){
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
                 off = ds_a.ihi % DRA[handle].chunk1; 
                 if(off && (ds_a.ihi != DRA[handle].dim1) && (algn_flag == ON)){

                        if(MAX_UNLG<=u) 
                           dai_error("dai_decomp_sect:insufficient nu",u); 
                        chunk_units = ds_a.ihi / DRA[handle].chunk1;
 
                        cover[u][ ILO ] = chunk_units*DRA[handle].chunk1 + 1;
                        cover[u][ IHI ] = MIN(cover[u][ ILO ] +
                                          DRA[handle].chunk1-1, DRA[handle].dim1);

                        unaligned[u][ ILO ] = cover[u][ ILO ];
                        unaligned[u][ IHI ] = ds_a.ihi;
                        unaligned[u][ JLO ] = cover[u][ JLO ] = ds_a.jlo;
                        unaligned[u][ JHI ] = cover[u][ JHI ] = ds_a.jhi;

                        aligned[a][ IHI ] = MAX(1,unaligned[u][ ILO ]-1);
                        algn_flag=(DRA[handle].chunk1 == DRA[handle].dim1)?OFF:ON;
                        u++;
                 }
                 *nu = u;
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
Integer dra_write_(g_a, d_a, request)
        Integer *g_a;                      /*input:GA handle*/
        Integer *d_a;                      /*input:DRA handle*/
        Integer *request;                  /*output: handle to async oper. */
{
Integer gdim1, gdim2, gtype, handle=*d_a+DRA_OFFSET;
logical transp = FALSE;
Integer ilo, ihi, jlo, jhi;

        ga_sync_();

        /* usual argument/type/range checking stuff */

        dai_check_handleM(*d_a,"dra_write");
        if( !dai_write_allowed(*d_a))
             dai_error("dra_write: write not allowed to this array",*d_a);

        ga_inquire_(g_a, &gtype, &gdim1, &gdim2);
        if(DRA[handle].type != gtype)dai_error("dra_write: type mismatch",gtype);
        if(DRA[handle].dim1 != gdim1)dai_error("dra_write: dim1 mismatch",gdim1);
        if(DRA[handle].dim2 != gdim2)dai_error("dra_write: dim2 mismatch",gdim2);

        /* right now, naive implementation just calls dra_write_section */
        ilo = 1; ihi = DRA[handle].dim1; 
        jlo = 1; jhi = DRA[handle].dim2; 
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
      if(ds_chunk->jlo && DRA[handle].chunk2>1) 
         ds_chunk->jlo -= (ds_chunk->jlo -1) % DRA[handle].chunk2;
    
    retval = dai_next2d(&ds_chunk->ilo, list[ ILO ], list[ IHI ],
                                        DRA[handle].chunk1,
                        &ds_chunk->jlo, list[ JLO ], list[ JHI ],
                                        DRA[handle].chunk2);
    if(!retval) return(retval);

    ds_chunk->ihi = MIN(list[ IHI ], ds_chunk->ilo + DRA[handle].chunk1 -1);
    ds_chunk->jhi = MIN(list[ JHI ], ds_chunk->jlo + DRA[handle].chunk2 -1);

    if(INDEPFILES(ds_chunk->handle)) { 
         Integer jhi_temp =  ds_chunk->jlo + DRA[handle].chunk2 -1;
         jhi_temp -= jhi_temp % DRA[handle].chunk2;
         ds_chunk->jhi = MIN(ds_chunk->jhi, jhi_temp); 

         /*this line was absent from older version on bonnie that worked */
         if(ds_chunk->jlo < list[ JLO ]) ds_chunk->jlo = list[ JLO ]; 
    }

    return 1;
}


int dai_myturn(section_t ds_chunk)
{
Integer   ioprocs = dai_io_procs(ds_chunk.handle); 
Integer   iome    = dai_io_nodeid(ds_chunk.handle);
    
    if(INDEPFILES(ds_chunk.handle)){

      /* compute cardinal number for the current chunk */
      sect_to_blockM(ds_chunk, &_dra_turn);

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
   printf("\n ld=%d rows=%d cols=%d\n",ld,rows,cols);
 
   for (i=0; i<rows; i++){
   for (j=0; j<cols; j++)
   printf("%f ", buf[j*ld+i]);
   printf("\n");
   }
}

static Integer mode_move=0;
void dra_set_mode_(Integer* val)
{
  mode_move = *val;
}

void ga_move(int op, int trans, section_t gs_a, section_t ds_a, 
             section_t ds_chunk, void* buffer, Integer ldb)
{
    if(!trans && (gs_a.ilo- gs_a.ihi ==  ds_a.ilo- ds_a.ihi) ){
        /*** straight copy possible if there's no reshaping or transpose ***/

        /* determine gs_chunk corresponding to ds_chunk */
        section_t gs_chunk = gs_a;
        dai_dest_indicesM(ds_chunk.ilo, ds_chunk.jlo,   ds_a.ilo, ds_a.jlo, 
                ds_a.ihi-ds_a.ilo+1, &gs_chunk.ilo, &gs_chunk.jlo, 
                gs_a.ilo, gs_a.jlo,   gs_a.ihi -     gs_a.ilo + 1);
        dai_dest_indicesM(ds_chunk.ihi, ds_chunk.jhi,   ds_a.ilo,ds_a.jlo, 
                ds_a.ihi-ds_a.ilo+1, &gs_chunk.ihi, &gs_chunk.jhi,
                gs_a.ilo, gs_a.jlo,   gs_a.ihi -     gs_a.ilo + 1);

        /* move data */
        if(op==LOAD) ga_get_sectM(gs_chunk, buffer, ldb);
        else         ga_put_sectM(gs_chunk, buffer, ldb);

    }else{
        /** due to generality of this transformation scatter/gather is required **/

         Integer ihandle, jhandle, vhandle, iindex, jindex, vindex;
         Integer pindex, phandle;
         Integer type = DRA[ds_a.handle+DRA_OFFSET].type, nelem;
         Integer i, j, ii, jj, base;  
         char    *base_addr;

#        define ITERATOR_2D(i,j, base, ds_chunk)\
                for(j = ds_chunk.jlo, base=0, jj=0; j<= ds_chunk.jhi; j++,jj++)\
                  for(i = ds_chunk.ilo, ii=0; i<= ds_chunk.ihi; i++,ii++,base++)

#        define COPY_SCATTER(ADDR_BASE, TYPE, ds_chunk)\
		ITERATOR_2D(i,j, base, ds_chunk) \
		ADDR_BASE[base+vindex] = ((TYPE*)buffer)[ldb*jj + ii]

#        define COPY_GATHER(ADDR_BASE, TYPE, ds_chunk)\
                for(i=0; i< nelem; i++){\
                   Integer ldc = ds_chunk.ihi - ds_chunk.ilo+1;\
                   base = INT_MB[pindex+i]; jj = base/ldc; ii = base%ldc;\
                   ((TYPE*)buffer)[ldb*jj + ii] = ADDR_BASE[i+vindex];\
                }

#        define COPY_TYPE(OPERATION, MATYPE, ds_chunk)\
         switch(MATYPE){\
         case MT_F_DBL: COPY_ ## OPERATION(DBL_MB,DoublePrecision,ds_chunk);break;\
         case MT_F_INT: COPY_ ## OPERATION(INT_MB, Integer, ds_chunk); break;\
         case MT_F_DCPL:COPY_ ## OPERATION(DCPL_MB, DoubleComplex, ds_chunk);\
         }

         if(ga_nodeid_()==0) printf("DRA warning: using scatter/gather\n");

         nelem = (ds_chunk.ihi-ds_chunk.ilo+1)*(ds_chunk.jhi-ds_chunk.jlo+1);
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
              case  MT_F_DCPL: base_addr = (char*) (DCPL_MB+vindex);
         }
    
         if(trans==TRANS) 
           ITERATOR_2D(i,j, base, ds_chunk) {
              dai_dest_indicesM(j, i, ds_a.ilo, ds_a.jlo,  ds_a.ihi-ds_a.ilo+1, 
                                INT_MB+base+iindex, INT_MB+base+jindex,
                                gs_a.ilo, gs_a.jlo,  gs_a.ihi -  gs_a.ilo + 1);
           }
         else
           ITERATOR_2D(i,j, base, ds_chunk) {
              dai_dest_indicesM(i, j, ds_a.ilo, ds_a.jlo,  ds_a.ihi-ds_a.ilo+1, 
                                INT_MB+base+iindex, INT_MB+base+jindex,
                                gs_a.ilo, gs_a.jlo,  gs_a.ihi -  gs_a.ilo + 1);
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



/*\  executes callback function associated with completion of asynch. I/O
\*/
void dai_exec_callback(request_t *request)
{
args_t   *arg;

        if(request->callback==OFF)return;
        request->callback = OFF;
        arg = &request->args;
        ga_move(arg->op, arg->transp, arg->gs_a, arg->ds_a, arg->ds_chunk,
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
Integer   type = DRA[ds_a.handle+DRA_OFFSET].type;
section_t ds_chunk, ds_unlg;
char      *buffer; 

   ds_chunk =  ds_unlg = ds_a;

   for(next = 0; next < Requests[req].nu; next++){

      ds_chunk.ilo = ds_chunk.jlo = 0; /* init */
      while(dai_next_chunk(req, Requests[req].list_cover[next],&ds_chunk)){

          if(dai_myturn(ds_chunk)){

              dai_wait(req); /* needs free buffer to proceed */

             /*find corresponding to chunk of 'cover' unaligned sub-subsection*/
              ds_unlg.ilo = Requests[req].list_unlgn[next][ ILO ];
              ds_unlg.ihi = Requests[req].list_unlgn[next][ IHI ];
              ds_unlg.jlo = Requests[req].list_unlgn[next][ JLO ];
              ds_unlg.jhi = Requests[req].list_unlgn[next][ JHI ];

              if(!dai_section_intersect(ds_chunk, &ds_unlg))
                  dai_error("dai_transfer_unlgn: inconsistent cover",0);

             /* copy data from disk to DRA buffer */
              chunk_ld =  ds_chunk.ihi - ds_chunk.ilo + 1;
              dai_get(ds_chunk, _dra_buffer, chunk_ld, &Requests[req].id);
              elio_wait(&Requests[req].id); 

             /* determine location in the buffer where GA data should be */
              offset  = (ds_unlg.jlo - ds_chunk.jlo)*chunk_ld + 
                         ds_unlg.ilo - ds_chunk.ilo;
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
                   dai_io_nodeid(), gs_chunk.ilo, gs_chunk.ihi, gs_chunk.jlo, 
                   gs_chunk.jhi,
                   ds_unlg.ilo,  ds_unlg.ihi,  ds_unlg.jlo,  ds_unlg.jhi);
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
Integer   next, chunk_ld;
section_t ds_chunk = ds_a;

   for(next = 0; next < Requests[req].na; next++){

      ds_chunk.ilo = ds_chunk.jlo = 0; /* init */
      while(dai_next_chunk(req, Requests[req].list_algn[next], &ds_chunk)){

          if(dai_myturn(ds_chunk)){

              dai_wait(req); /* needs free buffer to proceed */

              chunk_ld =  ds_chunk.ihi - ds_chunk.ilo + 1;

              switch (opcode){

              case DRA_OP_WRITE:
                 /* copy data from g_a to DRA buffer */
                 ga_move(LOAD, transp, gs_a, ds_a, ds_chunk, _dra_buffer, chunk_ld);

                 /* copy data from DRA buffer to disk */
                 dai_put(ds_chunk, _dra_buffer, chunk_ld, &Requests[req].id);
                 break;

              case DRA_OP_READ:
                 /* copy data from disk to DRA buffer */
                 dai_get(ds_chunk, _dra_buffer, chunk_ld, &Requests[req].id);
                 elio_wait(&Requests[req].id);

                 /* copy data from DRA buffer to g_a */
/*                 ga_move(STORE, transp, gs_a, ds_a, ds_chunk, _dra_buffer, chunk_ld);*/
                 dai_callback(STORE, transp, gs_a, ds_a, ds_chunk,chunk_ld,req);
                 break;

              default:

                 dai_error("dai_transfer_algn: invalid opcode",(Integer)opcode);
              }

#             ifdef DEBUG
                fprintf(stderr,"%d transf algn g[%d:%d,%d:%d]-d[%d:%d,%d:%d]\n",
                   dai_io_nodeid(), gs_chunk.ilo, gs_chunk.ihi, gs_chunk.jlo, 
                   gs_chunk.jhi,
                   ds_chunk.ilo,  ds_chunk.ihi,  ds_chunk.jlo,  ds_chunk.jhi);
#             endif
          }
      }
   }
}



/*\ WRITE SECTION g_a[gilo:gihi, gjlo:gjhi] TO d_a[dilo:dihi, djlo:djhi]
\*/
Integer dra_write_section_(transp, g_a, gilo, gihi, gjlo, gjhi,
                                  d_a, dilo, dihi, djlo, djhi, request)
        logical *transp;                   /*input:transpose operator*/
        Integer *g_a;                      /*input:GA handle*/ 
        Integer *d_a;                      /*input:DRA handle*/ 
        Integer *gilo;                     /*input*/
        Integer *gihi;                     /*input*/
        Integer *gjlo;                     /*input*/
        Integer *gjhi;                     /*input*/
        Integer *dilo;                     /*input*/
        Integer *dihi;                     /*input*/
        Integer *djlo;                     /*input*/
        Integer *djhi;                     /*input*/
        Integer *request;                  /*output: async. request id*/ 
{
Integer gdim1, gdim2, gtype, handle=*d_a+DRA_OFFSET;
section_t d_sect, g_sect;
  
   ga_sync_();

   /* usual argument/type/range checking stuff */

   dai_check_handleM(*d_a,"dra_write_sect");
   ga_inquire_(g_a, &gtype, &gdim1, &gdim2);
   if(!dai_write_allowed(*d_a))dai_error("dra_write_sect: write not allowed",*d_a);
   if(DRA[handle].type != gtype)dai_error("dra_write_sect: type mismatch",gtype);
   dai_check_rangeM(*gilo,*gihi, gdim1, "dra_write_sect: g_a dim1 error");
   dai_check_rangeM(*gjlo,*gjhi, gdim2, "dra_write_sect: g_a dim2 error");
   dai_check_rangeM(*dilo,*dihi,DRA[handle].dim1,"dra_write_sect:d_a dim1 error");
   dai_check_rangeM(*djlo,*djhi,DRA[handle].dim2,"dra_write_sect:d_a dim2 error");

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
Integer dra_read_section_(transp, g_a, gilo, gihi, gjlo, gjhi,
                                 d_a, dilo, dihi, djlo, djhi, request)
        logical *transp;                   /*input:transpose operator*/
        Integer *g_a;                      /*input:GA handle*/ 
        Integer *d_a;                      /*input:DRA handle*/ 
        Integer *gilo;                     /*input*/
        Integer *gihi;                     /*input*/
        Integer *gjlo;                     /*input*/
        Integer *gjhi;                     /*input*/
        Integer *dilo;                     /*input*/
        Integer *dihi;                     /*input*/
        Integer *djlo;                     /*input*/
        Integer *djhi;                     /*input*/
        Integer *request;               /*output: request id*/ 
{
Integer gdim1, gdim2, gtype, handle=*d_a+DRA_OFFSET;
section_t d_sect, g_sect;
 
   ga_sync_();

   /* usual argument/type/range checking stuff */
   dai_check_handleM(*d_a,"dra_read_sect");
   if(!dai_read_allowed(*d_a))dai_error("dra_read_sect: read not allowed",*d_a);
   ga_inquire_(g_a, &gtype, &gdim1, &gdim2);
   if(DRA[handle].type != gtype)dai_error("dra_read_sect: type mismatch",gtype);
   dai_check_rangeM(*gilo, *gihi, gdim1, "dra_read_sect: g_a dim1 error");
   dai_check_rangeM(*gjlo, *gjhi, gdim2, "dra_read_sect: g_a dim2 error");
   dai_check_rangeM(*dilo, *dihi,DRA[handle].dim1,"dra_read_sect:d_a dim1 error");
   dai_check_rangeM(*djlo, *djhi,DRA[handle].dim2,"dra_read_sect:d_a dim2 error");

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
Integer dra_read_(Integer* g_a, Integer* d_a, Integer* request)
{
Integer gdim1, gdim2, gtype, handle=*d_a+DRA_OFFSET;
logical transp = FALSE;
Integer ilo, ihi, jlo, jhi;

        ga_sync_();

        /* usual argument/type/range checking stuff */
        dai_check_handleM(*d_a,"dra_read");
        if(!dai_read_allowed(*d_a))dai_error("dra_read: read not allowed",*d_a);
        ga_inquire_(g_a, &gtype, &gdim1, &gdim2);
        if(DRA[handle].type != gtype)dai_error("dra_read: type mismatch",gtype);
        if(DRA[handle].dim1 != gdim1)dai_error("dra_read: dim1 mismatch",gdim1);
        if(DRA[handle].dim2 != gdim2)dai_error("dra_read: dim2 mismatch",gdim2);

        /* right now, naive implementation just calls dra_read_section */
        ilo = 1; ihi = DRA[handle].dim1;
        jlo = 1; jhi = DRA[handle].dim2;
        return(dra_read_section_(&transp, g_a, &ilo, &ihi, &jlo, &jhi,
                                         d_a, &ilo, &ihi, &jlo, &jhi, request));
}



/*\ WAIT FOR COMPLETION OF DRA OPERATION ASSOCIATED WITH request
\*/ 
Integer dra_wait_(Integer* request)
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
Integer dra_probe_(
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
Integer dra_inquire(d_a, type, dim1, dim2, name, filename)
        Integer *d_a;                      /*input:DRA handle*/ 
        Integer *type;                     /*output*/
        Integer *dim1;                     /*output*/
        Integer *dim2;                     /*output*/
        char    *name;                     /*output*/
        char    *filename;                 /*output*/
{
Integer handle=*d_a+DRA_OFFSET;

        dai_check_handleM(*d_a,"dra_inquire");

        *type = DRA[handle].type;
        *dim1 = DRA[handle].dim1;
        *dim2 = DRA[handle].dim2;
        strcpy(name, DRA[handle].name);
        strcpy(filename, DRA[handle].fname);
 
        return(ELIO_OK);
}


/*\ DELETE DISK ARRAY  -- relevant file(s) gone
\*/
Integer dra_delete_(Integer* d_a)            /*input:DRA handle */
{
Integer handle = *d_a+DRA_OFFSET;

        ga_sync_();

        dai_check_handleM(*d_a,"dra_delete");
        dai_delete_param(DRA[handle].fname,*d_a);

        if(dai_file_master(*d_a))
          if(INDEPFILES(*d_a)){ 
             sprintf(dummy_fname,"%s%d",DRA[handle].fname,dai_io_nodeid(*d_a));
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
Integer dra_terminate_()
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
