#include "elio.h"
#include "sf.h"

#define _max_shared_files 10
#define SF_OFFSET 3000
#define SF_FAIL (Integer)1

typedef struct{
        Integer handle;
        Integer actv;
        SFsize_t soft_size; 
        SFsize_t hard_size; 
        Fd_t fd;
        char fname[200];
} SF_t;

SF_t SF[_max_shared_files];

#include "coms.h"

/******************** internal macros ********************************/

#define sfi_check_handleM(_handle, msg)\
{\
        if(_handle+SF_OFFSET >=_max_shared_files||_handle+SF_OFFSET<0)\
        {fprintf(stderr,"%s, %ld --",msg, _max_shared_files);         \
            ERROR("invalid SF handle",_handle);}                      \
        if( SF[(_handle+SF_OFFSET)].actv == 0)                        \
        {fprintf(stderr,"%s:",msg);                                   \
            ERROR("disk array not active",_handle);}                  \
}


/*********************************************************************/

Integer sfi_get_handle()
{
Integer sf_handle =-1, candidate = 0;

        do{
            if(!SF[candidate].actv){
               sf_handle=candidate;
               SF[candidate].actv =1;
            }
            candidate++;
        }while(candidate < _max_shared_files && sf_handle == -1);
        return(sf_handle);
}


void sfi_release_handle(handle)
        Integer *handle;
{
     SF[*handle+SF_OFFSET].actv =0;
     *handle = 0;
}


Integer sf_create(fname, size_hard_limit, size_soft_limit, 
                   req_size, handle)
        char *fname;
        SFsize_t *size_hard_limit, *size_soft_limit, *req_size;
        Integer *handle;
{
Integer hndl;
        SYNC();

        /*** Get next free SF handle ***/
        if( (hndl = sfi_get_handle()) == -1)
           ERROR("sf_create: too many shared files ",_max_shared_files);
        *handle = hndl - SF_OFFSET;

        /*generate file name(s) */
        sprintf(SF[hndl].fname,"%s.%d",fname, hndl);

#       ifdef  PARAGON
          SF[hndl].fd = elio_gopen(SF[hndl].fname,ELIO_RW);
#       else
          SF[hndl].fd = elio_open(SF[hndl].fname,ELIO_RW, ELIO_SHARED);
#       endif

        SF[hndl].soft_size = *size_soft_limit;
        SF[hndl].hard_size = *size_hard_limit;
        SYNC();
        return(ELIO_OK);
}


Integer sf_destroy_(s_a)
        Integer *s_a;                      /*input:SF handle */
{
Integer handle = *s_a+SF_OFFSET;

        SYNC();

        sfi_check_handleM(*s_a,"sf_delete");
        if(ME() == 0)elio_delete(SF[handle].fname);
        sfi_release_handle(s_a);

        SYNC();
        return(ELIO_OK);
}


/*\ asynchronous write to shared file
\*/
Integer sf_write_(s_a, offset, bytes, buffer, req_id)
        Integer *s_a;
        SFsize_t *offset, *bytes;
        char *buffer;
        Integer *req_id;
{
Integer handle = *s_a+SF_OFFSET;
int status;
io_request_t id;

        sfi_check_handleM(*s_a,"sf_write");
        status = elio_awrite(SF[handle].fd, (off_t)*offset, buffer, 
                            (Size_t)*bytes, &id);
        *req_id = (Integer)id;
        return((Integer)status);
/*
 status = elio_write(SF[handle].fd, (off_t)*offset, buffer,(Size_t)*bytes);
                *req_id = (Integer)ELIO_DONE;
        if(status != (int)*bytes)
              return((Integer)SF_FAIL);
        else
              return((Integer)ELIO_OK);
*/
}
        

/*\ asynchronous read from shared file
\*/
Integer sf_read_(s_a, offset, bytes, buffer, req_id)
        Integer *s_a;
        SFsize_t *offset, *bytes;
        char *buffer;
        Integer *req_id;
{
Integer handle = *s_a+SF_OFFSET;
int status;
io_request_t id;

        sfi_check_handleM(*s_a,"sf_read");
        status = elio_aread(SF[handle].fd, (off_t)*offset, buffer, 
                           (Size_t)*bytes, &id);
        *req_id = (Integer)id;
        return((Integer)status);
}


/*\ block calling process until I/O operation completes
\*/
Integer sf_wait_(req_id)
        Integer *req_id;
{
int status;
io_request_t id = (io_request_t) *req_id;
        status = elio_wait(&id);
        *req_id = (Integer)id;
        return((Integer)status);
}


/*\ block calling process until all I/O operations associated
 *  with id in the list complete
\*/
Integer sf_waitall_(list, num)
        Integer *list, *num;
{
int i;
int status, fail=0;

        for(i=0;i<*num;i++){
           io_request_t id = (io_request_t) list[i];
           status = elio_wait(&id);
           if(status != ELIO_OK) fail++;
           list[i] = (Integer)id;
        }
        if (fail)return((Integer)SF_FAIL);
        else     return((Integer)ELIO_OK);
}
