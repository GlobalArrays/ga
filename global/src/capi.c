/* $Id: capi.c,v 1.55 2003-03-31 21:11:04 d3g293 Exp $ */
#include "ga.h"
#include "globalp.h"
#include <stdio.h>
#include <stdlib.h>

Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM], _ga_work[MAXDIM];
Integer _ga_dims[MAXDIM], _ga_map_capi[MAX_NPROC];
Integer _ga_width[MAXDIM];
Integer _ga_skip[MAXDIM];

Integer _ga_alo[MAXDIM], _ga_ahi[MAXDIM];
Integer _ga_blo[MAXDIM], _ga_bhi[MAXDIM];
Integer _ga_clo[MAXDIM], _ga_chi[MAXDIM];
Integer _ga_mlo[MAXDIM], _ga_mhi[MAXDIM];

Integer _ga_xxlo[MAXDIM], _ga_xxhi[MAXDIM];
Integer _ga_vvlo[MAXDIM], _ga_vvhi[MAXDIM];
Integer _ga_xxlllo[MAXDIM], _ga_xxllhi[MAXDIM];
Integer _ga_xxuulo[MAXDIM], _ga_xxuuhi[MAXDIM];


#ifdef USE_FAPI
#  define COPYC2F(carr, farr, n){\
   int i; for(i=0; i< (n); i++)(farr)[i]=(Integer)(carr)[i];} 
#  define COPYF2C(farr, carr, n){\
   int i; for(i=0; i< (n); i++)(carr)[i]=(int)(farr)[i];} 
#else
#  define COPYC2F(carr, farr, n){\
   int i; for(i=0; i< (n); i++)(farr)[n-i-1]=(Integer)(carr)[i];} 
#  define COPYF2C(farr, carr, n){\
   int i; for(i=0; i< (n); i++)(carr)[n-i-1]=(int)(farr)[i];} 
#define BASE_0
#endif

#define COPY(CAST,src,dst,n) {\
   int i; for(i=0; i< (n); i++)(dst)[i]=(CAST)(src)[i];} 

#ifdef BASE_0 
#  define COPYINDEX_C2F(carr, farr, n){\
   int i; for(i=0; i< (n); i++)(farr)[n-i-1]=(Integer)(carr)[i]+1;}
#  define COPYINDEX_F2C(farr, carr, n){\
   int i; for(i=0; i< (n); i++)(carr)[n-i-1]=(int)(farr)[i] -1;}
#else
#  define COPYINDEX_F2C COPYF2C
#  define COPYINDEX_C2F COPYC2F
#endif

int GA_Uses_fapi(void)
{
#ifdef USE_FAPI
return 1;
#else
return 0;
#endif
}


void GA_Initialize_ltd(size_t limit)
{
Integer lim = (Integer)limit;
     ga_initialize_ltd_(&lim);
}
    

int NGA_Create(int type, int ndim,int dims[], char *name, int *chunk)
{
    Integer *ptr, g_a; 
    logical st;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    if(!chunk)ptr=(Integer*)0;  
    else {
         COPYC2F(chunk,_ga_work, ndim);
         ptr = _ga_work;
    }
    st = nga_create((Integer)type, (Integer)ndim, _ga_dims, name, ptr, &g_a);
    if(st==TRUE) return (int) g_a;
    else return 0;
}

int NGA_Create_config(int type, int ndim,int dims[], char *name, int chunk[],
                      int p_handle)
{
    Integer *ptr, g_a; 
    logical st;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    if(!chunk)ptr=(Integer*)0;  
    else {
         COPYC2F(chunk,_ga_work, ndim);
         ptr = _ga_work;
    }
    st = nga_create_config((Integer)type, (Integer)ndim, _ga_dims, name, ptr,
                    (Integer)p_handle, &g_a);
    if(st==TRUE) return (int) g_a;
    else return 0;
}


int NGA_Create_irreg(int type,int ndim,int dims[],char *name,int block[],int map[])
{
    Integer *ptr, g_a;
    logical st;
    int d, base_map=0, base_work, b;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    COPYC2F(block,_ga_work, ndim);

    /* copy might swap only order of dimensions for blocks in map */
#ifdef  USE_FAPI
        base_work = 0;
#else
        base_work =MAX_NPROC;
#endif

    for(d=0; d<ndim; d++){
#ifndef  USE_FAPI
        base_work -= block[d];
        if(base_work <0)GA_Error("GA C api: error in block",d);
#endif
        for(b=0; b<block[d]; b++){

            _ga_map_capi[base_work + b] = (Integer)map[base_map +b]; /*****/
#ifdef BASE_0
            _ga_map_capi[base_work + b]++;
#endif
        }
        base_map += block[d];

#ifdef  USE_FAPI
        base_work += block[d];
        if(base_work >MAX_NPROC)GA_Error("GA (c): error in block",base_work);
#endif
     }

#ifdef  USE_FAPI
     ptr = _ga_map_capi;
#else
     ptr = _ga_map_capi + base_work;
#endif

    st = nga_create_irreg(type, (Integer)ndim, _ga_dims, name, ptr, _ga_work, &g_a);

    if(st==TRUE) return (int) g_a;
    else return 0;
}

int NGA_Create_irreg_config(int type,int ndim,int dims[],char *name,int block[],
                            int map[], int p_handle)
{
    Integer *ptr, g_a;
    logical st;
    int d, base_map=0, base_work, b;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    COPYC2F(block,_ga_work, ndim);

    /* copy might swap only order of dimensions for blocks in map */
#ifdef  USE_FAPI
        base_work = 0;
#else
        base_work =MAX_NPROC;
#endif

    for(d=0; d<ndim; d++){
#ifndef  USE_FAPI
        base_work -= block[d];
        if(base_work <0)GA_Error("GA C api: error in block",d);
#endif
        for(b=0; b<block[d]; b++){

            _ga_map_capi[base_work + b] = (Integer)map[base_map +b]; /*****/
#ifdef BASE_0
            _ga_map_capi[base_work + b]++;
#endif
        }
        base_map += block[d];

#ifdef  USE_FAPI
        base_work += block[d];
        if(base_work >MAX_NPROC)GA_Error("GA (c): error in block",base_work);
#endif
     }

#ifdef  USE_FAPI
     ptr = _ga_map_capi;
#else
     ptr = _ga_map_capi + base_work;
#endif

    st = nga_create_irreg_config(type, (Integer)ndim, _ga_dims, name, ptr,
                                 _ga_work, (Integer)p_handle, &g_a);

    if(st==TRUE) return (int) g_a;
    else return 0;
}

int NGA_Create_ghosts_irreg(int type,int ndim,int dims[],int width[],char *name,
    int block[],int map[])
{
    Integer *ptr, g_a;
    logical st;
    int d, base_map=0, base_work, b;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    COPYC2F(block,_ga_work, ndim);
    COPYC2F(width,_ga_width, ndim);

    /* copy might swap only order of dimensions for blocks in map */
#ifdef  USE_FAPI
        base_work = 0;
#else
        base_work =MAX_NPROC;
#endif

    for(d=0; d<ndim; d++){
#ifndef  USE_FAPI
        base_work -= block[d];
        if(base_work <0)GA_Error("GA C api: error in block",d);
#endif
        for(b=0; b<block[d]; b++){

            _ga_map_capi[base_work + b] = (Integer)map[base_map +b]; /*****/
#ifdef BASE_0
            _ga_map_capi[base_work + b]++;
#endif
        }
        base_map += block[d];

#ifdef  USE_FAPI
        base_work += block[d];
        if(base_work >MAX_NPROC)GA_Error("GA (c): error in block",base_work);
#endif
     }

#ifdef  USE_FAPI
     ptr = _ga_map_capi;
#else
     ptr = _ga_map_capi + base_work;
#endif

    st = nga_create_ghosts_irreg(type, (Integer)ndim, _ga_dims, _ga_width, name, ptr,
        _ga_work, &g_a);

    if(st==TRUE) return (int) g_a;
    else return 0;
}

int NGA_Create_ghosts_irreg_config(int type, int ndim, int dims[], int width[],
    char *name, int block[], int map[], int p_handle)
{
    Integer *ptr, g_a;
    logical st;
    int d, base_map=0, base_work, b;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    COPYC2F(block,_ga_work, ndim);
    COPYC2F(width,_ga_width, ndim);

    /* copy might swap only order of dimensions for blocks in map */
#ifdef  USE_FAPI
        base_work = 0;
#else
        base_work =MAX_NPROC;
#endif

    for(d=0; d<ndim; d++){
#ifndef  USE_FAPI
        base_work -= block[d];
        if(base_work <0)GA_Error("GA C api: error in block",d);
#endif
        for(b=0; b<block[d]; b++){

            _ga_map_capi[base_work + b] = (Integer)map[base_map +b]; /*****/
#ifdef BASE_0
            _ga_map_capi[base_work + b]++;
#endif
        }
        base_map += block[d];

#ifdef  USE_FAPI
        base_work += block[d];
        if(base_work >MAX_NPROC)GA_Error("GA (c): error in block",base_work);
#endif
     }

#ifdef  USE_FAPI
     ptr = _ga_map_capi;
#else
     ptr = _ga_map_capi + base_work;
#endif

    st = nga_create_ghosts_irreg_config(type, (Integer)ndim, _ga_dims, _ga_width,
         name, ptr, _ga_work, (Integer)p_handle, &g_a);

    if(st==TRUE) return (int) g_a;
    else return 0;
}


int NGA_Create_ghosts(int type, int ndim,int dims[], int width[], char *name,
    int chunk[])
{
    Integer *ptr, g_a; 
    logical st;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    COPYC2F(width,_ga_width, ndim);
    if(!chunk)ptr=(Integer*)0;  
    else {
         COPYC2F(chunk,_ga_work, ndim);
         ptr = _ga_work;
    }
    st = nga_create_ghosts((Integer)type, (Integer)ndim, _ga_dims,
        _ga_width, name, ptr, &g_a);
    if(st==TRUE) return (int) g_a;
    else return 0;
}

int NGA_Create_ghosts_config(int type, int ndim,int dims[], int width[], char *name,
    int chunk[], int p_handle)
{
    Integer *ptr, g_a; 
    logical st;
    if(ndim>MAXDIM)return 0;

    COPYC2F(dims,_ga_dims, ndim);
    COPYC2F(width,_ga_width, ndim);
    if(!chunk)ptr=(Integer*)0;  
    else {
         COPYC2F(chunk,_ga_work, ndim);
         ptr = _ga_work;
    }
    st = nga_create_ghosts_config((Integer)type, (Integer)ndim, _ga_dims,
        _ga_width, name, ptr, (Integer)p_handle, &g_a);
    if(st==TRUE) return (int) g_a;
    else return 0;
}

void GA_Update_ghosts(int g_a)
{
    Integer a=(Integer)g_a;
    ga_update_ghosts_(&a);
}

void GA_Merge_mirrored(int g_a)
{
    Integer a=(Integer)g_a;
    ga_merge_mirrored_(&a);
}

int NGA_Update_ghost_dir(int g_a, int dimension, int dir, int flag)
{
    Integer a=(Integer)g_a;
    Integer idim = (Integer)dimension;
    Integer idir = (Integer)dir;
    logical iflag = (logical)flag;
    logical st;
    idim++;
    st = nga_update_ghost_dir_(&a,&idim,&idir,&iflag);
    return (int)st;
}

void NGA_NbGet_ghost_dir(int g_a, int mask[], ga_nbhdl_t nbhandle)
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(mask,_ga_lo, ndim);
    nga_nbget_ghost_dir_(&a, _ga_lo,(Integer *)nbhandle);
}

int GA_Has_ghosts(int g_a)
{
    Integer a=(Integer)g_a;
    logical st;
    st = ga_has_ghosts_(&a);
    return (int)st;
}

void GA_Mask_sync(int first, int last)
{
    Integer ifirst = (Integer)first;
    Integer ilast = (Integer)last;
    ga_mask_sync_(&ifirst,&ilast);
}

int GA_Duplicate(int g_a, char* array_name)
{
    logical st;
    Integer a=(Integer)g_a, b;
    st = ga_duplicate(&a, &b, array_name);
    if(st==TRUE) return (int) b;
    else return 0;
}


void GA_Destroy(int g_a)
{
    logical st;
    Integer a=(Integer)g_a;
    st = ga_destroy_(&a);
    if(st==FALSE)GA_Error("GA (c) destroy failed",g_a);
}

void GA_Set_memory_limit(size_t limit)
{
Integer lim = (Integer)limit;
     ga_set_memory_limit_(&lim);
}

void GA_Zero(int g_a)
{
    Integer a=(Integer)g_a;
    ga_zero_(&a);
}

int GA_Default_config()
{
    int value = (int)ga_default_config_();
    return value;
}

int GA_Mirror_config()
{
    int value = (int)ga_mirror_config_();
    return value;
}

int GA_Idot(int g_a, int g_b)
{
    int value;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    gai_dot(C_INT, &a, &b, &value);
    return value;
}


long GA_Ldot(int g_a, int g_b)
{
    long value;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    gai_dot(C_LONG, &a, &b, &value);
    return value;
}

     
double GA_Ddot(int g_a, int g_b)
{
    double value;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    gai_dot(C_DBL, &a, &b, &value);
    return value;
}


DoubleComplex GA_Zdot(int g_a, int g_b)
{
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    return ga_zdot(&a,&b);
}

float GA_Fdot(int g_a, int g_b)
{
    float sum;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    gai_dot(C_FLOAT, &a, &b, &sum);
    return sum;
}    

void GA_Fill(int g_a, void *value)
{
    Integer a=(Integer)g_a;
    ga_fill_(&a, value);
}


void GA_Scale(int g_a, void *value)
{
    Integer a=(Integer)g_a;
    ga_scale_(&a,value);
}


void GA_Add(void *alpha, int g_a, void* beta, int g_b, int g_c)
{
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    Integer c=(Integer)g_c;
    ga_add_(alpha, &a, beta, &b, &c);
}


void GA_Copy(int g_a, int g_b)
{
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    ga_copy_(&a, &b);
}


void NGA_Get(int g_a, int lo[], int hi[], void* buf, int ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    nga_get_common(&a, _ga_lo, _ga_hi, buf, _ga_work,NULL);
}

void NGA_NbGet(int g_a, int lo[], int hi[], void* buf, int ld[],
               ga_nbhdl_t* nbhandle)
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    nga_get_common(&a, _ga_lo, _ga_hi, buf, _ga_work,(Integer *)nbhandle);
}

void NGA_Put(int g_a, int lo[], int hi[], void* buf, int ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    nga_put_common(&a, _ga_lo, _ga_hi, buf, _ga_work,(Integer *)NULL);
}    

void NGA_NbPut(int g_a, int lo[], int hi[], void* buf, int ld[],
               ga_nbhdl_t* nbhandle)
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    nga_put_common(&a, _ga_lo, _ga_hi, buf, _ga_work,(Integer *)nbhandle);
}

int NGA_NbWait(ga_nbhdl_t* nbhandle)
{
    return(nga_wait_internal((Integer *)nbhandle));
}

int GA_NbWait(ga_nbhdl_t* nbhandle)
{
    return(nga_wait_internal((Integer *)nbhandle));
}

void NGA_Strided_put(int g_a, int lo[], int hi[], int skip[],
                     void* buf, int ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    COPYC2F(skip, _ga_skip, ndim);
    nga_strided_put_(&a, _ga_lo, _ga_hi, _ga_skip, buf, _ga_work);
}    


void NGA_Acc(int g_a, int lo[], int hi[], void* buf,int ld[], void* alpha)
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    nga_acc_(&a, _ga_lo, _ga_hi, buf, _ga_work, alpha);
}    

void NGA_Periodic_get(int g_a, int lo[], int hi[], void* buf, int ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    ngai_periodic_(&a, _ga_lo, _ga_hi, buf, _ga_work, NULL, PERIODIC_GET);
}

void NGA_Periodic_put(int g_a, int lo[], int hi[], void* buf, int ld[])
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    ngai_periodic_(&a, _ga_lo, _ga_hi, buf, _ga_work, NULL, PERIODIC_PUT);
}    


void NGA_Periodic_acc(int g_a, int lo[], int hi[], void* buf,int ld[], void* alpha)
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    COPYC2F(ld,_ga_work, ndim-1);
    ngai_periodic_(&a, _ga_lo, _ga_hi, buf, _ga_work, alpha, PERIODIC_ACC);
}

long NGA_Read_inc(int g_a, int subscript[], long inc)
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    Integer in=(Integer)inc;
    COPYINDEX_C2F(subscript, _ga_lo, ndim);
    return (long)nga_read_inc_(&a, _ga_lo, &in);
}

void NGA_Distribution(int g_a, int iproc, int lo[], int hi[])
{
     Integer a=(Integer)g_a;
     Integer p=(Integer)iproc;
     Integer ndim = ga_ndim_(&a);
     nga_distribution_(&a, &p, _ga_lo, _ga_hi);
     COPYINDEX_F2C(_ga_lo,lo, ndim);
     COPYINDEX_F2C(_ga_hi,hi, ndim);
}

void NGA_Select_elem(int g_a, char* op, void* val, int* index)
{
     Integer a=(Integer)g_a;
     Integer ndim = ga_ndim_(&a);
     nga_select_elem_(&a, op, val, _ga_lo);
     COPYINDEX_F2C(_ga_lo,index,ndim);
}

int GA_Compare_distr(int g_a, int g_b)
{
    logical st;
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    st = ga_compare_distr_(&a,&b);
    if(st == TRUE) return 0;
    else return 1;
}

void NGA_Distribution_no_handle(int ndim, const int dims[], const int nblock[], const int mapc[], int iproc, int lo[], int hi[])
{
     Integer p=(Integer)iproc;
     Integer _ndim = ndim;
     Integer _dims[MAXDIM];
     Integer _nblock[MAXDIM];
     Integer _mapc[MAXDIM];
     COPYINDEX_C2F(dims, _dims, ndim);
     COPYINDEX_C2F(nblock, _nblock, ndim);
     COPYINDEX_C2F(mapc, _mapc, ndim);
     nga_distribution_no_handle_(&_ndim, _dims, _nblock, mapc, &p, _ga_lo, _ga_hi);
     COPYINDEX_F2C(_ga_lo,lo, ndim);
     COPYINDEX_F2C(_ga_hi,hi, ndim);
}


void NGA_Access(int g_a, int lo[], int hi[], void *ptr, int ld[])
{
     Integer a=(Integer)g_a;
     Integer ndim = ga_ndim_(&a);
     COPYINDEX_C2F(lo,_ga_lo,ndim);
     COPYINDEX_C2F(hi,_ga_hi,ndim);

     nga_access_ptr(&a,_ga_lo, _ga_hi, ptr, _ga_work);
     COPYF2C(_ga_work,ld, ndim-1);
}

void NGA_Access_ghosts(int g_a, int dims[], void *ptr, int ld[])
{
     Integer a=(Integer)g_a;
     Integer ndim = ga_ndim_(&a);

     nga_access_ghost_ptr(&a, _ga_lo, ptr, _ga_work);
     COPYF2C(_ga_work,ld, ndim-1);
     COPYF2C(_ga_lo, dims, ndim);
}

void NGA_Access_ghost_element(int g_a, void *ptr, int subscript[], int ld[])
{
     Integer a=(Integer)g_a;
     Integer ndim = ga_ndim_(&a);
     Integer _subscript[MAXDIM];
     Integer _ld[MAXDIM];

     COPYINDEX_C2F(subscript, _subscript, ndim);
     COPYINDEX_C2F(ld, _ld, ndim-1);
     nga_access_ghost_element_(&a, ptr, _subscript, _ld);
}

void NGA_Release(int g_a, int lo[], int hi[])
{
     Integer a=(Integer)g_a;
     Integer ndim = ga_ndim_(&a);
     COPYINDEX_C2F(lo,_ga_lo,ndim);
     COPYINDEX_C2F(hi,_ga_hi,ndim);

     nga_release_(&a,_ga_lo, _ga_hi);
}

int NGA_Locate(int g_a, int subscript[])
{
    logical st;
    Integer a=(Integer)g_a, owner;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(subscript,_ga_lo,ndim);

    st = nga_locate_(&a,_ga_lo,&owner);
    if(st == TRUE) return (int)owner;
    else return -1;
}


int NGA_Locate_region(int g_a,int lo[],int hi[],int map[],int procs[])
{
     logical st;
     Integer a=(Integer)g_a, np;
     Integer ndim = ga_ndim_(&a);
     Integer *tmap;
     int i;
     tmap = (Integer *)malloc( (int)(GA_Nnodes()*2*ndim *sizeof(Integer)));
     if(!map)GA_Error("NGA_Locate_region: unable to allocate memory",g_a);
     COPYINDEX_C2F(lo,_ga_lo,ndim);
     COPYINDEX_C2F(hi,_ga_hi,ndim);

     st = nga_locate_region_(&a,_ga_lo, _ga_hi, tmap, _ga_map_capi, &np);
     if(st==FALSE){
       free(tmap);
       return 0;
     }

     COPY(int,_ga_map_capi,procs, np);

        /* might have to swap lo/hi when copying */

     for(i=0; i< np*2; i++){
        Integer *ptmap = tmap+i*ndim;
        int *pmap = map +i*ndim;
        COPYINDEX_F2C(ptmap, pmap, ndim);  
     }
     free(tmap);
     return (int)np;
}


void NGA_Inquire(int g_a, int *type, int *ndim, int dims[])
{
     Integer a=(Integer)g_a;
     Integer ttype, nndim;
     nga_inquire(&a,&ttype, &nndim, _ga_dims);
     COPYF2C(_ga_dims, dims,nndim);  
     *ndim = (int)nndim;
     *type = (int)ttype;
}



char* GA_Inquire_name(int g_a)
{
     Integer a=(Integer)g_a;
     char *ptr;
     ga_inquire_name(&a, &ptr);
     return(ptr);
}

int GA_Memory_limited(void)
{
    if(ga_memory_limited_() ==TRUE) return 1;
    else return 0;
}

void NGA_Proc_topology(int g_a, int proc, int coord[])
{
     Integer a=(Integer)g_a;
     Integer p=(Integer)proc;
     Integer ndim = ga_ndim_(&a);
     nga_proc_topology_(&a, &p, _ga_work);
     COPY(int,_ga_work, coord,ndim);  
}


void GA_Check_handle(int g_a, char* string)
{
     Integer a=(Integer)g_a;
     ga_check_handle(&a,string);
}

int GA_Create_mutexes(int number)
{
     Integer n = (Integer)number;
     if(ga_create_mutexes_(&n) == TRUE)return 1;
     else return 0;
}

int GA_Destroy_mutexes(void) {
  if(ga_destroy_mutexes_() == TRUE) return 1;
  else return 0;
}

void GA_Lock(int mutex)
{
     Integer m = (Integer)mutex;
     ga_lock_(&m);
}

void GA_Unlock(int mutex)
{
     Integer m = (Integer)mutex;
     ga_unlock_(&m);
}

void GA_Brdcst(void *buf, int lenbuf, int root)
{
  Integer type=GA_TYPE_BRD;
  Integer len = (Integer)lenbuf;
  Integer orig = (Integer)root;
  ga_msg_brdcst(type, buf, len, orig);
}
   
void GA_Dgop(double x[], int n, char *op)
{
  Integer type=GA_TYPE_GOP;
  Integer len = (Integer)n;
  ga_dgop(type, x, len, op);
}
  
void GA_Lgop(long x[], int n, char *op)
{
  Integer type=GA_TYPE_GOP;
  Integer len = (Integer)n;
  ga_lgop(type, x, len, op);
}
  
void GA_Igop(Integer x[], int n, char *op)
{
  Integer type=GA_TYPE_GOP;
  Integer len = (Integer)n;
  ga_igop(type, x, len, op);
}

void GA_Fgop(float x[], int n, char *op)
{
  Integer type=GA_TYPE_GOP;
  Integer len = (Integer)n;
  ga_fgop(type, x, len, op);
}       

/*********** to do *******/
/*
void GA_Print_patch(int g_a,int ilo,int ihi,int jlo,int jhi,int pretty)
void GA_Copy_patch(ta,int g_a, int ailo, int aihi, int ajlo, int ajhi,
                             int g_b, int bilo, int bihi, int bjlo,int bjhi)
*/
void NGA_Scatter(int g_a, void *v, int* subsArray[], int n)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
    Integer ndim = ga_ndim_(&a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+i] = subsArray[idx][i] + 1;
    
    nga_scatter_(&a, v, _subs_array , &nv);
    
    free(_subs_array);
}

void NGA_Scatter_acc(int g_a, void *v, int* subsArray[], int n, void *alpha)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
    Integer ndim = ga_ndim_(&a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+i] = subsArray[idx][i] + 1;
    
    nga_scatter_acc_(&a, v, _subs_array , &nv, alpha);
    
    free(_subs_array);
}

void NGA_Gather(int g_a, void *v, int* subsArray[], int n)
{
    int idx, i;
    Integer a = (Integer)g_a;
    Integer nv = (Integer)n;
    Integer ndim = ga_ndim_(&a);
    Integer *_subs_array;
    _subs_array = (Integer *)malloc((int)ndim* n * sizeof(Integer));
    if(_subs_array == NULL) GA_Error("Memory allocation failed.", 0);

    /* adjust the indices for fortran interface */
    for(idx=0; idx<n; idx++)
        for(i=0; i<ndim; i++)
            _subs_array[idx*ndim+i] = subsArray[idx][i] + 1;
    
    nga_gather_(&a, v, _subs_array , &nv);
    
    free(_subs_array);
}


void GA_Dgemm(char ta, char tb, int m, int n, int k,
              double alpha, int g_a, int g_b, double beta, int g_c )
{
  Integer Ta = (ta=='t' || ta=='T');
  Integer Tb = (tb=='t' || tb=='T');
  Integer M = m;
  Integer N = n;
  Integer K = k;
  Integer G_a = g_a;
  Integer G_b = g_b;
  Integer G_c = g_c;

  Integer ailo = 1;
  Integer aihi = m;
  Integer ajlo = 1;
  Integer ajhi = k;
  
  Integer bilo = 1;
  Integer bihi = k;
  Integer bjlo = 1;
  Integer bjhi = n;
  
  Integer cilo = 1;
  Integer cihi = m;
  Integer cjlo = 1;
  Integer cjhi = n;
  
  ga_matmul_patch(&ta, &tb, (DoublePrecision *)&alpha,(DoublePrecision *)&beta,
		  &G_a, &ailo, &aihi, &ajlo, &ajhi,
		  &G_b, &bilo, &bihi, &bjlo, &bjhi,
		  &G_c, &cilo, &cihi, &cjlo, &cjhi);
}

void GA_Zgemm(char ta, char tb, int m, int n, int k,
              DoubleComplex alpha, int g_a, int g_b, 
	      DoubleComplex beta, int g_c )
{
  Integer Ta = (ta=='t' || ta=='T');
  Integer Tb = (tb=='t' || tb=='T');
  Integer M = m;
  Integer N = n;
  Integer K = k;
  Integer G_a = g_a;
  Integer G_b = g_b;
  Integer G_c = g_c;

  Integer ailo = 1;
  Integer aihi = m;
  Integer ajlo = 1;
  Integer ajhi = k;
  
  Integer bilo = 1;
  Integer bihi = k;
  Integer bjlo = 1;
  Integer bjhi = n;
  
  Integer cilo = 1;
  Integer cihi = m;
  Integer cjlo = 1;
  Integer cjhi = n;
  
  ga_matmul_patch(&ta, &tb, (DoublePrecision *)&alpha,(DoublePrecision *)&beta,
		  &G_a, &ailo, &aihi, &ajlo, &ajhi,
		  &G_b, &bilo, &bihi, &bjlo, &bjhi,
		  &G_c, &cilo, &cihi, &cjlo, &cjhi);
}

void GA_Sgemm(char ta, char tb, int m, int n, int k,
              float alpha, int g_a, int g_b, 
	      float beta,  int g_c )
{
  Integer Ta = (ta=='t' || ta=='T');
  Integer Tb = (tb=='t' || tb=='T');
  Integer M = m;
  Integer N = n;
  Integer K = k;
  Integer G_a = g_a;
  Integer G_b = g_b;
  Integer G_c = g_c;

  Integer ailo = 1;
  Integer aihi = m;
  Integer ajlo = 1;
  Integer ajhi = k;
  
  Integer bilo = 1;
  Integer bihi = k;
  Integer bjlo = 1;
  Integer bjhi = n;
  
  Integer cilo = 1;
  Integer cihi = m;
  Integer cjlo = 1;
  Integer cjhi = n;
  
  ga_matmul_patch(&ta, &tb, (DoublePrecision*)&alpha, (DoublePrecision*)&beta,
		  &G_a, &ailo, &aihi, &ajlo, &ajhi,
		  &G_b, &bilo, &bihi, &bjlo, &bjhi,
		  &G_c, &cilo, &cihi, &cjlo, &cjhi);
}

/* Patch related */

void NGA_Copy_patch(char trans, int g_a, int alo[], int ahi[],
                                int g_b, int blo[], int bhi[])
{
    Integer a=(Integer)g_a;
    Integer andim = ga_ndim_(&a);

    Integer b=(Integer)g_b;
    Integer bndim = ga_ndim_(&b);
    
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    nga_copy_patch(&trans, &a, _ga_alo, _ga_ahi, &b, _ga_blo, _ga_bhi);
}

void GA_Matmul_patch(char transa, char transb, void* alpha, void *beta,
		     int g_a, int ailo, int aihi, int ajlo, int ajhi,
		     int g_b, int bilo, int bihi, int bjlo, int bjhi,
		     int g_c, int cilo, int cihi, int cjlo, int cjhi)

{
    Integer a=(Integer)g_a;
    Integer b=(Integer)g_b;
    Integer c=(Integer)g_c;

    Integer ga_ailo=(Integer)(ailo+1);
    Integer ga_aihi=(Integer)(aihi+1);
    Integer ga_ajlo=(Integer)(ajlo+1); 
    Integer ga_ajhi=(Integer)(ajhi+1);  
    
    Integer ga_bilo=(Integer)(bilo+1);
    Integer ga_bihi=(Integer)(bihi+1);
    Integer ga_bjlo=(Integer)(bjlo+1); 
    Integer ga_bjhi=(Integer)(bjhi+1);  
    
    Integer ga_cilo=(Integer)(cilo+1);
    Integer ga_cihi=(Integer)(cihi+1);
    Integer ga_cjlo=(Integer)(cjlo+1); 
    Integer ga_cjhi=(Integer)(cjhi+1);  
   
    ga_matmul_patch(&transa, &transb, alpha, beta,
		    &a, &ga_ailo, &ga_aihi, &ga_ajlo, &ga_ajhi,
		    &b, &ga_bilo, &ga_bihi, &ga_bjlo, &ga_bjhi,
		    &c, &ga_cilo, &ga_cihi, &ga_cjlo, &ga_cjhi);
}

void NGA_Matmul_patch(char transa, char transb, void* alpha, void *beta,
		      int g_a, int alo[], int ahi[], 
		      int g_b, int blo[], int bhi[], 
		      int g_c, int clo[], int chi[]) 

{
    Integer a=(Integer)g_a;
    Integer andim = ga_ndim_(&a);
    
    Integer b=(Integer)g_b;
    Integer bndim = ga_ndim_(&b);
    
    Integer c=(Integer)g_c;
    Integer cndim = ga_ndim_(&c);
    
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
    
    COPYINDEX_C2F(clo,_ga_clo, cndim);
    COPYINDEX_C2F(chi,_ga_chi, cndim);
    
    nga_matmul_patch(&transa, &transb, alpha, beta,
		     &a, _ga_alo, _ga_ahi,
		     &b, _ga_blo, _ga_bhi,
		     &c, _ga_clo, _ga_chi);
}

int NGA_Idot_patch(int g_a, char t_a, int alo[], int ahi[],
                   int g_b, char t_b, int blo[], int bhi[])
{
    Integer res;
    Integer a=(Integer)g_a;
    Integer andim = ga_ndim_(&a);

    Integer b=(Integer)g_b;
    Integer bndim = ga_ndim_(&b);
    
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    res = nga_idot_patch(&a, &t_a, _ga_alo, _ga_ahi,
                         &b, &t_b, _ga_blo, _ga_bhi);

    return ((int) res);
}

double NGA_Ddot_patch(int g_a, char t_a, int alo[], int ahi[],
                   int g_b, char t_b, int blo[], int bhi[])
{
    DoublePrecision res;
    Integer a=(Integer)g_a;
    Integer andim = ga_ndim_(&a);

    Integer b=(Integer)g_b;
    Integer bndim = ga_ndim_(&b);

    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);

    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    res = nga_ddot_patch(&a, &t_a, _ga_alo, _ga_ahi,
                         &b, &t_b, _ga_blo, _ga_bhi);

    return ((double)res);
}

DoubleComplex NGA_Zdot_patch(int g_a, char t_a, int alo[], int ahi[],
                             int g_b, char t_b, int blo[], int bhi[])
{
    DoubleComplex res;
    
    Integer a=(Integer)g_a;
    Integer andim = ga_ndim_(&a);
    
    Integer b=(Integer)g_b;
    Integer bndim = ga_ndim_(&b);
    
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
    
    res = nga_zdot_patch(&a, &t_a, _ga_alo, _ga_ahi,
                         &b, &t_b, _ga_blo, _ga_bhi);
    
    return (res);
}

float NGA_Fdot_patch(int g_a, char t_a, int alo[], int ahi[],
                   int g_b, char t_b, int blo[], int bhi[])
{
    float res;
    Integer a=(Integer)g_a;
    Integer andim = ga_ndim_(&a);
 
    Integer b=(Integer)g_b;
    Integer bndim = ga_ndim_(&b);
 
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
 
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
 
    res = nga_fdot_patch(&a, &t_a, _ga_alo, _ga_ahi,
                         &b, &t_b, _ga_blo, _ga_bhi);
 
    return (res);
}                                           

void NGA_Fill_patch(int g_a, int lo[], int hi[], void *val)
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);

    nga_fill_patch_(&a, _ga_lo, _ga_hi, val);
}

void NGA_Zero_patch(int g_a, int lo[], int hi[])
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);

    nga_zero_patch_(&a, _ga_lo, _ga_hi);
}

void NGA_Scale_patch(int g_a, int lo[], int hi[], void *alpha)
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);

    nga_scale_patch_(&a, _ga_lo, _ga_hi, alpha);
}

void NGA_Add_patch(void * alpha, int g_a, int alo[], int ahi[],
                   void * beta,  int g_b, int blo[], int bhi[],
                   int g_c, int clo[], int chi[])
{
    Integer a=(Integer)g_a;
    Integer andim = ga_ndim_(&a);

    Integer b=(Integer)g_b;
    Integer bndim = ga_ndim_(&b);

    Integer c=(Integer)g_c;
    Integer cndim = ga_ndim_(&c);

    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);

    COPYINDEX_C2F(clo,_ga_clo, cndim);
    COPYINDEX_C2F(chi,_ga_chi, cndim);

    nga_add_patch_(alpha, &a, _ga_alo, _ga_ahi, beta, &b, _ga_blo, _ga_bhi,
                   &c, _ga_clo, _ga_chi);
}

void NGA_Print_patch(int g_a, int lo[], int hi[], int pretty)
{
    Integer a=(Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    Integer p = (Integer)pretty;
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
  
    nga_print_patch_(&a, _ga_lo, _ga_hi, &p);
}

void GA_Print(int g_a)
{
    Integer a=(Integer)g_a;
    ga_print_(&a);
}

void GA_Print_file(FILE *file, int g_a)
{
  Integer G_a = g_a;
  ga_print_file(file, &G_a);
}

void GA_Diag_seq(int g_a, int g_s, int g_v, void *eval)
{
    Integer a = (Integer)g_a;
    Integer s = (Integer)g_s;
    Integer v = (Integer)g_v;

    ga_diag_seq_(&a, &s, &v, eval);
}

void GA_Diag_std_seq(int g_a, int g_v, void *eval)
{
    Integer a = (Integer)g_a;
    Integer v = (Integer)g_v;

    ga_diag_std_seq_(&a, &v, eval);
}

void GA_Diag(int g_a, int g_s, int g_v, void *eval)
{
    Integer a = (Integer)g_a;
    Integer s = (Integer)g_s;
    Integer v = (Integer)g_v;

    ga_diag_(&a, &s, &v, eval);
}

void GA_Diag_std(int g_a, int g_v, void *eval)
{
    Integer a = (Integer)g_a;
    Integer v = (Integer)g_v;

    ga_diag_std_(&a, &v, eval);
}

void GA_Diag_reuse(int reuse, int g_a, int g_s, int g_v, void *eval)
{
    Integer r = (Integer)reuse;
    Integer a = (Integer)g_a;
    Integer s = (Integer)g_s;
    Integer v = (Integer)g_v;

    ga_diag_reuse_(&r, &a, &s, &v, eval);
}

void GA_Lu_solve(char tran, int g_a, int g_b)
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;

    Integer t;

    if(tran == 't' || tran == 'T') t = 1;
    else t = 0;

    ga_lu_solve_alt_(&t, &a, &b);
}

int GA_Llt_solve(int g_a, int g_b)
{
    Integer res;
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;

    res = ga_llt_solve_(&a, &b);

    return((int)res);
}

int GA_Solve(int g_a, int g_b)
{
    Integer res;
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;

    res = ga_solve_(&a, &b);

    return((int)res);
}

int GA_Spd_invert(int g_a)
{
    Integer res;
    Integer a = (Integer)g_a;

    res = ga_spd_invert_(&a);

    return((int)res);
}

void GA_Summarize(int verbose)
{
    Integer v = (Integer)verbose;

    ga_summarize_(&v);
}

void GA_Symmetrize(int g_a)
{
    Integer a = (Integer)g_a;

    ga_symmetrize_(&a);
}

void GA_Transpose(int g_a, int g_b)
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;

    ga_transpose_(&a, &b);
}


void GA_Print_distribution(int g_a)
{
extern void gai_print_distribution(int fstyle, Integer g_a);

#ifdef USE_FAPI
    gai_print_distribution(1,(Integer)g_a);
#else
    gai_print_distribution(0,(Integer)g_a);
#endif
}


void NGA_Release_update(int g_a, int lo[], int hi[])
{
}


int GA_Ndim(int g_a)
{
    Integer a = (Integer)g_a;
    return (int)ga_ndim_(&a);
}


/*Limin's functions */


void GA_Step_max2(int g_xx, int g_vv, int g_xxll, int g_xxuu,  double *step2)
{
    Integer xx = (Integer)g_xx;
    Integer vv = (Integer)g_vv;
    Integer xxll = (Integer)g_xxll;
    Integer xxuu = (Integer)g_xxuu;
    ga_step_max2_(&xx, &vv, &xxll, &xxuu, step2);
}

void GA_Step_max2_patch(int g_xx, int xxlo[], int xxhi[],  int g_vv, int vvlo[], int vvhi[], int g_xxll, int xxlllo[], int xxllhi[], int g_xxuu,  int xxuulo[], int xxuuhi[], double *step2)
{
    Integer xx = (Integer)g_xx;
    Integer vv = (Integer)g_vv;
    Integer xxll = (Integer)g_xxll;
    Integer xxuu = (Integer)g_xxuu;
    Integer ndim = ga_ndim_(&xx);
    COPYINDEX_C2F(xxlo,_ga_xxlo, ndim);
    COPYINDEX_C2F(xxhi,_ga_xxhi, ndim);
    COPYINDEX_C2F(vvlo,_ga_vvlo, ndim);
    COPYINDEX_C2F(vvhi,_ga_vvhi, ndim);
    COPYINDEX_C2F(xxlllo,_ga_xxlllo, ndim);
    COPYINDEX_C2F(xxllhi,_ga_xxllhi, ndim);
    COPYINDEX_C2F(xxuulo,_ga_xxuulo, ndim);
    COPYINDEX_C2F(xxuuhi,_ga_xxuuhi, ndim);
    ga_step_max2_patch_(&xx, _ga_xxlo, _ga_xxhi, &vv, _ga_vvlo, _ga_vvhi, &xxll, _ga_xxlllo, _ga_xxllhi, &xxuu, _ga_xxuulo, _ga_xxuuhi , step2);
}

void GA_Step_max(int g_a, int g_b, double *step)
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;
    ga_step_max_(&a, &b, step);
}

void GA_Step_max_patch(int g_a, int alo[], int ahi[], int g_b, int blo[], int bhi[], double *step)
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(alo,_ga_alo, ndim);
    COPYINDEX_C2F(ahi,_ga_ahi, ndim);
    COPYINDEX_C2F(blo,_ga_blo, ndim);
    COPYINDEX_C2F(bhi,_ga_bhi, ndim);
    ga_step_max_patch_(&a, _ga_alo, _ga_ahi, &b, _ga_blo, _ga_bhi, step);
}
void GA_Abs_value(int g_a)
{
    Integer a = (Integer)g_a;
    ga_abs_value_(&a);
}

void GA_Add_constant(int g_a, void *alpha)
{
    Integer a = (Integer)g_a;
    ga_add_constant_(&a, alpha);
}


void GA_Recip(int g_a)
{
    Integer a = (Integer)g_a;
    ga_recip_(&a);
}

void GA_Elem_multiply(int g_a, int g_b, int g_c)
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;
    Integer c = (Integer)g_c;
    ga_elem_multiply_(&a, &b, &c);
}

void GA_Elem_divide(int g_a, int g_b, int g_c)
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;
    Integer c = (Integer)g_c;
    ga_elem_divide_(&a, &b, &c);
}

void GA_Elem_maximum(int g_a, int g_b, int g_c)
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;
    Integer c = (Integer)g_c;
    ga_elem_maximum_(&a, &b, &c);
}


void GA_Elem_minimum(int g_a, int g_b, int g_c)
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;
    Integer c = (Integer)g_c;
    ga_elem_minimum_(&a, &b, &c);
}


void GA_Abs_value_patch(int g_a, int *lo, int *hi)
{
    Integer a = (Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    ga_abs_value_patch_(&a,_ga_lo, _ga_hi);
}

void GA_Add_constant_patch(int g_a, int *lo, int* hi, void *alpha)
{
    Integer a = (Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    ga_add_constant_patch_(&a, _ga_lo, _ga_hi, alpha);
}

void GA_Recip_patch(int g_a, int *lo, int *hi)
{
    Integer a = (Integer)g_a;
    Integer ndim = ga_ndim_(&a);
    COPYINDEX_C2F(lo,_ga_lo, ndim);
    COPYINDEX_C2F(hi,_ga_hi, ndim);
    ga_recip_patch_(&a,_ga_lo, _ga_hi);
}



void GA_Elem_multiply_patch(int g_a, int alo[], int ahi[],  int g_b, int blo[], int bhi[], int g_c, int clo[], int chi[])
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;
    Integer c = (Integer)g_c;
    Integer andim = ga_ndim_(&a);
    Integer bndim = ga_ndim_(&b);
    Integer cndim = ga_ndim_(&c);
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
    COPYINDEX_C2F(clo,_ga_clo, cndim);
    COPYINDEX_C2F(chi,_ga_chi, cndim);
    ga_elem_multiply_patch_(&a, _ga_alo, _ga_ahi, &b, _ga_blo, _ga_bhi, &c, _ga_clo, _ga_chi);
}

void GA_Elem_divide_patch(int g_a, int alo[], int ahi[],  int g_b, int blo[], int bhi[], int g_c, int clo[], int chi[])
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;
    Integer c = (Integer)g_c;
    Integer andim = ga_ndim_(&a);
    Integer bndim = ga_ndim_(&b);
    Integer cndim = ga_ndim_(&c);
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
    COPYINDEX_C2F(clo,_ga_clo, cndim);
    COPYINDEX_C2F(chi,_ga_chi, cndim);
    ga_elem_divide_patch_(&a, _ga_alo, _ga_ahi, &b, _ga_blo, _ga_bhi, &c, _ga_clo, _ga_chi);
}


void GA_Elem_maximum_patch(int g_a, int alo[], int ahi[],  int g_b, int blo[], int bhi[], int g_c, int clo[], int chi[])
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;
    Integer c = (Integer)g_c;
    Integer andim = ga_ndim_(&a);
    Integer bndim = ga_ndim_(&b);
    Integer cndim = ga_ndim_(&c);
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
    COPYINDEX_C2F(clo,_ga_clo, cndim);
    COPYINDEX_C2F(chi,_ga_chi, cndim);
    ga_elem_maximum_patch_(&a, _ga_alo, _ga_ahi, &b, _ga_blo, _ga_bhi, &c, _ga_clo, _ga_chi);
}

void GA_Elem_minimum_patch(int g_a, int alo[], int ahi[],  int g_b, int blo[], int bhi[], int g_c, int clo[], int chi[])
{
    Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;
    Integer c = (Integer)g_c;
    Integer andim = ga_ndim_(&a);
    Integer bndim = ga_ndim_(&b);
    Integer cndim = ga_ndim_(&c);
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
    COPYINDEX_C2F(clo,_ga_clo, cndim);
    COPYINDEX_C2F(chi,_ga_chi, cndim);
    ga_elem_minimum_patch_(&a, _ga_alo, _ga_ahi, &b, _ga_blo, _ga_bhi, &c, _ga_clo, _ga_chi);
}

void GA_Shift_diagonal(int g_a, void *c){
 Integer a = (Integer )g_a;
 ga_shift_diagonal_(&a, c);
}

void GA_Set_diagonal(int g_a, int g_v){
 Integer a = (Integer )g_a;
 Integer v = (Integer )g_v;
 ga_set_diagonal_(&a, &v);
}

void GA_Zero_diagonal(int g_a){
 Integer a = (Integer )g_a;
 ga_zero_diagonal_(&a);
}
void GA_Add_diagonal(int g_a, int g_v){
 Integer a = (Integer )g_a;
 Integer v = (Integer )g_v;
 ga_add_diagonal_(&a, &v);
}

void GA_Get_diag(int g_a, int g_v){
 Integer a = (Integer )g_a;
 Integer v = (Integer )g_v;
 ga_get_diag_(&a, &v);
}

void GA_Scale_rows(int g_a, int g_v){
 Integer a = (Integer )g_a;
 Integer v = (Integer )g_v;
 ga_scale_rows_(&a, &v);
}

void GA_Scale_cols(int g_a, int g_v){
 Integer a = (Integer )g_a;
 Integer v = (Integer )g_v;
 ga_scale_cols_(&a, &v);
}
void GA_Norm1(int g_a, double *nm){
 Integer a = (Integer )g_a;
 ga_norm1_(&a, nm);
}

void GA_Norm_infinity(int g_a, double *nm){
 Integer a = (Integer )g_a;
 ga_norm_infinity_(&a, nm);
}


void GA_Median(int g_a, int g_b, int g_c, int g_m){
 Integer a = (Integer )g_a;
 Integer b = (Integer )g_b;
 Integer c= (Integer )g_c;
 Integer m = (Integer )g_m;
 ga_median_(&a, &b, &c, &m);
}

void GA_Median_patch(int g_a, int *alo, int *ahi, int g_b, int *blo, int *bhi, int g_c, int *clo, int *chi, int g_m, int *mlo, int *mhi){

   Integer a = (Integer)g_a;
    Integer b = (Integer)g_b;
    Integer c = (Integer)g_c;
    Integer m = (Integer )g_m;
    Integer andim = ga_ndim_(&a);
    Integer bndim = ga_ndim_(&b);
    Integer cndim = ga_ndim_(&c);
    Integer mndim = ga_ndim_(&m);
    COPYINDEX_C2F(alo,_ga_alo, andim);
    COPYINDEX_C2F(ahi,_ga_ahi, andim);
    COPYINDEX_C2F(blo,_ga_blo, bndim);
    COPYINDEX_C2F(bhi,_ga_bhi, bndim);
    COPYINDEX_C2F(clo,_ga_clo, cndim);
    COPYINDEX_C2F(chi,_ga_chi, cndim);
    COPYINDEX_C2F(mlo,_ga_mlo, mndim);
    COPYINDEX_C2F(mhi,_ga_mhi, mndim);
    ga_median_patch_(&a, _ga_alo, _ga_ahi, &b, _ga_blo, _ga_bhi, &c, _ga_clo, _ga_chi, &m, _ga_mlo, _ga_mhi);
}

#ifdef WIN32
#include <armci.h>
#else
#include <../../armci/src/armci.h>
#endif

int GA_Cluster_nnodes()
{
    return armci_domain_count(ARMCI_DOMAIN_SMP);
} 

int GA_Cluster_nodeid() 
{
    return armci_domain_my_id(ARMCI_DOMAIN_SMP);
}

int GA_Cluster_nprocs(int x) 
{
    return armci_domain_nprocs(ARMCI_DOMAIN_SMP,x);
}

int GA_Cluster_procid(int node, int loc_proc)
{
    return armci_domain_glob_proc_id(ARMCI_DOMAIN_SMP,
                                     node, loc_proc);
}
