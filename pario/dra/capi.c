#include "global.h"
#include "dra.h"
#include "drap.h"
#include <stdio.h>
#include <stdlib.h>

Integer _da_lo[MAXDIM], _da_hi[MAXDIM], _ga_work[MAXDIM];
Integer _da_dims[MAXDIM];
Integer _da_reqdims[MAXDIM];

Integer _ga_lo[MAXDIM], _ga_hi[MAXDIM];
Integer _da_blo[MAXDIM], _da_bhi[MAXDIM];
Integer _da_clo[MAXDIM], _da_chi[MAXDIM];

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

int DRA_uses_fapi(void)
{
#ifdef USE_FAPI
return 1;
#else
return 0;
#endif
}


int NDRA_Create(int type, int ndim, int dims[], char *name, char* filename,
    int mode, int reqdims[], int *d_a)
{
    Integer ttype, nndim, dd_a, mmode; 
    logical st;
    if (ndim>MAXDIM) return 0;

    COPYC2F(dims, _da_dims, ndim);
    COPYC2F(reqdims, _da_reqdims, ndim);
    ttype = (Integer)type;
    nndim = (Integer)ndim;
    mmode = (Integer)mode;
    
    st = ndra_create(&ttype, &nndim, _da_dims, name,
        filename, &mmode, _da_reqdims, &dd_a);
    *d_a = (int)dd_a;
    if(st==TRUE) return 1;
    else return 0;
}

int NDRA_Inquire(int d_a, int *type, int *ndim, int dims[], char *name,
    char* filename)
{
   Integer  dd_a, ttype, nndim, status;
   dd_a = (Integer)d_a;
   status = ndra_inquire(&dd_a, &ttype, &nndim, _da_dims, name, filename);
   COPYF2C(_da_dims, dims, nndim);
   *type = (Integer)ttype;
   *ndim = (Integer)nndim;
   return (int)status;
}

int NDRA_Write(int g_a, int d_a, int *request)
{
   Integer status, gg_a, dd_a, rrequest;
   gg_a = (Integer)g_a;
   dd_a = (Integer)d_a;
   rrequest = (Integer)*request;
   status = ndra_write_(&gg_a, &dd_a, &rrequest);
   *request = (int)rrequest;
   return (int)status;
}

int NDRA_Read(int g_a, int d_a, int *request)
{
   Integer status, gg_a, dd_a, rrequest;
   gg_a = (Integer)g_a;
   dd_a = (Integer)d_a;
   rrequest = (Integer)*request;
   status = ndra_read_(&gg_a, &dd_a, &rrequest);
   *request = (int)rrequest;
   return (int)status;
}

int NDRA_Write_section(logical transp, int g_a, int glo[], int ghi[],
                       int d_a, int dlo[], int dhi[], int *request)
{
   Integer status;
   Integer ttransp, gg_a, dd_a, rrequest;
   Integer ndim;
   ttransp = (Integer)transp;
   gg_a = (Integer)g_a;
   ndim = ga_ndim_(&gg_a);
   dd_a = (Integer)d_a;
   rrequest = (Integer)*request;

   COPYINDEX_C2F(glo, _ga_lo, ndim);
   COPYINDEX_C2F(ghi, _ga_hi, ndim);
   COPYINDEX_C2F(dlo, _da_lo, ndim);
   COPYINDEX_C2F(dhi, _da_hi, ndim);
   status = ndra_write_section_(&ttransp, &gg_a, _ga_lo, _ga_hi, &dd_a, _da_lo,
          _da_hi, &rrequest);
   *request = (int)rrequest;
   return (int)status;
}

int NDRA_Read_section(logical transp, int g_a, int glo[], int ghi[],
                       int d_a, int dlo[], int dhi[], int *request)
{
   Integer status;
   Integer ttransp, gg_a, dd_a, rrequest;
   Integer ndim;
   ttransp = (Integer)transp;
   gg_a = (Integer)g_a;
   ndim = ga_ndim_(&gg_a);
   dd_a = (Integer)d_a;
   rrequest = (Integer)*request;

   COPYINDEX_C2F(glo, _ga_lo, ndim);
   COPYINDEX_C2F(ghi, _ga_hi, ndim);
   COPYINDEX_C2F(dlo, _da_lo, ndim);
   COPYINDEX_C2F(dhi, _da_hi, ndim);
   status = ndra_read_section_(&ttransp, &gg_a, _ga_lo, _ga_hi, &dd_a, _da_lo,
          _da_hi, &rrequest);
   *request = (int)rrequest;
   return (int)status;
}

int DRA_Init(int max_arrays, double max_array_size,
             double total_disk_space, double max_memory)
{
  Integer mmax_arrays, status;
  DoublePrecision mmax_array_size, ttotal_disk_space, mmax_memory;
  mmax_arrays = (Integer)max_arrays;
  mmax_array_size = (DoublePrecision)max_array_size;
  ttotal_disk_space = (DoublePrecision)total_disk_space;
  mmax_memory = (DoublePrecision)max_memory;
  status = dra_init_(&mmax_arrays, &mmax_array_size,
                     &ttotal_disk_space, &mmax_memory);
  return (int)status;
}

int DRA_Terminate()
{
  Integer status;
  status = dra_terminate_();
  return (int)status;
}

int DRA_Open(char* filename, int mode, int *d_a)
{
  Integer mmode, dd_a, status;
  mmode = (Integer)mode;
  printf ("DRA_Open: DRA handle is %d\n", *d_a);
  status = dra_open(filename, &mode, &dd_a);
  *d_a = (int)dd_a;
  return (int)status;
}

int DRA_Probe(int request, int *compl_status)
{
  Integer rrequest, ccompl_status, status;
  rrequest = (Integer)request;
  status = dra_probe_(&rrequest, &ccompl_status);
  *compl_status = (int)ccompl_status;
  return (int)status;
}

void DRA_Set_debug(logical flag)
{
  dra_set_debug_(&flag);
  return;
}

int DRA_Wait(int request)
{
  Integer rrequest, status;
  rrequest = (Integer)request;
  status = dra_wait_(&rrequest);
  return (int)status;
}

int DRA_Delete(int d_a)
{
  Integer dd_a, status;
  dd_a = (Integer)d_a;
  status = dra_delete_(&dd_a);
  return (int)status;
}

int DRA_Close(int d_a)
{
  Integer dd_a, status;
  printf ("DRA_Close: DRA handle is %d\n", d_a);
  dd_a = (Integer)d_a;
  status = dra_close_(&dd_a);
  return (int)status;
}

void DRA_Flick()
{
  dra_flick_();
}
