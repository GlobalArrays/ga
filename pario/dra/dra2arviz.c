#include <stdio.h>
#include <stdlib.h>
#include "global.h"
#include "../../tcgmsg/ipcv4.0/sndrcv.h"
#include "../../tcgmsg/ipcv4.0/msgtypesc.h"
#include "macommon.h"
#include "dra.h"

#define ERROR(msg,code){printf("ERROR:%s\0",(msg)); fflush(stdout); exit(1);}

#define BUFSIZE 8000000
int main(argc, argv)
int argc;
char **argv;
{
Integer heap=400000, stack=400000;
Integer me, nproc;
char buf[BUFSIZE];
Integer max_arrays=2;
double  max_sz=1e8, max_disk=2e8, max_mem=1e6;
Integer d_a, mode=DRA_R;
Integer g_a,dim1,dim2,rows,cols,block=-1;
char    name[1024], fname[1024];
Integer transp=0, reqid, one=1;
Integer index, ld;
Integer i,j,ilo,ihi,jlo,jhi,type;
size_t size, nitems;

    if(argc<6){
      printf("Usage: dra2arviz <dra_filename> <ilo> <ihi> <jlo> <jhi>\n");
      printf("       dra_filename is the meta-file name for disk resident array\n");
      printf("       [ilo:ihi, jlo:jhi]  array section to read\0\n\n");
      return(1);
    }

    PBEGIN_(argc, argv);
    me=NODEID_();
    nproc=NNODES_();

    heap /= nproc;
    stack /= nproc;
    if(! MA_init((Integer)MT_F_DBL, stack, heap))
       ERROR_("MA_init failed",stack+heap);     /* initialize memory allocator*/
    GA_initialize();                            /* initialize GA */

    if(nproc != 1)ERROR("Error: does not run in parallel",nproc);

    if(DRA_init(&max_arrays, &max_sz, &max_disk, &max_mem))
       ERROR("initialization failed",0);

    if(DRA_open(argv[1],&mode, &d_a)) ERROR("dra_open failed",0);

    ilo = atoi(argv[2]); ihi = atoi(argv[3]);
    jlo = atoi(argv[4]); jhi = atoi(argv[5]);
    rows = ihi - ilo +1; 
    cols = jhi - jlo +1; 


    if(DRA_inquire(&d_a, &type, &dim1, &dim2, name, fname))
        ERROR("dra_inquire failed",0);

    switch (type) {
     case  MT_F_INT:  size = sizeof(Integer); break;
     case  MT_F_DBL:  size = sizeof(DoublePrecision); break;
     case  MT_F_DCPL:  size = sizeof(DoubleComplex); break;
       default: ERROR("type not supported",type);
    }

    if(!GA_create(&type, &rows, &cols, "temp", &block, &block, &g_a))
        ERROR("ga_create failed:",0);

    if(DRA_read_section(&transp, &g_a, &one, &rows, &one, &cols,
                                 &d_a, &ilo, &ihi, &jlo, &jhi, &reqid))
        ERROR("dra_read_section failed",0);

    if(DRA_wait(&reqid)) ERROR("dra_wait failed",0);

    GA_access(&g_a, &one, &rows, &one, &cols, &index, &ld);
    index --; /* adjustment for C addressing */

    if(ld != rows) ERROR("ld != rows",ld); 

    fwrite("OK\0",1,3,stdout);
    nitems = (size_t)rows;
    /* write data by columns */
    for(i=0; i<cols; i++){
       if(type == MT_F_DBL)fwrite(&DBL_MB[index + i*ld],size,nitems,stdout);
       else 
       if(type == MT_F_DCPL)fwrite(&DCPL_MB[index+i*ld],size,nitems,stdout);
       else fwrite(&INT_MB[index + i*ld],size,nitems,stdout);
    }
    fflush(stdout);
    
    GA_destroy(&g_a);

    GA_terminate();
    PEND_();
    return 0;
}
