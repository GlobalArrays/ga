#include <stdio.h>
#include "global.h"
#include "macommon.h"


#if defined(__STDC__) || defined(__cplusplus)
# define ARGS_(s) s
#else
# define ARGS_(s) ()
#endif
extern double sqrt ARGS_((double));
#undef ARGS_


#define FNAM        35              /* length of Fortran names   */
#define DEBUG         0
#define GAinitialized 1
#define MAX_NPROC     2000
#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))

static int GAnproc, GAme;


/*\ CREATE A GLOBAL ARRAY
 *  Fortran version
\*/
#ifdef CRAY_T3D
logical ga_create2_(type, dim1, dim2, array_name, chunk1, chunk2, g_a)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a;
     _fcd array_name;
#else
logical ga_create2_(type, dim1, dim2, array_name, chunk1, chunk2, g_a, slen)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a;
     char* array_name;
     int slen;
#endif
{
char buf[FNAM];
  extern logical ga_create3();
#ifdef CRAY_T3D
      f2cstring(_fcdtocp(array_name), _fcdlen(array_name), buf, FNAM);
#else
      f2cstring(array_name ,slen, buf, FNAM);
#endif

  GAnproc = (int)ga_nnodes_();
  GAme    = (int)ga_nodeid_();
  return(ga_create3(type, dim1, dim2, buf, chunk1, chunk2, g_a));
}

logical ga_create2(type, dim1, dim2, array_name, chunk1, chunk2, g_a)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a; 
     char *array_name;

{
  extern logical ga_create3();

  GAnproc = (int)ga_nnodes_();
  GAme    = (int)ga_nodeid_();
  return(ga_create3(type, dim1, dim2, array_name, chunk1, chunk2, g_a));
}


/* !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 *                               Delete stuff above this  comment.
 * !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
 */


/*\ CREATE A GLOBAL ARRAY
\*/
logical ga_create3(type, dim1, dim2, array_name, chunk1, chunk2, g_a)
     Integer *type, *dim1, *dim2, *chunk1, *chunk2, *g_a; 
     char *array_name;
     /*
      * array_name    - a unique character string [input]
      * type          - MA type [input]
      * dim1/2        - array(dim1,dim2) as in FORTRAN [input]
      * chunk1/2      - minimum size that dimensions should
      *                 be chunked up into [input]
      *                 setting chunk1=dim1 gives distribution by rows
      *                 setting chunk2=dim2 gives distribution by columns 
      *                 Actual chunk sizes are modified so that they are
      *                 at least the min size and each process has either
      *                 zero or one chunk. 
      *                 chunk1/2 <=1 yields even distribution
      * g_a           - Integer handle for future references [output]
      */
{
register int   i, nprocx, nprocy, fchunk1, fchunk2;
static Integer map1[MAX_NPROC], map2[MAX_NPROC];
Integer nblock1, nblock2;

      if(!GAinitialized) ga_error("GA not initialized ", 0);

      ga_sync_();

      if(*type != MT_F_DBL && *type != MT_F_INT)
         ga_error("ga_create: type not yet supported ",  *type);
      else if( *dim1 <= 0 )
         ga_error("ga_create: array dimension1 invalid ",  *dim1);
      else if( *dim2 <= 0)
         ga_error("ga_create: array dimension2 invalid ",  *dim2);

      /*** figure out chunking ***/
      if(*chunk1 <= 1 && *chunk2 <= 1){
        if(*dim1 == 1)      { nprocx =1; nprocy=(int)GAnproc;}
        else if(*dim2 == 1) { nprocy =1; nprocx=(int)GAnproc;}
        else {
           nprocx= (int)sqrt((double)GAnproc);
           for(i=nprocx;i>0&& (GAnproc%i);i--);
           nprocx =(int)i; nprocy=(int)GAnproc/nprocx;
        }

        fchunk1 = (int) MAX(1, *dim1/nprocx);
        fchunk2 = (int) MAX(1, *dim2/nprocy);

        fchunk1 = (int)MIN(fchunk1, *dim1);
        fchunk2 = (int)MIN(fchunk2, *dim2);

        nblock1 = MIN( *dim1, nprocx );
        nblock2 = MIN( *dim2, nprocy );

        map1[0] = 1;
        if( nblock1 > 1 )
          for(i=1; i<= *dim1%nblock1; i++ ) map1[i]=map1[i-1]+fchunk1+1;
        for(i=*dim1%nblock1+1; i< nblock1; i++) map1[i]=map1[i-1]+fchunk1;

        map2[0] = 1;
        if( nblock2 > 1 )
          for(i=1; i<= *dim2%nblock2; i++ ) map2[i]=map2[i-1]+fchunk2+1;
        for(i=*dim2%nblock2+1; i< nblock2; i++) map2[i]=map2[i-1]+fchunk2;

        return( ga_create_irreg(type, dim1, dim2, array_name, map1, &nblock1,
                           map2, &nblock2, g_a) );

      }else if(*chunk1 <= 1){
        fchunk1 = (int) MAX(1, (*dim1 * *dim2)/(GAnproc* *chunk2));
        fchunk2 = (int) *chunk2;
      }else if(*chunk2 <= 1){
        fchunk1 = (int) *chunk1;
        fchunk2 = (int) MAX(1, (*dim1 * *dim2)/(GAnproc* *chunk1));
      }else{
        fchunk1 = (int) MAX(1,  *chunk1);
        fchunk2 = (int) MAX(1,  *chunk2);
      }

      fchunk1 = (int)MIN(fchunk1, *dim1);
      fchunk2 = (int)MIN(fchunk2, *dim2);

      /*** chunk size correction for load balancing ***/
      while(((*dim1-1)/fchunk1+1)*((*dim2-1)/fchunk2+1) >GAnproc){
           if(fchunk1 == *dim1 && fchunk2 == *dim2) 
                     ga_error("ga_create: chunking failed !! ", 0L);
           if(fchunk1 < *dim1) fchunk1 ++; 
           if(fchunk2 < *dim2) fchunk2 ++; 
      }

      /* Now build map arrays */
      for(i=0, nblock1=0; i< *dim1; i += fchunk1, nblock1++) map1[nblock1]=i+1;
      for(i=0, nblock2=0; i< *dim2; i += fchunk2, nblock2++) map2[nblock2]=i+1;   
      if(GAme==0&& DEBUG){
         fprintf(stderr,"blocks (%d,%d)\n",nblock1, nblock2);
         fprintf(stderr,"chunks (%d,%d)\n",fchunk1, fchunk2);
         if(GAme==0){
           for (i=0;i<nblock1;i++)fprintf(stderr," %d ",map1[i]);
           for (i=0;i<nblock2;i++)fprintf(stderr," .%d ",map2[i]);
         }
      }

      return( ga_create_irreg(type, dim1, dim2, array_name, map1, &nblock1,
                         map2, &nblock2, g_a) );
}

