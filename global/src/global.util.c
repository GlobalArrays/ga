/*$Id: global.util.c,v 1.25 1999-07-28 00:39:19 d3h325 Exp $*/
/*
 * module: global.util.c
 * author: Jarek Nieplocha
 * last modification: Tue Dec 20 09:41:55 PDT 1994
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

#include "global.h"
#include "globalp.h"
#include <stdio.h>
#include <string.h>
#ifndef WIN32
#include <unistd.h>
#endif


#ifdef CRAY
#include <fortran.h>
#endif

#if defined(SUN)
  void fflush();
#endif




/*\ PRINT g_a[ilo:ihi, jlo:jhi]
\*/
void FATR ga_print_patch_(g_a, ilo, ihi, jlo, jhi, pretty)
        Integer *g_a, *ilo, *ihi, *jlo, *jhi, *pretty;
/*
  Pretty = 0 ... spew output out with no formatting
  Pretty = 1 ... format output so that it is readable
*/  
{
#define DEV stdout
#define BUFSIZE 6
#define FLEN 80 
Integer i, j,jj, dim1, dim2, type, ibuf[BUFSIZE], jmax, ld=1, bufsize ;
DoublePrecision  dbuf[BUFSIZE];
char *name;

  ga_sync_();
  ga_check_handle(g_a, "ga_print");
  if(ga_nodeid_() == 0){

     ga_inquire_(g_a,  &type, &dim1, &dim2);
/*     name[FLEN-1]='\0';*/
     ga_inquire_name(g_a,  &name);
     if (*ilo <= 0 || *ihi > dim1 || *jlo <= 0 || *jhi > dim2){
                      fprintf(stderr,"%d %d %d %d dims: [%d,%d]\n", 
                             *ilo,*ihi, *jlo,*jhi, dim1, dim2);
                      ga_error(" ga_print: indices out of range ", *g_a);
     }

     fprintf(DEV,"\n global array: %s[%d:%d,%d:%d],  handle: %d \n",
             name, *ilo, *ihi, *jlo, *jhi, (int)*g_a);

     bufsize = (type==MT_F_DCPL)? BUFSIZE/2 : BUFSIZE;


     if (!*pretty) {
       for (i=*ilo; i <*ihi+1; i++){
         for (j=*jlo; j <*jhi+1; j+=bufsize){
           jmax = MIN(j+bufsize-1,*jhi);
           switch(type){
              case MT_F_INT:
                   ga_get_(g_a, &i, &i, &j, &jmax, ibuf, &ld);
                   for(jj=0; jj<(jmax-j+1); jj++)
                     fprintf(DEV," %8d",ibuf[jj]);
                   break;

              case MT_F_DBL:
                   ga_get_(g_a, &i, &i, &j, &jmax, dbuf, &ld);
                   for(jj=0; jj<(jmax-j+1); jj++)
                     fprintf(DEV," %11.5f",dbuf[jj]);
                   break;

              case MT_F_DCPL:
                   ga_get_(g_a, &i, &i, &j, &jmax, dbuf, &ld);
                   for(jj=0; jj<(jmax-j+1); jj+=2)
                     fprintf(DEV," %11.5f,%11.5f",dbuf[jj], dbuf[jj+1]);
                   break;
              default: ga_error("ga_print: wrong type",0);
           }
         }
         fprintf(DEV,"\n");
       }
       fflush(DEV);

     } else {

        for (j=*jlo; j<*jhi+1; j+=bufsize){
        jmax = MIN(j+bufsize-1,*jhi);

           fprintf(DEV, "\n"); fprintf(DEV, "\n");

           /* Print out column headers */

           fprintf(DEV, "      ");
           switch(type){
              case MT_F_INT:
                   for (jj=j; jj<=jmax; jj++) fprintf(DEV, "%6d  ", jj);
                   fprintf(DEV,"\n      ");
                   for (jj=j; jj<=jmax; jj++) fprintf(DEV," -------");
                   break;
              case MT_F_DCPL:
                   for (jj=j; jj<=jmax; jj++) fprintf(DEV,"%20d    ", jj);
                   fprintf(DEV,"\n      ");
                   for (jj=j; jj<=2*jmax; jj++) fprintf(DEV," -----------");
                   break;
              case MT_F_DBL:
                   for (jj=j; jj<=jmax; jj++) fprintf(DEV,"%8d    ", jj);
                   fprintf(DEV,"\n      ");
                   for (jj=j; jj<=jmax; jj++) fprintf(DEV," -----------");
           }
           fprintf(DEV,"\n");

           for(i=*ilo; i <*ihi+1; i++){
              fprintf(DEV,"%4i  ",i);

              switch(type){
                 case MT_F_INT:
                      ga_get_(g_a, &i, &i, &j, &jmax, ibuf, &ld);
                      for(jj=0; jj<(jmax-j+1); jj++)
                        fprintf(DEV," %8d",ibuf[jj]);
                      break;

                 case MT_F_DBL:
                      ga_get_(g_a, &i, &i, &j, &jmax, dbuf, &ld);
                      for(jj=0; jj<(jmax-j+1); jj++)
                        fprintf(DEV," %11.5f",dbuf[jj]);
                      break;

                 case MT_F_DCPL:
	              ga_get_(g_a, &i, &i, &j, &jmax, dbuf, &ld);
	              for(jj=0; jj<(jmax-j+1); jj+=2)
	                fprintf(DEV," %11.5f,%11.5f",dbuf[jj], dbuf[jj+1]);
                      break;
                 default: ga_error("ga_print: wrong type",0);
	     }
	     fprintf(DEV,"\n");
         }
         fflush(DEV);
      }
    }
  }
       
  ga_sync_();
}


void FATR ga_print_stats_()
{
int i;
     GAstat_arr = (long*)&GAstat;
     printf("\n                         GA Statistics for process %4d\n",ga_nodeid_());
     printf("                         ------------------------------\n\n");
     printf("       create   destroy   get      put      acc     scatter   gather  read&inc\n");

     printf("calls: ");
     for(i=0;i<8;i++) 
        if(GAstat_arr[i] < 9999) printf("%4ld     ",GAstat_arr[i]);
        else                   printf("%.2e ",(double)GAstat_arr[i]);
     printf("\n");

     printf("bytes total:             %.2e %.2e %.2e %.2e %.2e %.2e\n",
                   GAbytes.gettot, GAbytes.puttot, GAbytes.acctot,
                   GAbytes.scatot, GAbytes.gattot, GAbytes.rditot);

     printf("bytes remote:            %.2e %.2e %.2e %.2e %.2e %.2e\n",
                   GAbytes.gettot - GAbytes.getloc, 
                   GAbytes.puttot - GAbytes.putloc,
                   GAbytes.acctot - GAbytes.accloc,
                   GAbytes.scatot - GAbytes.scaloc,
                   GAbytes.gattot - GAbytes.gatloc,
                   GAbytes.rditot - GAbytes.rdiloc);
     printf("Max memory consumed for GA by this process: %ld bytes\n",GAstat.maxmem);
     if(GAstat.numser)
        printf("Number of requests serviced: %ld\n",GAstat.numser);
     fflush(stdout);
}

   
 

/*\  ERROR TERMINATION
 *   C-version
\*/
void ga_error(string, icode)
     char     *string;
     Integer  icode;
{
extern void Error();
#ifdef SYSV
   extern int SR_caught_sigint;
#endif
#ifdef CRAY_T3D 
#  define FOUT stdout
#else
#  define FOUT stderr
#endif
#define ERR_LEN 400
    int level;
    char error_buffer[ERR_LEN];

    ga_clean_resources(); 

    /* print GA names stack */
    sprintf(error_buffer,"%d:", ga_nodeid_());
    for(level = 0; level < GA_stack_size; level++){
       strcat(error_buffer,GA_name_stack[level]);
       strcat(error_buffer,":");
    }
    strcat(error_buffer,string);
    strcat(error_buffer,":");
       
    if (ga_nnodes_() > 1) Error(error_buffer, icode);
    else{
      fprintf(FOUT,"%s %ld\n",error_buffer,icode);
      perror("system message:");
      fflush(FOUT);
      exit(1);
    }
}




/*\  ERROR TERMINATION
 *   Fortran version
\*/
#ifdef CRAY_T3D
void FATR ga_error_(string, icode)
     _fcd        string;
#else
void FATR ga_error_(string, icode, slen)
     char        *string;
     int         slen;
#endif
     Integer     *icode;
{
#define FMSG 256
char buf[FMSG];
#ifdef CRAY_T3D
      f2cstring(_fcdtocp(string), _fcdlen(string), buf, FMSG);
#else
      f2cstring(string,slen, buf, FMSG);
#endif
      ga_error(buf,*icode);
}





/************** Fortran - C conversion routines for strings ************/

/*\ converts C strings to  Fortran strings
\*/
void c2fstring( cstring, fstring, flen)
     char *cstring, *fstring;
     Integer flen;
{
    int clen = strlen(cstring);

    /* remove terminal \n character if any */

    if(cstring[clen] == '\n') clen--;

    /* Truncate C string into Fortran string */

    if (clen > flen) clen = flen;

    /* Copy characters over */

    flen -= clen;
    while (clen--)
	*fstring++ = *cstring++;

    /* Now terminate with blanks */

    while (flen--)
	*fstring++ = ' ';
}


/*\
 * Strip trailing blanks from fstring and copy it to cstring,
 * truncating if necessary to fit in cstring, and ensuring that
 * cstring is NUL-terminated.
\*/
void f2cstring(fstring, flength, cstring, clength)
    char        *fstring;       /* FORTRAN string */
    Integer      flength;        /* length of fstring */
    char        *cstring;       /* C buffer */
    Integer      clength;        /* max length (including NUL) of cstring */
{
    /* remove trailing blanks from fstring */
    while (flength-- && fstring[flength] == ' ') ;

    /* the postdecrement above went one too far */
    flength++;

    /* truncate fstring to cstring size */
    if (flength >= clength)
        flength = clength - 1;

    /* ensure that cstring is NUL-terminated */
    cstring[flength] = '\0';

    /* copy fstring to cstring */
    while (flength--)
        cstring[flength] = fstring[flength];
}


void ga_debug_suspend()
{
#ifdef SYSV
#  include <sys/types.h>
#  include <unistd.h>

   fprintf(stdout,"ga_debug: process %ld ready for debugging\n",
           (long)getpid());
   fflush(stdout);
   pause();

#endif
}








#ifdef ARMCI

/*********************************************************************
 *        N-dimensional operations                                   *
 *********************************************************************/


/*\ print range of n-dimensional array with two strings before and after
\*/
static void gai_print_range(char *pre,int ndim, 
                            Integer lo[], Integer hi[], char* post)
{
        int i;

        printf("%s[",pre);
        for(i=0;i<ndim;i++){
                printf("%d:%d",lo[i],hi[i]);
                if(i==ndim-1)printf("] %s",post);
                else printf(",");
        }
}



/*\ print array distribution to stdout
\*/
void FATR ga_print_distribution_(Integer* g_a)
{
Integer ndim, i, proc, type, nproc=ga_nnodes_();
Integer dims[MAXDIM], lo[MAXDIM], hi[MAXDIM];
char msg[100];

    ga_sync_();

    nga_inquire_(g_a, &type, &ndim, dims);
    printf("Array handle=%d name:'%s' ",g_a, ga_inquire_name_(g_a));
    printf("data type:");
    switch(type){
      case MT_F_DBL: printf("double"); break;
      case MT_F_INT: printf("integer"); break;
      case MT_F_DCPL: printf("double complex"); break;
      default: ga_error("ga_print_distribution: type not supported",type);
    }
    printf(" dimensions:");
    for(i=0; i<ndim-1; i++)printf("%dx",dims[i]);
    printf("%d\n",dims[ndim-1]);

    /* now everybody prints array range it owns */
    for(proc = 0; proc < nproc; proc++){
        nga_distribution_(g_a,&proc,lo,hi);
        sprintf(msg,"proc=%d\t owns array section: ",proc);
        gai_print_range(msg,(int)ndim,lo,hi,"\n");
    }
    fflush(stdout);

    ga_sync_();
}


/*
 * Jialin added nga_print and nga_print_patch on Jun 28, 1999
 */

/*\ PRINT g_a[ilo, jlo]
\*/
void FATR nga_print_patch_(g_a, lo, hi, pretty)
        Integer *g_a, *lo, *hi, *pretty;
/*
  Pretty = 0 ... spew output out with no formatting
  Pretty = 1 ... format output so that it is readable
*/  
{
#define DEV stdout
#define BUFSIZE 6
#define FLEN 80 

    Integer i, j, jj, jmax;
    Integer type;
    char *name;
    Integer ndim, dims[MAXDIM], ld[MAXDIM];
    Integer bufsize;
    Integer ibuf[BUFSIZE], ibuf_2d[BUFSIZE*BUFSIZE];
    DoublePrecision dbuf[BUFSIZE], dbuf_2d[BUFSIZE*BUFSIZE];
    Integer lop[MAXDIM], hip[MAXDIM];
    Integer done, status_2d, status_3d;
    ga_sync_();
    ga_check_handle(g_a, "nga_print");

    /* only the first process print the array */
    if(ga_nodeid_() == 0) {
        
        nga_inquire_(g_a,  &type, &ndim, dims);
        ga_inquire_name(g_a,  &name);
        
        /* check the boundary */
        for(i=0; i<ndim; i++)
            if(lo[i] <= 0 || hi[i] > dims[i]) 
                ga_error("g_a indices out of range ", *g_a);
        
        /* print the general information */
        fprintf(DEV,"\n global array: %s[", name);
        for(i=0; i<ndim; i++)
            if(i != (ndim-1)) fprintf(DEV, "%d:%d,", lo[i], hi[i]);
            else fprintf(DEV, "%d:%d", lo[i], hi[i]);
        fprintf(DEV,"],  handle: %d \n", (int)*g_a);
        
        bufsize = (type==MT_F_DCPL)? BUFSIZE/2 : BUFSIZE;
        
        for(i=0; i<ndim; i++) ld[i] = bufsize;
        
        if(!*pretty) {
            done = 1;
            for(i=0; i<ndim; i++) {
                lop[i] = lo[i]; hip[i] = lo[i];
            }
            hip[0] = MIN(lop[0]+bufsize-1, hi[0]);
            while(done) {
                switch(type) {
                    case MT_F_INT: nga_get_(g_a, lop, hip, ibuf, ld); break;
                    case MT_F_DBL: nga_get_(g_a, lop, hip, dbuf, ld); break;
                    case MT_F_DCPL: nga_get_(g_a, lop, hip, dbuf, ld); break;
                    default: ga_error("ga_print: wrong type",0);
                }
                
                /* print the array */
                for(i=0; i<(hip[0]-lop[0]+1); i++) {
                    fprintf(DEV,"%s(", name);
                    for(j=0; j<ndim; j++)
                        if((j == 0) && (j == (ndim-1)))
                            fprintf(DEV, "%d", lop[j]+i);
                        else if((j != 0) && (j == (ndim-1)))
                            fprintf(DEV, "%d", lop[j]);
                        else if((j == 0) && (j != (ndim-1)))
                            fprintf(DEV, "%d,", lop[j]+i);
                        else fprintf(DEV, "%d,", lop[j]);
                    switch(type) {
                        case MT_F_INT: fprintf(DEV,") = %d\n", ibuf[i]);break;
                        case MT_F_DBL:
                            if((double)dbuf[i]<100000.0)
                                fprintf(DEV,") = %f\n", dbuf[i]);
                            else fprintf(DEV,") = %e\n", dbuf[i]);
                            break;
                        case MT_F_DCPL:
                            if(((double)dbuf[i*2]<100000.0) &&
                               ((double)dbuf[i*2+1]<100000.0))
                                fprintf(DEV,") = (%f,%f)\n",
                                        dbuf[i*2],dbuf[i*2+1]);
                            else
                                fprintf(DEV,") = (%e,%e)\n",
                                        dbuf[i*2],dbuf[i*2+1]);
                    }
                }
                
                fflush(DEV);
                
                lop[0] = hip[0]+1; hip[0] = MIN(lop[0]+bufsize-1, hi[0]);
                
                for(i=0; i<ndim; i++)
                    if(lop[i] > hi[i]) 
                        if(i == (ndim-1)) done = 0;
                        else {
                            lop[i] = lo[i];
                            if(i == 0) hip[i] = MIN(lop[i]+bufsize-1, hi[i]);
                            else hip[i] = lo[i];
                            lop[i+1]++; hip[i+1]++;
                        }
            }
        }
        else {
            /* pretty print */
            done = 1;
            for(i=0; i<ndim; i++) {
                lop[i] = lo[i];
                if((i == 0) || (i == 1))
                    hip[i] = MIN(lop[i]+bufsize-1, hi[i]);
                else 
                    hip[i] = lo[i];
            }
            
            status_2d = 1; status_3d = 1;
            
            while(done) {
                if(status_3d && (ndim > 2)) { /* print the patch info */
                    fprintf(DEV,"\n -- patch: %s[", name);
                    for(i=0; i<ndim; i++)
                        if(i < 2)
                            if(i != (ndim-1))
                                fprintf(DEV, "%d:%d,", lo[i], hi[i]);
                            else fprintf(DEV, "%d:%d", lo[i], hi[i]);
                        else
                            if(i != (ndim-1)) fprintf(DEV, "%d,", lop[i]);
                            else fprintf(DEV, "%d", lop[i]);
                    fprintf(DEV,"]\n"); status_3d = 0;
                }
                
                if(status_2d &&(ndim > 1)) {
                    fprintf(DEV, "\n"); 
                    switch(type) {
                        case MT_F_INT:
                            fprintf(DEV, "     ");
                            for (i=lop[1]; i<=hip[1]; i++)
                                fprintf(DEV, "%7d  ", i);
                            fprintf(DEV,"\n      ");
                            for (i=lop[1]; i<=hip[1]; i++)
                                fprintf(DEV," --------");
                            break;
                        case MT_F_DBL:
                            fprintf(DEV, "   ");
                            for (i=lop[1]; i<=hip[1]; i++)
                                fprintf(DEV, "%10d  ", i);
                            fprintf(DEV,"\n      ");
                            for (i=lop[1]; i<=hip[1]; i++)
                                fprintf(DEV," -----------");
                            break;
                        case MT_F_DCPL:
                            for (i=lop[1]; i<=hip[1]; i++)
                                fprintf(DEV, "%22d  ", i);
                            fprintf(DEV,"\n      ");
                            for (i=lop[1]; i<=hip[1]; i++)
                                fprintf(DEV," -----------------------");
                    }
                    
                    fprintf(DEV,"\n");
                    status_2d = 0;
                }
                
                switch(type) {
                    case MT_F_INT: nga_get_(g_a, lop, hip, ibuf_2d, ld); break;
                    case MT_F_DBL: nga_get_(g_a, lop, hip, dbuf_2d, ld); break;
                    case MT_F_DCPL: nga_get_(g_a, lop, hip, dbuf_2d, ld);break;
                    default: ga_error("ga_print: wrong type",0);
                }
                
                for(i=0; i<(hip[0]-lop[0]+1); i++) {
                    fprintf(DEV,"%4i  ", (lop[0]+i));
                    switch(type) {
                        case MT_F_INT:
                            if(ndim > 1)
                                for(j=0; j<(hip[1]-lop[1]+1); j++)
                                    fprintf(DEV," %8d", ibuf_2d[j*bufsize+i]);
                            else fprintf(DEV," %8d", ibuf_2d[i]);
                            break;
                        case MT_F_DBL:
                            if(ndim > 1)
                                for(j=0; j<(hip[1]-lop[1]+1); j++)
                                    if((double)dbuf_2d[j*bufsize+i]<100000.0)
                                        fprintf(DEV," %11.5f",
                                                dbuf_2d[j*bufsize+i]);
                                    else
                                        fprintf(DEV," %.5e",
                                                dbuf_2d[j*bufsize+i]);
                            else
                                if((double)dbuf_2d[i]<100000.0)
                                    fprintf(DEV," %11.5f",dbuf_2d[i]);
                                else
                                    fprintf(DEV," %.5e",dbuf_2d[i]);
                            break;
                        case MT_F_DCPL:
                            if(ndim > 1)
                                for(j=0; j<(hip[1]-lop[1]+1); j++)
                                    if(((double)dbuf_2d[(j*bufsize+i)*2]<100000.0)&&((double)dbuf_2d[(j*bufsize+i)*2+1]<100000.0))
                                        fprintf(DEV," %11.5f,%11.5f",
                                                dbuf_2d[(j*bufsize+i)*2],
                                                dbuf_2d[(j*bufsize+i)*2+1]);
                                    else
                                        fprintf(DEV," %.5e,%.5e",
                                                dbuf_2d[(j*bufsize+i)*2],
                                                dbuf_2d[(j*bufsize+i)*2+1]);
                            else
                                if(((double)dbuf_2d[i*2]<100000.0) &&
                                   ((double)dbuf_2d[i*2+1]<100000.0))
                                    fprintf(DEV," %11.5f,%11.5f",
                                            dbuf_2d[i*2], dbuf_2d[i*2+1]);
                                else
                                    fprintf(DEV," %.5e,%.5e",
                                            dbuf_2d[i*2], dbuf_2d[i*2+1]);
                    }
                    
                    fprintf(DEV,"\n");
                }
                
                lop[0] = hip[0]+1; hip[0] = MIN(lop[0]+bufsize-1, hi[0]);
                
                for(i=0; i<ndim; i++)
                    if(lop[i] > hi[i]) 
                        if(i == (ndim-1)) done = 0;
                        else {
                            lop[i] = lo[i];
                            
                            if((i == 0) || (i == 1))
                                hip[i] = MIN(lop[i]+bufsize-1, hi[i]);
                            else hip[i] = lo[i];
                            
                            if(i == 0) {
                                lop[i+1] = hip[i+1]+1;
                                hip[i+1] = MIN(lop[i+1]+bufsize-1, hi[i+1]);
                            }
                            else {
                                lop[i+1]++; hip[i+1]++;
                            }
                            
                            if(i == 0) status_2d = 1;
                            if(i == 1) status_3d = 1;
                        }
            }
        }
    }
    
    ga_sync_();
}

#endif

void FATR ga_print_(Integer *g_a)
{
#ifdef ARMCI    
    Integer i;
    Integer type, ndim, dims[MAXDIM];
    Integer lo[MAXDIM];
    Integer pretty = 1;

    nga_inquire_(g_a, &type, &ndim, dims);

    for(i=0; i<ndim; i++) lo[i] = 1;

    nga_print_patch_(g_a, lo, dims, &pretty);

#else
    Integer type, dim1, dim2;
    Integer ilo=1, jlo=1;
    Integer pretty = 1;
    
    ga_inquire_(g_a, &type, &dim1, &dim2);
    
    ga_print_patch_(g_a, &ilo, &dim1, &jlo, &dim2, &pretty);
#endif    
}
  
