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
#include "macommon.h"
#include <stdio.h>


#ifdef CRAY_T3D
#include <fortran.h>
#endif

#if !(defined(SGI)||defined(AIX))
#ifndef CRAY_T3D
  extern int fprintf();
#endif
#endif
#if defined(SUN)
  void fflush();
#endif



/*\ COPY ONE GLOBAL ARRAY INTO ANOTHER
\*/
void ga_copy_(g_a, g_b)
     Integer *g_a, *g_b;
{
Integer atype, btype, adim1, adim2, bdim1, bdim2;
Integer ilo, ihi, jlo, jhi;
Integer me= ga_nodeid_(), index, ld;

   ga_sync_();

#ifdef GA_TRACE
       trace_stime_();
#endif

   ga_check_handle(g_a, "ga_copy");
   ga_check_handle(g_b, "ga_copy");

   if(*g_a == *g_b) ga_error("ga_copy: arrays have to different ", 0L);

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype || (atype != MT_F_DBL && atype != MT_F_INT))
               ga_error("ga_copy: wrong types ", 0L);

   if(adim1 != bdim1 || adim2!=bdim2 )
               ga_error("ga_copy: arrays not conformant", 0L);

   ga_distribution_(g_a, &me, &ilo, &ihi, &jlo, &jhi);

   if (  ihi>0 && jhi>0 ){
      ga_access_(g_a, &ilo, &ihi, &jlo, &jhi,  &index, &ld);
      if(atype == MT_F_DBL)
           ga_put_(g_b, &ilo, &ihi, &jlo, &jhi, DBL_MB+index-1, &ld);
      else
           ga_put_(g_b, &ilo, &ihi, &jlo, &jhi, INT_MB+index-1, &ld);
      ga_release_(g_a, &ilo, &ihi, &jlo, &jhi);
   }

#ifdef GA_TRACE
   trace_etime_();
   op_code = GA_OP_COP;
   trace_genrec_(g_a, ilo, ihi, jlo, jhi, &op_code);
#endif

   ga_sync_();
}



/*\ PRINT g_a[ilo:ihi, jlo:jhi]
\*/
void ga_print_patch_(g_a, ilo, ihi, jlo, jhi, pretty)
        Integer *g_a, *ilo, *ihi, *jlo, *jhi, *pretty;
/*
  Pretty = 0 ... spew output out with no formatting
  Pretty = 1 ... format output so that it is readable
*/  
{
#define DEV stdout
#define BUFSIZE 6
#define FLEN 80 
Integer i, j,jj, dim1, dim2, type, ibuf[BUFSIZE], jmax, ld=1 ;
DoublePrecision  dbuf[BUFSIZE];
char name[80];

  ga_sync_();
  ga_check_handle(g_a, "ga_print");
  if(ga_nodeid_() == 0){

     ga_inquire_(g_a,  &type, &dim1, &dim2);
     name[FLEN-1]='\0';
     ga_inquire_name(g_a,  name);
     if (*ilo <= 0 || *ihi > dim1 || *jlo <= 0 || *jhi > dim2){
                      fprintf(stderr,"%d %d %d %d dims: [%d,%d]\n", 
                             *ilo,*ihi, *jlo,*jhi, dim1, dim2);
                      ga_error(" ga_print: indices out of range ", *g_a);
     }

     fprintf(DEV,"\n global array: %s[%d:%d,%d:%d],  handle: %d \n",
             name, *ilo, *ihi, *jlo, *jhi, (int)*g_a);

     if (!*pretty) {
       for (i=*ilo; i <*ihi+1; i++){
         for (j=*jlo; j <*jhi+1; j+=BUFSIZE){
	   jmax = MIN(j+BUFSIZE-1,*jhi);
	   if(type == MT_F_INT){
	     ga_get_(g_a, &i, &i, &j, &jmax, ibuf, &ld);
	     for(jj=0; jj<(jmax-j+1); jj++)
	       fprintf(DEV," %8d",ibuf[jj]);
	     
	   }else if(type == MT_F_DBL){
	     ga_get_(g_a, &i, &i, &j, &jmax, dbuf, &ld);
	     for(jj=0; jj<(jmax-j+1); jj++)
	       fprintf(DEV," %12.6f",dbuf[jj]);
	     
	   }else ga_error("ga_print: wrong type",0);
	   
	 }
	 fprintf(DEV,"\n");
       }
       fflush(DEV);
     }
     else {
       for (j=*jlo; j<*jhi+1; j+=BUFSIZE){
	 jmax = MIN(j+BUFSIZE-1,*jhi);
	 fprintf(DEV, "\n"); fprintf(DEV, "\n");

	 /* Print out column headers */

	 fprintf(DEV, "      ");
	 if (type == MT_F_INT) {
	   for (jj=j; jj<=jmax; jj++)
	     fprintf(DEV, "%6d  ", jj);
	   fprintf(DEV,"\n      ");
	   for (jj=j; jj<=jmax; jj++)
	     fprintf(DEV," -------");
	 }
	 else {
	   for (jj=j; jj<=jmax; jj++)
	     fprintf(DEV,"%8d    ", jj);
	   fprintf(DEV,"\n      ");
	   for (jj=j; jj<=jmax; jj++)
	     fprintf(DEV," -----------");
	 }
	 fprintf(DEV,"\n");
	   
	 for (i=*ilo; i <*ihi+1; i++){
	   fprintf(DEV,"%4i  ",i);
	   if(type == MT_F_INT){
	     ga_get_(g_a, &i, &i, &j, &jmax, ibuf, &ld);
	     for(jj=0; jj<(jmax-j+1); jj++)
	       fprintf(DEV,"%8d",ibuf[jj]);
	     
	   }else if(type == MT_F_DBL){
	     ga_get_(g_a, &i, &i, &j, &jmax, dbuf, &ld);
	     for(jj=0; jj<(jmax-j+1); jj++)
	       fprintf(DEV,"%12.6f",dbuf[jj]);
	     
	   }else ga_error("ga_print: wrong type",0);
	   fprintf(DEV,"\n");
	 }
       }
       fflush(DEV);
     }
   }
       
  ga_sync_();
}


void ga_print_(g_a)
     Integer *g_a;
{
  Integer type, dim1, dim2;
  Integer ilo=1, jlo=1;
  Integer pretty = 1;

  ga_inquire_(g_a, &type, &dim1, &dim2);

  ga_print_patch_(g_a, &ilo, &dim1, &jlo, &dim2, &pretty);
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

    ga_clean_mem(); 

    if (ga_nnodes_() > 1){
#      ifdef SYSV
          /* TCGMSG */
          if (SR_caught_sigint)fprintf(stderr,"%s %ld",string,icode);
#      endif
       Error(string,  icode);
    }
    fprintf(stderr,"%s %ld",string,icode);
    fflush(stderr);
#   if defined(SUN) || defined(SGI)
       abort();
#   else
       exit(1);
#   endif
}




/*\  ERROR TERMINATION
 *   Fortran version
\*/
#ifdef CRAY_T3D
void ga_error_(string, icode)
     _fcd        string;
#else
void ga_error_(string, icode, slen)
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
char *strncpy();
int clen = strlen(cstring);
    strncpy(fstring, cstring, flen);
    /* remove \n character if any */
    if(flen-- >clen)fstring[clen]=' ';
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


