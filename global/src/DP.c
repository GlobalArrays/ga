/* $Id: DP.c,v 1.10 1999-07-28 00:27:02 d3h325 Exp $ */
#include "global.h"
#include "globalp.h"
#include "macommon.h"

#ifdef CRAY_T3D
#      include <fortran.h>
#      define cptofcd(fcd)  _cptofcd((fcd),1)
#else
#      define cptofcd(fcd) (fcd)
#endif


/*\ check if I own the patch
\*/
static logical own_patch(g_a, ilo, ihi, jlo, jhi)
     Integer *g_a, ilo, ihi, jlo, jhi;
{
   Integer ilop, ihip, jlop, jhip, me=ga_nodeid_();

   ga_distribution_(g_a, &me, &ilop, &ihip, &jlop, &jhip);
   if(ihip != ihi || ilop != ilo || jhip != jhi || jlop != jlo) return(FALSE);
   else return(TRUE);
}

static logical patch_intersect(ilo, ihi, jlo, jhi, ilop, ihip, jlop, jhip)
     Integer *ilo, *ihi, *jlo, *jhi;
     Integer *ilop, *ihip, *jlop, *jhip;
{
     /* check consistency of patch coordinates */
     if( *ihi < *ilo || *jhi < *jlo)     return FALSE; /* inconsistent */
     if( *ihip < *ilop || *jhip < *jlop) return FALSE; /* inconsistent */

     /* find the intersection and update (ilop: ihip, jlop: jhip) */
     if( *ihi < *ilop || *ihip < *ilo) return FALSE; /* don't intersect */
     if( *jhi < *jlop || *jhip < *jlo) return FALSE; /* don't intersect */
     *ilop = MAX(*ilo,*ilop);
     *ihip = MIN(*ihi,*ihip);
     *jlop = MAX(*jlo,*jlop);
     *jhip = MIN(*jhi,*jhip);

     return TRUE;
}


/*\ COPY A PATCH 
 *
 *  . identical shapes 
 *  . copy by column order - Fortran convention
\*/
void ga_copy_patch_dp(t_a, g_a, ailo, aihi, ajlo, ajhi,
                   g_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;
     char *t_a;
{
Integer atype, btype, adim1, adim2, bdim1, bdim2;
Integer ilos, ihis, jlos, jhis;
Integer ilod, ihid, jlod, jhid, corr, nelem;
Integer me= ga_nodeid_(), index, ld, i,j;
Integer indexT, handleT, ldT;
char transp;


   ga_check_handle(g_a, "ga_copy_patch_dp");
   ga_check_handle(g_b, "ga_copy_patch_dp");

   /* if(*g_a == *g_b) ga_error("ga_copy_patch_dp: arrays have to different ", 0L); */

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype || (atype != MT_F_DBL ))
      ga_error("ga_copy_patch_dp: wrong types ", 0L);

   /* check if patch indices and dims match */
   if (*ailo <= 0 || *aihi > adim1 || *ajlo <= 0 || *ajhi > adim2)
       ga_error(" ga_copy_patch_dp: g_a indices out of range ", 0L);
   if (*bilo <= 0 || *bihi > bdim1 || *bjlo <= 0 || *bjhi > bdim2)
       ga_error(" ga_copy_patch_dp: g_b indices out of range ", 0L);

   /* check if numbers of elements in two patches match each other */
   if (((*bihi - *bilo + 1)  != (*aihi - *ailo + 1)) || 
      ( (*bjhi - *bjlo + 1)  != (*ajhi - *ajlo + 1)) )
       ga_error(" ga_copy_patch_dp: shapes two of patches do not match ", 0L);

    /* is transpose operation required ? */
   transp = (*t_a == 'n' || *t_a =='N')? 'n' : 't';

   /* now find out cordinates of a patch of g_a that I own */
   ga_distribution_(g_a, &me, &ilos, &ihis, &jlos, &jhis);

   if(patch_intersect(ailo, aihi, ajlo, ajhi, &ilos, &ihis, &jlos, &jhis)){
      ga_access_(g_a, &ilos, &ihis, &jlos, &jhis, &index, &ld);

      nelem = (ihis-ilos+1)*(jhis-jlos+1);
      index --;     /* fortran to C conversion */

      if ( transp == 'n' ) {
	  corr  = *bilo - *ailo;
	  ilod  = ilos + corr; 
	  ihid  = ihis + corr;
	  corr  = *bjlo - *ajlo;
	  jlod  = jlos + corr; 
	  jhid  = jhis + corr;
      } else {
	  /* If this is a transpose copy, we need local scratch space */
	  if ( !MA_push_get(MT_F_DBL, nelem, "ga_copy_patch_dp",
			    &handleT, &indexT))
	      ga_error(" ga_copy_patch_dp: MA failed ", 0L);

	  /* Copy from the source into this local array, transposed */
	  ldT = jhis-jlos+1;
	  
	  for(j=0; j< jhis-jlos+1; j++)
	      for(i=0; i< ihis-ilos+1; i++)
		  *(DBL_MB+indexT + i*ldT + j) = *(DBL_MB+index + j*ld + i);

	  /* Now we can reset index to point to the transposed stuff */
	  index = indexT;
	  ld = ldT;

	  /* And finally, figure out what the destination indices are */
	  corr  = *bilo - *ajlo;
	  ilod  = jlos + corr; 
	  ihid  = jhis + corr;
	  corr  = *bjlo - *ailo;
	  jlod  = ilos + corr; 
	  jhid  = ihis + corr;
      }
	  
      /* Put it where it belongs */
      ga_put_(g_b, &ilod, &ihid, &jlod, &jhid, DBL_MB + index, &ld);

      /* Get rid of local memory if we used it */
      if( transp == 't') MA_pop_stack(handleT);
  }
}

/*\ COPY A PATCH
 *  Fortran interface
\*/
void ga_copy_patch_dp_(trans, g_a, ailo, aihi, ajlo, ajhi,
                    g_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;
#ifdef CRAY_T3D
     _fcd    trans;
{ga_copy_patch_dp(_fcdtocp(trans),g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi);}
#else 
     char*   trans;
{  ga_copy_patch_dp(trans,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi); }
#endif



DoublePrecision ga_ddot_patch_dp(g_a, t_a, ailo, aihi, ajlo, ajhi,
                                  g_b, t_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     char    *t_a, *t_b;                          /* transpose operators */
{
Integer atype, btype, adim1, adim2, bdim1, bdim2;
Integer iloA, ihiA, jloA, jhiA, indexA, ldA;
Integer iloB, ihiB, jloB, jhiB, indexB, ldB;
Integer g_A = *g_a;
Integer me= ga_nodeid_(), i, j, temp_created=0;
Integer handleB, corr, nelem;
char    transp, transp_a, transp_b;
DoublePrecision  sum = 0.;


   ga_check_handle(g_a, "ga_ddot_patch_dp");
   ga_check_handle(g_b, "ga_ddot_patch_dp");

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype || (atype != MT_F_DBL ))
      ga_error("ga_ddot_patch_dp: wrong types ", 0L);

  /* check if patch indices and g_a dims match */
   if (*ailo <= 0 || *aihi > adim1 || *ajlo <= 0 || *ajhi > adim2)
      ga_error(" ga_ddot_patch_dp: g_a indices out of range ", 0L);

   /* check if patch indices and g_b dims match */
   if (*bilo <= 0 || *bihi > bdim1 || *bjlo <= 0 || *bjhi > bdim2)
       ga_error(" ga_ddot_patch_dp: g_b indices out of range ", 0L);


   /* is transpose operation required ? */
   /* -- only if for one array transpose operation requested*/
   transp_a = (*t_a == 'n' || *t_a =='N')? 'n' : 't';
   transp_b = (*t_b == 'n' || *t_b =='N')? 'n' : 't';
   transp   = (transp_a == transp_b)? 'n' : 't';
   if(transp == 't')
          ga_error(" ga_ddot_patch_dp: transpose operators don't match: ", me);


   /* find out coordinates of patches of g_A and g_B that I own */
   ga_distribution_(&g_A, &me, &iloA, &ihiA, &jloA, &jhiA);

   if (patch_intersect(ailo, aihi, ajlo, ajhi, &iloA, &ihiA, &jloA, &jhiA)){
       ga_access_(&g_A, &iloA, &ihiA, &jloA, &jhiA, &indexA, &ldA);
       indexA --;
       nelem = (ihiA-iloA+1)*(jhiA-jloA+1);

       corr  = *bilo - *ailo;
       iloB  = iloA + corr;
       ihiB  = ihiA + corr;
       corr  = *bjlo - *ajlo;
       jloB  = jloA + corr;
       jhiB  = jhiA + corr;

      if(own_patch(g_b, iloB, ihiB, jloB, jhiB)){
         /* all the data is local */
         ga_access_(g_b, &iloB, &ihiB, &jloB, &jhiB, &indexB, &ldB);
         indexB--;
      }else{
         /* data is remote -- get it to temp storage*/
         temp_created =1;
         if(!MA_push_get(MT_F_DBL,nelem, "ddot_dp_b", &handleB, &indexB))
             ga_error(" ga_ddot_patch_dp: MA failed ", 0L);
         /* no need to adjust index (indexB--;) -- we got it from MA*/

         ldB   = ihiB-iloB+1; 
         ga_get_(g_b, &iloB, &ihiB, &jloB, &jhiB, DBL_MB+indexB, &ldB);
      }

      sum = 0.;
      for(j=0; j< jhiA-jloA+1; j++)
          for(i=0; i< ihiA-iloA+1; i++)
             sum += *(DBL_MB+indexA + j*ldA + i) * 
                    *(DBL_MB+indexB + j*ldB + i);

      if(temp_created)MA_pop_stack(handleB);
   }
   return sum;
}

      
/*\ compute DOT PRODUCT of two patches
 *  Fortran interface
\*/
DoublePrecision ga_ddot_patch_dp_(g_a, t_a, ailo, aihi, ajlo, ajhi,
                               g_b, t_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */

#ifdef CRAY_T3D
     _fcd   t_a, t_b;                          /* transpose operators */
{ return ga_ddot_patch_dp(g_a, _fcdtocp(t_a), ailo, aihi, ajlo, ajhi,
                       g_b, _fcdtocp(t_b), bilo, bihi, bjlo, bjhi);}
#else 
     char    *t_a, *t_b;                          /* transpose operators */
{ return ga_ddot_patch_dp(g_a, t_a, ailo, aihi, ajlo, ajhi,
                       g_b, t_b, bilo, bihi, bjlo, bjhi);}
#endif
