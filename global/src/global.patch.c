/*$Id: global.patch.c,v 1.24 1999-11-23 19:40:45 jju Exp $*/
#include "global.h"
#include "globalp.h"
#include <math.h>


#if defined(CRAY) || defined(WIN32)
#   define cptofcd(fcd)  _cptofcd((fcd),1)
#else
#      define cptofcd(fcd) (fcd)
#endif


#define DEST_INDICES(is,js, ilos,jlos, lds, id,jd, ilod, jlod, ldd) \
{ \
    int _index_;\
    _index_ = (lds)*((js)-(jlos)) + (is)-(ilos);\
    id = (_index_)%(ldd) + (ilod);\
    jd = (_index_)/(ldd) + (jlod);\
}


/*\ check if dimensions of two patches are divisible 
\*/
logical patches_conforming(ailo, aihi, ajlo, ajhi, bilo, bihi, bjlo, bjhi)
     Integer *ailo, *aihi, *ajlo, *ajhi;
     Integer *bilo, *bihi, *bjlo, *bjhi;
{
Integer mismatch;
Integer adim1, bdim1, adim2, bdim2;
     adim1 = *aihi - *ailo +1;
     adim2 = *ajhi - *ajlo +1;
     bdim1 = *bihi - *bilo +1;
     bdim2 = *bjhi - *bjlo +1;
     mismatch  = (adim1<bdim1) ? bdim1%adim1 : adim1%bdim1; 
     mismatch += (adim2<bdim2) ? bdim2%adim2 : adim2%bdim2; 
     if(mismatch)return(FALSE);
     else return(TRUE);
} 



/*\ check if patches are identical 
\*/
static logical comp_patch(ilo, ihi, jlo, jhi, ilop, ihip, jlop, jhip)
     Integer ilo, ihi, jlo, jhi;
     Integer ilop, ihip, jlop, jhip;
{
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


/*\ COMPARE DISTRIBUTIONS of two global arrays
 *  now implemented elsewhere using internal GA API
\*/
logical FATR ga_compare_distr_disabled(g_a, g_b)
     Integer *g_a, *g_b;
{
Integer me= ga_nodeid_();
Integer iloA, ihiA, jloA, jhiA;
Integer iloB, ihiB, jloB, jhiB;
DoublePrecision mismatch;
Integer type = GA_TYPE_GSM, len = 1;

   ga_sync_();
   GA_PUSH_NAME("ga_compare_distr");

   ga_distribution_(g_a, &me, &iloA, &ihiA, &jloA, &jhiA);
   ga_distribution_(g_b, &me, &iloB, &ihiB, &jloB, &jhiB);

   mismatch = comp_patch(iloA, ihiA, jloA, jhiA, iloB, ihiB, jloB, jhiB)?0. :1.;
   ga_dgop(type, &mismatch, len, "+");
   ga_sync_();
   GA_POP_NAME;

   if(mismatch) return (FALSE);
   else return(TRUE); 
}



/*\ COPY A PATCH AND POSSIBLY RESHAPE
 *
 *  . the element capacities of two patches must be identical
 *  . copy by column order - Fortran convention
\*/
void ga_copy_patch(char *trans, Integer *g_a, Integer *ailo, Integer *aihi,
                   Integer *ajlo, Integer *ajhi, Integer *g_b, Integer *bilo,
                   Integer *bihi, Integer *bjlo, Integer *bjhi)
{
    Integer alo[2], ahi[2], blo[2], bhi[2];

    alo[0] = *ailo; alo[1] = *ajlo;
    ahi[0] = *aihi; ahi[1] = *ajhi;
    blo[0] = *bilo; blo[1] = *bjlo;
    bhi[0] = *bihi; bhi[1] = *bjhi;

    nga_copy_patch(trans, g_a, alo, ahi, g_b, blo, bhi);
}


/*\ COPY A PATCH AND POSSIBLY RESHAPE
 *  Fortran interface
\*/
void FATR ga_copy_patch_(trans, g_a, ailo, aihi, ajlo, ajhi,
                    g_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;
#if defined(CRAY) || defined(WIN32)
     _fcd    trans;
{ga_copy_patch(_fcdtocp(trans),g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi);}
#else 
     char*   trans;
{  ga_copy_patch(trans,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi); }
#endif


/*\ generic dot product routine
\*/
void gai_dot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                   g_b, t_b, bilo, bihi, bjlo, bjhi, retval)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     char    *t_a, *t_b;                          /* transpose operators */
     DoublePrecision *retval;
{
    Integer alo[2], ahi[2], blo[2], bhi[2];

    alo[0] = *ailo; alo[1] = *ajlo;
    ahi[0] = *aihi; ahi[1] = *ajhi;
    blo[0] = *bilo; blo[1] = *bjlo;
    bhi[0] = *bihi; bhi[1] = *bjhi;

    ngai_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, (void *)retval);
}



/*\ compute Double Precision DOT PRODUCT of two patches
 *
 *          . different shapes and distributions allowed but not recommended
 *          . the same number of elements required
\*/
DoublePrecision ga_ddot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                              g_b, t_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     char    *t_a, *t_b;                          /* transpose operators */
{
Integer atype, btype, adim1, adim2, bdim1, bdim2;
DoublePrecision  sum = 0.;

   ga_sync_();
   GA_PUSH_NAME("ga_ddot_patch");

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype || (atype != MT_F_DBL )) ga_error(" wrong types ", 0L);

   gai_dot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                 g_b, t_b, bilo, bihi, bjlo, bjhi, &sum);

   GA_POP_NAME;
   return (sum);
}


/*\ compute Double Complex DOT PRODUCT of two patches
 *
 *          . different shapes and distributions allowed but not recommended
 *          . the same number of elements required
\*/
DoubleComplex ga_zdot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                            g_b, t_b, bilo, bihi, bjlo, bjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     char    *t_a, *t_b;                          /* transpose operators */
{
Integer atype, btype, adim1, adim2, bdim1, bdim2;
DoubleComplex  sum;

   ga_sync_();
   GA_PUSH_NAME("ga_zdot_patch");

   ga_inquire_(g_a, &atype, &adim1, &adim2);
   ga_inquire_(g_b, &btype, &bdim1, &bdim2);

   if(atype != btype || (atype != MT_F_DCPL )) ga_error(" wrong types ", 0L);

   gai_dot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                 g_b, t_b, bilo, bihi, bjlo, bjhi, (DoublePrecision*)&sum);

   GA_POP_NAME;
   return (sum);
}


/*\ compute DOT PRODUCT of two patches
 *  Fortran interface
\*/
void FATR gai_dot_patch_(g_a, t_a, ailo, aihi, ajlo, ajhi,
                    g_b, t_b, bilo, bihi, bjlo, bjhi, retval)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     DoublePrecision *retval;

#if defined(CRAY) || defined(WIN32)
     _fcd   t_a, t_b;                          /* transpose operators */
{  gai_dot_patch(g_a, _fcdtocp(t_a), ailo, aihi, ajlo, ajhi,
                 g_b, _fcdtocp(t_b), bilo, bihi, bjlo, bjhi, retval);}
#else 
     char    *t_a, *t_b;                          /* transpose operators */
{ gai_dot_patch(g_a, t_a, ailo, aihi, ajlo, ajhi,
                g_b, t_b, bilo, bihi, bjlo, bjhi, retval);}
#endif




/*\ FILL IN ARRAY WITH VALUE 
\*/
void FATR ga_fill_patch_(g_a, ilo, ihi, jlo, jhi, val)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
     Void    *val;
{
    Integer lo[2], hi[2];

    lo[0] = *ilo; lo[1] = *jlo;
    hi[0] = *ihi; hi[1] = *jhi;

    nga_fill_patch_(g_a, lo, hi, val);
}



/*\ SCALE ARRAY 
\*/
void FATR ga_scale_patch_(g_a, ilo, ihi, jlo, jhi, alpha)
     Integer *g_a, *ilo, *ihi, *jlo, *jhi;
     DoublePrecision     *alpha;
{
    Integer lo[2], hi[2];

    lo[0] = *ilo; lo[1] = *jlo;
    hi[0] = *ihi; hi[1] = *jhi;

    nga_scale_patch_(g_a, lo, hi, (void *)alpha);
}


/*\  SCALED ADDITION of two patches
\*/
void FATR ga_add_patch_(alpha, g_a, ailo, aihi, ajlo, ajhi,
                    beta,  g_b, bilo, bihi, bjlo, bjhi,
                           g_c, cilo, cihi, cjlo, cjhi)
     Integer *g_a, *ailo, *aihi, *ajlo, *ajhi;    /* patch of g_a */
     Integer *g_b, *bilo, *bihi, *bjlo, *bjhi;    /* patch of g_b */
     Integer *g_c, *cilo, *cihi, *cjlo, *cjhi;    /* patch of g_c */
     DoublePrecision      *alpha, *beta;
{
    Integer alo[2], ahi[2], blo[2], bhi[2], clo[2], chi[2];

    alo[0] = *ailo; alo[1] = *ajlo;
    ahi[0] = *aihi; ahi[1] = *ajhi;
    blo[0] = *bilo; blo[1] = *bjlo;
    bhi[0] = *bihi; bhi[1] = *bjhi;
    clo[0] = *cilo; clo[1] = *cjlo;
    chi[0] = *cihi; chi[1] = *cjhi;
    
    nga_add_patch_(alpha, g_a, alo, ahi, beta, g_b, blo, bhi, g_c, clo, chi);
}

