/******************  operations on sections of 2D arrays ************/


#include "global.h"       /* used only to define datatypes */
#include "drap.h"

#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))


/*\ check if two patches are conforming (dimensions are divisible)
\*/
logical dai_patches_conforming(
        Integer* ailo, Integer* aihi, Integer* ajlo, Integer* ajhi, 
        Integer* bilo, Integer* bihi, Integer* bjlo, Integer* bjhi)
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
logical dai_comp_patch(
        Integer* ilo, Integer* ihi, Integer* jlo, Integer* jhi, 
        Integer* ilop, Integer* ihip, Integer* jlop, Integer* jhip)
{
   if(*ihip != *ihi || *ilop != *ilo || *jhip != *jhi || *jlop != *jlo)
        return(FALSE);
   else return(TRUE);
}


/*\ check if patches have a nontrivial intersection
 *        if yes, find the intersection and update [ilop:ihip, jlop:jhip]
\*/
logical dai_patch_intersect(
        Integer ilo, Integer ihi, Integer jlo, Integer jhi,
        Integer* ilop, Integer* ihip, Integer* jlop, Integer* jhip)
{
     /* check consistency of patch coordinates */
     if( ihi < ilo || jhi < jlo)     return FALSE; /* inconsistent */
     if( *ihip < *ilop || *jhip < *jlop) return FALSE; /* inconsistent */

     /* find the intersection and update (ilop: ihip, jlop: jhip) */
     if( ihi < *ilop || *ihip < ilo) return FALSE; /* don't intersect */
     if( jhi < *jlop || *jhip < jlo) return FALSE; /* don't intersect */
     *ilop = MAX(ilo,*ilop);
     *ihip = MIN(ihi,*ihip);
     *jlop = MAX(jlo,*jlop);
     *jhip = MIN(jhi,*jhip);

     return(TRUE);
}


/*\ check if sections have a nontrivial intersection
 *        if yes, find the intersection and update [ilop:ihip, jlop:jhip]
 *        section format
\*/
logical dai_section_intersect(section_t sref, section_t *sadj)
{
     /* check consistency of patch coordinates */
     if( sref.hi[0] < sref.lo[0] || sref.hi[1] < sref.lo[1])     
                                  return FALSE; /* inconsistent */
     if( sadj->hi[0] < sadj->lo[0] || sadj->hi[1] < sadj->lo[1]) 
                                  return FALSE; /* inconsistent */

     /* find the intersection and update (ilop: ihip, jlop: jhip) */
     if( sref.hi[0] < sadj->lo[0] || sadj->hi[0] < sref.lo[0]) 
                                  return FALSE; /* don't intersect */
     if( sref.hi[1] < sadj->lo[1] || sadj->hi[1] < sref.lo[1]) 
                                  return FALSE; /* don't intersect */
     sadj->lo[0] = MAX(sref.lo[0],sadj->lo[0]);
     sadj->hi[0] = MIN(sref.hi[0],sadj->hi[0]);
     sadj->lo[1] = MAX(sref.lo[1],sadj->lo[1]);
     sadj->hi[1] = MIN(sref.hi[1],sadj->hi[1]);

     return(TRUE);
}

