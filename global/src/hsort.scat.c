/**************************************************************************\
 
 Sort routines for scatter and gather.
 scatter requires sorting of index and value arrays.
 gather requires sorting of index arrays only.

\**************************************************************************/

  
#include "typesf2c.h"
#include "macommon.h"
extern void ga_error();

#define GT(a,b) (*(a) > *(b))
#define GE(a,b) (*(a) >= *(b))


#define INDEX_SORT(base,pn,SWAP){\
  unsigned gap, g;\
  Integer *p, *q, n=*pn;\
  Integer *hi, *base0=base - 1;\
\
  gap = n >>1;\
  hi = base0 + gap + gap;\
  if (n & 1) hi ++;\
\
  for ( ; gap != 1; gap--) {\
    for (p = base0 + (g = gap) ; (q = p + g) <= hi ; p = q) {\
      g += g;\
      if (q != hi && GT(q+1, q)) {\
        q++;\
        g++;\
      }\
      if (GE(p,q)) break;\
      SWAP(p , q);\
    }\
  }\
\
  for ( ; hi != base ; hi--) {\
    p = base;\
    for (g = 1 ; (q = p + g) <= hi ; p = q) {\
      g += g;\
      if (q != hi && GT(q+1,q)) {\
        q++;\
        g++;\
      }\
      if (GE(p,q)) break;\
      SWAP(p, q);\
    }\
    SWAP(base, hi);\
  }\
}




void ga_sort_scat_dcpl_(pn, v, i, j, base)
     Integer *pn;
     DoubleComplex *v;
     Integer *i;
     Integer *j;
     Integer *base;
{

  if (*pn < 2) return;

#  define SWAP(a,b) { \
    Integer ltmp; \
    DoubleComplex dtmp; \
    int ia = a - base; \
    int ib = b - base; \
    ltmp=*a; *a=*b; *b=ltmp; \
    dtmp=v[ia]; v[ia]=v[ib]; v[ib]=dtmp; \
    ltmp=i[ia]; i[ia]=i[ib]; i[ib]=ltmp; \
    ltmp=j[ia]; j[ia]=j[ib]; j[ib]=ltmp; \
  }
  INDEX_SORT(base,pn,SWAP);
}

void ga_sort_permutation(pn, index, base)
     Integer *pn;
     Integer *index;
     Integer *base;
{
  if (*pn < 2) return;
#  undef SWAP  
#  define SWAP(a,b) { \
    Integer ltmp; \
    Integer itmp;\
    int ia = a - base; \
    int ib = b - base; \
    ltmp=*a; *a=*b; *b=ltmp; \
    itmp=index[ia]; index[ia]=index[ib]; index[ib] = itmp;\
   }
  INDEX_SORT(base,pn,SWAP);
}

     


void ga_sort_scat_dbl_(pn, v, i, j, base)
     Integer *pn;
     DoublePrecision *v;
     Integer *i;
     Integer *j;
     Integer *base;
{
  
  if (*pn < 2) return;

#  undef SWAP  
#  define SWAP(a,b) { \
    Integer ltmp; \
    DoublePrecision dtmp; \
    int ia = a - base; \
    int ib = b - base; \
    ltmp=*a; *a=*b; *b=ltmp; \
    dtmp=v[ia]; v[ia]=v[ib]; v[ib]=dtmp; \
    ltmp=i[ia]; i[ia]=i[ib]; i[ib]=ltmp; \
    ltmp=j[ia]; j[ia]=j[ib]; j[ib]=ltmp; \
  }
  INDEX_SORT(base,pn,SWAP);
}


void ga_sort_scat_int_(pn, v, i, j, base)
     Integer *pn;
     Integer *v;
     Integer *i;
     Integer *j;
     Integer *base;
{

  if (*pn < 2) return;

#  undef SWAP  
#  define SWAP(a,b) { \
    Integer ltmp; \
    Integer dtmp; \
    int ia = a - base; \
    int ib = b - base; \
    ltmp=*a; *a=*b; *b=ltmp; \
    dtmp=v[ia]; v[ia]=v[ib]; v[ib]=dtmp; \
    ltmp=i[ia]; i[ia]=i[ib]; i[ib]=ltmp; \
    ltmp=j[ia]; j[ia]=j[ib]; j[ib]=ltmp; \
  }
  INDEX_SORT(base,pn,SWAP);
}



void ga_sort_scat(pn, v, i, j, base, type)
     Integer *pn;
     Void    *v;
     Integer *i;
     Integer *j;
     Integer *base;
     Integer type;
{ 
   switch (type){
     case MT_F_DBL:  ga_sort_scat_dbl_(pn, (DoublePrecision*)v, i,j,base);break;
     case MT_F_DCPL: ga_sort_scat_dcpl_(pn, (DoubleComplex*)v, i,j,base); break;
     case MT_F_INT:  ga_sort_scat_int_(pn, (Integer*)v, i, j, base); break;
     default:        ga_error("ERROR:ga_sort_scat: wrong type",type);
   } 
}



void ga_sort_gath_(pn, i, j, base)
     Integer *pn;
     Integer *i;
     Integer *j;
     Integer *base;
{

  if (*pn < 2) return;
  
#  undef SWAP  
#  define SWAP(a,b) { \
    Integer ltmp; \
    int ia = a - base; \
    int ib = b - base; \
    ltmp=*a; *a=*b; *b=ltmp; \
    ltmp=i[ia]; i[ia]=i[ib]; i[ib]=ltmp; \
    ltmp=j[ia]; j[ia]=j[ib]; j[ib]=ltmp; \
  }
  INDEX_SORT(base,pn,SWAP);
}
