/*$Id: hsort.scat.c,v 1.3 1995-02-02 23:13:39 d3g681 Exp $*/
#include "types.f2c.h"

#define GT(a,b) (*(a) > *(b))
#define GE(a,b) (*(a) >= *(b))



void ga_sort_scat_dbl_(pn, v, i, j, base)
     Integer *pn;
     DoublePrecision *v;
     Integer *i;
     Integer *j;
     Integer *base;
{
  Integer *p, *q, *base0=base - 1, *hi, n=*pn;

  unsigned gap , g;
  if (n < 2)
    return;
  
  gap = n >>1;
  hi = base0 + gap + gap;
  if (n & 1)
    hi ++;
  
#define SWAP(a,b) { \
  Integer ltmp; \
  DoublePrecision dtmp; \
  int ia = a - base; \
  int ib = b - base; \
  ltmp=*a; *a=*b; *b=ltmp; \
  dtmp=v[ia]; v[ia]=v[ib]; v[ib]=dtmp; \
  ltmp=i[ia]; i[ia]=i[ib]; i[ib]=ltmp; \
  ltmp=j[ia]; j[ia]=j[ib]; j[ib]=ltmp; \
}

  for ( ; gap != 1; gap--) {
    for (p = base0 + (g = gap) ; (q = p + g) <= hi ; p = q) {
      g += g;
      if (q != hi && GT(q+1, q)) {
	q++;
	g++;
      }
      if (GE(p,q))
	break;
      
      SWAP(p , q);
    }
  }
  
  for ( ; hi != base ; hi--) {
    p = base;
    for (g = 1 ; (q = p + g) <= hi ; p = q) {
      g += g;
      if (q != hi && GT(q+1,q)) {
	q++;
	g++;
      }
      if (GE(p,q))
	break;
      
      SWAP(p, q);
    }
    SWAP(base, hi);
  }
}


void ga_sort_scat_int_(pn, v, i, j, base)
     Integer *pn;
     Integer *v;
     Integer *i;
     Integer *j;
     Integer *base;
{
  Integer *p, *q, *base0=base - 1, *hi, n=*pn;

  unsigned gap , g;
  if (n < 2)
    return;
  
  gap = n >>1;
  hi = base0 + gap + gap;
  if (n & 1)
    hi ++;
#undef SWAP  
#define SWAP(a,b) { \
  Integer ltmp; \
  Integer dtmp; \
  int ia = a - base; \
  int ib = b - base; \
  ltmp=*a; *a=*b; *b=ltmp; \
  dtmp=v[ia]; v[ia]=v[ib]; v[ib]=dtmp; \
  ltmp=i[ia]; i[ia]=i[ib]; i[ib]=ltmp; \
  ltmp=j[ia]; j[ia]=j[ib]; j[ib]=ltmp; \
}

  for ( ; gap != 1; gap--) {
    for (p = base0 + (g = gap) ; (q = p + g) <= hi ; p = q) {
      g += g;
      if (q != hi && GT(q+1, q)) {
	q++;
	g++;
      }
      if (GE(p,q))
	break;
      
      SWAP(p , q);
    }
  }
  
  for ( ; hi != base ; hi--) {
    p = base;
    for (g = 1 ; (q = p + g) <= hi ; p = q) {
      g += g;
      if (q != hi && GT(q+1,q)) {
	q++;
	g++;
      }
      if (GE(p,q))
	break;
      
      SWAP(p, q);
    }
    SWAP(base, hi);
  }
}



void ga_sort_gath_(pn, i, j, base)
     Integer *pn;
     Integer *i;
     Integer *j;
     Integer *base;
{
  Integer *p, *q, *base0=base - 1, *hi, n=*pn;

  unsigned gap , g;
  if (n < 2)
    return;
  
  gap = n >>1;
  hi = base0 + gap + gap;
  if (n & 1)
    hi ++;
  
#undef SWAP  
#define SWAP(a,b) { \
  Integer ltmp; \
  int ia = a - base; \
  int ib = b - base; \
  ltmp=*a; *a=*b; *b=ltmp; \
  ltmp=i[ia]; i[ia]=i[ib]; i[ib]=ltmp; \
  ltmp=j[ia]; j[ia]=j[ib]; j[ib]=ltmp; \
}

  for ( ; gap != 1; gap--) {
    for (p = base0 + (g = gap) ; (q = p + g) <= hi ; p = q) {
      g += g;
      if (q != hi && GT(q+1, q)) {
	q++;
	g++;
      }
      if (GE(p,q))
	break;
      
      SWAP(p , q);
    }
  }
  
  for ( ; hi != base ; hi--) {
    p = base;
    for (g = 1 ; (q = p + g) <= hi ; p = q) {
      g += g;
      if (q != hi && GT(q+1,q)) {
	q++;
	g++;
      }
      if (GE(p,q))
	break;
      
      SWAP(p, q);
    }
    SWAP(base, hi);
  }
}

