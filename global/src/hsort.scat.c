
#define GT(a,b) (*(a) > *(b))
#define GE(a,b) (*(a) >= *(b))

void ga_sort_scat2_(pn, v, i, j, base)
     long *pn;
     double *v;
     long *i;
     long *j;
     long *base;
{
  long *p, *q, *base0=base - 1, *hi, n=*pn;

  unsigned gap , g;
  if (n < 2)
    return;
  
  gap = n >>1;
  hi = base0 + gap + gap;
  if (n & 1)
    hi ++;
  
#define SWAP(a,b) { \
  long ltmp; \
  double dtmp; \
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
