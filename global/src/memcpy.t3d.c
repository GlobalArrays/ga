/*$Id: memcpy.t3d.c,v 1.2 1995-02-02 23:13:45 d3g681 Exp $*/
#include <stdio.h>
#include <memory.h>

void copyto(const unsigned char *src, unsigned char *dest, long n)
/*
  A copy optimized for DESTINATIONS in shared memory that
  are aligned and data is to be read by other processes.

*/
{
  if (n < 32 || (dest - src) & 7) {

    /* small n, or
       not possible to get src and dest even word aligned */

    while (n--)
      *dest++ = *src++;
/*    memcpy(dest, src, (size_t) n); */
    return;
  }
  
  /* Read ahead so that dest is aligned on a page boundary */
  
  {
    register long nbytes = (127 & (unsigned long) dest);
    if (nbytes > 0) nbytes = 128 - nbytes;
    if (nbytes > n) nbytes = n;
    n -= nbytes;

    while (nbytes--)
      *dest++ = *src++;

    if (n == 0) return;
  }

  {
    /* src is at least word aligned and dest is subpage aligned */
    
    register long npage = n>>7;
    register const unsigned long *from = (unsigned long *) src;
    register unsigned long *to = (unsigned long *) dest;
    register unsigned long a, b, c, d, e, f, g, h;

    src  += npage<<7;
    dest += npage<<7;
    n    -= npage<<7;
    
    while (npage--) {

      a = from[0];
      b = from[1];
      c = from[2];
      d = from[3];
      e = from[4];
      f = from[5];
      g = from[6];
      h = from[7];
      to[0] = a;
      to[1] = b;
      to[2] = c;
      to[3] = d;
      to[4] = e;
      to[5] = f;
      to[6] = g;
      to[7] = h;

      a = from[8];
      b = from[9];
      c = from[10];
      d = from[11];
      e = from[12];
      f = from[13];
      g = from[14];
      h = from[15];
      to[8]  = a;
      to[9]  = b;
      to[10] = c;
      to[11] = d;
      to[12] = e;
      to[13] = f;
      to[14] = g;
      to[15] = h;

/*       _pstsp((char *) to); */
      
      to += 16; from+= 16;
    }
  }

  {
    register long nbytes = n;
    register const unsigned char *from = (unsigned char *) src;
    register unsigned char *to = (unsigned char *) dest;

    while (nbytes--)
      *to++ = *from++;
  }
}

void copyfrom(const unsigned char *src, unsigned char *dest, long n)
/*
  A copy optimized for SOURCES in shared memory that
  are aligned.

*/
{
  if (n < 32 || (dest - src) & 7) {

    /* small n, or
       not possible to get src and dest even word aligned */

    while (n--)
      *dest++ = *src++;
/*    memcpy(dest, src, (size_t) n);*/
    return;
  }
  
  /* Read ahead so that src is aligned on a page boundary */
  
  {
    register long nbytes = (127 & (unsigned long) src);
    if (nbytes > 0) nbytes = 128 - nbytes;
    if (nbytes > n) nbytes = n;
    n -= nbytes;

    while (nbytes--)
      *dest++ = *src++;

    if (n == 0) return;
  }

  {
    /* dest is at least word aligned and src is subpage aligned */
    
    register long npage = n>>7;
    register const unsigned long *from = (unsigned long *) src;
    register unsigned long *to = (unsigned long *) dest;
    register unsigned long a, b, c, d, e, f, g, h;

    src  += npage<<7;
    dest += npage<<7;
    n    -= npage<<7;
    
    while (npage--) {

      a = from[0];
      b = from[1];
      c = from[2];
      d = from[3];
      e = from[4];
      f = from[5];
      g = from[6];
      h = from[7];
      to[0] = a;
      to[1] = b;
      to[2] = c;
      to[3] = d;
      to[4] = e;
      to[5] = f;
      to[6] = g;
      to[7] = h;

      a = from[8];
      b = from[9];
      c = from[10];
      d = from[11];
      e = from[12];
      f = from[13];
      g = from[14];
      h = from[15];
      to[8]  = a;
      to[9]  = b;
      to[10] = c;
      to[11] = d;
      to[12] = e;
      to[13] = f;
      to[14] = g;
      to[15] = h;

/*       _pstsp((char *) to); */
      
      to += 16; from+= 16;
    }
  }

  {
    register long nbytes = n;
    register const unsigned char *from = (unsigned char *) src;
    register unsigned char *to = (unsigned char *) dest;

    while (nbytes--)
      *to++ = *from++;
  }
}

