/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/memalign.c,v 1.4 1995-02-24 02:17:25 d3h325 Exp $ */

#if !defined(SUN) && !defined(CRAY)

#if defined(ULTRIX) || defined(SGI) || defined(NEXT) || defined(HPUX) || \
defined(DECOSF)
extern void *malloc();
#else
#ifndef IPSC
extern char *malloc();
#endif
#endif

char *memalign(alignment, size)
     unsigned alignment;
     unsigned size;
/*
  Cut from SUN man-page

     memalign() allocates size bytes  on  a  specified  alignment
     boundary, and returns a pointer to the allocated block.  The
     value of the returned address is guaranteed to  be  an  even
     multiple of alignment.  Note: the value of alignment must be
     a power of two, and must be greater than  or  equal  to  the
     size of a word.

     No checking is done on the value of alignment ... should really.
*/
{
  union screwup {
    unsigned long integer;
    char *address;
  } fiddle;

  unsigned long offset;
  
  alignment += alignment;   /* Actually align on twice requested boundary */

  fiddle.address = malloc((unsigned) (alignment+size));

  if (fiddle.address != (char *) 0) {
    offset = fiddle.integer & (alignment-1);
    
    if (offset != 0)
      fiddle.address += alignment - offset;
  }

  return fiddle.address;
}

#endif

#if defined(SUN)

/* Use the system routine */
void AvoidNullSymbolTable()
{}

#endif

#if defined(CRAY)

/* Alignment is not really an issue on the cray except when casting
   a character pointer to another type when information may be lost */

extern char *malloc();

char *memalign(alignment, size)
     unsigned alignment;
     unsigned size;
{
  return malloc(size);
}

#endif
