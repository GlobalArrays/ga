#ifdef NX 

extern long masktrap(long);
#define ga_mask(new, old) {*(old) = masktrap((new));}


#elif defined(SP1)

#ifdef EUIH
#  include "mpctof.c"
#  define mp_rcvncall rcvncall
#  define mp_lockrnc  lockrnc
   extern long  mpc_wait(long*,long *);
   extern long  lockrnc(long*,long *);
   extern long  rcvncall(char*,long *, long*,long*,long*,void*());
#else
#  include <mpproto.h>
#endif


#define ga_mask(new, old) { long __new = new; mp_lockrnc(& __new, (old));}

#endif
