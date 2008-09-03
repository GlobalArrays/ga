#include "sf.h"
#include <string.h>

#if defined(CRAY) && defined(__crayx1)
#undef CRAY
#endif
 
#ifdef CRAY
#      include <fortran.h>
#endif
 
#define MAX_NAME 256
char cname[MAX_NAME+1];

static void f2cstring(fstring, flength, cstring, clength)
    char        *fstring;       /* FORTRAN string */
    int          flength;        /* length of fstring */
    char        *cstring;       /* C buffer */
    int          clength;        /* max length (including NUL) of cstring */
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


#if defined(CRAY) || defined(WIN32)
Integer FATR SF_CREATE(fname, size_hard_limit, size_soft_limit, req_size, handle)
        _fcd fname;
        SFsize_t *size_hard_limit, *size_soft_limit, *req_size;
        Integer *handle;
#else
#  if defined(F2C2_)
#    define sf_create_   sf_create__
#  endif

Integer FATR sf_create_(fname, size_hard_limit, size_soft_limit, req_size,handle,len)
        char *fname;
        SFsize_t *size_hard_limit, *size_soft_limit, *req_size;
        Integer *handle;
        int len;

#endif
{
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(fname), _fcdlen(fname), cname, MAX_NAME);
#else
      f2cstring(fname, len, cname, MAX_NAME);
#endif
return sf_create(cname, size_hard_limit, size_soft_limit,
                   req_size, handle);
}


#if defined(CRAY) || defined(WIN32)
Integer FATR SF_CREATE_SUFFIX(fname, size_hard_limit, size_soft_limit, req_size, handle, suffix)
        _fcd fname;
        SFsize_t *size_hard_limit, *size_soft_limit, *req_size;
        Integer *handle;
        Integer *suffix;
#else
#  if defined(F2C2_)
#    define sf_create_suffix_   sf_create_suffix__
#  endif

Integer FATR sf_create_suffix_(fname, size_hard_limit, size_soft_limit, req_size, handle, suffix, len)
        char *fname;
        SFsize_t *size_hard_limit, *size_soft_limit, *req_size;
        Integer *handle;
        Integer *suffix;
        int len;

#endif
{
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(fname), _fcdlen(fname), cname, MAX_NAME);
#else
      f2cstring(fname, len, cname, MAX_NAME);
#endif
return sf_create_suffix(cname, size_hard_limit, size_soft_limit,
                   req_size, handle, suffix);
}

/*****************************************************************************/

static int string_to_fortchar(char *f, const int flen, const char *buf)
{
  const int len = (int) strlen(buf);
  int i;

  if (len > flen)
    return 0;                   /* Won't fit */

  for (i=0; i<len; i++)
    f[i] = buf[i];
  for (i=len; i<flen; i++)
    f[i] = ' ';

  return 1;
}


#if defined(CRAY) || defined(WIN32)
void FATR sf_errmsg_(Integer *code,  _fcd m)
{
    char *msg = _fcdtocp(m);
    int msglen = _fcdlen(m);
#else
void FATR sf_errmsg_(Integer *code, char *msg, int msglen)
{
#endif
    char buf[80];

    sf_errmsg((int) *code, buf);

    (void) string_to_fortchar(msg, msglen, buf);
}
