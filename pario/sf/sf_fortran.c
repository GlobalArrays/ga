#include "sf.h"
#ifdef CRAY_T3D
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


#ifdef CRAY
Integer SF_CREATE(fname, size_hard_limit, size_soft_limit, req_size, handle)
        _fcd fname;
        SFsize_t *size_hard_limit, *size_soft_limit, *req_size;
        Integer *handle;
#else
Integer sf_create_(fname, size_hard_limit, size_soft_limit, req_size,handle,len)
        char *fname;
        SFsize_t *size_hard_limit, *size_soft_limit, *req_size;
        Integer *handle;
        int len;

#endif
{
#ifdef CRAY
      f2cstring(_fcdtocp(fname), _fcdlen(fname), cname, MAX_NAME);
#else
      f2cstring(fname, len, cname, MAX_NAME);
#endif
return sf_create(cname, size_hard_limit, size_soft_limit,
                   req_size, handle);
}

