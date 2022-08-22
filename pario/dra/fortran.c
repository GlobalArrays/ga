#if HAVE_CONFIG_H
#   include "config.h"
#endif

/*************************** fortran  DRA interface **************************/

#include "dra.h"
#include "drap.h"
#include "draf2c.h"
#include "farg.h"
#include "typesf2c.h"
#include "ga-papi.h"
#include "global.h"

static char cname[DRA_MAX_NAME+1], cfilename[DRA_MAX_FNAME+1];

Integer FATR dra_create_(
#ifdef F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
        Integer *type,
        Integer *dim1,
        Integer *dim2,
        char    *name,
        char    *filename,
        Integer *mode,
        Integer *reqdim1,
        Integer *reqdim2,
        Integer *d_a,
        size_t  nlen,
        size_t  flen
#else
        Integer *type,
        Integer *dim1,
        Integer *dim2,
        char    *name,
        size_t  nlen,
        char    *filename,
        size_t  flen,
        Integer *mode,
        Integer *reqdim1,
        Integer *reqdim2,
        Integer *d_a
#endif
        )
{
    ga_f2cstring(name, nlen, cname, DRA_MAX_NAME);
    ga_f2cstring(filename, flen, cfilename, DRA_MAX_FNAME);
    return drai_create(type, dim1, dim2, cname, cfilename,
            mode, reqdim1, reqdim2,d_a);
}


Integer FATR ndra_create_(
#ifdef F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
        Integer *type,
        Integer *ndim,
        Integer  dims[],
        char    *name,
        char    *filename,
        Integer *mode,
        Integer  reqdims[],
        Integer *d_a,
        size_t  nlen,
        size_t  flen
#else
        Integer *type,
        Integer *ndim,
        Integer  dims[],
        char    *name,
        size_t  nlen,
        char    *filename,
        size_t  flen,
        Integer *mode,
        Integer  reqdims[],
        Integer *d_a
#endif
        )
{
    ga_f2cstring(name, nlen, cname, DRA_MAX_NAME);
    ga_f2cstring(filename, flen, cfilename, DRA_MAX_FNAME);
    return ndrai_create(type, ndim, dims, cname, cfilename, mode, reqdims, d_a);
}


Integer FATR dra_open_(
#ifdef F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
        char    *filename,
        Integer *mode,
        Integer *d_a,
        size_t      flen
#else
        char    *filename,
        size_t      flen,
        Integer *mode,
        Integer *d_a
#endif
        )
{
    ga_f2cstring(filename, flen, cfilename, DRA_MAX_FNAME);
    return drai_open(cfilename, mode, d_a);
}


Integer FATR dra_inquire_(
#ifdef F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
        Integer *d_a,
        Integer *type,
        Integer *dim1,
        Integer *dim2,
        char    *name,
        char    *filename,
        size_t      nlen,
        size_t      flen
#else
        Integer *d_a,
        Integer *type,
        Integer *dim1,
        Integer *dim2,
        char    *name,
        size_t      nlen,
        char    *filename,
        size_t      flen
#endif
        )
{
    Integer stat = drai_inquire(d_a, type, dim1, dim2, cname, cfilename);
    *type = pnga_type_c2f(*type);
    ga_c2fstring(cname, name, nlen);
    ga_c2fstring(cfilename, filename, flen);
    return stat;
}


Integer FATR ndra_inquire_(
#ifdef F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
        Integer *d_a,
        Integer *type,
        Integer *ndim,
        Integer  dims[],
        char    *name,
        char    *filename,
        size_t      nlen,
        size_t      flen
#else
        Integer *d_a,
        Integer *type,
        Integer *ndim,
        Integer  dims[],
        char    *name,
        size_t      nlen,
        char    *filename,
        size_t      flen
#endif
        )
{
    Integer stat = ndrai_inquire(d_a, type, ndim, dims, cname, cfilename);
    *type = pnga_type_c2f(*type);
    ga_c2fstring(cname, name, nlen);
    ga_c2fstring(cfilename, filename, flen);
    return stat;
}


Integer ndra_create_config_(
#ifdef F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
        Integer *type,
        Integer *ndim,
        Integer  dims[],
        char    *name,
        char    *filename,
        Integer *mode, 
        Integer  reqdims[],
        Integer *numfiles, 
        Integer *numioprocs,
        Integer *d_a,
        size_t      nlen,
        size_t      flen
#else
        Integer *type,
        Integer *ndim,
        Integer  dims[],
        char    *name,
        size_t      nlen,
        char    *filename,
        size_t      flen,
        Integer *mode, 
        Integer  reqdims[],
        Integer *numfiles, 
        Integer *numioprocs,
        Integer *d_a
#endif
        )
{
    ga_f2cstring(name, nlen, cname, DRA_MAX_NAME);
    ga_f2cstring(filename, flen, cfilename, DRA_MAX_FNAME);
    return ndrai_create_config(type, ndim, dims, cname, cfilename,
            mode, reqdims, numfiles, numioprocs, d_a);
}
