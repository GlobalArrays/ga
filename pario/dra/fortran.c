/**************************** fortran  DRA interface **************************/

#include "global.h"
#include "drap.h"
#include "dra.h"

#if defined(__STDC__) || defined(__cplusplus) || defined(WIN32)
# define _ARGS_(s) s
#else
# define _ARGS_(s) ()
#endif

extern void f2cstring   _ARGS_((char*, Integer, char*, Integer));
extern void c2fstring   _ARGS_((char*, char*, Integer));

#undef _ARGS_

#ifdef CRAY_T3D
#      include <fortran.h>
#endif


static char cname[DRA_MAX_NAME+1], cfilename[DRA_MAX_FNAME+1];


#if defined(CRAY) || defined(WIN32)
Integer FATR dra_create_(type, dim1, dim2, name, filename, mode, reqdim1, reqdim2,d_a)
        Integer *d_a;                      /*input:DRA handle*/
        Integer *type;                     /*input*/
        Integer *dim1;                     /*input*/
        Integer *dim2;                     /*input*/
        Integer *reqdim1;                  /*input: dim1 of typical request*/
        Integer *reqdim2;                  /*input: dim2 of typical request*/
        Integer *mode;                     /*input*/
        _fcd    name;                      /*input*/
        _fcd    filename;                  /*input*/
#else
Integer FATR dra_create_(type, dim1, dim2, name, filename, mode, reqdim1, reqdim2,d_a,
                   nlen, flen)
        Integer *d_a;                      /*input:DRA handle*/
        Integer *type;                     /*input*/
        Integer *dim1;                     /*input*/
        Integer *dim2;                     /*input*/
        Integer *reqdim1;                  /*input: dim1 of typical request*/
        Integer *reqdim2;                  /*input: dim2 of typical request*/
        Integer *mode;                     /*input*/
        char    *name;                     /*input*/
        char    *filename;                 /*input*/

        int     nlen;
        int     flen;

#endif
{
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(name), _fcdlen(name), cname, DRA_MAX_NAME);
      f2cstring(_fcdtocp(filename), _fcdlen(filename), cfilename, DRA_MAX_FNAME);
#else
      f2cstring(name, nlen, cname, DRA_MAX_NAME);
      f2cstring(filename, flen, cfilename, DRA_MAX_FNAME);
#endif
return dra_create(type, dim1, dim2,cname,cfilename, mode, reqdim1, reqdim2,d_a);
}

#if defined(CRAY) || defined(WIN32)
Integer FATR ndra_create_(type, ndim, dims, name, filename, mode, reqdims, d_a)
        Integer *d_a;                      /*input:DRA handle*/
        Integer *type;                     /*input*/
        Integer *ndim;                     /*input*/
        Integer dims[];                    /*input*/
        Integer reqdims[];                 /*input: dims of typical request*/
        Integer *mode;                     /*input*/
        _fcd    name;                      /*input*/
        _fcd    filename;                  /*input*/
#else
Integer FATR ndra_create_(type, ndim, dims, name, filename, mode, reqdims, d_a,
                   nlen, flen)
        Integer *d_a;                      /*input:DRA handle*/
        Integer *type;                     /*input*/
        Integer *ndim;                     /*input*/
        Integer dims[];                    /*input*/
        Integer reqdims[];                 /*input: dims of typical request*/
        Integer *mode;                     /*input*/
        char    *name;                     /*input*/
        char    *filename;                 /*input*/

        int     nlen;
        int     flen;

#endif
{
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(name), _fcdlen(name), cname, DRA_MAX_NAME);
      f2cstring(_fcdtocp(filename), _fcdlen(filename), cfilename, DRA_MAX_FNAME);
#else
      f2cstring(name, nlen, cname, DRA_MAX_NAME);
      f2cstring(filename, flen, cfilename, DRA_MAX_FNAME);
#endif
return ndra_create(type, ndim, dims, cname, cfilename, mode, reqdims, d_a);
}



#if defined(CRAY) || defined(WIN32)
Integer FATR dra_open_(filename, mode, d_a)
        _fcd  filename;                  /*input*/
        Integer *mode;                   /*input*/
        Integer *d_a;                    /*input*/ 
#else
Integer FATR dra_open_(filename, mode, d_a, flen)
        char *filename;                  /*input*/
        Integer *mode;                   /*input*/
        Integer *d_a;                    /*input*/
        int     flen;
#endif
{
#if defined(CRAY) || defined(WIN32)
      f2cstring(_fcdtocp(filename), _fcdlen(filename), cfilename, DRA_MAX_FNAME);
#else
      f2cstring(filename, flen, cfilename, DRA_MAX_FNAME);
#endif
      return dra_open(cfilename, mode, d_a);
}



#if defined(CRAY) || defined(WIN32)
Integer FATR dra_inquire_(d_a, type, dim1, dim2, name, filename)
        Integer *d_a;                      /*input:DRA handle*/
        Integer *type;                     /*output*/
        Integer *dim1;                     /*output*/
        Integer *dim2;                     /*output*/
        _fcd    name;                      /*output*/
        _fcd    filename;        
#else
Integer FATR dra_inquire_(d_a, type, dim1, dim2, name, filename, nlen, flen)
        Integer *d_a;                      /*input:DRA handle*/
        Integer *type;                     /*output*/
        Integer *dim1;                     /*output*/
        Integer *dim2;                     /*output*/
        char    *name;                     /*output*/
        char    *filename;

        int     nlen;
        int     flen;
#endif
{
Integer stat = dra_inquire(d_a, type, dim1, dim2, cname, cfilename);
*type = (Integer)ga_type_c2f((int)*type);
#if defined(CRAY) || defined(WIN32)
   c2fstring(cname, _fcdtocp(name), _fcdlen(name));
   c2fstring(cfilename, _fcdtocp(filename), _fcdlen(filename));
#else
   c2fstring(cname, name, nlen);
   c2fstring(cfilename, filename, flen);
#endif
   return stat;
}

#if defined(CRAY) || defined(WIN32)
Integer FATR ndra_inquire_(d_a, type, ndim, dims, name, filename)
        Integer *d_a;                      /*input:DRA handle*/
        Integer *type;                     /*output*/
        Integer *ndim;                     /*output*/
        Integer dims[];                    /*output*/
        _fcd    name;                      /*output*/
        _fcd    filename;        
#else
Integer FATR ndra_inquire_(d_a, type, ndim, dims, name, filename, nlen, flen)
        Integer *d_a;                      /*input:DRA handle*/
        Integer *type;                     /*output*/
        Integer *ndim;                     /*output*/
        Integer dims[];                    /*output*/
        char    *name;                     /*output*/
        char    *filename;

        int     nlen;
        int     flen;
#endif
{
Integer stat = ndra_inquire(d_a, type, ndim, dims, cname, cfilename);
*type = (Integer)ga_type_c2f((int)*type);
#if defined(CRAY) || defined(WIN32)
   c2fstring(cname, _fcdtocp(name), _fcdlen(name));
   c2fstring(cfilename, _fcdtocp(filename), _fcdlen(filename));
#else
   c2fstring(cname, name, nlen);
   c2fstring(cfilename, filename, flen);
#endif
   return stat;
}
