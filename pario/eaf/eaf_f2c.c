#include "eaf.h"
#include "eafP.h"
#include "typesf2c.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#if defined(CRAY) && defined(__crayx1)
#undef CRAY
#endif

#if defined(CRAY)
#include <fortran.h>
#endif

#if defined(CRAY) || defined(WIN32)
#define eaf_write_ 	EAF_WRITE
#define eaf_awrite_ 	EAF_AWRITE
#define eaf_read_ 	EAF_READ
#define eaf_aread_ 	EAF_AREAD
#define eaf_wait_ 	EAF_WAIT
#define eaf_probe_ 	EAF_PROBE
#define eaf_open_ 	EAF_OPEN
#define eaf_close_ 	EAF_CLOSE
#define eaf_delete_ 	EAF_DELETE
#define eaf_length_ 	EAF_LENGTH
#define eaf_truncate_ 	EAF_TRUNCATE
#define eaf_stat_ 	EAF_STAT
#define eaf_eof_ 	EAF_EOF
#define eaf_error_ 	EAF_ERROR
#define eaf_print_stats_ EAF_PRINT_STATS
#define eaf_errmsg_ 	EAF_ERRMSG
#define eaf_util_szint_ EAF_UTIL_SZINT
#define eaf_util_random_ EAF_UTIL_RANDOM
#elif defined(F2C2_)
#define eaf_write_ 	eaf_write__ 	
#define eaf_awrite_ 	eaf_awrite__ 	
#define eaf_read_ 	eaf_read__ 	
#define eaf_aread_ 	eaf_aread__ 	
#define eaf_wait_ 	eaf_wait__ 	
#define eaf_probe_ 	eaf_probe__ 	
#define eaf_open_ 	eaf_open__ 	
#define eaf_close_ 	eaf_close__ 	
#define eaf_delete_ 	eaf_delete__ 	
#define eaf_length_ 	eaf_length__ 	
#define eaf_truncate_ 	eaf_truncate__ 	
#define eaf_stat_ 	eaf_stat__ 	
#define eaf_eof_ 	eaf_eof__ 	
#define eaf_error_ 	eaf_error__ 	
#define eaf_print_stats_ eaf_print_stats__  
#define eaf_errmsg_ 	eaf_errmsg__ 	
#define eaf_util_szint_ eaf_util_szint__   
#define eaf_util_random_ eaf_util_random__
#endif

static int fortchar_to_string(const char *f, int flen, char *buf, 
			      const int buflen)
{
  while (flen-- && f[flen] == ' ')
    ;

  if ((flen+1) >= buflen)
    return 0;			/* Won't fit */

  flen++;
  buf[flen] = 0;
  while(flen--)
    buf[flen] = f[flen];

  return 1;
}

static int string_to_fortchar(char *f, const int flen, const char *buf)
{
  const int len = (int) strlen(buf);
  int i;

  if (len > flen) 
    return 0;			/* Won't fit */

  for (i=0; i<len; i++)
    f[i] = buf[i];
  for (i=len; i<flen; i++)
    f[i] = ' ';

  return 1;
}

static int valid_offset(double offset)
{
    return ((offset - ((double) ((eaf_off_t) offset))) == 0.0);
}
	
Integer FATR eaf_write_(Integer *fd, double *offset, const void *buf, 
		   Integer *bytes)
{
    if (!valid_offset(*offset)) return EAF_ERR_NONINTEGER_OFFSET;
    return (Integer) eaf_write((int) *fd, (eaf_off_t) *offset, buf, 
			       (size_t) *bytes);
}

Integer FATR eaf_awrite_(Integer *fd, double *offset, const void *buf, 
		    Integer *bytes, Integer *req_id)
{
    int req, status;

    if (!valid_offset(*offset)) return EAF_ERR_NONINTEGER_OFFSET;
    status = eaf_awrite((int) *fd, (eaf_off_t) *offset, buf, 
			(size_t) *bytes, &req);
    *req_id = (Integer) req;
    return (Integer) status;
}

Integer FATR eaf_read_(Integer *fd, double *offset, void *buf, Integer *bytes)
{
    if (!valid_offset(*offset)) return EAF_ERR_NONINTEGER_OFFSET;
    return (Integer) eaf_read((int) *fd, (eaf_off_t) *offset, buf, 
			      (size_t) *bytes);
}

Integer FATR eaf_aread_(Integer *fd, double *offset, void *buf, 
		    Integer *bytes, Integer *req_id)
{
    int req, status;

    if (!valid_offset(*offset)) return EAF_ERR_NONINTEGER_OFFSET;
    status = eaf_aread((int) *fd, (eaf_off_t) *offset, buf, 
		       (size_t) *bytes, &req);
    *req_id = (Integer) req;
    return (Integer) status;
}

Integer FATR eaf_wait_(Integer *fd, Integer *id)
{
    return (Integer) eaf_wait((int) *fd, (int) *id);
}

void FATR eaf_print_stats_(Integer *fd)
{
    eaf_print_stats((int) *fd);
}

Integer FATR eaf_truncate_(Integer *fd, double *length)
{
    return (Integer) eaf_truncate((int) *fd, (eaf_off_t) *length);
}

Integer FATR eaf_probe_(Integer *id, Integer *status)
{
    int s, code;

    code = eaf_probe((int) *id, &s);
    *status = (Integer) s;

    return (Integer) code;
}


Integer FATR eaf_close_(Integer *fd)
{
    return (Integer) eaf_close((int) *fd);
}


Integer FATR eaf_length_(Integer *fd, double *length)
{
    eaf_off_t len;
    int code;

    code = eaf_length((int) *fd, &len);
    if (!code) *length = (double) len;

    return code;
}

logical eaf_eof_(Integer *code)
{
    return (logical) eaf_eof((int) *code);
}


#if defined(CRAY) || defined(WIN32)
Integer FATR  eaf_open_(_fcd f, Integer *type, Integer *fd)
{
    char *fname = _fcdtocp(f);
    int flen = _fcdlen(f);
#else
Integer FATR eaf_open_(const char *fname, Integer *type, Integer *fd, int flen)
{
#endif
    char buf[1024];
    int code, tmp;

    if (!fortchar_to_string(fname, flen, buf, sizeof(buf)))
	return (Integer) EAF_ERR_TOO_LONG;

    code = eaf_open(buf, (int) *type, &tmp);
    *fd = (Integer) tmp;

    return (Integer)code;
}

#if defined(CRAY) || defined(WIN32)
Integer FATR eaf_delete_(_fcd f)
{
    char *fname = _fcdtocp(f);
    int flen = _fcdlen(f);
#else
Integer FATR eaf_delete_(const char *fname, int flen)
{
#endif
    char buf[1024];

    if (!fortchar_to_string(fname, flen, buf, sizeof(buf)))
	return (Integer) EAF_ERR_TOO_LONG;

    return (Integer) eaf_delete(buf);
}

#if defined(CRAY) || defined(WIN32)
Integer FATR eaf_stat_(_fcd p, Integer *avail_kb, _fcd fst)
{
    char *path = _fcdtocp(p);
    int pathlen = _fcdlen(p);
    char *fstype = _fcdtocp(fst);
    int fslen = _fcdlen(fst);
#else
Integer FATR eaf_stat_(const char *path, Integer *avail_kb, char *fstype, 
		  int pathlen, int fslen)
{
#endif
    char pbuf[1024];
    char fbuf[32];

    int code, kb;

    if (!fortchar_to_string(path, pathlen, pbuf, sizeof(pbuf)))
	return (Integer) EAF_ERR_TOO_LONG;

    code = eaf_stat(pbuf, &kb, fbuf, sizeof(fbuf));

    if (!code) {
	if (!string_to_fortchar(fstype, fslen, fbuf))
	    return (Integer) EAF_ERR_TOO_SHORT;
	*avail_kb = (double) kb;
    }

    return code;
}
    
#if defined(CRAY) || defined(WIN32)
void FATR eaf_errmsg_(Integer *code,  _fcd m)
{
    char *msg = _fcdtocp(m);
    int msglen = _fcdlen(m);
#else
void FATR eaf_errmsg_(Integer *code, char *msg, int msglen)
{
#endif
    char buf[80];

    eaf_errmsg((int) *code, buf);

    (void) string_to_fortchar(msg, msglen, buf);
}


double FATR eaf_util_random_(Integer* seed)
{
#ifdef NEC
  if(*seed) srand((unsigned) *seed);
  return ((double) rand())*4.6566128752458e-10;
#else
  if(*seed) srandom((unsigned) *seed);
  return ((double) random())*4.6566128752458e-10;
#endif
}

Integer FATR eaf_util_szint_()
{
  return (Integer)sizeof(Integer);
}

