#include <sys/types.h>
#include "eaf.h"
#include "eafP.h"
#include "types.f2c.h"

#if defined(CRAY)
#include <fortran.h>
#define eaf_write_ EAF_WRITE
#define eaf_awrite_ EAF_AWRITE
#define eaf_read_ EAF_READ
#define eaf_aread_ EAF_AREAD
#define eaf_wait_ EAF_WAIT
#define eaf_probe_ EAF_PROBE
#define eaf_open_ EAF_OPEN
#define eaf_close_ EAF_CLOSE
#define eaf_delete_ EAF_DELETE
#define eaf_length_ EAF_LENGTH
#define eaf_truncate_ EAF_TRUNCATE
#define eaf_stat_ EAF_STAT
#define eaf_eof_ EAF_EOF
#define eaf_error_ EAF_ERROR
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
	
Integer eaf_write_(Integer *fd, double *offset, const void *buf, 
		   Integer *bytes)
{
    if (!valid_offset(*offset)) return EAF_ERR_NONINTEGER_OFFSET;
    return (Integer) eaf_write((int) *fd, (eaf_off_t) *offset, buf, 
			       (size_t) *bytes);
}

Integer eaf_awrite_(Integer *fd, double *offset, const void *buf, 
		    Integer *bytes, Integer *req_id)
{
    int req, status;

    if (!valid_offset(*offset)) return EAF_ERR_NONINTEGER_OFFSET;
    status = eaf_awrite((int) *fd, (eaf_off_t) *offset, buf, 
			(size_t) *bytes, &req);
    *req_id = (Integer) req;
    return (Integer) status;
}

Integer eaf_read_(Integer *fd, double *offset, void *buf, Integer *bytes)
{
    if (!valid_offset(*offset)) return EAF_ERR_NONINTEGER_OFFSET;
    return (Integer) eaf_read((int) *fd, (eaf_off_t) *offset, buf, 
			      (size_t) *bytes);
}

Integer eaf_aread_(Integer *fd, double *offset, void *buf, 
		    Integer *bytes, Integer *req_id)
{
    int req, status;

    if (!valid_offset(*offset)) return EAF_ERR_NONINTEGER_OFFSET;
    status = eaf_aread((int) *fd, (eaf_off_t) *offset, buf, 
		       (size_t) *bytes, &req);
    *req_id = (Integer) req;
    return (Integer) status;
}

Integer eaf_wait_(Integer *fd, Integer *id)
{
    return (Integer) eaf_wait((int) *fd, (int) *id);
}

void eaf_print_stats_(Integer *fd)
{
    eaf_print_stats((int) *fd);
}

Integer eaf_truncate_(Integer *fd, double *length)
{
    return (Integer) eaf_truncate((int) *fd, (eaf_off_t) *length);
}

Integer eaf_probe_(Integer *id, Integer *status)
{
    int s, code;

    code = eaf_probe((int) *id, &s);
    *status = (Integer) s;

    return (Integer) code;
}


Integer eaf_close_(Integer *fd)
{
    return (Integer) eaf_close((int) *fd);
}

Integer eaf_length_(Integer *fd, double *length)
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

#if defined(CRAY) || defined(CRAY_T3D)
int eaf_open_(fcd *f, Integer *type, Integer *fd)
{
    char *fname = _fcdtocp(f);
    int flen = _fcdtolen(f);
#else
int eaf_open_(const char *fname, Integer *type, Integer *fd, int flen)
{
#endif
    char buf[1024];
    int code, tmp;

    if (!fortchar_to_string(fname, flen, buf, sizeof(buf)))
	return (Integer) EAF_ERR_TOO_LONG;

    code = eaf_open(buf, (int) *type, &tmp);
    *fd = (Integer) tmp;

    return code;
}

#ifdef CRAY
Integer eaf_delete_(fcd f)
{
    char *fname = _fcdtocp(f);
    int flen = _fcdtolen(f);
#else
Integer eaf_delete_(const char *fname, int flen)
#endif
{
    char buf[1024];

    if (!fortchar_to_string(fname, flen, buf, sizeof(buf)))
	return (Integer) EAF_ERR_TOO_LONG;

    return (Integer) eaf_delete(buf);
}

#ifdef CRAY
Integer eaf_stat_(fcd p, Integer *avail_kb, fcd fst)
    char *path = _fcdtocp(p);
    int pathlen = _fcdtolen(p);
    char *fstype = _fcdtocp(fst);
    int fslen = _fcdtolen(fst);
{
#else
Integer eaf_stat_(const char *path, int *avail_kb, char *fstype, 
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
    
#ifdef CRAY
void eaf_errmsg_(Integer *code,  fcd m)
{
    char *msg = _fcdtocp(m);
    int msglen = _fcdtolen(m);
#else
void eaf_errmsg_(Integer *code, char *msg, int msglen)
{
#endif
    char buf[80];

    eaf_errmsg((int) *code, buf);

    (void) string_to_fortchar(msg, msglen, buf);
}


