/* $Header: /tmp/hpctools/ga/tcgmsg-mpi/evon.c,v 1.4 2003-07-10 15:12:01 d3h325 Exp $ */
#ifdef __crayx1
#undef CRAY
#endif

/* Crude FORTRAN interface to C event logging routines.
   See evlog.c for more details.

   FORTRAN character variables are so unportable that guaranteeing
   that U can parse a variable length argument list is next to impossible.

   This provides very basic event logging functionality.

   CALL EVON()

      enable logging.

   CALL EVOFF()
 
      disable logging.

   CALL EVBGIN("event description")

      push event onto state stack

   CALL EVEND("event description")

      pop event off state stack

   CALL EVENT("event description")

      log occurence of event that doesn't change state stack
*/

#ifdef IPSC
#define bcopy(a, b, n) memcpy((b), (a), (n))
#endif

#if defined(ULTRIX) || defined(SGI) || defined(NEXT) || defined(HPUX) || \
    defined(KSR)    || defined(DECOSF)
extern void *malloc();
#else
extern char *malloc();
#endif

#ifdef WIN32
#include "winf2c.h"
#else
#define FATR 
#endif

#include "evlog.h"

/* These to get portable FORTRAN interface ... these routines
   will not be called from C which has the superior evlog interface */

#if (defined(AIX) || defined(NEXT) || defined(HPUX)) && !defined(EXTNAME)
#define evon_     evon
#define evoff_    evoff
#define evbgin_   evbgin
#define evend_    evend
#define event_    event
#endif

#if defined(CRAY) || defined(ARDENT) || defined(WIN32) || defined(HITACHI)
#define evon_     EVON
#define evoff_    EVOFF
#define evbgin_   EVBGIN
#define evend_    EVEND
#define event_    EVENT
#endif

/* Define crap for handling FORTRAN character arguments */

#ifdef CRAY
#include <fortran.h>
#endif
#ifdef ARDENT
struct char_desc {
  char *string;
  int len;
};
#endif

void FATR evon_()
{
#ifdef EVENTLOG
  evlog(EVKEY_ENABLE, EVKEY_LAST_ARG);
#endif
}

void FATR evoff_()
{
#ifdef EVENTLOG
  evlog(EVKEY_DISABLE, EVKEY_LAST_ARG);
#endif
}

#if defined(ARDENT)
void evbgin_(arg)
     struct char_desc *arg;
{
  char *string = arg->string;
  int   len = arg->len;
#elif defined(CRAY) || defined(WIN32)
void FATR evbgin_(arg)
     _fcd arg;
{
  char *string = _fcdtocp(arg);
  int len = _fcdlen(arg);
#else
void evbgin_(string, len)
  char *string;
  int   len;
{
#endif
#ifdef EVENTLOG
  char *value = malloc( (unsigned) (len+1) );

  if (value) {
    (void) bcopy(string, value, len);
    value[len] = '\0';
    evlog(EVKEY_BEGIN, value, EVKEY_LAST_ARG);
    (void) free(value);
  }
#endif
}

#ifdef ARDENT
void evend_(arg)
     struct char_desc *arg;
{
  char *string = arg->string;
  int   len = arg->len;
#elif defined(CRAY) || defined(WIN32)
void FATR evend_(arg)
     _fcd arg;
{
  char *string = _fcdtocp(arg);
  int len = _fcdlen(arg);
#else
void FATR evend_(string, len)
  char *string;
  int   len;
{
#endif
#ifdef EVENTLOG
  char *value = malloc( (unsigned) (len+1) );

  if (value) {
    (void) bcopy(string, value, len);
    value[len] = '\0';
    evlog(EVKEY_END, value, EVKEY_LAST_ARG);
    (void) free(value);
  }
#endif
}
  
#ifdef ARDENT
void event_(arg)
     struct char_desc *arg;
{
  char *string = arg->string;
  int   len = arg->len;
#elif defined(CRAY) || defined(WIN32)
void FATR event_(arg)
     _fcd arg;
{
  char *string = _fcdtocp(arg);
  int len = _fcdlen(arg);
#else
void FATR event_(string, len)
  char *string;
  int   len;
{
#endif
#ifdef EVENTLOG
  char *value = malloc( (unsigned) (len+1) );

  if (value) {
    (void) bcopy(string, value, len);
    value[len] = '\0';
    evlog(EVKEY_EVENT, value, EVKEY_LAST_ARG);
    (void) free(value);
  }
#endif
}
