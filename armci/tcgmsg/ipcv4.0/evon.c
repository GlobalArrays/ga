#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/evon.c,v 1.4 1995-02-24 02:17:17 d3h325 Exp $ */

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

#include <stdlib.h>

#include "evlog.h"

/* These to get portable FORTRAN interface ... these routines
   will not be called from C which has the superior evlog interface */

#if defined(AIX) && !defined(EXTNAME)
#define evon_     evon
#define evoff_    evoff
#define evbgin_   evbgin
#define evend_    evend
#define event_    event
#endif

void evon_()
{
#ifdef EVENTLOG
  evlog(EVKEY_ENABLE, EVKEY_LAST_ARG);
#endif
}

void evoff_()
{
#ifdef EVENTLOG
  evlog(EVKEY_DISABLE, EVKEY_LAST_ARG);
#endif
}

void evbgin_(string, len)
  char *string;
  int   len;
{
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

void evend_(string, len)
  char *string;
  int   len;
{
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

void event_(string, len)
  char *string;
  int   len;
{
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
