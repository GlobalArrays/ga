/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/strdup.c,v 1.5 2004-04-01 02:04:57 manoj Exp $ */

#include <stdlib.h>
extern char *strcpy();
extern size_t strlen();

char *strdup(s)
    char *s;
{
  char *new;

  if ((new = malloc((size_t) (strlen(s)+1))))
     (void) strcpy(new,s);

  return new;
}
