/*$Id: strdup.c,v 1.2 1995-02-02 23:25:55 d3g681 Exp $*/
/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/strdup.c,v 1.2 1995-02-02 23:25:55 d3g681 Exp $ */

#if defined(ULTRIX) || defined(SGI) || defined(NEXT) || defined(HPUX) \
                    || defined(DECOSF)
extern void *malloc();
#else
extern char *malloc();
#endif
extern char *strcpy();

char *strdup(s)
    char *s;
{
  char *new;

  if (new = malloc((unsigned) (strlen(s)+1)))
    (void) strcpy(new,s);

  return new;
}
