/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/strtok.c,v 1.5 2004-04-01 02:04:57 manoj Exp $ */

/*
  Primitive version of strtok for alliant etc who don't have it.

  I think it works .... ?
*/

#undef NULL
#define NULL 0

static int InSet(a, set)
     char *a, *set;
/*
  Return 1 if in set
         0 otherwise
*/
{
  register char test;
  register char b = (*a);

  while ( (test = *set++) )
    if (test == b)
      return 1;

  return 0;
}

static char *NextNotInSet(string, set)
     char *string, *set;
/*
  Return pointer to next character in string not in set or
  return NULL pointer if no such character
*/
{
  if (string == (char *) NULL)    /* Return NULL if given NULL */
    return (char *) NULL;

  while (*string) {
    if (InSet(string, set))
      string++;
    else
      break;
  }

  if (*string)
    return string;
  else
    return (char *) NULL;
}

static char *NextInSet(string, set)
     char *string, *set;
/*
  Return pointer to next character in string in set or
  return NULL pointer if no such character
*/
{
  if (string == (char *) NULL)    /* Return NULL if given NULL */
    return (char *) NULL;

  while (*string) {
    if (InSet(string, set))
      break;
    else
      string++;
  }

  if (*string)
    return string;
  else
    return (char *) NULL;  
}

char *strtok(s1, s2)
     char *s1, *s2;
/*
  Naive version of strtok for the alliant
*/
{
  static char *string = (char *) NULL;
  char *start, *end;
  
  if (s1 != (char *) NULL)          /* Initialize on first call */
    string = s1;
  
  start = NextNotInSet(string, s2); /* Find start of next token */

  end = NextInSet(start, s2);       /* Find end of this token */

  if (end == (char *) NULL)
    string = (char *) NULL;
  else {
    string = end + 1;
    *end = '\0';
  }

  return start;
}
