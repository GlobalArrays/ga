#ifndef TCGMSG_H
#define TCGMSG_H

#include <stdio.h>

typedef long Integer;
extern void Error(const char *, long);

extern void SYNCH_(Integer *);
extern Integer NNODES_(void);
extern Integer NODEID_(void);
extern void PBEGIN_(int, char **);
extern void PEND_(void);

extern int printf(const char *, ...);
extern int fprintf(FILE *, const char *, ...);
extern int fflush(FILE *);

#endif




