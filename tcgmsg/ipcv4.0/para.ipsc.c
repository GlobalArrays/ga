/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/para.ipsc.c,v 1.4 1995-02-24 02:17:34 d3h325 Exp $ */

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <string.h>
#include <signal.h>
#include <cube.h>

extern long errno;
extern char *malloc();
extern char *getenv();

static char *cubename = "defaultname";

static int attached = 0;       /* True if a cube is attached */

static int do_getcube = 0;     /* True if we did/need to do a getcube */
static char *cubet = (char *) NULL;  /* User input for cubetype for getcube */
static int wait_getcube = 0;   /* True if we should wait for availability */

void Error(s, i)
     char *s;
     long i;
/* Usual error handler ... tidy up the cube if there is one */
{
  (void) fflush(stdout);
  (void) fprintf(stderr, "%s (%ld)\n", s, i);
  (void) fflush(stderr);
  (void) perror("system message");
  (void) fflush(stderr);

  if (attached)
    (void) _killcube(-1, -1);

  if (do_getcube)
    (void) _relcube(cubename);

  (void) exit(1);
}

/*ARGSUSED*/
void SigintHandler(sig, code, scp, addr)
     int sig, code;
     struct sigcontext *scp;
     char *addr;
{
  Error("SigintHandler: signal was caught",(long) code);
}

void TrapSigint()
/*
  Trap the signal SIGINT
*/
{
  if ( signal(SIGINT, SigintHandler) == (void (*)()) -1)
    Error("TrapSigint: error from signal setting SIGINT",(long) SIGINT);
}

static char *ProcgrpFile(argument)
     char *argument;
/*
  Find the name of the procgrp file from

  1) an argument from the command line with .p appended
  2) as 1) but also prepending $HOME/pdir/
  2) the translation of the environmental variable PROCGRP
  3) the file PROCGRP in the current directory
*/
{
  char *tmp, *home;
  int len;
  struct stat buf;

  if (argument != (char *) NULL) {
    len = strlen(argument);
    tmp = malloc((unsigned) (len+3) );
    (void) strcpy(tmp, argument);
    (void) strcpy(tmp+len, ".p");

    if (stat(tmp, &buf) == 0)     /* try ./arg1.p */
      return tmp;
    else
      (void) free(tmp);
    
    if ( (home = getenv("HOME")) != (char *) NULL ) {
      tmp = malloc((unsigned) (strlen(home) + len + 9));
      (void) strcpy(tmp, home);
      (void) strcpy(tmp+strlen(home),"/pdir/");
      (void) strcpy(tmp+strlen(home)+6,argument);
      (void) strcpy(tmp+strlen(home)+6+len,".p");

      (void) printf("tmp = %s\n",tmp);

      if (stat(tmp, &buf) == 0)     /* try $HOME/pdir/arg1.p */
	return tmp;
      else
	(void) free(tmp);
    }
  }

  if ( (tmp = getenv("PROCGRP")) != (char *) NULL )
    if (stat(tmp, &buf) == 0)
      return tmp;
 
  return strdup("PROCGRP");
}
  
static void SkipPastEOL(fp)
  FILE *fp;
/*
  Read past first newline character
*/
{
  int test;

   while ( (char) (test = getc(fp)) != '\n')
     if (test == EOF)
       break;
}

static char *GetProcgrp(filename, len_procgrp)
     char *filename;
     long *len_procgrp;
/*
  Read the entire contents of the PROCGRP into a NULL terminated
  character string. Be lazy and read the file twice, first to
  count the number of characters (ftell cannot be beleived?).
*/
{
  FILE *file;
  char *tmp, *procgrp;
  int status;

  if ( (file = fopen(filename,"r")) == (FILE *) NULL ) {
    (void) fprintf(stderr,"Master: PROCGRP = %s\n",filename);
    Error("GetProcgrp: failed to open PROCGRP", (long) 0);
  }

  *len_procgrp = 0;
  while ( (status = getc(file)) != EOF) {
    if (status == '#')
      SkipPastEOL(file);
    else
      (*len_procgrp)++;
  }

  (*len_procgrp)++;

  if ( (tmp = procgrp = malloc((unsigned) *len_procgrp)) == (char *) NULL )
    Error("GetProcgrp: failed in malloc",  (long) *len_procgrp);

  (void) fseek(file, 0L, (int) 0);   /* Seek to beginning of file */

  while ( (status = getc(file)) != EOF) {
    if (status == '#')
      SkipPastEOL(file);
    else
      *tmp++ = (char) status;
  }
  
  *tmp = '\0';

  if ( (int) (tmp - procgrp + 1) != *len_procgrp )
    Error("GetProcgrp: screwup dimensioning procgrp",  (long) *len_procgrp);

  (void) fclose(file);

  return procgrp;
}

int main(argc, argv)
     int argc;
     char **argv;
/*
  Generic host program for ipsc.

  parallel [-w] [-t cubetype] [-C cubename] [procgrp]

  if '-t cubetype' is specified
     then getcube/relcube are performed internally.
  else
     it is assumed that a getcube has been performed previously
     and the cube should not be released on exit

  if '-w' is specified in addition to '-t cubetype'
     then the getcube will wait until the requested type is available
  else
     then the getcube will return with an error
*/
{
  long node, lo, hi;
  long len_procgrp;
  char **dirlist;
  char *white = " \t\n";
  char *contents, *alo, *ahi, *image, *workdir;
  char *procgrp = (char *) NULL;

  /* Catch SIGINT to remove need for manual killcube */

  TrapSigint();

  /* Parse the argument list */

  argc--; argv++;  /* Throw away the name of the command */
  while (argc) {

    if (!strcmp(*argv, "-c") || !strcmp(*argv,"-C")) { /* Look for -C/-c */
      argc--; argv++;
      if (argc) {
	cubename = *argv;               /* cubename is next argument */
	if (strlen(cubename) >= NAMELEN)
	  Error("parallel: cubename must be less than 16 chars",
		(long) strlen(cubename));
      }
      else                             /* cubename not there! */
	Error("parallel: -c specified but no cubename", (long) argc);
    }
    else if (!strcmp(*argv, "-w")) {  /* look for -w */
      wait_getcube = 1;
    }
    else if (!strcmp(*argv, "-t")) {  /* look for -t */
      do_getcube = 1;
      argc--; argv++;
      if (argc)
	cubet = *argv;                /* cubetype is next argument */
      else
	Error("parallel: -t requires cubetype as argument > 0", (long) -1);
    }
    else {
      procgrp = *argv;
    }
    argc--; argv++;
  }

  /* Determine the name of the procgrp file and read its contents
     stripping off comments etc */

  procgrp = ProcgrpFile(procgrp);
  contents = GetProcgrp(procgrp, &len_procgrp);

  /* If -t was specified need to do a getcube and a relcube
     In addition if -w was specified we should wait if the cube is
     not initially available */

    if (do_getcube) {
      if (wait_getcube) {
        while(_getcube(cubename, cubet, (char *) NULL, 0)) {
	  (void) printf("Requested cubetype not available ... waiting\n");
	  (void) fflush(stdout);
	  (void) sleep(60);
        }
      }
      else {
        if (_getcube(cubename, cubet, (char *) NULL, 0))
          Error("parallel: getcube failed", (long) -1);
      }
    }

  /* Try and attach to the cube */

  if(_attachcube(cubename))
    Error("parallel: attachcube failed", (long) -1);

  if(_setpid(0))
    Error("parallel: setpid failed", (long) -1);

  (void) printf("cubename=%s, procgrp=%s, numnodes=%ld\n", 
		cubename, procgrp, numnodes());
  (void) fflush(stdout);

  /* Do a killcube to make sure all is clean */

  if(_killcube(-1, -1))
    Error("parallel: initial killcube failed", (long) -1);

  /* Allocate an array to make sure that we are not double loading
     into processors */

  if (!(dirlist = (char **) malloc(sizeof(char *)*numnodes())))
    Error("parallel: failed to malloc work space",(long) 4*numnodes());

  for (node=0; node<numnodes(); node++)
    dirlist[node] = (char *) 0;

  /* Parse the contents of the procgrp file */

  while (1) {
    
    alo = strtok(contents, white);
    contents = (char *) NULL;
    if (!alo)
      break;
    
    ahi = strtok(contents, white);
    image = strtok(contents, white);
    workdir = strtok(contents, white);
    
    if ( !alo || !ahi || !image || !workdir )
      Error("parallel: failed to parse procgrp file", (long) -1);
    
    if (!strcmp(alo, "$"))
      lo = numnodes()-1;
    else if (!strcmp(alo, "$-1"))
      lo = numnodes()-2;
    else
      lo = atoi(alo);
    
    if (!strcmp(ahi, "$"))
      hi = numnodes()-1;
    else if (!strcmp(ahi, "$-1"))
      hi = numnodes()-2;
    else
      hi = atoi(ahi);
    
    if ( (lo < 0) || (lo > numnodes()-1) || 
	(hi < lo) || (hi > numnodes()-1) )
      Error("parallel: low-high processor range invalid",(long) hi*1000+lo);
    
    /* Check that these nodes are not in use already */
    
    for (node=lo; node<=hi; node++) {
      if (dirlist[node])
	Error("parallel: node is already loaded", node);
      dirlist[node] = workdir;
    }
    
    /* Actually load the executables */
    
    (void) printf("load nodes %d-%d with %s, workdir=%s\n",
		  lo, hi, image, workdir);
    (void) fflush(stdout);
    if ( (lo == 0) && (hi == (numnodes()-1)) ) {
      if(_load(image, -1, 0))
	Error("parallel: error loading all nodes", (long) -1);
    }
    else {
      for(node=lo; node<=hi; node++)
        if(_load(image, node, 0))
          Error("parallel: error loading node", node);
    }
  }

  for (node=0; node<numnodes(); node++)
    if (dirlist[node]) {
      if(_csend(2, dirlist[node], strlen(dirlist[node])+1, node, 0))
	Error("parallel: error sending workdir to node", node);
    }
    else {
      (void) printf("parallel: warning ... nothing loaded on node %ld\n",
		    node);
      (void) fflush(stdout);
    }

  /* Now wait for the node processes to complete */

  if(_waitall(-1,0))
    Error("parallel: error waiting for processes to complete\n", (long) -1);

  /* Do a killcube for tidiness */

  if(_killcube(-1, -1))
    Error("parallel: killcube failed after successful finish", (long) -1);

  if (do_getcube)
    if( _relcube(cubename) )
      Error("parallel: relcube failed after successful finished",
	    (long) -1);

  return 0;
}
