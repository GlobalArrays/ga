/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/cluster.c,v 1.6 1995-10-11 23:46:21 d3h325 Exp $ */

#include <stdio.h>

#ifdef SEQUENT
#include <strings.h>
#else
#include <string.h>
#endif

#include "sndrcvP.h"
#include "defglobals.h"

#if defined(ALLIANT) || defined(ENCORE) || defined(SEQUENT)|| defined(AIX)  \
                     || defined(CONVEX) || defined(ARDENT) || defined(ULTRIX) \
                     || defined(NEXT)
extern char *strdup();
extern char *strtok();
#endif

#ifdef SWTCH
#include "sw.h"
#endif

extern void Error();

void InitClusInfoNotParallel()
{
int SR_n_clus = 0;

    SR_clus_info[SR_n_clus].user = "?";
    SR_clus_info[SR_n_clus].hostname = "?";
    SR_clus_info[SR_n_clus].nslave = 1;
    SR_clus_info[SR_n_clus].image = "?";
    SR_clus_info[SR_n_clus].workdir = "?";
    SR_clus_info[SR_n_clus].masterid = 0;
}
  
void InitClusInfo(procgrp, masterhostname)
     char *procgrp, *masterhostname;
/*
  Initialize the SR_clus_info structure, SR_n_clus and SR_n_proc
  by parsing the PROCGRP info.

  The procgrp file consists of white space separated records.
  user host nslave image workdir 

  Masterhostname is the name of the host running the parallel command.

  This routine could do with some more error checking.
  
*/
{
  char *user, *host, *nslave, *image, *workdir;
  char *white = " \t\n";
  char *tmp = strdup(procgrp);
  int i;

  SR_n_clus = 0;
  SR_n_proc = 0;

  if (!tmp) Error("InitClusInfo: no memory", 0L);

  while (1) {
    user = strtok(tmp, white);
    tmp = (char *) NULL;
    if (user == (char *) NULL)
      break;
    host = strtok(tmp, white);
    nslave = strtok(tmp, white);
    image = strtok(tmp, white);
    workdir = strtok(tmp, white);
    if (workdir == (char *) NULL)
	Error("InitClusInfo: error parsing PROCGRP, line=",SR_n_clus+1);

    if (SR_n_clus == MAX_CLUSTER)
      Error("InitClusInfo: maximum no. of clusters exceeded",
            (long) MAX_CLUSTER);

    SR_clus_info[SR_n_clus].user = strdup(user);
    SR_clus_info[SR_n_clus].hostname = strdup(host);
    SR_clus_info[SR_n_clus].nslave = atoi(nslave);
    SR_clus_info[SR_n_clus].image = strdup(image);
    SR_clus_info[SR_n_clus].workdir = strdup(workdir);
    SR_clus_info[SR_n_clus].masterid = SR_n_proc;
#if defined(ALLIANT) && defined(SWTCH)
    SR_clus_info[SR_n_clus].swtchport = sw_port_by_name(host);
#endif

    if (!SR_clus_info[SR_n_clus].user || !SR_clus_info[SR_n_clus].hostname ||
        !SR_clus_info[SR_n_clus].image || !SR_clus_info[SR_n_clus].workdir)
      Error("InitClusInfo: no memory 2 ", 0L);

    for (i=0; i<atoi(nslave); i++)
      SR_proc_info[SR_n_proc+i].clusid = SR_n_clus;

    SR_n_proc += SR_clus_info[SR_n_clus].nslave;
    SR_n_clus++;
  }

  

  /* Define info about the parallel command process */
  SR_proc_info[SR_n_proc].clusid   = SR_n_clus;
  SR_clus_info[SR_n_clus].hostname = strdup(masterhostname);
  SR_clus_info[SR_n_clus].user     = "?";
  SR_clus_info[SR_n_clus].workdir  = "?";
  SR_clus_info[SR_n_clus].image    = "parallel";
  if (!SR_clus_info[SR_n_clus].hostname)
    Error("InitClusInfo: no memory 3 ", 0L);
#if defined(ALLIANT) && defined(SWTCH)
  SR_clus_info[SR_n_clus].swtchport = sw_port_by_name(masterhostname);
#endif

  free(tmp);
}


void PrintClusInfo()
{
  long i, clus_to_print;
  
  clus_to_print = SR_parallel ? SR_n_clus+1: SR_n_clus;

  printf("No. Clusters: %d\n", SR_n_clus);
  for (i=0; i<clus_to_print; i++)
#ifndef SWTCH
    (void) printf("Cluster %ld {\n  user = %s\n  host = %s\n  nslave = %ld\n\
  image = %s\n  workdir = %s\n  masterid = %ld}\n",
#else
    (void) printf("Cluster %ld {\n  user = %s\n  host = %s\n  nslave = %ld\n\
  image = %s\n  workdir = %s\n  swport = %ld  masterid = %ld}\n",
#endif
		  i,
		  SR_clus_info[i].user,
		  SR_clus_info[i].hostname,
		  SR_clus_info[i].nslave,
		  SR_clus_info[i].image,
		  SR_clus_info[i].workdir,
#ifdef SWTCH
		  SR_clus_info[i].swtchport,
#endif
		  SR_clus_info[i].masterid);
  printf("SR_clus_info = %d size=%d\n",(long) SR_clus_info, sizeof(struct cluster_info_struct));
  (void) fflush(stdout);
}

void InitGlobal()
/*
  Initialize all the globals to something appropriate
*/
{
  long i;

  SR_n_clus = 1;
  SR_n_proc = 1;

  SR_clus_id = 0;
  SR_proc_id = 0;

  SR_debug = FALSE;

  SR_exit_on_error = TRUE;

  SR_error = FALSE;

  SR_nchild = 0;
  for (i=0; i<MAX_SLAVE; i++)
    SR_pids[i] = 0;

  for (i=0; i<MAX_CLUSTER; i++) {
    SR_clus_info[i].user = (char *) NULL;
    SR_clus_info[i].hostname = (char *) NULL;
    SR_clus_info[i].nslave = 0;
    SR_clus_info[i].image = (char *) NULL;
    SR_clus_info[i].workdir = (char *) NULL;
    SR_clus_info[i].masterid = 0;
#if defined(ALLIANT) && defined(SWTCH)
    SR_clus_info[i].swtchport = -1;
#endif
  }

  for (i=0; i<MAX_PROCESS; i++) {
    SR_proc_info[i].clusid = 0;
    SR_proc_info[i].slaveid = 0;
    SR_proc_info[i].local = 0;
    SR_proc_info[i].sock = -1;
    SR_proc_info[i].shmem = (char *) NULL;
    SR_proc_info[i].shmem_size = 0;
    SR_proc_info[i].shmem_id = -1;
    SR_proc_info[i].buffer = (char *) NULL;
    SR_proc_info[i].buflen = 0;
    SR_proc_info[i].header = (MessageHeader *) 0;
    SR_proc_info[i].semid = -1;
    SR_proc_info[i].sem_pend = -1;
    SR_proc_info[i].sem_read = -1;
    SR_proc_info[i].sem_written = -1;
    SR_proc_info[i].n_rcv = 0;
    SR_proc_info[i].nb_rcv = 0;
    SR_proc_info[i].t_rcv = 0;
    SR_proc_info[i].n_snd = 0;
    SR_proc_info[i].nb_snd = 0;
    SR_proc_info[i].t_snd = 0;
    SR_proc_info[i].peeked = FALSE;
  }
}

