/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/sndrcvP.h,v 1.4 1995-02-24 02:17:48 d3h325 Exp $ */

/*
  This include file contains definitions PRIVATE to the message
  passing routines and not for public use. These items should not
  be directly manipulated even in the message passing routines, except
  by the appropriate lowlevel routines.

  Actual instances of the extern data items are declared in defglobals.h
  which is included by cluster.c.
*/

#define SNDRCVP

#ifndef MSGTYPES
#include "msgtypesc.h"
#endif

/******************************
  Defines and macro definitions
  *****************************/

#define MAX_CLUSTER 32           /* Maximum no. of clusters */
#define MAX_SLAVE   32           /* Maximum no. of slaves per cluster */
#define MAX_PROCESS 255          /* Maximum no. of processes */

#define TYPE_SETUP 32768         /* used for setup communication */
#define TYPE_CHECK 32769         /* used for checking communication */
#define TYPE_END   32770         /* used for propagating end message */
#define TYPE_NXTVAL (MSGINT | 32771) /* Used in nxtval */
#define TYPE_CONNECT (MSGINT | 32772) /* Used in RemoteConnect */
#define TYPE_BEGIN 32773         /* Used in pbegin and parallel */
#define TYPE_CLOCK_SYNCH 32774;  /* Used to synch clocks */

#define BIG_MESSAGE 41943040     /* 40Mb max message only for safety check.
				    Change as needed.*/

/* Shared memory allocated per process .. make even multiple of
   page size ... usually 4096 */
#ifdef KSR
#define SHMEM_BUF_SIZE 524288
#endif
#ifdef ALLIANT
#define SHMEM_BUF_SIZE 524288
#endif
#ifdef ENCORE
#define SHMEM_BUF_SIZE 4096
#endif
#ifdef SEQUENT
#define SHMEM_BUF_SIZE 16384
#endif
#ifdef KSR_NATIVE
#include "ksr.h"
#undef SHMEM_BUF_SIZE
#define SHMEM_BUF_SIZE KSR_SHMEM_BUF_SIZE
#endif
#if !defined(SHMEM_BUF_SIZE)
#define SHMEM_BUF_SIZE 65536
#endif

#define SR_SOCK_BUF_SIZE  32768       /* Size that system buffers set to */

#define PACKET_SIZE SR_SOCK_BUF_SIZE  /* Internal packet size over sockets */

#define TIMEOUT_ACCEPT 180         /* timeout for connection in secs */

#define TRUE 1
#define FALSE 0
#define DEBUG_ SR_debug           /* substitute name of debug flag */

/*********************************************************
  Global information and structures ... all begin with SR_
  ********************************************************/

extern long SR_n_clus;                   /* No. of clusters */
extern long SR_n_proc;                   /* No. of processes excluding dummy
				     master process */

extern long SR_clus_id;                  /* Logical id of current cluster */
extern long SR_proc_id;                  /* Logical id of current process */

extern long SR_debug;                    /* flag for debug output */

extern long SR_parallel;		/* True if job started with parallel */
extern long SR_exit_on_error;            /* flag to exit on error */
extern long SR_error;                    /* flag indicating error has been called
                                     with SR_exit_on_error == FALSE */

extern long SR_nchild;                   /* no. of forked processes */
extern long SR_pids[MAX_SLAVE];          /* pids of forked processes */


/* This is used to store info from the PROCGRP file about each
   cluster of processes */

struct cluster_info_struct {
  char *user;                     /* user name */
  char *hostname;                 /* hostname */
  long nslave;                    /* no. slave on this host */
  char *image;                    /* path executable image */
  char *workdir;                  /* work directory */
  long masterid;                  /* process no. of cluster master */
  int  swtchport;                 /* Switch port for alliant hippi */
};

extern struct cluster_info_struct SR_clus_info[MAX_CLUSTER];

typedef struct message_header_struct {
  long nodefrom;                  /* originating node of message */
  long nodeto;                    /* target node of message */
  long type;                      /* user defined type */
  long length;                    /* length of message in bytes */
  long tag;                       /* No. of this message for id */
} MessageHeader;

/* This is used to store all info about processes */

struct process_info_struct {
  long clusid;                     /* cluster no. for this process */
  long slaveid;                    /* slave no. in cluster 0,1,...,nslave */
  long local;                      /* boolean flag for local/remote */
  int sock;                        /* socket to remote process */
  char *shmem;                     /* shared memory region */
  long shmem_size;                 /* shared memory region size */
  long shmem_id;                   /* shared memory region id */
  char *buffer;                    /* shared memory message buffer */
  long buflen;                     /* shared memory message buffer size */
  MessageHeader *header;           /* shared memory message header */
  long semid;                      /* semaphore group id */
  long sem_pend;                   /* semaphore no. posted when data pending */
  long sem_read;                   /* semaphore no. posted when data read */
  long sem_written;                /* semaphore no. posted when data written */
  long n_rcv;                      /* No. of messages received */
  double nb_rcv;		   /* No. of bytes received */
  double t_rcv;			   /* Time spent receiving in sec */
  long n_snd;                      /* No. of messages sent */
  double nb_snd;		   /* No. of bytes sent */
  double t_snd;			   /* Time spent sending in sec */
  long peeked;                     /* True if have peeked at socket */
  MessageHeader head_peek;         /* Header that we peeked at */
  long *buffer_full;		   /* Flag indicating full buffer */
};

extern struct process_info_struct SR_proc_info[MAX_PROCESS];
