#include "tcgmsg.h"
#include "srftoc.h"

#include <stdlib.h>
#include <string.h>
#include <signal.h>

#ifdef SHMEM
#include "shmem.h"
#endif

#define    MAX_PROC 256

#ifdef PBEGIN_C
/* This stupidity to avoid multiple defininitions in the SGI linker */
#define EXTERN /**/
#else
#define EXTERN extern
#endif

extern void USleep(long);

EXTERN long TCGMSG_nodeid;	/* The id of this process */
EXTERN long TCGMSG_nnodes;	/* Total no. of processes */

EXTERN char *TCGMSG_shmem;	/* Pointer to shared-memory segment */
EXTERN long  TCGMSG_shmem_id;	/* ID of shared-memory segment */
EXTERN long  TCGMSG_shmem_size;	/* Size of shared-memory segment */

EXTERN long TCGMSG_caught_sigint; /* True if SIGINT was trapped */

/* Structure defines shared memory buffer ... each process has
   one for every process that can send to it via shared memory.

   Adjust SHMEM_BUF_SIZE so that sizeof(ShmemBuf) is an integer
   multiple of page sizes. Structure of this buffer is exploited
   in T3D code. */

#define SHMEM_BUF_SIZE (8096  - 4*sizeof(long))
typedef struct {
  long info[4];                 /* 0=type, 1=length, 2=tag, 3=full */
  char buf[SHMEM_BUF_SIZE];	/* Message buffer */
} ShmemBuf;

/* Structure defines an entry in the send q */

typedef struct {
  long msgid;			/* Message id for msg_status */
  long type;			/* Message type */
  long node;			/* Destination node */
  long tag;			/* Message tag */
  char *buf;			/* User or internally malloc'd buffer */
  long lenbuf;			/* Length of user buffer in bytes */
  long written;			/* Amount already sent */
  long buffer_number;           /* No. of buffers alread sent */
  long free_buf_on_completion;	/* Boolean true if free buffer using free */
  void *next;			/* Pointer to next entry in linked list */

  void *next_in_ring;		/* Pointer to next entry in ring of free entries */
  long active;			/* 0/1 if free/allocated */
} SendQEntry;

/* This structure holds basically all process specific information */

#define COMM_MODE_NONE  0
#define COMM_MODE_SHMEM 1
#define COMM_MODE_SOCK  2

typedef struct {
  ShmemBuf *sendbuf;		/* Shared-memory buffer for sending to node*/
  ShmemBuf *recvbuf;		/* Shared-memory buffer for receiving from */
  int sock;			/* Socket for send/receive */
  int comm_mode;		/* Defines communication info */
  pid_t pid;			/* Unix process id (or 0 if unknown) */
  long n_snd;			/* No. of messages sent from this process */
  long n_rcv;			/* No. of messages recv from this process */
  SendQEntry *sendq;		/* Queue of messages to be sent */
} ProcInfo;

EXTERN ProcInfo *TCGMSG_proc_info; /* Will point to array of structures */

EXTERN SendQEntry *TCGMSG_sendq_ring; /* Circular ring of SendQEntry structures 
					 for fast allocation/free */

#ifdef CRAY_T3D  /* on this machine we don't use SYS V shared memory */
       ShmemBuf  TCGMSG_receive_buffer[MAX_PROC];
       void t3d_gops_init();
#      pragma _CRI cache_align TCGMSG_receive_buffer 
#endif

