/*$Id: message.h,v 1.8 1996-01-02 18:41:31 d3h325 Exp $*/

#ifdef MPI
#  include "mpi.h"
#endif

/* flags to specify blocking/nonblocking communication in TCGMSG */
#define SYNC  1
#define ASYNC 0
   
/* communicators for broadcast and global ops */ 
#define ALL_GRP         0         /* all GA processes */
#define CLUST_GRP       1         /* all compute processes */
#define ALL_CLUST_GRP   2         /* all processes inc. data server in cluster*/
#define INTER_CLUST_GRP 3         /* cluster masters */

/*#   define MSG_BUF_SIZE      262152 */
/* constants for send and receive buffers to handle remote requests */
#if defined(NX) || defined(SP1)
#   ifdef IWAY
#      define MSG_BUF_SIZE    129000
#   else
#      define MSG_BUF_SIZE    122840
#   endif
#elif defined(SYSV)
#   define MSG_BUF_SIZE      262152 
#else
#   define MSG_BUF_SIZE      16384 
#endif

/* limit the buffer size on SP when unexpected messages arrive (IWAY) */
#define IWAY_MSG_BUF_SIZE    8000 

#define REQ_FIELDS 10
#define MSG_HEADER_SIZE  (REQ_FIELDS*sizeof(Integer))
#define TOT_MSG_SIZE     (MSG_BUF_SIZE + MSG_HEADER_SIZE) 
#define MSG_BUF_DBL_SIZE ((TOT_MSG_SIZE + sizeof(double)-1)/sizeof(double))

/* size of buffer used to send/broadcast shared memory ids in ga_create */
#ifdef SUN
#  define SHMID_BUF_SIZE 400
#else
#  define SHMID_BUF_SIZE 80
#endif


struct message_struct{
       Integer g_a;
       Integer ilo; 
       Integer ihi; 
       Integer jlo; 
       Integer jhi; 
       Integer to; 
       Integer type; 
       Integer operation; 
       Integer from; 
       Integer tag; 
       char    buffer[MSG_BUF_SIZE];
};

/* max number of machines running data servers */
#ifdef DATA_SERVER
#  define MAX_CLUST 256
#else
#  define MAX_CLUST 2 
#endif
#define HOSTNAME_LEN 128

/* message id for nonblocking receive */
#if defined(MPI) && !(defined(SP1) || defined(NX))
       typedef MPI_Request msgid_t;      
#else
       typedef Integer msgid_t;      
#endif

/*
 * cluster_info_t corresponds to a subset of the TCGMSG cluster_info_struct 
 * nslave -- no. of processes in a cluster (includes master)
 * masterid -- process id of the cluster master
 * hostname -- machine name
 */ 
typedef struct {
  Integer nslave;               /* no. slaves on this host */
  Integer masterid;             /* process no. of cluster master */
  char hostname[HOSTNAME_LEN];  /* host name */ 
} cluster_info_t;

extern cluster_info_t GA_clus_info[MAX_CLUST];
extern Integer  GA_proc_id;       /* process id of current process --
                                   * this is NOT the GA process id
                                   */
extern Integer  GA_n_proc;            /* No. of processes */
extern Integer  GA_n_clus;            /* No. of clusters */
extern Integer  GA_clus_id;           /* Logical id of current cluster */

extern struct message_struct *MessageSnd, *MessageRcv;
extern Integer cluster_master;
extern Integer cluster_server;
extern Integer cluster_nodes;
extern Integer cluster_compute_nodes;
extern Integer ClusterMode;
extern Integer *NumRecReq;



#if !defined(NX) &&  defined(__STDC__) || defined(__cplusplus)
# define ARGS_(s) s
#else
# define ARGS_(s) ()
#endif

extern void ClustInfoInit ARGS_(( void));
extern Integer ClusterID  ARGS_((Integer ));
extern Integer DataServer ARGS_((Integer ));

extern Integer ga_msg_nnodes_  ARGS_((void));
extern Integer ga_msg_nodeid_  ARGS_((void));

extern void ga_msg_snd      ARGS_((Integer type, Void *buffer, Integer bytes, 
                                 Integer to));
extern void ga_msg_rcv      ARGS_((Integer type, Void *buffer, Integer buflen,
                                 Integer *msglen, Integer from,
                                 Integer *whofrom));
extern msgid_t ga_msg_ircv  ARGS_((Integer type, Void *buffer,Integer buflen,
                                 Integer from ));
extern Integer ga_msg_probe ARGS_((Integer type, Integer from));

extern void ga_msg_wait   ARGS_((msgid_t msgid, Integer *msglen, 
                                 Integer *whofrom));
extern void ga_msg_brdcst ARGS_((Integer type, Void* buffer, Integer len, 
                                 Integer root));
extern void ga_msg_sync_  ARGS_((void));

extern void ga_snd_req    ARGS_((Integer, Integer, Integer, Integer, Integer,
                                 Integer nbytes, Integer data_type,Integer oper,
                                 Integer, Integer to));
extern void ga_SERVER     ARGS_((Integer));
extern void ga_igop_clust ARGS_((Integer, Integer *, Integer, char *, Integer));
extern void ga_brdcst_clust ARGS_((Integer, Void*, Integer, Integer, Integer));
extern void ga_dgop_clust   ARGS_((Integer , DoublePrecision *, Integer, char *,
                                   Integer ));
extern void ga_wait_server  ARGS_(( void));
extern void        waitcom_ ARGS_((Integer*));

#undef ARGS_

