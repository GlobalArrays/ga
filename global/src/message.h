/*$Id: message.h,v 1.3 1995-02-02 23:13:46 d3g681 Exp $*/

/* flags to specify blocking/nonblocking communication in TCGMSG */
#define SYNC  1
#define ASYNC 0
   
/* communicators for broadcast and global ops */ 
#define ALL_GRP         0
#define CLUST_GRP       1
#define ALL_CLUST_GRP   2
#define INTER_CLUST_GRP 3

#define MSG_BUF_SIZE    16384 
#define MSG_HEADER_SIZE 10*sizeof(Integer)
#define TOT_MSG_SIZE    (MSG_BUF_SIZE + MSG_HEADER_SIZE) 

/* size of buffer used to send/broadcast shared memory ids */
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
       Integer val; 
       Integer tag; 
       char    buffer[MSG_BUF_SIZE];
};

extern struct message_struct *MessageSnd, *MessageRcv;
extern Integer cluster_master;
extern Integer num_clusters;
extern Integer cluster_id;
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
extern void ga_snd_msg    ARGS_((Integer type, Void *buffer, Integer bytes, 
                                 Integer to, Integer sync));
extern void ga_rcv_msg    ARGS_((Integer type, Void *buffer, Integer buflen,
                                 Integer *msglen, Integer from,Integer *whofrom,
                                 Integer sync));
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

