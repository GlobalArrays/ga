
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


void ClusterInitInfo();
Integer DataServer();
