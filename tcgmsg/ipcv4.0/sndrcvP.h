/** @file
 * This include file contains definitions PRIVATE to the message
 * passing routines and not for public use. These items should not
 * be directly manipulated even in the message passing routines, except
 * by the appropriate lowlevel routines.
 *
 * Actual instances of the extern data items are declared in defglobals.h
 * which is included by cluster.c.
 */
#ifndef SNDRCVP_H_
#define SNDRCVP_H_

#include "typesf2c.h"
#include "msgtypesc.h"

#define MAX_CLUSTER 128               /**< Maximum no. of clusters */
#define MAX_SLAVE   512               /**< Maximum no. of slaves per cluster */
#define MAX_PROCESS 8192              /**< Maximum no. of processes */

#define TYPE_SETUP 32768              /**< used for setup communication */
#define TYPE_CHECK 32769              /**< used for checking communication */
#define TYPE_END   32770              /**< used for propagating end message */
#define TYPE_NXTVAL (MSGINT | 32771)  /**< Used in nxtval */
#define TYPE_CONNECT (MSGINT | 32772) /**< Used in RemoteConnect */
#define TYPE_BEGIN 32773              /**< Used in pbegin and parallel */
#define TYPE_CLOCK_SYNCH 32774;       /**< Used to synch clocks */

#ifdef BIG_MESSAGE_PROTECTION
/** 40Mb max message only for safety check. Change as needed.*/
#   define BIG_MESSAGE 41943040ul
#else
#   define BIG_MESSAGE 2147483647ul /**< 2GB */
#endif

/** Shared memory allocated per process.
 * Make even multiple of page size. Usually 4096
 */
#if !defined(SHMEM_BUF_SIZE)
#   define SHMEM_BUF_SIZE 131072
#endif

#define SR_SOCK_BUF_SIZE 32768       /**< Size that system buffers set to */
#define PACKET_SIZE SR_SOCK_BUF_SIZE /**< Internal packet size over sockets */
#define TIMEOUT_ACCEPT 180           /**< timeout for connection in secs */

#define TRUE 1
#define FALSE 0
#define DEBUG_ SR_debug /**< substitute name of debug flag */

/*********************************************************
  Global information and structures ... all begin with SR_
 ********************************************************/

extern Integer SR_n_clus; /**< No. of clusters */
extern Integer SR_n_proc; /**< No. of processes excluding dummy master process */

extern Integer SR_clus_id; /**< Logical id of current cluster */
extern Integer SR_proc_id; /**< Logical id of current process */

extern Integer SR_debug; /**< flag for debug output */

extern Integer SR_parallel; /**< True if job started with parallel */
extern Integer SR_exit_on_error; /**< flag to exit on error */
extern Integer SR_error; /**< flag indicating error has been called with SR_exit_on_error == FALSE */

extern Integer SR_numchild; /**< no. of forked processes */
extern Integer SR_pids[MAX_SLAVE]; /**< pids of forked processes */
extern int SR_socks[MAX_PROCESS]; /**< Sockets used for each process */
extern int SR_socks_proc[MAX_PROCESS]; /**< Process associated with a given socket */
extern int SR_nsock; /**< No. of sockets in the list */
extern Integer SR_using_shmem; /**< 1=if shmem is used for an process, 0 if all processes are connected to this one by sockets */

/** This is used to store info from the PROCGRP file about each
 * cluster of processes.
 */
struct cluster_info_struct {
    char *user;     /**< user name */
    char *hostname; /**< hostname */
    Integer nslave;    /**< no. slave on this host */
    char *image;    /**< path executable image */
    char *workdir;  /**< work directory */
    Integer masterid;  /**< process no. of cluster master */
    int  swtchport; /**< Switch port for alliant hippi */
};

extern struct cluster_info_struct SR_clus_info[MAX_CLUSTER];

typedef struct message_header_struct {
    Integer nodefrom; /**< originating node of message */
    Integer nodeto;   /**< target node of message */
    Integer type;     /**< user defined type */
    Integer length;   /**< length of message in bytes */
    Integer tag;      /**< No. of this message for id */
} MessageHeader;

/** This is used to store all info about processes. */
struct process_info_struct {
    Integer clusid;             /**< cluster no. for this process */
    Integer slaveid;            /**< slave no. in cluster 0,1,...,nslave */
    Integer local;              /**< boolean flag for local/remote */
    int sock;                /**< socket to remote process */
    char *shmem;             /**< shared memory region */
    Integer shmem_size;         /**< shared memory region size */
    Integer shmem_id;           /**< shared memory region id */
    char *buffer;            /**< shared memory message buffer */
    Integer buflen;             /**< shared memory message buffer size */
    MessageHeader *header;   /**< shared memory message header */
    Integer semid;              /**< semaphore group id */
    Integer sem_pend;           /**< semaphore no. posted when data pending */
    Integer sem_read;           /**< semaphore no. posted when data read */
    Integer sem_written;        /**< semaphore no. posted when data written */
    Integer n_rcv;              /**< No. of messages received */
    DoublePrecision nb_rcv;           /**< No. of bytes received */
    DoublePrecision t_rcv;            /**< Time spent receiving in sec */
    Integer n_snd;              /**< No. of messages sent */
    DoublePrecision nb_snd;           /**< No. of bytes sent */
    DoublePrecision t_snd;            /**< Time spent sending in sec */
    Integer peeked;             /**< True if have peeked at socket */
    MessageHeader head_peek; /**< Header that we peeked at */
    Integer *buffer_full;       /**< Flag indicating full buffer */
};

extern struct process_info_struct SR_proc_info[MAX_PROCESS];

#endif /* SNDRCVP_H_ */
