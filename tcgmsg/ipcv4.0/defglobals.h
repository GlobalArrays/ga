/** @file
 * Actual definition of these globals ... need this once in
 * any executable ... included by cluster.c
 */
#ifndef DEFGLOBALS_H_
#define DEFGLOBALS_H_

#include "sndrcvP.h"

/*********************************************************
  Global information and structures ... all begin with SR_
  ********************************************************/

Integer SR_n_clus; /**< No. of clusters */
Integer SR_n_proc; /**< No. of processes excluding dummy master process */

int  SR_socks[MAX_PROCESS];
int  SR_socks_proc[MAX_PROCESS];
int  SR_nsock;
Integer SR_using_shmem;

Integer SR_clus_id; /**< Logical id of current cluster */
Integer SR_proc_id; /**< Logical id of current process */

Integer SR_debug; /**< flag for debug output */

Integer SR_parallel; /**< True if job started with parallel */

Integer SR_exit_on_error; /**< flag to exit on error */
Integer SR_error; /**< flag indicating error has been called
                  with SR_exit_on_error == FALSE */

Integer SR_numchild; /**< no. of forked processes */
Integer SR_pids[MAX_SLAVE]; /**< pids of forked processes */


/** This is used to store info from the PROCGRP file about each
 * cluster of processes
 */
struct cluster_info_struct SR_clus_info[MAX_CLUSTER];

struct process_info_struct SR_proc_info[MAX_PROCESS];

#endif /* DEFGLOBALS_H_ */
