#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_SIGNAL_H
#   include <signal.h>
#endif
#if HAVE_UNISTD_H
#   include <unistd.h>
#endif
#if HAVE_STRINGS_H
#   include <strings.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif
#if HAVE_SYS_TYPES_H
#   include <sys/types.h>
#endif
#if HAVE_SYS_TIME_H
#   include <sys/time.h>
#endif
#if HAVE_SYS_CNX_TYPES_H
#   include <sys/cnx_types.h>
#endif
#if HAVE_SYS_WAIT_H
#   include <sys/wait.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif

extern int atoi(const char *nptr);

#if defined(SOLARIS)
/* See notes below on processor binding */
/*#include <sys/processor.h>*/
/*#include <sys/procset.h>*/
#endif

#include "cluster.h"
#include "signals.h"
#include "sndrcv.h"
#include "sndrcvP.h"
#include "sockets.h"

#if HAVE_SYS_IPC_H
#   include "sema.h"
#   include "shmem.h"
#endif

#ifdef EVENTLOG
#   include "evlog.h"
#endif

#ifdef USE_VAMPIR
#   include "tcgmsg_vampir.h"
#endif

extern void exit();
extern void InitClusInfoNotParallel();
extern int WaitAll(Integer nchild);

#define max(A, B) ( (A) > (B) ? (A) : (B) )
#define min(A, B) ( (A) < (B) ? (A) : (B) )

static int SR_initialized=0;

Integer TCGREADY_()
{
    return (Integer)SR_initialized;
}


static void ConnectAll()
{
    Integer j, k, clus1, clus2, node1, node2, nslave1, nslave2;

    for (clus1=1; clus1 < SR_n_clus; clus1++) {

        node1 = SR_clus_info[clus1].masterid;
        nslave1 = SR_clus_info[clus1].nslave;

        for (clus2=0; clus2 < clus1; clus2++) {

            node2 = SR_clus_info[clus2].masterid;

            RemoteConnect(node1, node2, SR_n_proc);  /* connect masters */

#if HAVE_SYS_IPC_H
            nslave2 = SR_clus_info[clus2].nslave;

            for (j=1; j<nslave1; j++) {
                RemoteConnect(node1+j, node2, node1);
                for (k=1; k<nslave2; k++) {
                    RemoteConnect(node1+j, node2+k, node2);
                }
            }
            for (k=1; k<nslave2; k++) {
                RemoteConnect(node1, node2+k, node2);
            }
#endif 
        }
    }

    /* Connect local slaves to master soley for next value service */

    for (clus1=0; clus1 < SR_n_clus; clus1++) {
        for (j=1; j<SR_clus_info[clus1].nslave; j++) {
            RemoteConnect(SR_n_proc,
                    SR_clus_info[clus1].masterid + j,
                    SR_clus_info[clus1].masterid);
        }
    }
}

static void PrintArgs(int argc, char **argv)
{
    int i;

    for (i=0; i<argc; i++) {
        (void) printf("argv[%d] = %s\n",i, argv[i]);
    }

    (void) fflush(stdout);
}


/**
 * first thing to call on entering program
 *
 * if the argument list contains 
 *   '-master hostname port nclus nproc clusid procid'
 * then this is running in parallel, else we are just running as
 * a single process.
 */
void tcgi_pbegin(int argc, char **argv)
{
    Integer i, masterid, len_pgrp, lenbuf, lenmes, sync=1;
    Integer nodesel, nodefrom, type, me, nslave, status;
    char *masterhostname, *cport;
    char *procgrp;
#ifdef EVENTLOG
    Integer start=MTIME_();
    char *eventfile;
#endif
#if HAVE_SYS_IPC_H
    Integer *flags;
#endif
#ifdef USE_VAMPIR
    vampir_init(argc,argv,__FILE__,__LINE__);
    tcgmsg_vampir_init(__FILE__,__LINE__);
    vampir_begin(TCGMSG_PBEGINF,__FILE__,__LINE__);
#endif

    if (SR_initialized) {
        Error("TCGMSG initialized already???",-1);
    }
    else {
        SR_initialized=1;
    }

    if (DEBUG_) {
        (void) printf("In pbegin .. print the arguments\n");
        PrintArgs(argc, argv);
        (void) fflush(stdout);
    }

    /* First initialize the globals as if only one process */

    if (DEBUG_) {
        (void) printf("pbegin: InitGlobal\n");
        (void) fflush(stdout);
    }
    InitGlobal();

    /* Set up handler for SIGINT and SIGCHLD */

    TrapSigint();
    TrapSigchld();

    /* If '-master host port' is not present return, else extract the
       master's hostname and port number */

    if (DEBUG_) {
        (void) printf("pbegin: look for -master in arglist\n");
        (void) fflush(stdout);
    }

    for (i=1; i<argc; i++) {
        if (strcmp(argv[i],"-master") == 0) {
            if ( (i+6) >= argc ) {
                Error("pbegin: -master present but not other arguments",
                        (Integer) argc);
            }
            break;
        }
    }

    if ( (i+6) >= argc ) {
        SR_parallel = FALSE;
        InitClusInfoNotParallel();
        SR_n_clus=1;
        return;
    }
    else {
        SR_parallel = TRUE;
    }

    if (DEBUG_) {
        (void) printf("pbegin: assign argument values\n");
        (void) fflush(stdout);
    }

    masterhostname = strdup(argv[i+1]);
    cport = strdup(argv[i+2]);
    SR_n_clus = atoi(argv[i+3]);
    SR_n_proc = atoi(argv[i+4]);
    SR_clus_id = atoi(argv[i+5]);
    SR_proc_id = atoi(argv[i+6]);

    /* Check out some of this info */

    if ((SR_n_clus >= MAX_CLUSTER) || (SR_n_clus < 1)) {
        Error("pbegin: invalid no. of clusters", SR_n_clus);
    }
    if ((SR_n_proc >= MAX_PROCESS) || (SR_n_proc < 1)) {
        Error("pbegin: invalid no. of processes", SR_n_proc);
    }
    if ((SR_clus_id >= SR_n_clus) || (SR_clus_id < 0)) {
        Error("pbegin: invalid cluster id", SR_clus_id);
    }
    if ((SR_proc_id >= SR_n_proc) || (SR_proc_id < 0)) {
        Error("pbegin: invalid process id", SR_proc_id);
    }

    /* Close all files we don't need. Process 0 keeps stdin/out/err.
       All others only stdout/err. */

    if (SR_clus_id != 0) {
        (void) fclose(stdin);
    }
    for (i=3; i<64; i++) {
        (void) close((int) i);
    }

    /* Connect to the master process which will have process id
       equal to the number of processes */

    if (DEBUG_) {
        (void) printf("pbegin: %ld CreateSocketAndConnect\n",NODEID_());
        (void) fflush(stdout);
    }
    masterid = SR_n_proc;
    SR_proc_info[SR_n_proc].sock = CreateSocketAndConnect(masterhostname,
            cport);

    /* Now we have initialized this info we should be able to use the
       standard interface routines rather than accessing the SR variables
       directly */

    /* Get the procgrp from the master process

       Note that byteordering and word length start to be an issue. */

    if (DEBUG_) {
        (void) printf("pbegin: %ld get len_pgrp\n",NODEID_());
        (void) fflush(stdout);
    }
    type  = TYPE_BEGIN | MSGINT;
    lenbuf = sizeof(Integer);
    nodesel = masterid;
    RCV_(&type, (char *) &len_pgrp, &lenbuf, &lenmes, &nodesel, &nodefrom,
            &sync);
    if (DEBUG_) {
        (void) printf("len_pgrp = %ld\n",len_pgrp); (void) fflush(stdout);
    }
    if ( (procgrp = malloc((unsigned) len_pgrp)) == (char *) NULL ) {
        Error("pbegin: failed to allocate procgrp",len_pgrp);
    }

    if (DEBUG_) {
        (void) printf("pbegin: %ld get progcrp len=%ld\n",NODEID_(),len_pgrp);
        (void) fflush(stdout);
    }
    type = TYPE_BEGIN | MSGCHR;
    RCV_(&type, procgrp, &len_pgrp, &lenmes, &nodesel, &nodefrom, &sync);
    if (DEBUG_) {
        (void) printf("procgrp:\n%55s...\n",procgrp); (void) fflush(stdout);
        (void) fflush(stdout);
    }

    /* Parse the procgrp to fill out SR_clus_info ... it also again works out
       SR_n_clus and SR_n_proc ... ugh */

    InitClusInfo(procgrp, masterhostname);

    if (DEBUG_) {
        PrintClusInfo();
        (void) fflush(stdout);
    }

    /* Change to desired working directory ... forked processes
       will inherit it */

    if(chdir(SR_clus_info[SR_clus_id].workdir) != 0) {
        Error("pbegin: failed to switch to work directory", (Integer) -1);
    }

    if (DEBUG_) {
        printf("%2ld: pbegin: changed to working directory %s\n",
                NODEID_(), SR_clus_info[SR_clus_id].workdir);
        (void) fflush(stdout);
    }

    /* If we have more than 1 process in this cluster we have to
       create the shared memory and semaphores and fork the processes
       partitioning out the resources */

    SR_using_shmem = 0;
#if HAVE_SYS_IPC_H
    me = NODEID_();
    nslave = SR_clus_info[SR_clus_id].nslave;
    if (nslave > 1) {
        SR_proc_info[me].shmem_size = nslave*SHMEM_BUF_SIZE + 
            (nslave+1)*sizeof(Integer);
        SR_proc_info[me].shmem_size = 
            ((SR_proc_info[me].shmem_size - 1)/4096)*4096 + 4096;
        if (DEBUG_) {
            (void) printf("pbegin: %ld allocate shmem, nslave=%ld\n",
                          NODEID_(), nslave);
            (void) fflush(stdout);
        }
        SR_using_shmem = 1;
        SR_proc_info[me].shmem = CreateSharedRegion(&SR_proc_info[me].shmem_id,
                &SR_proc_info[me].shmem_size);
        if (DEBUG_) {
            (void) printf("pbegin: %ld allocate sema, nslave=%ld\n",
                          NODEID_(), nslave);
            (void) fflush(stdout);
        }

        flags = (Integer *) (SR_proc_info[me].shmem + nslave*SHMEM_BUF_SIZE);

        (void) bzero(SR_proc_info[me].shmem, SR_proc_info[me].shmem_size);

        for (i=0; i<nslave; i++) {
            ((MessageHeader *) 
             (SR_proc_info[me].shmem + i * SHMEM_BUF_SIZE))->nodeto = -1;
            flags[i] = FALSE;
        }

        SR_proc_info[me].semid = SemSetCreate((Integer) 3*nslave, (Integer) 0);

#   ifdef SOLARIS
        /* If there fewer processes than processors it appears beneficial
           to bind processes to processors.  It also appears useful to
           leave the lowest numbered processors free (???).  
           BUT ... this code is not general enough since the configured
           processors are not necessarily numbered consecutively and
           we also need to add logic to determine the list of processors
           that have not already been bound to a process. 

           Need to also modify the code below for binding slaves and enable
           the include of processor.h and procset.h */

        /* printf("binding master process %d to processor %d\n", getpid(), 31-0);
           if (processor_bind(P_PID, P_MYID, 31-0, (void *) NULL))
           printf("binding to %d failed\n", 31-0); */
#   endif /* SOLARIS */

        for (i=1; i<nslave; i++) {
            if (DEBUG_) {
                (void) printf("pbegin: %ld fork process, i=%ld\n", NODEID_(), nslave);
                (void) fflush(stdout);
            }
#   if HAVE_SYS_CNX_TYPES_H
            status=i/8; /* on SPP-1200 there are eight processors per hypernode */
            status = cnx_sc_fork(CNX_INHERIT_SC,status);
#   else /* HAVE_SYS_CNX_TYPES_H */
            status = fork();
#   endif /* HAVE_SYS_CNX_TYPES_H */
            if (status < 0) {
                Error("pbegin: error forking process",status);
            }
            else if (status == 0) {

                /* Child process */
                me = SR_proc_id += i;               /* change process id */

#   ifdef SOLARIS
                /*printf("binding slave process %d to processor %d\n", getpid(), 31-i);
                  if (processor_bind(P_PID, P_MYID, 31-i, (void *) NULL))
                  printf("binding to %d failed\n", 31-i); */
#   endif /* SOLARIS */

                /* Tidy up files */
                if (SR_clus_id == 0) { /* if not 0 is shut already */
                    (void) fclose(stdin);
                }
                (void) close(SR_proc_info[SR_n_proc].sock);
                SR_proc_info[SR_n_proc].sock = -1;  /* eliminate connection */
                break;
            }
            else if (status > 0) {
                SR_pids[SR_numchild++] = status;
            }
        }

        masterid = SR_clus_info[SR_clus_id].masterid;

        for (i=masterid; i<(masterid+nslave); i++) {
            Integer slaveid = i - masterid;
            SR_proc_info[i].slaveid = slaveid;
            SR_proc_info[i].local = 1;
            SR_proc_info[i].sock = -1;
            SR_proc_info[i].shmem = SR_proc_info[masterid].shmem;
            SR_proc_info[i].shmem_size = SR_proc_info[masterid].shmem_size;
            SR_proc_info[i].shmem_id = SR_proc_info[masterid].shmem_id;
            SR_proc_info[i].header = (MessageHeader *)
                (SR_proc_info[i].shmem + slaveid * SHMEM_BUF_SIZE);
            /*      SR_proc_info[i].header->nodeto = -1; */
            SR_proc_info[i].buffer = ((char *) SR_proc_info[i].header) + 
                sizeof(MessageHeader) + (sizeof(MessageHeader) % 8);
            SR_proc_info[i].buflen = SHMEM_BUF_SIZE - sizeof(MessageHeader) -
                (sizeof(MessageHeader) % 8);
            SR_proc_info[i].semid = SR_proc_info[masterid].semid;
            SR_proc_info[i].sem_pend = 3*slaveid;
            SR_proc_info[i].sem_read = 3*slaveid + 1;
            SR_proc_info[i].sem_written = 3*slaveid + 2;
            SR_proc_info[i].buffer_full = (Integer*)(flags + slaveid);
            /*      *SR_proc_info[i].buffer_full = FALSE;*/
        }

        /* Post read semaphore to make sends partially asynchronous */
        SemPost(SR_proc_info[me].semid, SR_proc_info[me].sem_read);
    }

#else /* HAVE_SYS_IPC_H */
    if (SR_clus_info[SR_clus_id].nslave != 1)
        Error("pbegin: no shared memory on this host ... nslave=1 only",
                SR_clus_info[SR_clus_id].nslave);
#endif /* HAVE_SYS_IPC_H */

    /* Now have to connect everyone together */

    ConnectAll();

    /* If we are only using sockets we can block in select when waiting for a message */
    SR_nsock = 0;
    for (i=0; i<(SR_n_proc+1); i++) {
        if (SR_proc_info[i].sock >= 0) {
            SR_socks[SR_nsock] = SR_proc_info[i].sock;
            SR_socks_proc[SR_nsock] = i;
            SR_nsock++;
        }
    }
    /* Synchronize timers before returning to application 
       or logging any events */

    (void) TCGTIME_();
    type = TYPE_CLOCK_SYNCH;
    SYNCH_(&type);
    MtimeReset();

    /* If logging events make the file events.<nodeid> */

#ifdef EVENTLOG
    if (eventfile=malloc((unsigned) 32)) {
        (void) sprintf(eventfile, "events.%03ld", NODEID_());
        evlog(EVKEY_ENABLE, EVKEY_FILENAME, eventfile,
                EVKEY_BEGIN, EVENT_PROCESS,
                EVKEY_STR_INT, "Startup used (cs)", (int) (MTIME_()-start),
                EVKEY_STR_INT, "No. of processes", (int) NNODES_(),
                EVKEY_DISABLE,
                EVKEY_LAST_ARG);
        (void) free(eventfile);
        SYNCH_(&type);
    }
#endif

    if (DEBUG_) {
        printf("pbegin: %2ld: Returning to application\n",NODEID_());
        fflush(stdout);
    }
#ifdef USE_VAMPIR
    vampir_end(TCGMSG_PBEGINF,__FILE__,__LINE__);
#endif
}


/**
 * Call this to tidy up after parallel section.
 * The cluster master is responsible for tidying up any shared
 * memory/semaphore resources. Everyone else can just quit.
 *
 * Woops ... everyone should return so that FORTRAN can tidy up
 * after itself.
 */
void PEND_()
{
    Integer me = NODEID_();
    Integer masterid = SR_clus_info[SR_clus_id].masterid;
    Integer nslave = SR_clus_info[SR_clus_id].nslave;
    Integer zero = 0;
    Integer status;
#ifdef EVENTLOG
    Integer start=MTIME_();
#endif
#ifdef USE_VAMPIR
    vampir_begin(TCGMSG_PEND,__FILE__,__LINE__);
#endif

    SR_initialized = 0;
    if (!SR_parallel) {
        return;
    }

    (void) signal(SIGCHLD, SIG_DFL); /* Death of children now OK */
    (void) NXTVAL_(&zero);  /* Send termination flag to nxtval server */

    if (me != masterid) {
        status = 0;
    }
    else {
        status = WaitAll(nslave-1);       /* Wait for demise of children */
        if (nslave > 1) {
            (void) SemSetDestroyAll();      /* Ex the semaphores and shmem */
            (void) DeleteSharedRegion(SR_proc_info[me].shmem_id);
        }
    }

    ShutdownAll();    /* Close sockets for machines with static kernel */


    /* If logging events log end of process and dump trace */
#ifdef EVENTLOG
    evlog(EVKEY_ENABLE,
            EVKEY_END, EVENT_PROCESS,
            EVKEY_STR_INT, "Time (cs) waiting to finish", (int) (MTIME_()-start),
            EVKEY_DUMP,
            EVKEY_LAST_ARG);
#endif
#ifdef USE_VAMPIR
    vampir_end(TCGMSG_PEND,__FILE__,__LINE__);
    vampir_finalize(__FILE__,__LINE__);
#endif
    /* Return to calling program unless we had an error */

    if (status) {
        exit((int) status);
    }
}


void tcgi_alt_pbegin(int *argc, char **argv[])
{
    tcgi_pbegin(*argc, *argv);
}
