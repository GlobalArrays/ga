#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
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
#if HAVE_UNISTD_H
#   include <unistd.h> 
#endif
#if HAVE_SYS_STAT_H
#   include <sys/stat.h>
#endif
#if HAVE_SYS_SOCKET_H
#   include <sys/socket.h>
#endif
#if HAVE_NETDB_H
#   include <netdb.h>
#endif
#if HAVE_SYS_WAIT_H
#   include <sys/wait.h>
#endif
#if HAVE_STDLIB_H
#   include <stdlib.h>
#endif

extern void free(void *ptr);
extern char *getenv(const char *name);

#include "cluster.h"
#include "signals.h"
#include "sndrcv.h"
#include "sndrcvP.h"
#include "sockets.h"

extern void NextValueServer();
extern void Error();
extern int WaitAll(Integer nchild);


/**
 * Find the name of the procgrp file from
 *
 * 1) the first argument on the command line with .p appended
 * 2) as 1) but also prepending $HOME/pdir/
 * 2) the translation of the environmental variable PROCGRP
 * 3) the file PROCGRP in the current directory
 */
static char *ProcgrpFile(int argc, char **argv)
{
    char *tmp, *home;
    int len;
    struct stat buf;

    if (argc > 1) {
        len = strlen(argv[1]);
        tmp = malloc((unsigned) (len+3) );
        (void) strcpy(tmp, argv[1]);
        (void) strcpy(tmp+len, ".p");

        if (stat(tmp, &buf) == 0) {   /* try ./arg1.p */
            return tmp;
        }
        else {
            (void) free(tmp);
        }

        if ( (home = getenv("HOME")) != (char *) NULL ) {
            tmp = malloc((unsigned) (strlen(home) + len + 9));
            (void) strcpy(tmp, home);
            (void) strcpy(tmp+strlen(home),"/pdir/");
            (void) strcpy(tmp+strlen(home)+6,argv[1]);
            (void) strcpy(tmp+strlen(home)+6+len,".p");

            (void) printf("tmp = %s\n",tmp);

            if (stat(tmp, &buf) == 0) {   /* try $HOME/pdir/arg1.p */
                return tmp;
            }
            else {
                (void) free(tmp);
            }
        }
    }

    if ( (tmp = getenv("PROCGRP")) != (char *) NULL ) {
        if (stat(tmp, &buf) == 0) {
            return tmp;
        }
    }

    return strdup("PROCGRP");
}

  
/**
 * Read past first newline character.
 */
static void SkipPastEOL(FILE *fp)
{
    int test;

    while ( (char) (test = getc(fp)) != '\n') {
        if (test == EOF) {
            break;
        }
    }
}


/**
 * Read the entire contents of the PROCGRP into a NULL terminated
 * character string. Be lazy and read the file twice, first to
 * count the number of characters (ftell cannot be beleived?).
 */
static char *GetProcgrp(char *filename, Integer *len_procgrp)
{
    FILE *file;
    char *tmp, *procgrp;
    int status;

    if ( (file = fopen(filename,"r")) == (FILE *) NULL ) {
        (void) fprintf(stderr,"Master: PROCGRP = %s\n",filename);
        Error("Master: failed to open PROCGRP", (Integer) 0);
    }

    *len_procgrp = 0;
    while ( (status = getc(file)) != EOF) {
        if (status == '#') {
            SkipPastEOL(file);
        }
        else {
            (*len_procgrp)++;
        }
    }

    (*len_procgrp)++;

    if ( (tmp = procgrp = malloc((unsigned) *len_procgrp)) == (char *) NULL ) {
        Error("GetProcgrp: failed in malloc",  (Integer) *len_procgrp);
    }

    (void) fseek(file, 0L, (int) 0);   /* Seek to beginning of file */

    while ( (status = getc(file)) != EOF) {
        if (status == '#') {
            SkipPastEOL(file);
        }
        else {
            *tmp++ = (char) status;
        }
    }

    *tmp = '\0';

    if ( (int) (tmp - procgrp + 1) != *len_procgrp ) {
        Error("GetProcgrp: screwup dimensioning procgrp",  (Integer) *len_procgrp);
    }

    (void) fclose(file);

    return procgrp;
}


/**
 * Use gethostbyname and return the canonicalized name.
 */
char *Canonical(char *name)
{
    struct hostent *host;

    if ( (host = gethostbyname(name)) != (struct hostent *) NULL ) {
        return strdup(host->h_name);
    }
    else {
        return (char *) NULL;
    }
}


/**
 * Using rsh create a process on remote_hostname running the
 * executable in the remote file remote_executable. Through
 * arguments pass it my hostname and the port number of a socket
 * to conenct on. Also propagate the arguments which this program
 * was invoked with.
 *
 * Listen for a connection to be established. The return value of
 * RemoteCreate is the filedescriptor of the socket connecting the 
 * processes together. 
 *
 * Rsh should ensure that the standard output of the remote
 * process is connected to the local standard output and that
 * local interrupts are propagated to the remote process.
 */
static Integer RemoteCreate(
        char *remote_hostname,
        char *remote_username, 
        char *remote_executable,
        int argc,
        char **argv,
        Integer n_clus,
        Integer n_proc,
        Integer clus_id,
        Integer proc_id)
{
    char  local_hostname[256], c_port[8];
    char  c_n_clus[8], c_n_proc[8], c_clus_id[8], c_proc_id[8];
    char  *argv2[256];
    int sock, port, i, pid;
    char *tmp;

    /* Create and bind socket to wild card internet name */

    CreateSocketAndBind(&sock, &port);

    /* create remote process using rsh passing master hostname and
       port as arguments */

    if (gethostname(local_hostname, 256) || strlen(local_hostname) == 0) {
        Error("RemoteCreate: gethostname failed", (Integer) 0);
    }

    (void) sprintf(c_port, "%d", port);
    (void) sprintf(c_n_clus, "%ld", n_clus);
    (void) sprintf(c_n_proc, "%ld", n_proc);
    (void) sprintf(c_clus_id, "%ld", clus_id);
    (void) sprintf(c_proc_id, "%ld", proc_id);

    (void) printf(" Creating: host=%s, user=%s,\n  file=%s, port=%s\n",
                  remote_hostname, remote_username, remote_executable, c_port);

    pid = fork();
    if (pid == 0) {
        /* In child process */

        sleep(1); /* So that parallel can make the sockets */

        if (proc_id != 0) {      /* Close all uneeded files */
            (void) fclose(stdin);
        }
        for (i=3; i<64; i++) {
            (void) close(i);
        }

        /* Overlay the desired executable */

        if (strcmp(remote_hostname, local_hostname) != 0) {
            argv2[0      ] = "rsh";
            argv2[1      ] = remote_hostname;
            argv2[2      ] = "-l";
            argv2[3      ] = remote_username;
            argv2[4      ] = "-n";
            argv2[5      ] = remote_executable;
            argv2[6      ] = " ";
            for (i=2; i<argc; i++)
                argv2[i+5] = argv[i];
            argv2[argc+5 ] = "-master";
            argv2[argc+6 ] = local_hostname;
            argv2[argc+7 ] = c_port;
            argv2[argc+8 ] = c_n_clus;
            argv2[argc+9 ] = c_n_proc;
            argv2[argc+10] = c_clus_id;
            argv2[argc+11] = c_proc_id;
            argv2[argc+12] = (char *) NULL;

            if ( (tmp = getenv("TCGRSH")) != (char *) NULL ) {
                (void) execv(tmp,argv2);
            }
            else {
                (void) execv(TCGMSG_RSH,argv2);
            }
        }
        else {
            argv2[0     ] = remote_executable;
            for (i=1; i<(argc-1); i++) /* Don't copy the .p file name over */
                argv2[i] = argv[i+1];
            argv2[i+0] = "-master";
            argv2[i+1] = Canonical(local_hostname);
            argv2[i+2] = c_port;
            argv2[i+3] = c_n_clus;
            argv2[i+4] = c_n_proc;
            argv2[i+5] = c_clus_id;
            argv2[i+6] = c_proc_id;
            argv2[i+7] = (char *) NULL;

            (void) execv(remote_executable, argv2);
        }

        Error("RemoteCreate: in child after execv", (Integer) -1);
    }
    else if (pid > 0) {
        SR_pids[SR_numchild++] = pid;
    }
    else {
        Error("RemoteCreate: failed forking process", (Integer) pid);
    }

    /* accept one connection */

    return ListenAndAccept(sock);
}


/**
 * This is the master process of the cluster network.
 *
 * a) read the procgrp file. This is found by trying in turn:
 *    
 *         1) the first argument on the command line with .p appended
 *         2) the translation of the environmental variable PROCGRP
 *         3) the file PROCGRP in the current directory
 *
 * b) create the remote processes specified in this file, connecting
 *    to them via sockets and pass them the entire contents of the
 *    PROCGRP file in ascii
 *
 * c) Navigate messages to establish connections between the remote
 *    processes
 *
 * d) wait for all the children to finish and exit with the appropriate
 *    status
 */
int main(int argc, char **argv)
{
    char  hostname[256];     /* Me */
    char *filename;          /* The name of PROCGRP file */
    char *procgrp;           /* The contents of PROCGRP */
    Integer len_procgrp;        /* The length of PROCGRP */
    Integer i, j, node, type, lenbuf, status=0, sync=1;

    /* Initialize all the globals */

    InitGlobal();

    /* Set up handler for SIGINT  and SIGCHLD */

    TrapSigint();
    TrapSigchld();
    TrapSigterm();

    /* on Solaris parallel gets SIGSEGV interrupted while polling in NxtVal */
#ifdef SOLARIS
    TrapSigsegv();
#endif

    /* Generate a name for the PROCGRP file */

    filename = ProcgrpFile(argc, argv);
    if (DEBUG_) {
        (void) printf("PROCGRP = %s\n",filename);
    }

    /* Read in the entire contents of the PROCGRP file */

    procgrp = GetProcgrp(filename, &len_procgrp);

    /* Parse the procgrp info filling in the ClusterInfo structure and
       computing the number of clusters */

    if (gethostname(hostname, sizeof hostname) || strlen(hostname) == 0) {
        Error("parallel: gethostname failed?", (Integer) sizeof hostname);
    }

    InitClusInfo(procgrp, hostname);

    if (DEBUG_) {
        PrintClusInfo();
    }

    /* I am the master process so I have the highest ids */

    SR_proc_id = SR_n_proc;

    /* Now create the remote cluster master processes */

    for (i=0; i<SR_n_clus; i++) {
        node = SR_clus_info[i].masterid;
        SR_proc_info[node].sock = RemoteCreate(SR_clus_info[i].hostname,
                SR_clus_info[i].user,
                SR_clus_info[i].image,
                argc, argv,
                SR_n_clus,
                SR_n_proc,
                i,
                node);
        type = TYPE_BEGIN | MSGINT;
        lenbuf = sizeof(Integer);
        SND_(&type, (char *) &len_procgrp, &lenbuf, &node, &sync);
        type = TYPE_BEGIN | MSGCHR;
        SND_(&type, procgrp, &len_procgrp, &node, &sync);
    }

    /* Now have to route messages between the cluster masters as they connect */

    for (i=1; i< SR_n_clus; i++) {
        for (j=0; j < i; j++) {
            RemoteConnect(SR_clus_info[i].masterid, 
                    SR_clus_info[j].masterid,
                    NODEID_());
        }
    }

    /* Now for the next value service I need to connect to everyone else */

    for (i=0; i < SR_n_clus; i++) {
        for (j=1; j<SR_clus_info[i].nslave; j++) {
            RemoteConnect(SR_proc_id,
                    SR_clus_info[i].masterid+j,
                    SR_clus_info[i].masterid);
        }
    }

    /* Since we only using sockets we can block in select
     * when waiting for a message */
    SR_using_shmem = 0;
    SR_nsock = 0;
    for (i=0; i<(SR_n_proc+1); i++) {
        if (SR_proc_info[i].sock >= 0) {
            SR_socks[SR_nsock] = SR_proc_info[i].sock;
            SR_socks_proc[SR_nsock] = i;
            SR_nsock++;
        }
    }

    /* Provide the next value service ... exit gracefully when get termination
       message from everyone or detect error */

    NextValueServer();

    /* Now wait patiently for everything to finish, then close all
       sockets and return */

    status = WaitAll(SR_n_clus);

    if (SR_error) {
        status = 1;
    }

    ShutdownAll();

    return status;
}
