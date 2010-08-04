#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
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
#if HAVE_SYS_WAIT_H
#   include <sys/wait.h>
#endif
#if HAVE_SYS_SELECT_H
#   include <sys/select.h>
#endif
#if HAVE_ERRNO_H
#   include <errno.h>
#endif
#if HAVE_SYS_ERRNO_H
#   include <sys/errno.h>
#endif
#if HAVE_SYS_TIME_H
#   include <sys/time.h>
#endif
#if HAVE_SYS_TYPES_H
#   include <sys/types.h>
#endif
#if HAVE_SYS_SOCKET_H
#   include <sys/socket.h>
#endif
#if HAVE_NETINET_IN_H
#   include <netinet/in.h>
#endif
#if HAVE_NETINET_TCP_H
#   include <netinet/tcp.h>
#endif
#if HAVE_NETDB_H
#   include <netdb.h>
#endif
#if HAVE_MEMORY_H
#   include <memory.h>
#endif

extern int atoi(const char *nptr);

#include "sndrcv.h"
#include "sndrcvP.h"


/**
 * Wait until one or more sockets are ready or have an exception.
 *
 * Returns the number of ready sockets and sets corresponding 
 * numbers in list.  I.e., list[i]=k meaning sock[k] is ready.
 */
Integer WaitForSockets(int nsock, int *socks, int *list)
{
    fd_set ready;
    int i;
    Integer nready;
    int sockmax = 0;

again:
    FD_ZERO(&ready);
    for (i=0; i<nsock; i++) {
        FD_SET(socks[i], &ready);
        if (socks[i] > sockmax) sockmax = socks[i];
    }
    nready = (Integer) select(sockmax+1, &ready, (fd_set *) NULL,
            (fd_set *) NULL, (struct timeval *) NULL);
    if (nready < 0) {
        if (errno == EINTR) {
            /*fprintf(stderr,"wait in sockets got interrupted\n");*/
            goto again;
        }
        else {
            Error("WaitForSockets: error from select",  0L);
        }
    }
    else {
        int n = 0;
        for (i=0; i<nsock; i++) {
            if (FD_ISSET(socks[i],&ready)) list[n++] = i;
        }
    }    

    return nready;
}


/**
 * Poll the socket for available input.
 *
 * Return 1 if data is available, 0 otherwise.
 */
Integer PollSocket(int sock)
{
    fd_set ready;
    struct timeval timelimit;
    int nready;

    if (sock < 0) {
        return 0;
    }

again:
    FD_ZERO(&ready);
    FD_SET(sock, &ready);
    timelimit.tv_sec = 0;
    timelimit.tv_usec = 0;

    nready = (Integer) select(sock+1, &ready, (fd_set *) NULL,
            (fd_set *) NULL, &timelimit);
    if (nready < 0) {
        if (errno == EINTR) {
            goto again;
        } else {
            Error("PollSocket: error from select",  (Integer) sock);
        }
    }

    return nready;
}


/**
 * Turn off waiting for more input to improve buffering 
 * by TCP layer ... improves performance for small messages by
 * a factor of 30 or more. Slightly degrades performance for
 * large messages.
 */
void TcpNoDelay(int sock)
{
    int status, level, value=1;
    struct protoent *proto = getprotobyname("TCP");

    if (proto == (struct protoent *) NULL) {
        Error("TcpNoDelay: getprotobyname on TCP failed!", (Integer) -1);
    }

    level = proto->p_proto;

    status = setsockopt(sock, level, TCP_NODELAY, &value, sizeof(int));

    if (status != 0) {
        Error("TcpNoDelay: setsockopt failed", (Integer) status);
    }
}


/**
 * close all sockets discarding any pending data in either direction.
 */
void ShutdownAll()
{
    int i;

    for (i=0; i<NNODES_(); i++) {
        if (SR_proc_info[i].sock >= 0) {
            (void) shutdown(SR_proc_info[i].sock, 2);
            (void) close(SR_proc_info[i].sock);
        }
    }
}


/**
 * Read from the socket until we get all we want.
 */
int ReadFromSocket(int sock, char *buf, Integer lenbuf)
{
    int nread, status;

    status = lenbuf;
    while (lenbuf > 0) {
again:
        if ( (nread = recv(sock, buf, (int) lenbuf, 0)) < 0) {
            if (errno == EINTR) {
                goto again;
            } else {
                (void) fprintf(stderr,"sock=%d, pid=%ld, nread=%d, len=%ld\n",
                               sock, NODEID_(), nread, lenbuf);
                (void) fflush(stderr);
                status = -1;
                break;
            }
        }
        buf += nread;
        lenbuf -= nread;
    }

    return status;
}


/*
 * Write to the socket in packets of PACKET_SIZE bytes.
 */
int WriteToSocket(int sock, char *buf, Integer lenbuf)
{
    int status = lenbuf;
    int nsent, len;

    while (lenbuf > 0) {

        len = (lenbuf > PACKET_SIZE) ? PACKET_SIZE : lenbuf;
        nsent = send(sock, buf, (int) len, 0);

        if (nsent < 0) { /* This is bad news */
            (void) fprintf(stderr,"sock=%d, pid=%ld, nsent=%d, len=%ld\n",
                           sock, NODEID_(), nsent, lenbuf);
            (void) fflush(stderr);
            status = -1; break;
        }

        buf += nsent;
        lenbuf -= nsent;
    }

    return status;
}


/**
 * Create a socket, bind it to a wildcard internet name and return
 * the info so that its port number may be advertised
 */
void CreateSocketAndBind(int *sock, int *port)
{
    socklen_t length;
    struct sockaddr_in server;
    int size = SR_SOCK_BUF_SIZE;
    int on = 1;
/*
#if defined(LINUX) && defined(__powerpc__)
    int dupsock;
#endif
*/

    length = sizeof (struct sockaddr_in);

    /* Create socket */

    if ( (*sock = socket(AF_INET, SOCK_STREAM, 0)) < 0) {
        Error("CreateSocketAndBind: socket creation failed", (Integer) *sock);
    }

/*
#if defined(LINUX) && defined(__powerpc__)
    if(*sock==0)
        dupsock = dup(*sock);
    *sock = dupsock;
#endif
*/

    if(setsockopt(*sock, SOL_SOCKET, SO_REUSEADDR, 
                (char *) &on, sizeof on) == -1) {
        Error("CreateSocketAndBind: error from setsockopt", (Integer) -1);
    }

    /* Increase size of socket buffers to improve Integer message
       performance and increase size of message that goes asynchronously */

    if(setsockopt(*sock, SOL_SOCKET, SO_RCVBUF, (char *) &size, sizeof size)) {
        Error("CreateSocketAndBind: error setting SO_RCVBUF", (Integer) size);
    }
    if(setsockopt(*sock, SOL_SOCKET, SO_SNDBUF, (char *) &size, sizeof size)) {
        Error("CreateSocketAndBind: error setting SO_SNDBUF", (Integer) size);
    }

    /* Name socket with wildcards */

    server.sin_family = AF_INET;
    server.sin_addr.s_addr = INADDR_ANY;
    server.sin_port = 0;
    if (bind(*sock, (struct sockaddr *) &server, length) < 0) {
        Error("CreateSocketAndBind: bind failed", (Integer) 0);
    }

    /* Find out port number etc. */

    if (getsockname(*sock, (struct sockaddr *) &server, &length) < 0) {
        Error("CreateSocketAndBind: getsockname failed", (Integer) 0);
    }

    *port = ntohs(server.sin_port);

}


/**
 * Listen for a connection on the specified socket
 * which was created with CreateSocketAndBind
 */
void ListenOnSock(int sock)
{
againlist:
    if (listen(sock, 1) < 0) {
        if (errno == EINTR) {
            goto againlist;
        } else {
            Error("ListenAndAccept: listen failed", (Integer) 0);
        }
    }

    if (DEBUG_) {
        (void) printf("process %ld out of listen on socket %d\n",NODEID_(),sock);
        (void) fflush(stdout);
    }
}


/**
 * Accept a connection on the specified socket
 * which was created with CreateSocketAndBind and
 * listen has been called.
 */
int AcceptConnection(int sock)
{
    fd_set ready;
    struct timeval timelimit;
    int msgsock, nready;
    int size = SR_SOCK_BUF_SIZE;

    /* Use select to wait for someone to try and establish a connection
       so that we can add a short timeout to avoid hangs */

againsel:
    FD_ZERO(&ready);
    FD_SET(sock, &ready);

    timelimit.tv_sec = TIMEOUT_ACCEPT;
    timelimit.tv_usec = 0;
    nready = select(sock+1, &ready, (fd_set *) NULL, (fd_set *) NULL,
            &timelimit);
    if ( (nready <= 0) && (errno == EINTR) ) {
        goto againsel;
    } else if (nready < 0) {
        Error("ListenAndAccept: error from select",  (Integer) nready);
    } else if (nready == 0) {
        Error("ListenAndAccept: timeout waiting for connection", 
                (Integer) nready);
    }

    if (!FD_ISSET(sock, &ready)) {
        Error("ListenAndAccept: out of select but not ready!",
                (Integer) nready);
    }

againacc:
    msgsock = accept(sock, (struct sockaddr *) NULL, (socklen_t *) NULL);
    if (msgsock == -1) {
        if (errno == EINTR) {
            goto againacc;
        } else {
            Error("ListenAndAccept: accept failed", (Integer) msgsock);
        }
    }

    if (DEBUG_) {
        (void) printf("process %ld out of accept on socket %d\n",
                      NODEID_(),msgsock);
        (void) fflush(stdout);
    }

    /* Increase size of socket buffers to improve Integer message
       performance and increase size of message that goes asynchronously */

    if(setsockopt(msgsock, SOL_SOCKET, SO_RCVBUF, (char *)&size, sizeof size)) {
        Error("ListenAndAccept: error setting SO_RCVBUF", (Integer) size);
    }
    if(setsockopt(msgsock, SOL_SOCKET, SO_SNDBUF, (char *)&size, sizeof size)) {
        Error("ListenAndAccept: error setting SO_SNDBUF", (Integer) size);
    }

    (void) close(sock); /* will not be needing this again */
    return msgsock;
}


/**
 * Listen and accept a connection on the specified socket
 * which was created with CreateSocketAndBind
 */
int ListenAndAccept(int sock)
{
    fd_set ready;
    struct timeval timelimit;
    int msgsock, nready;
    int size = SR_SOCK_BUF_SIZE;

againlist:
    if (listen(sock, 1) < 0) {
        if (errno == EINTR) {
            goto againlist;
        } else {
            Error("ListenAndAccept: listen failed", (Integer) 0);
        }
    }

    if (DEBUG_) {
        (void) printf("process %ld out of listen on socket %d\n",
                      NODEID_(),sock);
        (void) fflush(stdout);
    }

    /* Use select to wait for someone to try and establish a connection
       so that we can add a short timeout to avoid hangs */

againsel:
    FD_ZERO(&ready);
    FD_SET(sock, &ready);

    timelimit.tv_sec = TIMEOUT_ACCEPT;
    timelimit.tv_usec = 0;
    nready = select(sock+1, &ready, (fd_set *) NULL, (fd_set *) NULL,
            &timelimit);
    if ( (nready <= 0) && (errno == EINTR) ) {
        goto againsel;
    } else if (nready < 0) {
        Error("ListenAndAccept: error from select",  (Integer) nready);
    } else if (nready == 0) {
        Error("ListenAndAccept: timeout waiting for connection", 
                (Integer) nready);
    }

    if (!FD_ISSET(sock, &ready)) {
        Error("ListenAndAccept: out of select but not ready!",
                (Integer) nready);
    }

againacc:
    msgsock = accept(sock, (struct sockaddr *) NULL, (socklen_t *) NULL);
    if (msgsock == -1) {
        if (errno == EINTR) {
            goto againacc;
        } else {
            Error("ListenAndAccept: accept failed", (Integer) msgsock);
        }
    }

    if (DEBUG_) {
        (void) printf("process %ld out of accept on socket %d\n",
                      NODEID_(),msgsock);
        (void) fflush(stdout);
    }

    /* Increase size of socket buffers to improve Integer message
       performance and increase size of message that goes asynchronously */

    if(setsockopt(msgsock, SOL_SOCKET, SO_RCVBUF, (char *)&size, sizeof size)) {
        Error("ListenAndAccept: error setting SO_RCVBUF", (Integer) size);
    }
    if(setsockopt(msgsock, SOL_SOCKET, SO_SNDBUF, (char *)&size, sizeof size)) {
        Error("ListenAndAccept: error setting SO_SNDBUF", (Integer) size);
    }

    (void) close(sock); /* will not be needing this again */
    return msgsock;
}


/**
 * Return the file descriptor of the socket which connects me to the
 * remote process on hostname at port in string cport
 *
 * hostname = hostname of the remote process
 * cport    = asci string containing port number of remote socket
 */
int CreateSocketAndConnect(char *hostname, char *cport)
{
    int sock, status;
    struct sockaddr_in server;
    struct hostent *hp;
    int on = 1;
    int size = SR_SOCK_BUF_SIZE;
    struct hostent *gethostbyname();
/*
#if defined(LINUX) && defined(__powerpc__)
    int dupsock;
#endif
*/

    /* Create socket */

    if ( (sock = socket(AF_INET, SOCK_STREAM, 0)) < 0 ) {
        (void) fprintf(stderr,"trying to connect to host=%s, port=%s\n",
                       hostname, cport);
        Error("CreateSocketAndConnect: socket failed",  (Integer) sock);
    }

/*
#if defined(LINUX) && defined(__powerpc__)
    if(sock==0)
        dupsock = dup(sock);
    sock = dupsock;
#endif
*/

    if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, 
                (char *) &on, sizeof on) == -1) {
        Error("CreateSocketAndConnect: error setting REUSEADDR", (Integer) -1);
    }

    /* Increase size of socket buffers to improve Integer message
       performance and increase size of message that goes asynchronously */

    if(setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char *) &size, sizeof size)) {
        Error("CreateSocketAndConnect: error setting SO_RCVBUF",
                (Integer) size);
    }
    if(setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char *) &size, sizeof size)) {
        Error("CreateSocketAndConnect: error setting SO_SNDBUF",
                (Integer) size);
    }

    /* Connect socket */

    server.sin_family = AF_INET;
    hp = gethostbyname(hostname);
    if (hp == 0) {
        (void) fprintf(stderr,"trying to connect to host=%s, port=%s\n",
                       hostname, cport);
        Error("CreateSocketAndConnect: gethostbyname failed", (Integer) 0);
    }

    bcopy((char *) hp->h_addr, (char *) &server.sin_addr, hp->h_length);
    server.sin_port = htons((ushort) atoi(cport));

againcon:
    if ((status = connect(sock, (struct sockaddr *) &server, sizeof server)) < 0) {
        if (errno == EINTR) {
            goto againcon;
        } else {
            (void) fprintf(stderr,"trying to connect to host=%s, port=%s\n",
                           hostname, cport);
            Error("CreateSocketAndConnect: connect failed", (Integer) status);
        }
    }

    return sock;
}
