#include <stdio.h>

#ifdef SEQUENT
#include <strings.h>
#else
#include <string.h>
#endif
#if defined(SUN) || defined(ALLIANT) || defined(ENCORE) || \
                    defined(SEQUENT) || defined(AIX)    || defined(NEXT)
#include <sys/wait.h>
#endif

#ifdef PARAGON
#include <sys/ioctl.h>
#endif
#ifdef AIX
#include <sys/select.h>
#include <sys/ioctl.h>
#endif
#ifdef CONVEX
#include <errno.h>
#else
#include <sys/errno.h>
#endif
#include <sys/time.h>
#include <sys/types.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>

#ifdef DELTA
#include <CMC/ntoh.h>
#define bcopy(a,b,n) (void) memcpy(b,a,n)
extern char *strdup();
#endif

extern int errno;

#ifdef CRAY
#include <memory.h>
#endif

#define PACKET_SIZE    32768
#define TIMEOUT_ACCEPT 300

static int nproc;		/* Zero initialization assumed */
static int nsock;
static int proc_list[1024];
static int sock_list[1024];

void iw_Error(msg, code)
     char *msg;
     long code;
{
  (void) fflush(stdout);
  (void) fprintf(stderr, "%s %d(0x%x)\n", msg, code, code);
  (void) perror("system message");
  (void) fflush(stderr);

  /* Additional tidy up here */

  exit(1);
}

long iw_PollSocket(sock)
     int sock;
/*
  Poll the socket for available input.

  Return 1 if data is available, 0 otherwise.
*/
{
  fd_set ready;
  struct timeval timelimit;
  int nready;

  if (sock < 0)
    return 0;

again:
  FD_ZERO(&ready);
  FD_SET(sock, &ready);
  timelimit.tv_sec = 0;
  timelimit.tv_usec = 0;

  nready = (long) select(sock+1, &ready, (fd_set *) NULL, (fd_set *) NULL,
			 &timelimit);
  if (nready < 0) {
    if (errno == EINTR)
      goto again;
    else
      iw_Error("iw_PollSocket: error from select",  (long) sock);
  }

  return nready;
}

long iw_WriteableSocket(sock)
     int sock;
/*
  Poll the socket for possible output.

  Return 1 if data is writeable, 0 otherwise.
*/
{
  fd_set ready;
  struct timeval timelimit;
  int nready;

  if (sock < 0)
    return 0;

again:
  FD_ZERO(&ready);
  FD_SET(sock, &ready);
  timelimit.tv_sec = 0;
  timelimit.tv_usec = 0;

  nready = (long) select(sock+1, (fd_set *) NULL, &ready, (fd_set *) NULL,
			 &timelimit);
  if (nready < 0) {
    if (errno == EINTR)
      goto again;
    else
      iw_Error("iw_WriteableSocket: error from select",  (long) sock);
  }

  return nready;
}


void iw_SelectWrapper(nread, rsocks, rstat, nwrite, wsocks, wstat, usecs)
     int nread, *rsocks, *rstat, nwrite, *wsocks, *wstat, usecs;
/*
  Wait in select for the specified no. of micro-seconds for possible
  reads or writes on the given lists of sockets.  Return in the r/wstat
  arrays the status of each socket (0=read/write not possible,
  1 = read/write possible).

  A negative usecs means wait for ever.
*/
{
  fd_set read_set, write_set;
  struct timeval timelimit, *tp;
  int i, width=0, nready;

  if (usecs < 0)
    tp = NULL;
  else {
    timelimit.tv_sec  = usecs>>20;
    timelimit.tv_usec = usecs - timelimit.tv_sec;
    tp = &timelimit;
  }

again:
  FD_ZERO(&read_set);
  FD_ZERO(&write_set);

  for (i=0; i<nread; i++) {
    int sock = rsocks[i];
    FD_SET(sock, &read_set);
    if (width < (sock+1)) width = sock+1;
  }
  for (i=0; i<nwrite; i++) {
    int sock = wsocks[i];
    FD_SET(sock, &write_set);
    if (width < (sock+1)) width = sock+1;
  }

  nready = (long) select(width, &read_set, &write_set, (fd_set *) NULL, tp);

  if (nready < 0) {
    if (errno == EINTR)
      goto again;
    else
      iw_Error("iw_SelectWrapper: error from select",  (long) 1);
  }

  for (i=0; i<nread; i++) {
    int sock = rsocks[i];
    rstat[i] = FD_ISSET(sock, &read_set);
  }
  for (i=0; i<nwrite; i++) {
    int sock = wsocks[i];
    wstat[i] = FD_ISSET(sock, &write_set);
  }
}  
  
static void WaitForRead(sock)
     int sock;
/*
  Wait until data is available on the socket
*/
{
#ifdef PARAGONMAYBE
  /* Paragon seems to hang in select? */

  while (!iw_PollSocket(sock))
    ;
#else
  fd_set ready;
  int nready;

again:
  FD_ZERO(&ready); FD_SET(sock, &ready);

  nready = (long) select(sock+1, &ready, (fd_set *) NULL, (fd_set *) NULL,
			 NULL);
  if (nready < 0) {
    if (errno == EINTR)
      goto again;
    else
      iw_Error("WaitForRead: error from select",  (long) sock);
  }
#endif
}  

static void WaitForWrite(sock)
     int sock;
/*
  Wait until a write might not block
*/
{
  fd_set ready;
  int nready;

again:
  FD_ZERO(&ready); FD_SET(sock, &ready);

  nready = (long) select(sock+1, (fd_set *) NULL, &ready, (fd_set *) NULL,
			 NULL);
  if (nready < 0) {
    if (errno == EINTR)
      goto again;
    else
      iw_Error("WaitForWrite: error from select",  (long) sock);
  }
}  

static void SetNonBlocking(sock)
     int sock;
/*
  Mark the socket for non-blocking I/O
*/
{
#if defined(SOLARIS) || defined(PARAGON___)
  if (fcntl(sock, F_SETFL, O_NDELAY) < 0)
#elif defined(AIX) || defined(PARAGON)
  int on = 1;
  if (ioctl(sock, FIONBIO, &on))
#else
  if (fcntl(sock, F_SETFL, FNDELAY) < 0)
#endif
    iw_Error("SetNonBlocking: fcntl FNDELAY failed on socket", (long) sock);
}

static void TcpNoDelay(sock)
  int sock;
/*
  Turn off waiting for more input to improve buffering 
  by TCP layer ... improves performance for small messages by
  a factor of 30 or more. Slightly degrades performance for
  large messages.
*/
{
  int status, level, value=1;
#ifdef AIX
  struct protoent *proto = getprotobyname("tcp");
#else
  struct protoent *proto = getprotobyname("TCP");
#endif

#if defined(APOLLO) || defined(LINUX)
  if (value)
    return;
#endif

  if (proto == (struct protoent *) NULL)
    iw_Error("TcpNoDelay: getprotobyname on TCP failed!", (long) -1);

  level = proto->p_proto;

  status = setsockopt(sock, level, TCP_NODELAY, (char*)&value, sizeof(int));

  if (status != 0)
    iw_Error("TcpNoDelay: setsockopt failed", (long) status);
}

int iw_NonBlockingReadFromSocket(sock, buf, lenbuf)
     int sock;
     char *buf;
     long lenbuf;
/*
   Read from the socket until we we what we asked for
   or until we block.  Return the amount read or -1
   if there was an error.
*/
{
  int nread = 0;

#if defined(PARAGON__) || defined(PARAGONMAYBE)
  if (!iw_PollSocket(sock)) return;
#endif

/*  printf("NBRFS: reading lenbuf=%d\n", lenbuf);  fflush(stdout);*/
  nread = read(sock, buf, (int) lenbuf, 0);
/*  printf("NBRFS: out nread=%d\n", nread);  fflush(stdout);*/
 
  if (nread == -1 && (errno == EINTR || errno == EWOULDBLOCK || errno == EAGAIN))
      nread = 0;

  return nread;
}

int iw_NonBlockingWriteToSocket(sock, buf, lenbuf)
     int sock;
     char *buf;
     long lenbuf;
/*
  Write as much data as possible to the socket without blocking.
  Return the amount written or -1 if there was an error.
*/
{
  int nsent = 0;
#ifdef PARAGONMAYBE
  if (!iw_WriteableSocket(sock)) return;
#endif

  lenbuf = (lenbuf < 32768) ? lenbuf : 32768;
  nsent = send(sock, buf, (int) lenbuf, 0);
  if (nsent == -1 && (errno == EINTR || errno == EWOULDBLOCK))
    nsent = 0;
  return nsent;
}

int iw_BlockingReadFromSocket(sock, buf, lenbuf)
     int sock;
     char *buf;
     long lenbuf;
/*
   Read from the socket until we get all we asked for.
   Return the amount read or -1 on an error;
*/
{
  int status = lenbuf;
  
  while (lenbuf > 0) {
    int nread = iw_NonBlockingReadFromSocket(sock, buf, lenbuf);
    
    if (nread < 0)
      return -1;
    else if (nread != lenbuf)
      WaitForRead(sock);
    
    buf += nread;
    lenbuf -= nread;
  }
  
  return status;
}

int iw_BlockingWriteToSocket(sock, buf, lenbuf)
     int sock;
     char *buf;
     long lenbuf;
/*
   Write to the socket until we get send everything.
   Return the amount written or -1 on an error;
*/
{
  int status = lenbuf;
  
  while (lenbuf > 0) {
    int nsent = iw_NonBlockingWriteToSocket(sock, buf, lenbuf);
    
    if (nsent < 0)
      return -1;
    else if (nsent != lenbuf)
      WaitForWrite(sock);
    
    buf += nsent;
    lenbuf -= nsent;
  }
  
  return status;
}

void iw_CreateSocketAndBind(sock, port)
     int *sock;
     int *port;
/*
  Create a socket, bind it to a wildcard internet name and return
  the info so that its port number may be advertised
*/
{
  int length;
  struct sockaddr_in server;
  int size = PACKET_SIZE;
  int on = 1;

  length = sizeof (struct sockaddr_in);

  /* Create socket */

  if ( (*sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    iw_Error("iw_CreateSocketAndBind: socket creation failed", (long) *sock);

  if(setsockopt(*sock, SOL_SOCKET, SO_REUSEADDR, 
		(char *) &on, sizeof on) == -1)
	iw_Error("iw_CreateSocketAndBind: error from setsockopt", (long) -1);

  /* Increase size of socket buffers to improve long message
     performance and increase size of message that goes asynchronously */

#if !(defined(DELTA) || defined(IPSC))
  if(setsockopt(*sock, SOL_SOCKET, SO_RCVBUF, (char *) &size, sizeof size))
    iw_Error("iw_CreateSocketAndBind: error setting SO_RCVBUF", (long) size);
  if(setsockopt(*sock, SOL_SOCKET, SO_SNDBUF, (char *) &size, sizeof size))
    iw_Error("iw_CreateSocketAndBind: error setting SO_SNDBUF", (long) size);
#endif

  /* Name socket with wildcards */

  server.sin_family = AF_INET;
  server.sin_addr.s_addr = INADDR_ANY;
  server.sin_port = 0;
  if (bind(*sock, (struct sockaddr *) &server, length) < 0)
    iw_Error("iw_CreateSocketAndBind: bind failed", (long) 0);

  /* Find out port number etc. */

  if (getsockname(*sock, (struct sockaddr *) &server, &length) < 0)
    iw_Error("iw_CreateSocketAndBind: getsockname failed", (long) 0);

  *port = ntohs(server.sin_port);

#ifndef ARDENT
  TcpNoDelay(*sock);
#endif

  SetNonBlocking(*sock);
}

int iw_ListenAndAccept(sock)
  int sock;
/*
  Listen and accept a connection on the specified socket
  which was created with CreateSocketAndBind
*/
{
  fd_set ready;
  struct timeval timelimit;
  int msgsock, nready;
  int size = PACKET_SIZE;
  
againlist:
  if (listen(sock, 1) < 0) {
    if (errno == EINTR)
      goto againlist;
    else
      iw_Error("iw_ListenAndAccept: listen failed", (long) 0);
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
  if ( (nready <= 0) && (errno == EINTR) )
    goto againsel;
  else if (nready < 0)
    iw_Error("iw_ListenAndAccept: error from select",  (long) nready);
  else if (nready == 0)
    iw_Error("iw_ListenAndAccept: timeout waiting for connection", 
          (long) nready);

  if (!FD_ISSET(sock, &ready))
    iw_Error("iw_ListenAndAccept: out of select but not ready!", (long) nready);

againacc:
  msgsock = accept(sock, (struct sockaddr *) NULL, (int *) NULL);
  if (msgsock == -1) {
    if (errno == EINTR)
      goto againacc;
    else
      iw_Error("iw_ListenAndAccept: accept failed", (long) msgsock);
  }

  /* Increase size of socket buffers to improve long message
     performance and increase size of message that goes asynchronously */

#if !(defined(DELTA) && defined(IPSC))
  if(setsockopt(msgsock, SOL_SOCKET, SO_RCVBUF, (char *) &size, sizeof size))
    iw_Error("iw_ListenAndAccept: error setting SO_RCVBUF", (long) size);
  if(setsockopt(msgsock, SOL_SOCKET, SO_SNDBUF, (char *) &size, sizeof size))
    iw_Error("iw_ListenAndAccept: error setting SO_SNDBUF", (long) size);
#endif

#if !(defined(ARDENT) || defined(APOLLO) || defined(DELTA) || defined(IPSC))
  TcpNoDelay(msgsock);
#endif
  SetNonBlocking(msgsock);
/*  return msgsock; */

  (void) close(sock); /* will not be needing this again */
  return msgsock;
}

int iw_CreateSocketAndConnect(hostname, cport)
     char *hostname;
     char *cport;
/*
  Return the file descriptor of the socket which connects me to the
  remote process on hostname at port in string cport

  hostname = hostname of the remote process
  cport    = asci string containing port number of remote socket
*/
{
  int sock, status;
  struct sockaddr_in server;
  struct hostent *hp;
  int on = 1;
  int size = PACKET_SIZE;
#ifndef SGI
  struct hostent *gethostbyname();
#endif

  /* Create socket */

  if ( (sock = socket(AF_INET, SOCK_STREAM, 0)) < 0 ) {
    (void) fprintf(stderr,"trying to connect to host=%s, port=%s\n",
                   hostname, cport);
    iw_Error("iw_CreateSocketAndConnect: socket failed",  (long) sock);
  }

  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, 
		 (char *) &on, sizeof on) == -1)
      iw_Error("iw_CreateSocketAndConnect: error setting REUSEADDR", (long) -1);

  /* Increase size of socket buffers to improve long message
     performance and increase size of message that goes asynchronously */

#if !(defined(DELTA) && defined(IPSC))
  if(setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char *) &size, sizeof size))
    iw_Error("iw_CreateSocketAndConnect: error setting SO_RCVBUF", (long) size);
  if(setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char *) &size, sizeof size))
    iw_Error("iw_CreateSocketAndConnect: error setting SO_SNDBUF", (long) size);
#endif

  /* Connect socket */

  server.sin_family = AF_INET;
  hp = gethostbyname(hostname);
  if (hp == 0) {
    (void) fprintf(stderr,"trying to connect to host=%s, port=%s\n",
                   hostname, cport);
    iw_Error("CreateSocketAndConnect: gethostbyname failed", (long) 0);
  }

  bcopy((char *) hp->h_addr, (char *) &server.sin_addr, hp->h_length);
  server.sin_port = htons((ushort) atoi(cport));

againcon:
  if ((status = 
     connect(sock, (struct sockaddr *) &server, sizeof server)) < 0) {
    if (errno == EINTR)
      goto againcon;
    else {
      (void) fprintf(stderr,"trying to connect to host=%s, port=%s\n",
                   hostname, cport);
      iw_Error("iw_CreateSocketAndConnect: connect failed", (long) status);
    }
  }
  
#if !(defined(ARDENT) || defined(APOLLO) || defined(DELTA) || defined(IPSC))
  TcpNoDelay(sock);
#endif
  SetNonBlocking(sock);

  return sock;
}

static char *Canonical(name)
    char *name;
/*
  Use gethostbyname and return the canonicalized name.
*/
{
  struct hostent *host;

  if ( (host = gethostbyname(name)) != (struct hostent *) NULL )
    return strdup(host->h_name);
  else
    return (char *) NULL;
}

#if defined(DELTA) || defined(IPSC)
/*ARGSUSED*/
int iw_RemoteCreate(remote_hostname, remote_username, remote_executable, argv)
     char *remote_hostname;
     char *remote_username;
     char *remote_executable;
     char **argv;
{
  iw_Error("RemoteCreate not implemented", (long) 1);
}
#else
int RemoteMachineIsDelta(name)
	char *name;
{
  return (!strncmp(name,"delta",5) || !strncmp(name,"DELTA",5));
}

int iw_RemoteCreate(remote_hostname, remote_username, remote_executable, argv)
     char *remote_hostname;
     char *remote_username;
     char *remote_executable;
     char **argv;
/*
  Using rsh create a process on remote_hostname running the
  executable in the remote file remote_executable. Through
  arguments pass it my hostname and the port number of a socket
  to conenct on as "-master <hostname> <port>" appended to
  the provided null terminated list argv[].

  Listen for a connection to be established. The return value of
  RemoteCreate is the filedescriptor of the socket connecting the 
  processes together. 

  Rsh should ensure that the standard output of the remote
  process is connected to the local standard output and that
  local interrupts are propagated to the remote process.

  If the remote hostname matches the local machine name then
  the new executable is simply forked off and rsh is not used.
 */
{
  char local_hostname[256], c_port[8];
  int sock, port, i, pid;

  /* Create and bind socket to wild card internet name */

  iw_CreateSocketAndBind(&sock, &port);

  /* create remote process using rsh passing master hostname and
     port as arguments */

  if (gethostname(local_hostname, 256) != 0)
    iw_Error("iw_RemoteCreate: gethostname failed", (long) 0);

  (void) sprintf(c_port, "%d", port);

/*
  (void) printf(" Creating: host=%s, user=%s,\n\
           file=%s, port=%s\n",
                remote_hostname, remote_username, remote_executable, 
                c_port);
*/

  pid = fork();
  if (pid == 0) {
    char  *argv2[256];
    int argc = 0;
    /* In child process */

    for (i=3; i<64; i++)	/* Close uneeded files */
      (void) close(i);

    /* Overlay the desired executable */

    if (strcmp(remote_hostname, local_hostname) != 0) {
      argv2[argc++] = "rsh";
      argv2[argc++] = remote_hostname;
      argv2[argc++] = "-l";
      argv2[argc++] = remote_username;
      argv2[argc++] = "-n";
      argv2[argc++] = remote_executable;

      /* Copy in user argument list */

      if (argv)
	for (; *argv; argv++)
	  argv2[argc++] = *argv;

      if (RemoteMachineIsDelta(remote_hostname)) {
	/* This assumes that the last argument given to mexec
	   is the name of the executable with any arguments without
	   additional quoting */

        char tmp[256];
	(void) sprintf(tmp, "'%s -master %s %s'", *(argv-1),
		       local_hostname, c_port);
	argv2[argc-1] = strdup(tmp);
      }
      else {
	/* Generic rsh */
        argv2[argc++] = "-master";
        argv2[argc++] = local_hostname;
        argv2[argc++] = c_port;
        argv2[argc++] = (char *) NULL;
      }

      argv2[argc++] = (char *) NULL;

      for (i=0; i<(argc-1); i++)
	printf("%s ", argv2[i]);
      printf("\n");

#ifdef SGI
      (void) execv("/usr/bsd/rsh",argv2);
#endif
#ifdef HPUX
      (void) execv("/usr/bin/remsh",argv2);
#endif
#if !defined(SGI) && !defined(HPUX)
      (void) execv("/usr/ucb/rsh",argv2);
#endif
    }
    else {
      argv2[argc++] = remote_executable;

      /* Copy in user argument list */

      if (argv)
	for (; *argv; argv++)
	  argv2[argc++] = *argv;

      argv2[argc++] = "-master";
      argv2[argc++] = Canonical(local_hostname);
      argv2[argc++] = c_port;
      argv2[argc++] = (char *) NULL;

      
      (void) printf("(%s): ",local_hostname);
      for (i=0; i<(argc-1); i++)
	(void) printf("%s ", argv2[i]);
      (void) printf("\n");
      
      (void) execv(remote_executable, argv2);
    }
    
    iw_Error("iw_RemoteCreate: in child after execv", (long) -1);
  }
  else if (pid > 0)
    proc_list[nproc++] = pid;
  else
    iw_Error("iw_RemoteCreate: failed forking process", (long) pid);

  /* accept one connection */

  sock = iw_ListenAndAccept(sock);

  sock_list[nsock++] = sock;

  return sock;
}
#endif

double iw_Dclock()
/*
  Return elapsed time in seconds from some arbitary origin.
*/
{
#if defined(DELTA) || defined(IPSC) || defined(PARAGON)
  extern double dclock();
  return dclock();
#else
  struct timeval tp;
  struct timezone tzp;
  
  (void) gettimeofday(&tp, &tzp);

  return (double) (0x7fffffff & tp.tv_sec) + 0.000001 * (double) tp.tv_usec;
#endif
}
