/**************************************************************************
 This code was derived from the TCGMSG sockets.c by Robert Harrison
 *************************************************************************/

#include <stdio.h>
#include <string.h>
#include <sys/wait.h>

#ifdef AIX
#include <sys/select.h>
#endif

#include <errno.h>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <netdb.h>

#ifdef CRAY
#include <memory.h>
#endif

#if defined(AIX) || defined(LINUX)
  typedef size_t socklen_t;
#else
  typedef int socklen_t;
#endif

#include "sockets.h"

extern int armci_me, armci_nproc;
#define DEBUG_ 0


int armci_PollSocket(int sock)
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

  nready = select(sock+1, &ready, (fd_set *) NULL, (fd_set *) NULL, &timelimit);
  if (nready < 0) {
    if (errno == EINTR)
      goto again;
    else
      armci_die("armci_PollSocket: error from select",   sock);
  }

  return nready;
}


/*\ sleep in select until data appears on one of sockets
 *  return number of sockets ready and indicate which ones are in ready array 
\*/
int armci_WaitSock(int *socklist, int num, int *ready)
{

  int sock,maxsock;
  fd_set dset;
  struct timeval timelimit;
  int nready;

  if(num<0) armci_die("armci_WaitSock: num <0",num);

again:

  FD_ZERO(&dset);
  maxsock=0;
  for(sock=0; sock<num; sock++){
     if(socklist[sock] > maxsock)maxsock = socklist[sock];
     FD_SET(socklist[sock], &dset);
  }


  nready = select(maxsock+1, &dset, (fd_set*)NULL, (fd_set*)NULL, NULL); 

    if (nready < 0) {
    if (errno == EINTR){
      fprintf(stderr,"%d:interrupted in select\n",armci_me);
      goto again;
    } else
      armci_die("armci_WaitSocket: error from select",   sock);
  }

  
  for(sock=0; sock<num; sock++)
     if(FD_ISSET(socklist[sock],&dset)) ready[sock]=1;
     else ready[sock]=0;

  return nready;
}
    




void armci_TcpNoDelay( int sock)
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

#if  defined(LINUX)
  if (value) return;
#endif

  if (proto == (struct protoent *) NULL)
    armci_die("armci_TcpNoDelay: getprotobyname on TCP failed!",  -1);

  level = proto->p_proto;

  status = setsockopt(sock, level, TCP_NODELAY, &value, sizeof(int));

  if (status != 0)
    armci_die("armci_TcpNoDelay: setsockopt failed",  status);
}



void armci_ShutdownAll(int socklist[], int num)
/* 
   close all sockets discarding any pending data in either direction.
*/
{
   int i;

   for (i=0; i<num; i++)
      if (socklist[i] >= 0) {
         (void) shutdown(socklist[i], 2);
         (void) close(socklist[i]);
         socklist[i]=-1;
      }
}


int armci_ReadFromSocket(int sock, void* buffer, int lenbuf)
/*
   Read from the socket until we get all we want.
*/
{
   int nread, status;
   char *buf = (char*)buffer;

   status = lenbuf;
   while (lenbuf > 0) {
again:
     
     nread = recv(sock, buf, lenbuf, 0);
     /* on linux 0 can be returned if socket is closed  by sender */ 
     if(nread < 0 || ((nread ==  0) && errno ) ){
       if (errno == EINTR){
         fprintf(stderr,"%d:interrupted in recv\n",armci_me);
         goto again;
       }else {
         if(DEBUG_){
           (void) fprintf(stderr,"sock=%d, pid=%d, nread=%d, len=%d\n",
                               sock, armci_me, nread, lenbuf);
           if(errno)perror("armci_ReadFromSocket: recv failed");
         }
         status = -1;
         break;
       }
     }
     buf += nread;
     lenbuf -= nread;
   }
   
   return status;
}

int armci_WriteToSocket (int sock, void* buffer, int lenbuf)
/*
  Write to the socket in packets of PACKET_SIZE bytes
*/
{
  int status = lenbuf;
  int nsent, len;
  char *buf = (char*)buffer;
  
  if(DEBUG_){
    printf("%d armci_WriteToSocket sock=%d lenbuf=%d\n",armci_me,sock,lenbuf);
    fflush(stdout);
  }

  while (lenbuf > 0) {
    
    len = (lenbuf > PACKET_SIZE) ? PACKET_SIZE : lenbuf;
    nsent = send(sock, buf, len, 0);
    
    if (nsent < 0) { /* This is bad news */
      (void) fprintf(stderr,"sock=%d, pid=%d, nsent=%d, len=%d\n",
		     sock, armci_me, nsent, lenbuf);
      (void) fflush(stderr);
      status = -1; break;
    }

    buf += nsent;
    lenbuf -= nsent;
  }
  
  return status;
}


void armci_CreateSocketAndBind(int *sock, int *port)
/*
  Create a socket, bind it to a wildcard internet name and return
  the info so that its port number may be advertised
*/
{
  socklen_t  length;
  struct sockaddr_in server;
  int size = PACKET_SIZE;
  int on = 1;

  length = sizeof (struct sockaddr_in);

  /* Create socket */

  if ( (*sock = socket(AF_INET, SOCK_STREAM, 0)) < 0)
    armci_die("armci_CreateSocketAndBind: socket creation failed",  *sock);

  if(setsockopt(*sock, SOL_SOCKET, SO_REUSEADDR, (char *) &on, sizeof on) == -1)
	armci_die("armci_CreateSocketAndBind: error from setsockopt",  -1);

  /* Increase size of socket buffers to improve long message
     performance and increase size of message that goes asynchronously */

  if(setsockopt(*sock, SOL_SOCKET, SO_RCVBUF, (char *) &size, sizeof(size)))
    armci_die("armci_CreateSocketAndBind: error setting SO_RCVBUF", size);
  if(setsockopt(*sock, SOL_SOCKET, SO_SNDBUF, (char *) &size, sizeof(size)))
    armci_die("armci_CreateSocketAndBind: error setting SO_SNDBUF", size);

  armci_TcpNoDelay(*sock);

  /* Name socket with wildcards */

  server.sin_family = AF_INET;
  server.sin_addr.s_addr = INADDR_ANY;
  server.sin_port = 0;
  if (bind(*sock, (struct sockaddr *) &server, length) < 0)
    armci_die("armci_CreateSocketAndBind: bind failed", 0);

  /* Find out port number etc. */

  if (getsockname(*sock, (struct sockaddr *) &server, &length) < 0)
    armci_die("armci_CreateSocketAndBind: getsockname failed",  0);

  *port = ntohs(server.sin_port);

}


/*
 * Listen and accept a connection on the specified socket
 * which was created with CreateSocketAndBind
 */

void armci_ListenAndAcceptAll(int* socklist, int num)
{
  fd_set ready, fdzero;
  struct timeval timelimit;
  int maxsock, msgsock, nready, num_accept=0;
  int size = PACKET_SIZE, i;

  if(num<0)armci_die("armci_ListenAndAcceptAll invalid number of sockets",num);

  for(i=0; i< num; i++){ 
     againlist:
       if (listen(socklist[i], num) < 0) {
         if (errno == EINTR)
           goto againlist;
         else
           armci_die("armci_ListenAndAcceptAll: listen failed",  0);
       }
  }

  if (DEBUG_) {
    (void) printf("process %ld out of listen on %d sockets\n",armci_me,num);
    (void) fflush(stdout);
  }

  /* Use select to wait for someone to try and establish a connection
     so that we can add a short timeout to avoid hangs */

  FD_ZERO(&fdzero);

againsel:
  FD_ZERO(&ready);

  /* we negate socket number on the list to mark already connected */
  maxsock=0;
  for(i=0; i<num; i++){
     if(socklist[i] > maxsock)maxsock = socklist[i]; /* find largest value*/
     if(socklist[i]>0) FD_SET(socklist[i], &ready);
  }

  timelimit.tv_sec = TIMEOUT_ACCEPT;
  timelimit.tv_usec = 0;
  nready = select(maxsock+1, &ready, (fd_set *) NULL, (fd_set *) NULL,
                  &timelimit);

  /* error screening */
  if ( (nready <= 0) && (errno == EINTR) )
    goto againsel;
  else if (nready < 0)
    armci_die("armci_ListenAndAcceptAll: error from select",nready);
  else if (nready == 0)
    armci_die("armci_ListenAndAcceptAll:timeout waiting for connection",nready);

/*  if (bcmp(&ready,&fdzero,sizeof(fdzero)))*/
/*    armci_die("armci_ListenAndAcceptAll: out of select but not ready!",nready);*/

  /* accept connection from newly contacted clients */
  for(i=0; i< num; i++){ 
    int sock = socklist[i];
    if(sock<0) continue; /* accepted already */
    if(!FD_ISSET(sock, &ready)) continue; /* not contacted yet */

    againacc:

      msgsock = accept(sock, (struct sockaddr *) NULL, (socklen_t *) NULL);

      if (msgsock == -1) {
        if (errno == EINTR)
          goto againacc;
        else
          armci_die("armci_ListenAndAcceptAll: accept failed",  msgsock);
      }

    if(DEBUG_) {
       (void) printf("process %d out of accept socket=%d\n",armci_me,msgsock);
       (void) fflush(stdout);
    }

    /* Increase size of socket buffers to improve long message
       performance and increase size of message that goes asynchronously */

    if(setsockopt(msgsock, SOL_SOCKET, SO_RCVBUF, (char *) &size, sizeof size))
      armci_die("armci_ListenAndAcceptAll: error setting SO_RCVBUF",  size);
    if(setsockopt(msgsock, SOL_SOCKET, SO_SNDBUF, (char *) &size, sizeof size))
      armci_die("armci_ListenAndAcceptAll: error setting SO_SNDBUF",  size);

    armci_TcpNoDelay(sock);

    (void) close(sock); /* will not be needing this again */

    socklist[i] = -msgsock; /* negate connected socket on the list */

    num_accept++;
  }

  if(num_accept < num)
     goto againsel;

  for(i=0; i< num; i++) 
     if(socklist[i]>=0)
        armci_die("armci_ListenAndAcceptAll: not connected",socklist[i]);
     else
        socklist[i] = - socklist[i];

}


int armci_ListenAndAccept(int sock)
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
      armci_die("armci_ListenAndAccept: listen failed",  0);
  }

  if (DEBUG_) {
    (void) printf("process %ld out of listen on socket %d\n",armci_me,sock);
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
  if ( (nready <= 0) && (errno == EINTR) )
    goto againsel;
  else if (nready < 0)
    armci_die("armci_ListenAndAccept: error from select",  nready);
  else if (nready == 0)
    armci_die("armci_ListenAndAccept: timeout waiting for connection", nready);

  if (!FD_ISSET(sock, &ready))
    armci_die("armci_ListenAndAccept: out of select but not ready!",  nready);

againacc:
  msgsock = accept(sock, (struct sockaddr *) NULL, (socklen_t *) NULL);
  if (msgsock == -1) {
    if (errno == EINTR)
      goto againacc;
    else
      armci_die("armci_ListenAndAccept: accept failed",  msgsock);
  }

  if (DEBUG_) {
    (void) printf("process %ld out of accept on socket %d\n", armci_me,msgsock);
    (void) fflush(stdout);
  }

  /* Increase size of socket buffers to improve long message
     performance and increase size of message that goes asynchronously */

  if(setsockopt(msgsock, SOL_SOCKET, SO_RCVBUF, (char *) &size, sizeof size))
    armci_die("armci_ListenAndAccept: error setting SO_RCVBUF",  size);
  if(setsockopt(msgsock, SOL_SOCKET, SO_SNDBUF, (char *) &size, sizeof size))
    armci_die("armci_ListenAndAccept: error setting SO_SNDBUF",  size);

  armci_TcpNoDelay(sock);

  (void) close(sock); /* will not be needing this again */
  return msgsock;
}


int armci_CreateSocketAndConnect(char *hostname, int port)
/*
  Return the file descriptor of the socket which connects me to the
  remote process on hostname at port 

  hostname = hostname of the remote process
  port     =  port number of remote socket
*/
{
  int sock, status;
  struct sockaddr_in server;
  struct hostent *hp;
  int on = 1;
  int size = PACKET_SIZE;
  int trial;
#ifndef SGI
  struct hostent *gethostbyname();
#endif

  /* Create socket */

  if ( (sock = socket(AF_INET, SOCK_STREAM, 0)) < 0 ) {
    (void) fprintf(stderr,"trying to connect to host=%s, port=%d\n",
                   hostname, port);
    armci_die("armci_CreateSocketAndConnect: socket failed",  sock);
  }

  if (setsockopt(sock, SOL_SOCKET, SO_REUSEADDR, 
		 (char *) &on, sizeof on) == -1)
	armci_die("armci_CreateSocketAndConnect: error setting REUSEADDR",  -1);

  /* Increase size of socket buffers to improve long message
     performance and increase size of message that goes asynchronously */

  if(setsockopt(sock, SOL_SOCKET, SO_RCVBUF, (char *) &size, sizeof size))
    armci_die("armci_CreateSocketAndConnect: error setting SO_RCVBUF",  size);
  if(setsockopt(sock, SOL_SOCKET, SO_SNDBUF, (char *) &size, sizeof size))
    armci_die("armci_CreateSocketAndConnect: error setting SO_SNDBUF",  size);


  /* Connect socket */

  server.sin_family = AF_INET;
  hp = gethostbyname(hostname);
  if (hp == 0) {
    (void) fprintf(stderr,"trying to connect to host=%s, port=%d\n",
                   hostname, port);
    armci_die("armci_CreateSocketAndConnect: gethostbyname failed", 0);
  }

  bcopy((char *) hp->h_addr, (char *) &server.sin_addr, hp->h_length);
  server.sin_port = htons((ushort) port);

  trial = 0;
againcon:
  if ((status = 
     connect(sock, (struct sockaddr *) &server, sizeof server)) < 0) {
    if (errno == EINTR)
      goto againcon;
    else if(trial){
      
           (void) fprintf(stderr,"trying to connect to host=%s, port=%d\n",
                   hostname, port);
           armci_die("armci_CreateSocketAndConnect: connect failed", status);
       }else {

         trial =1;
         sleep(1);
         goto againcon;
       }
  }
  
  return sock;
}
