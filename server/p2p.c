#include <stdio.h>
#include "sockets.h"
#include "p2p.h"

static int id;
static int sock;

static int debug = 0;
static int master=0;
static int slave=0;
static int connected=0;

#define MIN(a,b) ((a) <= (b) ? (a) : (b))

char** find_args(int argc, char **argv)
{
   int i, found=0;

   printf("args=%d\n",argc);
   for(i=1; i< argc; i++){
       printf("arg[%d] = %s\n",i,argv[i]);
       fflush(stdout);
       if(strcmp(argv[i],"-Master")==0){ master=1; return argv + i;}
       if(strcmp(argv[i],"-Slave")==0) { slave=1;  return argv + i;}
   }

   return( (char**) 0);
/*
   printf("Command line error: -Master or -Slave hostname port args expected");
   exit(1);
*/
}

int server_clusid()
{
    if(master)return(0);
    else return(1);
}


int server_init(int argc, char **argv)
{

    const char *hostname, *cport;
    char **iway_argv;

    iway_argv = find_args(argc,argv);
    if(!iway_argv) return(0); 

    if (master) {
	int port;
	iw_CreateSocketAndBind(&sock, &port);

	printf("Master: connect to me at port number %d\n", port);
	fflush(stdout);
	id = 0;
	sock = iw_ListenAndAccept(sock);
    }
    else {
	if (argc < 3) exit(1);
        hostname = iway_argv[1];
        cport = iway_argv[slave+1];
        printf("Slave %d: connecting at port %s\n",slave,cport);
        fflush(stdout);
	id = 1;
	sock = iw_CreateSocketAndConnect(hostname, cport);
    }
    connected = 1;
    return(1);
}

void server_close(void)
{
    if(connected){
       printf("%d: Closing\n", id); fflush(stdout);
       close(sock);
    }
}

/*#define BUFFERSIZE (131072+16 + 4)*/
#define BUFFERSIZE (264000+16 + 4)
static nb=0;
static char rcvbuf[BUFFERSIZE+sizeof(int)];


int poll_server(void)
{
    if (nb)
	return 1;

    return iw_PollSocket(sock);
}
	

static void send_while_buffering(void *buff, int buflen)
{
  char *buf = buff;
  while (buflen) {
    int nfree = sizeof(rcvbuf) - nb;
    int nread;
    int nwrote = iw_NonBlockingWriteToSocket(sock, buf, buflen);
    
    if (nwrote < 0)
      iw_Error("send_w_buf: write of buflen failed", (long) sizeof(int));
    
    buflen -= nwrote;
    buf += nwrote;
    
    if (debug && nwrote) {
      printf("%d: wrote %d bytes\n", id, nwrote); 
      fflush(stdout);
    }
    
    if (nfree > 0) {
      if (debug) {
	printf("%d: calling nonblocking read, nfree=%d\n",id,nfree); 
	fflush(stdout);
      }
      nread = iw_NonBlockingReadFromSocket(sock, rcvbuf+nb, nfree);
      if (debug) {
	printf("%d: out of  nonblocking read\n",id); fflush(stdout);
      }
      
      if (nread < 0)
	iw_Error("send_w_buf: read of sock failed", (long) nfree);
      else if (nread > 0) {
	nb += nread;
	if (debug) {
	  printf("%d: while sending read %d bytes\n", id, nread); 
	  fflush(stdout);
	}
      }
    }
    else
      printf("%d: not reading since buffer is full\n", id);

  }
}

void send_to_server(void *buf, int buflen)
{
    send_while_buffering(&buflen, sizeof(buflen));
    send_while_buffering(buf, buflen);

    if (debug) {
	printf("%d: sent message of length %d\n", id, buflen);
	fflush(stdout);
    }
}

void recv_from_server(void *buff, int *buflen)
{
    char *buf = buff;
    if (!nb) {
	if (iw_BlockingReadFromSocket(sock, (char *) buflen, sizeof(int)) != 
	    sizeof(int))
	    iw_Error("recv_from_server: read of buflen failed", (long) id);
	if (iw_BlockingReadFromSocket(sock, buf, *buflen) != *buflen)
	    iw_Error("recv_from_server: read of buf failed", (long) id);
    }
    else {
	int ncopy, nread;
	if (nb < sizeof(int))
	    iw_Error("recv_from_server: split msg len", (long) id);

	memcpy((char *) buflen, rcvbuf, sizeof(int));
	ncopy = MIN(nb-sizeof(int), *buflen);;

	if (debug) {
	    printf("%d: nb=%d, buflen=%d, ncopy=%d\n", id, nb, *buflen, ncopy);
	    fflush(stdout);
	}
	
	memcpy(buf, rcvbuf+sizeof(int), ncopy);

	nread = *buflen - ncopy;

	if (nread) {
	    if (debug) {
		printf("%d: Reading additional bytes %d\n", id, nread); 
		fflush(stdout);
	    }
	    if (iw_BlockingReadFromSocket(sock, buf+ncopy, nread) != nread)
		iw_Error("recv_from_server: read of buf failed", (long) nread);
	    nb = 0;
	}
	else if (nb > (*buflen+sizeof(int))) {
	    int nused = *buflen+sizeof(int);
	    int nleft = nb - nused;
	    char *from = rcvbuf + nused;
	    int i;

	    if (debug) {
		printf("%d: Moving additional bytes %d\n", id, nleft);
		fflush(stdout);
	    }

	    for (i=0; i<nleft; i++)
		rcvbuf[i] = from[i];
	    /*bcopy(from, rcvbuf, nleft); */
	    nb = nb - nused;
	}
	else
	    nb = 0;
    }

    if (debug) {
	printf("%d: rcvd message of length %d, nb=%d\n", id, *buflen, nb);
	fflush(stdout);
    }

}

	
