#include <stdio.h>
#include "sockets.h"
#include "p2p.h"

#define N 32768
void master_test()
{
    int n = N, i, j;
    int sndbuf[N], rcvbuf[N], msglen;

    for (i=0; i<N; i++)
	sndbuf[i] = N-i-1;

    for (i=0; i<20; i++) {
	printf("master sending loop %d\n", i); fflush(stdout);
	for (j=0; j<N; j+=(N/4))
	    send_to_server((char *) (sndbuf+j), (N/4)*sizeof(int));
	recv_from_server((char *) rcvbuf, &msglen);
	if (msglen != sizeof(rcvbuf)) 
	    iw_Error("bad msglen in master\n", (long) msglen);
    }

    for (i=0; i<N; i++)
	if (rcvbuf[i] != i)
	    iw_Error("bad msg in master\n", (long) i);

    while (1) {
	if (poll_server()) {
	    recv_from_server(rcvbuf, &msglen);
	    printf("master got message\n"); fflush(stdout);
	    break;
	}
	printf("master sleeping\n");
	fflush(stdout);
	sleep(1);
    }
	
}


void slave_test()
{
    int n = N, i, j;
    int sndbuf[N], rcvbuf[N], msglen;

    for (i=0; i<N; i++)
	sndbuf[i] = i;

    /* Test buffering ... should be able to send */
       

    for (i=0; i<20; i++) {
	send_to_server((char *) sndbuf, sizeof(sndbuf));
	for (j=0; j<N; j+=(N/4))
	    recv_from_server((char *) (rcvbuf+j), &msglen);
	if (msglen != (N/4)*sizeof(int)) 
	    iw_Error("bad msglen in slave\n", (long) msglen);
    }

    for (i=0; i<N; i++)
	if (rcvbuf[i] != (N-i-1))
	    iw_Error("bad msg in slave\n", (long) i);

    printf("slave sleeping\n"); fflush(stdout);
    sleep(10);
    send_to_server(sndbuf, 4);
    printf("slave sent message\n"); fflush(stdout);
    sleep(2);
}

    

int main(int argc, char **argv)
{
    char msg[] = "Hi there!", ack[1];
    int msglen;

    if(!server_init(argc, argv)){
       printf("needs two clusters to run\n");
       exit(1);
    }

    if(server_clusid()==0){ /*master */
       printf("0:servers connected\n");
       fflush(stdout);
       send_to_server(msg, sizeof(msg));
       recv_from_server(ack, &msglen);

       master_test();
       sleep(5);
    }else{
       printf("1:servers connected\n");
       fflush(stdout);
       recv_from_server(msg, &msglen);
       printf("Message from master = %s\n", msg);
       fflush(stdout);

       send_to_server(ack, sizeof(ack));

       slave_test();
    }

    server_close();

    return 0;
}
