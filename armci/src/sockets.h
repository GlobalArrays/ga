#ifndef SOCKETS_H_
#define SOCKETS_H_

extern int armci_PollSocket(int sock);
extern int armci_WaitSock(int *socklist, int num, int *ready);
extern int armci_ReadFromSocket(int sock, void* buffer, int lenbuf);
extern int armci_WriteToSocket (int sock, void* buffer, int lenbuf);
extern void armci_ListenSockAll(int* socklist, int num);
extern void armci_AcceptSockAll(int* socklist, int num);
extern int armci_CreateSocketAndConnect(char *hostname, int port);
extern void armci_ShutdownAll(int socklist[], int num);
extern void armci_CreateSocketAndBind(int *sock, int *port);

#define PACKET_SIZE  32768
#define TIMEOUT_ACCEPT 60

#endif
