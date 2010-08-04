/** @file */
#ifndef SOCKETS_H_
#define SOCKETS_H_

#include "typesf2c.h"

extern void ShutdownAll();
extern int ReadFromSocket(int sock, char *buf, Integer lenbuf);
extern int WriteToSocket(int sock, char *buf, Integer lenbuf);
extern void CreateSocketAndBind(int *sock, int *port);
extern int ListenAndAccept(int sock);
extern int CreateSocketAndConnect(char *hostname, char *cport);
extern Integer PollSocket(int sock);
extern Integer WaitForSockets(int nsock, int *socks, int *list);

#endif /* SOCKETS_H_ */
