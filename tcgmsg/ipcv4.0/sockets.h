/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/sockets.h,v 1.1.1.1 1994-03-29 06:44:50 d3g681 Exp $ */

extern void ShutdownAll();
extern int ReadFromSocket();
extern int WriteToSocket();
extern void CreateSocketAndBind();
extern int ListenAndAccept();
extern int CreateSocketAndConnect();
extern long PollSocket();
