/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/sockets.h,v 1.4 1995-02-24 02:17:50 d3h325 Exp $ */

extern void ShutdownAll();
extern int ReadFromSocket();
extern int WriteToSocket();
extern void CreateSocketAndBind();
extern int ListenAndAccept();
extern int CreateSocketAndConnect();
extern long PollSocket();
