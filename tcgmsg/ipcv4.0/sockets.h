/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/sockets.h,v 1.3 1995-02-24 02:14:28 d3h325 Exp $ */

extern void ShutdownAll();
extern int ReadFromSocket();
extern int WriteToSocket();
extern void CreateSocketAndBind();
extern int ListenAndAccept();
extern int CreateSocketAndConnect();
extern long PollSocket();
