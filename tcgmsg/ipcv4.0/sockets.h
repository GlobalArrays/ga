/*$Id: sockets.h,v 1.2 1995-02-02 23:25:49 d3g681 Exp $*/
/* $Header: /tmp/hpctools/ga/tcgmsg/ipcv4.0/sockets.h,v 1.2 1995-02-02 23:25:49 d3g681 Exp $ */

extern void ShutdownAll();
extern int ReadFromSocket();
extern int WriteToSocket();
extern void CreateSocketAndBind();
extern int ListenAndAccept();
extern int CreateSocketAndConnect();
extern long PollSocket();
