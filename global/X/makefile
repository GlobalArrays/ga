#$Id: makefile,v 1.3 1995-02-02 23:12:57 d3g681 Exp $
#Specify location for X Windows 

    LOC = /msrc/apps/X11R5
  XLIBS = -L$(LOC)/lib -lXaw -lXmu -lXt -lX11 -lXau -lXdmcp -lXext

#
 CFLAGS =  -O -I$(LOC)/include

xregion0:	xregion0.o
	cc $(CFLAGS) -o xregion0 xregion0.o $(XLIBS) -lm

lint:
	lint -h -I$(LOC)/include xregion0.c

clean:
	/bin/rm -rf *.o xregion0
