ifndef TARGET
error:
	@echo "You must define TARGET"
	exit
endif

LIB_DEFINES =
LIB_DISTRIB = ../lib

LIBRARY = libserver.a
LIB_TARGETS = test.x master
OBJ = p2p.o sockets.o

include ../config/makefile.h
include ../config/makelib.h


ifeq ($(TARGET), SOLARIS)
  TARGET_LIBS += /usr/ucblib/libucb.a -lsocket -lrpcsvc -lnsl
endif

test.x: test.o $(LIBRARY_PATH)
	$(CC) $(CFLAGS) -o $@ $^ $(TARGET_LIBS)

all:  test.x 

$(patsubst %,$(LIBRARY_PATH)(%),p2p.o) : 	p2p.h
$(patsubst %,$(LIBRARY_PATH)(%),$(OBJ)):	sockets.h
test.o master.o:	p2p.h
