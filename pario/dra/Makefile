
     LIBRARY = libchemio.a
         OBJ = disk.arrays.o disk.param.o patch.util.o fortran.o
     HEADERS = dra.fh
LIB_INCLUDES = -I../elio

ifdef HPIODIR
  LIB_DEFINES = -DHPIODIR=\'$(HPIODIR)/\'
endif

# Uncomment the line below to enable the implementation based on a
# collection of independent files. Also set HPIODIR=/tmp on the SP.
#LIB_DEFINES  += -DINDEPFILES

include ../../config/makefile.h
include ../../config/makelib.h

# DO NOT EDIT BENEATH THIS LINE ... GENERATED AUTOMATICALLY
$(LIBRARY_PATH)(test_mxm.o):	$(INCDIR)/tcgmsg.fh $(INCDIR)/mafdecls.fh $(INCDIR)/global.fh ./dra.fh 
$(LIBRARY_PATH)(test.o):	$(INCDIR)/mafdecls.fh $(INCDIR)/global.fh ./dra.fh 
$(LIBRARY_PATH)(perf.o):	$(INCDIR)/tcgmsg.fh $(INCDIR)/mafdecls.fh $(INCDIR)/global.fh ./dra.fh 
$(LIBRARY_PATH)(patch.util.o):	$(INCDIR)/global.h ./drap.h 
$(LIBRARY_PATH)(global.unsup.o):	$(INCDIR)/macommon.h $(INCDIR)/global.h 
$(LIBRARY_PATH)(fortran.o):	./drap.h ./dra.h 
$(LIBRARY_PATH)(disk.param.o):	$(INCDIR)/global.h ./drap.h 
$(LIBRARY_PATH)(disk.arrays.o):	./drap.h ./dra.h 
$(LIBRARY_PATH)(dbg_write.o):	$(INCDIR)/macommon.h $(INCDIR)/global.h $(INCDIR)/DRAp.h $(INCDIR)/DRA.h 
$(LIBRARY_PATH)(dbg_read.o):	$(INCDIR)/macommon.h $(INCDIR)/global.h $(INCDIR)/DRAp.h $(INCDIR)/DRA.h 
