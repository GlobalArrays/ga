# make file for Shared Files library
#
# Jarek Nieplocha, Aug  1 1996
#

      LIBRARY = libchemio.a
          OBJ = shared.files.o fortran.o
      HEADERS = sf.fh
 LIB_INCLUDES = -I../elio
  LIB_TARGETS = 

# select the communication library for Shared Files
     COM_LIB = -ltcgmsg
ifeq ($(SFCOM), GA)
 LIB_DEFINES = -DGA
     COM_LIB = -lglobal -ltcgmsg
endif
ifeq ($(SFCOM), MPI)
 LIB_DEFINES = -DMPI
     COM_LIB = -lmpi
endif

ifdef HPIODIR
 LIB_DEFINES += -DHPIODIR=\'$(HPIODIR)/\'
endif

ifdef LARGE_FILES
LIB_DEFINES += $(shell uname -a | awk \
              '{ if ($$1 == "AIX" && ($$4 > 4 || ($$4 == 4 && $$3 > 1)))\
               print "-D_LARGE_FILES -D_LARGE_FILE_API" }')
endif


include ../../config/makefile.h
include ../../config/makelib.h

