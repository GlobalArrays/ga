
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

