# make file for Shared Files library
#
# Jarek Nieplocha, Aug  1 1996
#

  LIB_DISTRIB = ../../lib
      LIBRARY = libchemio.a
          OBJ = shared.files.o sf_fortran.o
      HEADERS = sf.fh sf.h
 LIB_INCLUDES = -I../elio
  LIB_TARGETS = 


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

$(patsubst %,$(LIBRARY_PATH)(%),shared.files.o): sf.h coms.h
$(patsubst %,$(LIBRARY_PATH)(%),fortran.o):      sf.h
