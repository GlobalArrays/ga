#$Id: makefile,v 1.11 1997-03-13 22:07:00 d3h325 Exp $

          OBJ = eaf.o eaf_f2c.o
      LIBRARY = libchemio.a
  LIB_DISTRIB = ../../lib
      HEADERS = eaf.fh eaf.h
  LIB_DEFINES = -DEAF_STATS
  LIB_TARGETS = test.x
 LIB_INCLUDES = -I../elio -I../../global/src

ifdef LARGE_FILES
LIB_DEFINES += $(shell uname -a | awk \
              '{ if ($$1 == "AIX" && ($$4 > 4 || ($$4 == 4 && $$3 > 1)))\
               print "-D_LARGE_FILES -D_LARGE_FILE_API" }')
endif

include ../../config/makefile.h
include ../../config/makelib.h

test.x:	test.o $(LIBRARY_PATH)
	$(LINK.f) -o $@ test.o $(LIBS)

