#$Id: makefile,v 1.4 1996-09-17 22:21:55 d3h325 Exp $
#
# ChemIO makefile for making ELIO
#
#   Written by Jace A Mogill
#              12 June 1996
#	       Argonne National Laboratory
#
#

 LIB_DISTRIB = ../../lib
     LIBRARY = libelio.a

         OBJ = elio.o stat.o
     HEADERS = elio.h chemio.h

ifdef PABLO
   LIB_CDEFS = -DPABLO
endif

ifdef NOPIOFS
   LIB_CDEFS += -DNOPIOFS
endif

include ../../config/makefile.h
include ../../config/makelib.h

$(OBJ) : $(HEADERS)

$(patsubst %,$(LIBRARY_PATH)(%),$(OBJ)):  $(HEADERS)
