#$Id: makefile,v 1.3 1996-07-26 21:06:23 d3h325 Exp $
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

         OBJ = elio.o
     LIBRARY = libelio.a
     HEADERS = elio.h chemio.h

ifdef PABLO
   LIB_CDEFS = -DPABLO
endif

include ../../config/makefile.h
include ../../config/makelib.h

$(OBJ) : $(HEADERS)

