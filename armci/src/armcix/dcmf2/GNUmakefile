# begin_generated_IBM_copyright_prolog                             #
#                                                                  #
# This is an automatically generated copyright prolog.             #
# After initializing,  DO NOT MODIFY OR MOVE                       #
#  --------------------------------------------------------------- #
#                                                                  #
# (C) Copyright IBM Corp.  2007, 2008                              #
# IBM BSD License                                                  #
#                                                                  #
#  --------------------------------------------------------------- #
#                                                                  #
# end_generated_IBM_copyright_prolog                               #

#$Id$
         FOPT = -O
         COPT = -O3

OBJ = armcix_impl.o armcix_wait.o armcix_fence.o armcix_rmw.o armcix_get.o armcix_put.o armcix_lock.o armcix_acc.o

#defines communication libs and symbols
include ../../../config/makemp.h
include ../../../config/makecoms.h


LIB_DEFINES += $(MP_DEFINES) $(COMM_DEFINES) -DARMCIX
LIB_INCLUDES += $(COMM_INCLUDES) $(MP_INCLUDES)
EXTRA_LIBS += $(MP_LIBS) $(COMM_LIBS)

#......................................................................
LIBRARY = libarmci.a
HARDCLEAN = yes

#add some extra Microsoft VC files to clean if any left 
NTSTUFF = *.ilk *.exe *.ncb
LIB_TARGETS = *.x shmtest $(NTSTUFF)
LIB_DISTRIB = ../../../lib

include ../../../config/makefile.h

OBJ += $(EXTRA_OBJ)
include ../../../config/makelib.h
LIBS += $(EXTRA_LIBS)



#......................................................................
%.x.p:%.x
	echo "`whoami` `hostname` 4 `pwd`/$(basename $@) /tmp" > $@

%.x:%.c  $(FULL_LIBRARY_PATH)
	$(CC) -g -I. $(COPT_REN) $(MP_INCLUDES) $(MP_DEFINES) -o $@ $^ $(LIBS) -lm

$(FULL_LIBRARY_PATH)(dcmf2/armcix_get.o): ../armcix.h ./armcix_impl.h
$(FULL_LIBRARY_PATH)(dcmf2/armcix_impl.o): ../armcix.h ./armcix_impl.h
$(FULL_LIBRARY_PATH)(dcmf2/armcix_fence.o): ../armcix.h ./armcix_impl.h
$(FULL_LIBRARY_PATH)(dcmf2/armcix_put.o): ../armcix.h ./armcix_impl.h
$(FULL_LIBRARY_PATH)(dcmf2/armcix_wait.o): ../armcix.h ./armcix_impl.h
$(FULL_LIBRARY_PATH)(dcmf2/armcix_rmw.o): ../armcix.h ./armcix_impl.h
$(FULL_LIBRARY_PATH)(dcmf2/armcix_lock.o): ../armcix.h ./armcix_impl.h
$(FULL_LIBRARY_PATH)(dcmf2/armcix_acc.o): ../armcix.h ./armcix_impl.h

%.x: %.c

