 
# Tiny makefile wrapper

include ../config/makefile.h

ifeq ($(TARGET),SP1)
    ALTDIR = $(LIBDIR:SP1=LAPI)
endif


export LIBDIR CNFDIR BINDIR

tcgmsg-mpi_stamp:	include_stamp 
	@$(MAKE) -f Makefile TARGET=$(TARGET)
ifeq ($(TARGET),SP1)
	test -d $(ALTDIR) || $(MKDIR) -p $(ALTDIR)
	$(MAKE) -f Makefile TARGET=LAPI LIBDIR=$(ALTDIR) BINDIR=../../bin/SP1 test.x 
endif


include_stamp:	msgtypesf.h
	cp msgtypesf.h  $(INCDIR) 
	touch include_stamp

update_archive:
	@$(MAKE) -f Makefile TARGET=$(TARGET) $@

%.x:
	$(MAKE) -f Makefile TARGET=$(TARGET) $@
	
realclean clean:	
	/bin/rm -f include_stamp
	-$(MAKE) -f Makefile clean
ifeq ($(TARGET),SP1)
	$(MAKE) -f Makefile TARGET=LAPI LIBDIR=$(ALTDIR) clean
endif

depend:
	echo No dependencies in TCGMSG-MPI 

.PHONY:	cleanF
cleanF:
	@echo Not done here
