
# Tiny makefile wrapper

include ../config/makefile.h


export LIBDIR CNFDIR

tcgmsg-mpi_stamp:	include_stamp 
	@$(MAKE) -f Makefile TARGET=$(TARGET)

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

depend:
	echo No dependencies in TCGMSG-MPI 

.PHONY:	cleanF
cleanF:
	@echo Not done here
