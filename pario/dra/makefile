#$Id: makefile,v 1.2 1996-08-30 00:58:39 vg038 Exp $

# Tiny makefile wrapper

include ../../config/makefile.h


export LIBDIR CNFDIR

dra_stamp:	include_stamp 
	$(MAKE) -f Makefile TARGET=$(TARGET)

include_stamp:	dra.fh file.modes.h
	cp dra.fh  $(INCDIR) 
	cp file.modes.h  $(INCDIR) 
	touch include_stamp

update_archive:
	@$(MAKE) -f Makefile TARGET=$(TARGET) $@

%.x:
	$(MAKE) -f Makefile TARGET=$(TARGET) $@
	
realclean clean:	
	/bin/rm -f include_stamp
	-$(MAKE) -f Makefile clean

depend:
	echo No dependencies in disk arrays 

ifdef USES_BLAS
sngl_to_dbl:
	$(CNFDIR)/sngl_to_dbl $(USES_BLAS)
dbl_to_sngl:
	$(CNFDIR)/dbl_to_sngl $(USES_BLAS)
else
sngl_to_dbl dbl_to_sngl:
	@echo $@ : no conversion necessary
endif

.PHONY:	cleanF
cleanF:
	@echo Not done here
