#$Id: makefile,v 1.18 1997-02-26 00:44:17 d3j191 Exp $

# Tiny makefile wrapper

include ../config/makefile.h

USES_BLAS = src/ga_solve_seq.F src/ga_dgemm.F src/ga_dgemm_seq.F

#
# This dependency is buggered since the actual GA makefile
# always CD's into the src directory to do the make ... no
# alternative but to do the make every time
#

export LIBDIR CNFDIR BINDIR INCDIR


global_stamp:	include_stamp 
	$(MAKE) -f Makefile TARGET=$(TARGET)\
		DIAG=PAR

include_stamp:	src/global.fh
	cp src/global.fh da/da.fh src/global.h src/types.f2c.h\
		src/c.names.h src/cray.names.h $(INCDIR) 
	touch include_stamp

%.x:
	$(MAKE) -f Makefile TARGET=$(TARGET) DIAG=PAR $@
	
realclean clean:	
	/bin/rm -f include_stamp
	-$(MAKE) -f Makefile clean

depend:
	echo No dependencies in global

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

src.F:
	
