
# Tiny makefile wrapper


include ../config/makefile.h

USES_BLAS = src/ga_solve_seq.F src/ga_dgemm.F src/ga_dgemm_seq.F

#
# This dependency is buggered since the actual GA makefile
# always CD's into the src directory to do the make ... no
# alternative but to do the make every time
#

global_stamp:	include_stamp 
	make -f makefile.GA TARGET=$(TARGET) VERSION=SHMEM \
		DIAG=PAR LU_SOLVE=PAR
	cp -p libglobal.a $(LIBDIR)

include_stamp:	src/global.fh
	cp src/global.fh $(INCDIR) 
	touch include_stamp

realclean clean:	
	/bin/rm -f include_stamp
	-$(MAKE) -f makefile.GA clean

ifdef USES_BLAS
sngl_to_dbl:
	$(CNFDIR)/sngl_to_dbl $(USES_BLAS)
dbl_to_sngl:
	$(CNFDIR)/dbl_to_sngl $(USES_BLAS)
else
sngl_to_dbl dbl_to_sngl:
	echo $@ : no conversion necessary
endif
