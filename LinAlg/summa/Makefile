ifdef USE_SUMMA
      OBJ = ga_create3.o     ga_summa_lay.o   summa_abt.o\
            ga_summa_al.o    ga_summa_lay2.o  summa_abt2.o\
            ga_summa_c.o     summa_ab.o       summa_atb.o\
            ga_summa_cc.o    summa_ab2.o      summa_atb2.o\
            summa_to_ga.o    ga_summa.o

LIB_DISTRIB = ../../lib
LIBRARY = liblinalg.a
LIB_TARGETS = test.x

INCPATH = -I../../global/src -I../../tcgmsg/ipcv4.0 -I../../ma

include ../../config/makefile.h
include ../../config/makelib.h
endif


ifdef USE_SUMMA
test.x:	test.F
	$(FC) $(FFLAGS) $(INCLUDES) -o $@ $^  -L$(LIBDIR)\
		-lglobal -ltcgmsg -lma -llinalg -lkmath  -nx
else
.PHONY: empty
empty:
	@echo 'summa is not built'

.PHONY: clean
clean:
	rm -f *.o *.x *.a core*


endif

