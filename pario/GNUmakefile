#$Id: GNUmakefile,v 1.2.14.2 2007-06-12 22:44:10 manoj Exp $

LIB_DISTRIB = ../lib
LIBRARY = libpario.a
LIBRARY_SHARED = libpario.so
SUBDIRS = elio eaf dra sf

ifdef GA_SHLIB
# NOTE: Skip shared library build in all subdirs. Build the monolithic
# shared library (libpario.so) here.
.PHONY: $(LIBRARY)
$(LIBRARY):
	@for dir in $(SUBDIRS); do \
		cd $$dir ; \
		$(MAKE) SKIP_SHLIB="Yes" || exit 1  ; \
		cd .. ;\
	done
	$(CC) $(SHLIB_LDFLAGS) -o $(SHLIB_DIR)/$(LIBRARY_SHARED) $(SHLIB_OBJ)
endif

include ../config/makefile.h
include ../config/makelib.h

ifdef GA_SHLIB
  ELIO_OBJ  := $(patsubst %, elio/$(OBJ_DIR)/%, elio.o stat.o)
  EAF_OBJ   := $(patsubst %,  eaf/$(OBJ_DIR)/%, eaf.o eaf_f2c.o)
  SF_OBJ    := $(patsubst %,   sf/$(OBJ_DIR)/%, sf_fortran.o shared.files.o)
  DRA_OBJ   := $(patsubst %,  dra/$(OBJ_DIR)/%, disk.arrays.o disk.param.o patch.util.o fortran.o env.o capi.o buffers.o)
  SHLIB_OBJ := $(ELIO_OBJ) $(EAF_OBJ) $(SF_OBJ) $(DRA_OBJ)
endif
