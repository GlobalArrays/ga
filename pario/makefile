#$Id: makefile,v 1.2 1996-07-17 15:57:18 mogill Exp $
#
# Parent makefile for ChemIO in NWChem tree
#
#   Written by Jace A Mogill
#              12 June 1996
#	       Argonne National Laboratory
#
#

include ../config/makefile.h


SUBDIRS = elio eaf # dra sf
MAKESUBDIRS = for dir in $(SUBDIRS); do $(MAKE) -C $$dir $@ || exit 1; done

global_stamp: include_stamp
	@echo "this is rule global_stamp"
	$(MAKE) -C elio TARGET=$(TARGET)
	$(MAKE) -C eaf  TARGET=$(TARGET)

include_stamp:
	@echo "this is rule include_stamp"
	/bin/cp -f elio/elio.h elio/chemio.h eaf/c2f.h eaf/eaf.h eaf/eaf.fh eaf/eaf_c2f.h $(INCDIR)
	$(MAKESUBDIRS)
	touch include_stamp

all : $(SUBDIRS)


elio:
	$(MAKE) -C elio


eaf:
	$(MAKE) -C eaf



dra:
	@echo "DRA not under this tree yet"

sf:
	@echo "SF does not exist yet"


clean:
	#(MAKESUBDIRS) clean

