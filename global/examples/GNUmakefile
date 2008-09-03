# $Id: GNUmakefile,v 1.1.2.1 2007-10-23 00:46:20 manoj Exp $

SUBDIRS     = boltzmann scf conjugate_gradient lennard-jones md_cluster
MAKESUBDIRS = for dir in $(SUBDIRS); do $(MAKE)  -C $$dir $@ || exit 1 ; done

clean:
	$(MAKESUBDIRS) $^
	$(RM) -f $(LIBRARY)

cleanstamp:
	$(MAKESUBDIRS) $^
