# $Id: GNUmakefile,v 1.2 2007-10-30 02:15:44 manoj Exp $

SUBDIRS     = boltzmann scf conjugate_gradient lennard-jones md_cluster
MAKESUBDIRS = for dir in $(SUBDIRS); do $(MAKE)  -C $$dir $@ || exit 1 ; done

clean:
	$(MAKESUBDIRS) $^
	$(RM) -f $(LIBRARY)

cleanstamp:
	$(MAKESUBDIRS) $^
