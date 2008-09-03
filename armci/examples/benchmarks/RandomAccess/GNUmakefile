:$Id: GNUmakefile,v 1.1.2.1 2007-06-20 17:41:20 vinod Exp $
all: simple smp_bucket

.PHONY : simple
simple:
	@echo '>>> Making simple RA only <<<'
	(cd simple;make)

.PHONY : smp_bucket
smp_bucket:
	@echo '>>> Making smp_bucket only <<<'
	(cd smp_bucket;make)

clean:
	(cd simple;make clean)
	(cd smp_bucket;make clean)
