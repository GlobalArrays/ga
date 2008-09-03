#$Id: GNUmakefile,v 1.1.2.1 2007-06-20 17:41:59 vinod Exp $
all: simple overlap

.PHONY : simple
simple:
	@echo '>>> Making simple only <<<'
	(cd simple;make)

.PHONY : overlap
overlap:
	@echo '>>> Making overlap only <<<'
	(cd overlap;make)

clean:
	(cd simple;make clean)
	(cd overlap;make clean)
