#$Id: GNUmakefile,v 1.1.2.1 2007-06-20 17:42:11 vinod Exp $
all: simple computation_impact

.PHONY : simple
simple:
	@echo '>>> Making simple only <<<'
	(cd simple;make)

.PHONY : computation_impact
computation_impact:
	@echo '>>> Making computation_impact only <<<'
	(cd computation_impact;make)

clean:
	(cd simple;make clean)
	(cd computation_impact;make clean)
