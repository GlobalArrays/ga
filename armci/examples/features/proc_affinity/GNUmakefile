#$Id: GNUmakefile,v 1.2 2007-10-30 02:15:43 manoj Exp $
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
