#$id$
all: simple multidma

.PHONY : simple
simple:
	@echo '>>> Making simple only <<<'
	(cd simple;make)

.PHONY : multidma
multidma:
	@echo '>>> Making multidma only <<<'
	(cd multidma;make)

clean:
	(cd simple;make clean)
	(cd multidma;make clean)
