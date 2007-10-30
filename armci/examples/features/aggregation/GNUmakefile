#$id$
all: simple sparse_matvecmul

.PHONY : simple
simple:
	@echo '>>> Making simple only <<<'
	(cd simple;make)

.PHONY : sparse_matvecmul
sparse_matvecmul:
	@echo '>>> Making sparse_matvecmul only <<<'
	(cd sparse_matvecmul;make)

clean:
	(cd simple;make clean)
	(cd sparse_matvecmul;make clean)
