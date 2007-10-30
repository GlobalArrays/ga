all: features benchmarks

.PHONY : features
features:
	@echo '>>> Making features only <<<'
	(cd features;make)

.PHONY : benchmarks
benchmarks:
	@echo '>>> Making benchmarks only <<<'
	(cd benchmarks;make)

clean:
	(cd features;make clean)
	(cd benchmarks;make clean)
