all: aggregation concurrency gpc non-blocking notification

.PHONY : aggregation
aggregation:
	@echo '>>> Making aggregation only <<<'
	(cd aggregation;make)

.PHONY : concurrency
concurrency:
	@echo '>>> Making concurrency only <<<'
	(cd concurrency;make)

.PHONY : gpc
gpc:
	@echo '>>> Making gpc only <<<'
	(cd gpc;make)

.PHONY : non-blocking
non-blocking:
	@echo '>>> Making non-blocking only <<<'
	(cd non-blocking;make)

.PHONY : notification
notification:
	@echo '>>> Making notification only <<<'
	(cd notification;make)
clean:
	(cd aggregation;make clean)
	(cd concurrency;make clean)
	(cd gpc;make clean)
	(cd non-blocking;make clean)
	(cd notification;make clean)
