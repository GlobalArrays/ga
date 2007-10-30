#$Id: GNUmakefile,v 1.2 2007-10-30 02:15:38 manoj Exp $
all: pthreads armci_blocking armci_nonblocking armci_multithreaded

.PHONY : pthreads
pthreads:
	@echo '>>> Making lu only <<<'
	(cd pthreads;make)

.PHONY : armci_blocking
armci_blocking:
	@echo '>>> Making lu only <<<'
	(cd armci_blocking;make)

.PHONY : armci_nonblocking
armci_nonblocking:
	@echo '>>> Making lu only <<<'
	(cd armci_nonblocking;make)

.PHONY : armci_multithreaded
armci_multithreaded:
	@echo '>>> Making lu only <<<'
	(cd armci_multithreaded;make)

clean:
	(cd pthreads;make clean)
	(cd armci_blocking;make clean)
	(cd armci_nonblocking;make clean)
	(cd armci_multithreaded;make clean)
