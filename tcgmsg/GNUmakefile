ifdef  TARGET
       MACHINE = $(TARGET)
endif

ifndef MACHINE
error:
	@echo "You must define MACHINE"
	exit
endif

VERSION5_TARGETS = CRAY-T3D CRAY-T3E LAPI LAPI64 CYGNUS CYGWIN INTERIX MACX

ifeq ($(MACHINE),$(findstring $(MACHINE),$(VERSION5_TARGETS)))
     SRCDIR = ipcv5.0
else
     SRCDIR = ipcv4.0
endif


export MACHINE TARGET BINDIR

test.x:
	(cd $(SRCDIR); $(MAKE) $@)

all:
	(cd $(SRCDIR); $(MAKE) $@)

realclean: clean

clean:
	(cd ipcv5.0; $(MAKE) $@)
	(cd ipcv4.0; $(MAKE) $@)


