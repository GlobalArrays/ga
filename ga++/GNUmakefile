# default version and target (to be overwritten on command line)
CUR_TARGET  =
CUR_VERSION =

SUBDIRS     = src testing
MAKESUBDIRS = for dir in $(SUBDIRS); do $(MAKE)  -C $$dir $@ || exit 1 ; done

ifndef TARGET 
       TARGET = $(CUR_TARGET)
endif
ifndef LIBDIR
     LIBDIR = ../lib/$(TARGET)
endif

LIBRARY = $(LIBDIR)/libga++.a
TESTS   = testc.x

export TARGET MSG_COMMS

.PHONY: $(LIBRARY)

$(LIBRARY): 
	(echo TARGET is $(TARGET); cd ./src; $(MAKE) || exit 1;) 

$(TESTS) %.x:	$(LIBRARY)
	(cd ./testing; $(MAKE) $@  || exit 1;)

all: $(TESTS)

clean:
	$(MAKESUBDIRS) $^
	$(RM) -f $(LIBRARY)

cleanstamp:
	$(MAKESUBDIRS) $^
