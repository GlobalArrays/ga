# Include file for makefiles in individual subdirectories 

# LIBDIR defines where the libraries should be built.
# LIB_DISTRIB defines default location for the GA distribution package
# when built as a standalone package.
#

ifndef LIBDIR
   ifdef LIB_DISTRIB 
     LIBDIR = $(LIB_DISTRIB)/$(TARGET)
   else
     LIBDIR = .
   endif
endif
  
# makefile in each directory might define error message for undefined 
# symbols etc.
# When error message is defined, it should be displayed and then
# make processing aborted
#

define print_error
        @echo $(ERRMSG)
        exit 1
        @echo
endef

ifdef ERRMSG
error:
	$(print_error)
endif

# Make sure that nothing gets compiled in case of error 
#
ifdef ERRMSG
      CC = $(print_error)
      FC = $(print_error)
endif

ifdef LIBRARY_STAMP
      SYMBOL_STAMP = $(TARGET).$(LIBRARY_STAMP)
else
      SYMBOL_STAMP = $(TARGET)
endif

FULL_LIBRARY_PATH := $(LIBDIR)/$(LIBRARY)

OBJECTS := $(OBJ) $(OBJ_OPTIMIZE)

 LIBOBJ := $(patsubst %,$(FULL_LIBRARY_PATH)(%),$(OBJECTS))
 LIBOBJ_OPT := $(patsubst %,$(FULL_LIBRARY_PATH)(%),$(OBJ_OPTIMIZE))

$(FULL_LIBRARY_PATH): $(HEADERS) $(SYMBOL_STAMP).stamp $(LIBOBJ) $(LIBOBJ_OPT)
	$(RANLIB) $@

$(SYMBOL_STAMP).stamp:
	if [ -f *.stamp ]; then\
		 $(MAKE) cleanstamp;\
	fi
ifdef HEADERS
	cp $(HEADERS) $(INCDIR)
endif
	echo "" > $(SYMBOL_STAMP).stamp

ifdef OBJ_OPTIMIZE
ifndef OPTIMIZE
.PHONY: $(LIBOBJ_OPT)
$(LIBOBJ_OPT):
	@$(MAKE) OPTIMIZE="Yes"
endif
endif


MAKESUBDIRS = for dir in $(SUBDIRS); do $(MAKE)  -C $$dir $@ || exit 1 ; done


ifdef SUBDIRS

$(FULL_LIBRARY_PATH):        subdirs

.PHONY: subdirs
subdirs:
	@for dir in $(SUBDIRS); do \
		cd $$dir ; \
		$(MAKE) || exit 1  ; \
		cd .. ;\
	done
endif
#		$(MAKE)  -C $$dir || exit 1 ;  \

.PHONY: clean
clean:
ifdef SUBDIRS
	$(MAKESUBDIRS)
endif
	-$(RM) -f *.o *.p *core *stamp *trace *.x *events* $(LIB_TARGETS)
	-$(RM) -rf ./obj *.obj *.exe
ifdef HARDCLEAN 
	-$(RM) -f $(FULL_LIBRARY_PATH)
else
	if [ -f $(FULL_LIBRARY_PATH) ] ; then \
		$(AR) d $(FULL_LIBRARY_PATH) $(OBJ) $(OBJ_OPTIMIZE) ; \
		if [ `$(AR) t $(FULL_LIBRARY_PATH) | wc | awk ' {print $$1;}'` -eq 0 ] ; then \
			$(RM) -f $(FULL_LIBRARY_PATH) ; \
		fi ; \
	fi ;
endif


.PHONY: realclean
realclean:      clean
ifdef SUBDIRS
	$(MAKESUBDIRS)
endif
	-$(RM) -rf *~ \#*\#

.PHONY: cleanstamp
cleanstamp: clean
ifdef SUBDIRS
	$(MAKESUBDIRS)
endif
	-$(RM) -rf *.stamp
