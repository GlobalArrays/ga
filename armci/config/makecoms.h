#provides definitions for protocols used by communication libraries
# COMM_DEFINES - identifies communication network protocol
# COMM_LIBS    - list of libs that need to be linked with 
# COMM_INCLUDES- include path to access communication protocol API
#
ifeq ($(ARMCI_NETWORK),GM)
  COMM_DEFINES = -DGM
  ifdef GM_INCLUDE
    COMM_INCLUDES = -I$(GM_INCLUDE)
  endif
  ifdef GM_LIB
    COMM_LIBS = -L$(GM_LIB)
  endif
  GM_LIB_NAME = -lgm
  COMM_LIBS += $(GM_LIB_NAME) -lpthread
endif

ifeq ($(ARMCI_NETWORK),VIA)
  COMM_DEFINES = -DVIA
  ifdef VIA_INCLUDE
    COMM_INCLUDES = -I$(VIA_INCLUDE)
  endif
  ifdef VIA_LIB
    COMM_LIBS = -L$(VIA_LIB)
  endif
  VIA_LIB_NAME = -lvia
  COMM_LIBS += $(VIA_LIB_NAME)
endif

ifeq ($(ARMCI_NETWORK),QUADRICS)
  COMM_DEFINES = -DQUADRICS
  ifdef QUADRICS_INCLUDE
    COMM_INCLUDES = -I$(QUADRICS_INCLUDE)
  else
    ifeq ($(TARGET),DECOSF)
       COMM_INCLUDES = -I/opt/rms/include
    endif
  endif
  ifdef QUADRICS_LIB
    COMM_LIBS = -L$(QUADRICS_LIB)
  else
    ifeq ($(TARGET),DECOSF)
      COMM_LIBS = -L/opt/rms/lib
    endif
  endif
  QUADRICS_LIB_NAME = -lshmem -lelan3 -lelan -lrt
  COMM_LIBS += $(QUADRICS_LIB_NAME)
endif


ifeq ($(TARGET),LAPI)
ifdef LAPI2
  COMM_DEFINES += -DLAPI2
  COMM_INCLUDES = -I/u2/d3h325/lapi_vector_beta
  COMM_LIBS += /u2/d3h325/lapi_vector_beta/liblapi_r_dbg.a
endif
endif

ifeq ($(TARGET),SOLARIS)
#  need gethostbyname from -lucb under earlier versions of Solaris
   COMM_LIBS += $(shell uname -r |\
                awk -F. '{ if ( $$1 == 5 && $$2 < 6 )\
                print "/usr/ucblib/libucb.a" }')
   COMM_LIBS +=  -lsocket -lrpcsvc -lnsl
endif
ifeq ($(TARGET),SOLARIS64)
   COMM_LIBS +=  -lsocket -lrpcsvc -lnsl
endif

ifeq ($(TARGET),FUJITSU-VPP)
   COMM_LIBS = -lmp -lgen -lpx -lelf -Wl,-J,-P
endif

ifeq ($(TARGET),FUJITSU-VPP64)
   COMM_LIBS = -lmp -lgen -lpx -lelf -Wl,-J,-P
endif
   
ifeq ($(TARGET),FUJITSU-AP)
   COMM_LIBS = -L/opt/FSUNaprun/lib -lmpl -lelf -lgen
endif

ifeq ($(TARGET),CRAY-YMP)
   COMM_LIBS = $(LIBCM)
endif

