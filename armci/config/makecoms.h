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

ifeq ($(TARGET),LAPI)
ifdef LAPI2
  COMM_DEFINES += -DLAPI2
  COMM_INCLUDES = -I/u2/d3h325/lapi_vector_beta
  COMM_LIBS += /u2/d3h325/lapi_vector_beta/liblapi_r_dbg.a
endif
endif

ifeq ($(TARGET),SOLARIS)
   COMM_LIBS += /usr/ucblib/libucb.a -lsocket -lrpcsvc -lnsl
endif
ifeq ($(TARGET),SOLARIS64)
   COMM_LIBS += /usr/ucblib/libucb.a -lsocket -lrpcsvc -lnsl
endif

ifeq ($(TARGET),FUJITSU-VPP)
   COMM_LIBS = -lmp -lgen -lpx -lelf -Wl,-J,-P
endif

ifeq ($(TARGET),FUJITSU-VPP64)
   COMM_LIBS = -lmp -lgen -lpx -lelf -Wl,-J,-P
endif
   
ifeq ($(TARGET),CRAY-YMP)
   COMM_LIBS = $(LIBCM)
endif

ifeq ($(TARGET),HPUX)
   COMM_LIBS = -lm
endif
