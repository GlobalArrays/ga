# Makelib.h, 01.26.94
#

       CORE_LIBS = -lglobal -lma

ifndef LIBMPI
   LIBMPI = -lmpi
endif

ifneq ($(MSG_COMMS),MPI)
  ifdef USE_MPI
       LIBCOM = -ltcgmsg-mpi
  else
       LIBCOM = -ltcgmsg
  endif
endif


#................................ SUN ..........................................
ifeq ($(TARGET),SOLARIS)
       EXTRA_LIBS = /usr/ucblib/libucb.a -lsocket -lrpcsvc -lnsl
endif
#................................ FUJITSU-VPP ..............................
ifeq ($(TARGET),FUJITSU-VPP)
#       EXTRA_LIBS = -L /opt/tools/lib/ -lmp2tv -lgen  -lpx -lelf -Wl,-J,-P
#MPlib 2.2.X and higher
        EXTRA_LIBS = /usr/local/lib/libmp2.a -L/opt/tools/lib/ -lgen  -lpx -lelf -Wl,-J,-P -L/usr/lang/lib -lblasvp
endif
#................................ KSR ......................................
#
ifeq ($(TARGET),KSR)
#
# KSR-2 running OSF 1.2.0.7
#
#    LIBCOM = $(SRC)/tcgmsg/ipcv4.0/libtcgmsg.a

       BLAS  = -lksrblas
 EXTRA_LIBS += -lrpc -para
endif
#................................ HPUX  .....................................
ifeq ($(TARGET),HPUX)
       EXTRA_LIBS = -lm 
endif
#................................ Intel .....................................
ifeq ($(INTEL),YES)
#
# all Intel machines
#
#................................ PARAGON ...................................
ifeq ($(TARGET),PARAGON)
       CLIB = -lm
#
       EXTRA_LIBS = -nx 
else
       EXTRA_LIBS = -node 
endif
       BLAS  = -lkmath
endif
#................................   SGI ....................................
ifeq ($(TARGET),SGITFP)
endif
ifeq ($(TARGET),SGI)
       BLAS = -lblas
endif
ifeq ($(TARGET),SGI_N32)
       BLAS = -lblas
endif
ifeq ($(TARGET),SGI64)
       BLAS = -lblas
endif
#.................................. IBM SP ..................................
ifeq ($(TARGET),SP1)
       BLAS = -lblas
endif
ifeq ($(TARGET),SP)
       BLAS = -lblas
endif

#........ LAPI ............
#
ifeq ($(TARGET),LAPI)
LIBLAPIDIR = /usr/lpp/ssp/css/lib
   LIBHAL = -lhal_r
  LIBLAPI = -llapi_r
# LIBCOM := -ltimer $(LIBCOM)
     BLAS = -lessl_r

ifdef LAPI2
EXTRA_LIBS = -L/u2/d3h325/lapi_vector_beta
endif

EXTRA_LIBS += -lxlf90_r -lxlf -lm

EXTRA_LIBS__ += \
   -bnso -bI:/usr/lib/syscalls.exp -L$(LIBLAPIDIR) $(LIBHAL) $(LIBLAPI) \
          -bI:/usr/lib/threads.exp /usr/lpp/ssp/css/libtb3/libmpci_r.a \
          -bI:/usr/lpp/ssp/css/libus/fs_ext.exp \
          /usr/lpp/ppe.poe/lib/libppe_r.a  -lm \
          -bl:/tmp/new.map -lpthreads -lxlf90_r -lxlf -lm
   LIBMPI = -lmpi_r
endif

#...........................................................................
ifeq ($(TARGET),IBM)
       BLAS = -lblas
endif
#...........................................................................
ifeq ($(TARGET),DECOSF)
     CLIB = -lfor -lots -lm
endif
#...........................................................................

#LIBS += $(BLAS) -llinalg $(BLAS)

ifndef OLD_GA
  LIBCOM += -larmci
endif

#LIBCOM += -ltrace

ifdef USE_MPI
   ifdef MPI_LIB
         LIBCOM += -L$(MPI_LIB) $(LIBMPI)
   else
         LIBCOM +=  $(LIBMPI)
   endif
endif

ifdef IWAY
  LIBCOM += -lserver 
endif

LIBCOM += $(EXTRA_LIBS)

ifdef USE_SCALAPACK
  LINALG = $(SCALAPACK)
endif

LINALG += $(BLAS) -llinalg $(BLAS)
