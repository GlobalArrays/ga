#
# settings dependent on selection of the message-passing library
# MP_LIBS     - library path 
# MP_INCLUDES - location of header files
# MP_DEFINES  - cpp symbols

#reference to the MPI library name, overwritten by LIBMPI environment variable
# you should also add to it names of any libs on which mpi depends
# e.g., on Compaq with Quadrics network LIBMPI should also add -lelan3
# LIBMPI = -lmpi -lelan3
# 
SKIP_MPILIB_TARGET = LAPI HITACHI BGML
SKIP_MPILIB = mpifrt mpfort mpif77 mpxlf mpif90 ftn
MPI_LIB_NAME = -lmpi
ifeq ($(TARGET),$(findstring $(TARGET),$(SKIP_MPILIB_TARGET)))
   MPI_LIB_NAME = 
endif
ifneq (,$(findstring $(notdir $(FLD)), $(SKIP_MPILIB)))
   MPI_LIB_NAME = 
endif
ifeq ($(notdir $(FC)),mpifrt)
   MPI_LIB_NAME =
endif

ifeq ($(ARMCI_NETWORK),LAPI)
   MPI_LIB_NAME = 
endif

ifeq ($(ARMCI_NETWORK),QUADRICS)
   ifeq ($(TARGET),DECOSF)
     MPI_LOC = /usr/opt/mpi
   else
     MPI_LOC = /usr/lib/mpi
   endif
   MPI_LIB_NAME = -lmpi -lelan
endif
#
#reference to the PVM library name, overwritten by LIBPVM environment variable
ifeq ($(TARGET),CRAY-T3E)
     PVM_LIB_NAME = -lpvm3
else
     PVM_LIB_NAME = -lgpvm3 -lpvm3
endif

ifeq ($(MSG_COMMS),PVM)
  ifdef PVM_INCLUDE
    MP_TMP_INCLUDES = $(PVM_INCLUDE)
  endif
  ifdef PVM_LIB
    MP_LIBS += -L$(PVM_LIB)
  endif
  ifdef LIBPVM
    PVM_LIB_NAME = $(LIBPVM)
  endif
  MP_LIBS += $(PVM_LIB_NAME)
  MP_DEFINES += -DPVM
endif
#
#
ifeq ($(MSG_COMMS),TCGMSG)
  ifdef TCG_INCLUDE
     MP_TMP_INCLUDES = $(TCG_INCLUDE)
  endif
  ifdef TCG_LIB
     MP_LIBS += -L$(TCG_LIB)
  endif
  MP_LIBS += -ltcgmsg
  MP_DEFINES += -DTCGMSG
endif
#
#
ifdef GA_USE_VAMPIR
   MP_DEFINES += -DGA_USE_VAMPIR
   ifdef VT_DEBUG
      MP_DEFINES += -DVT_DEBUG
   endif
   ifdef VT_LIB
      ifdef LIBVT
         MP_LIBS += -L$(VT_LIB) $(LIBVT)
      else
         MP_LIBS += -L$(VT_LIB) -lVT
      endif
#  else
#     echo "Setenv VT_PATH to -L<directory where libVT.a lives>"
   endif
   ifdef VT_INCLUDE
         INCLUDES += -I$(VT_INCLUDE)
   endif

endif

ifeq ($(MSG_COMMS), BGML)
   ifdef BGML_INCLUDE
      MP_TMP_INCLUDES = $(BGML_INCLUDE)
   endif
   ifdef BGML_LIB
      MP_LIBS += -L$(BGML_LIB)
   endif
   ifdef LIBBGML_LIB_NAME
      LIBBGML_LIB_NAME = $(LIBBGML)
   endif
   MP_LIBS += $(BGML_LIB_NAME)
   MP_DEFINES +=  -DBGML
endif

ifeq ($(MSG_COMMS), BGMLMPI)
    ifdef BGML_INCLUDE
       MP_TMP_INCLUDES = $(BGML_INCLUDE)
    endif
    ifdef BGML_LIB
       MP_LIBS += -L$(BGML_LIB)
    endif
    ifdef LIBBGML_LIB_NAME
       LIBBGML_LIB_NAME = $(LIBBGML)
    endif
    MP_LIBS += $(BGML_LIB_NAME)
    MP_DEFINES +=  -DBGML
    MP_DEFINES += -DBGML -DMPI
endif
       
ifeq ($(MSG_COMMS), DCMFMPI)
   ifdef DCMF_INCLUDE
      MP_TMP_INCLUDES = $(DCMF_INCLUDE)
  else
     MP_TMP_INCLUDES = $(MPI_INCLUDE)
  endif
   ifdef DCMF_LIB
      MP_LIBS += -L$(DCMF_LIB)
  else
     MP_LIBS = -L$(MPI_LIB)
   endif
   ifdef LIBDCMF_LIB_NAME
      LIBDCMF_LIB_NAME = $(LIBDCMF)
   endif
   MP_LIBS += $(DCMF_LIB_NAME)
   MP_DEFINES += -DDCMF -DMPI
endif


#
#
ifeq ($(MSG_COMMS),MPI)
  ifdef MPI_INCLUDE
    MP_TMP_INCLUDES = $(MPI_INCLUDE)
  endif
  ifdef MPI_LIB
    MP_LIBS += -L$(MPI_LIB)
  endif
  ifdef LIBMPI
      MPI_LIB_NAME = $(LIBMPI)
  endif
  MP_LIBS += $(MPI_LIB_NAME)
  MP_DEFINES += -DMPI
endif
#
#
ifdef MP_TMP_INCLUDES
  Comma:= ,
  Empty:=
  Space:= $(Empty) $(Empty)
  MP_INCLUDES += -I$(subst $(Comma), $(Space)-I,$(MP_TMP_INCLUDES))
endif
