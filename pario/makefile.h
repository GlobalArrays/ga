
OSNAME =  $(shell uname | awk '{ print $1}')

#under AIX, AIX52 is defined to use POSIX API in AIX 5.2.0.0 or greater
ifeq ($(OSNAME),AIX) 
     LIB_DEFINES += $(shell oslevel | awk -F. \
                      '{ if ($$1 > 5 || ($$1 == 5 && $$2 > 1))\
                      print "-DAIX52" }')
ifdef USE_OLDAIO
     LIB_DEFINES += -D_AIO_AIX_SOURCE
endif
endif

#under AIX, there can be problems with AIO and large files
ifdef LARGE_FILES

  ifeq ($(OSNAME),AIX)
    LIB_DEFINES += $(shell oslevel | awk -F. \
              '{ if ($$1 > 4 || ($$1 == 4 && $$2 > 1))\
               print "-D_LARGE_FILES -D_LARGE_FILE_API" }')

#   asynchronous I/O with large files supported starting with 4.2.1
#   However, there is a bug in IBM libs on PNNL system that prevents us
#   from using AIO under 4.2.1 :-)
#
    AIO_LARGE_FILES = $(shell oslevel | awk -F. \
               '{ if ($$1 == 4 && $$2 == 2 && $$3 <= 0 ) \
               print  "NO"}')
  endif
  ifeq ($(TARGET), SOLARIS)
    LIB_DEFINES += -D_LARGEFILE_SOURCE -D_FILE_OFFSET_BITS=64
    CC += $(shell getconf LFS_CFLAGS)
  endif  
#
# LINUX: kernel 2.4 is needed
#
  ifeq ($(TARGET), LINUX)
    LIB_DEFINES += -D_LARGEFILE64_SOURCE
    LIB_DEFINES += $(shell getconf LFS_CFLAGS)
  endif  

#
# HP targets tested on HPUX 11.0
#
  ifeq ($(TARGET), HPUX)
    LIB_DEFINES += -D_LARGEFILE64_SOURCE 
    LIB_DEFINES += $(shell getconf XBS5_ILP32_OFFBIG_CFLAGS)
  endif  

  ifeq ($(TARGET), HPUX64)
    LIB_DEFINES +=  -D_LARGEFILE64_SOURCE 
    LIB_DEFINES += $(shell getconf XBS5_LP64_OFF64_CFLAGS)
  endif  

  LIB_DEFINES += -DLARGE_FILES
endif

ifdef LIB_TARGETS
# HPIODIR is used as a filename prefix in test programs
ifdef HPIODIR
 LIB_DEFINES += -DHPIODIR=\'$(HPIODIR)/\'
endif
ifeq ($(TARGET), DECOSF)
  LOC_LIBS += -laio -lpthreads
endif
endif
