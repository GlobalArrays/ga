##############################################################################
# include makefile for nmake under Windows NT
#
##############################################################################

#path and name of MPI library
MPI_LIB_NAME=Cvwmpi.lib
MPI = $(MPI_LIB)\$(MPI_LIB_NAME)

GLOB_DEFINES =-DWIN32
DEFINES = $(GLOB_DEFINES) $(LOC_DEFINES)
GLOB_INCLUDES=
INCLUDES = $(GLOB_INCLUDES) $(LOC_INCLUDES)

AR = link.exe -lib -nologo
ARFLAGS = /out:$(LIBRARY_PATH)

CC = cl -nologo
#COPT =   -Zi
COPT =  -G5 -O2
CFLAGS = $(COPT) $(DEFINES) $(INCLUDES) -Fo"$(OBJDIR)/" -c

FC = fl32 -nologo
#FOPT = -Zi
FOPT = -G5 -Ox
FFLAGS = $(FOPT) -Fo"$(OBJDIR)/" -c

CPP   = $(CC) -EP
CPPFLAGS = $(INCLUDES) $(DEFINES)

.SUFFIXES:
.SUFFIXES:      .obj .s .f .F .c

.c{$(OBJDIR)}.obj:
	$(CC) $(CFLAGS) $<


.F{$(OBJDIR)}.obj: 
        $(CPP) $(CPPFLAGS) $< > $*.for
	$(FC) $(FFLAGS) $*.for 
	del $*.for


.F.for:
        $(CPP) $(CPPFLAGS) $< > $*.for
