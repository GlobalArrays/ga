# Please specify location of X include files and libs
# XLOC = -I/usr

  XLOC = /usr/X11R6

INCLUDES = -I$(XLOC)/include
    COPT = -g
  CFLAGS = $(COPT) $(INCLUDES)

ifeq ($(TARGET), SGI_N32)
  CFLAGS += -32
endif

  XLIBS = -L$(XLOC)/lib  -lXaw -lXmu -lXt -lX11 -lXau -lXdmcp -lXext
  LIBS = 
  RM = rm
OBJ_LOC = .

  CC = cc

THE_OBJS =	\
		$(OBJ_LOC)/xregion_colormap.o\
		$(OBJ_LOC)/xregion_dialog.o\
		$(OBJ_LOC)/xregion_display.o\
		$(OBJ_LOC)/xregion_fileio.o\
		$(OBJ_LOC)/xregion_main.o\
		$(OBJ_LOC)/xregion_pixregion.o\
		$(OBJ_LOC)/xregion_scrollbars.o\
		$(OBJ_LOC)/xregion_util.o\
		$(OBJ_LOC)/xregion_overview.o\
		$(OBJ_LOC)/xregion_view.o\
		$(OBJ_LOC)/xregion_xcmap.o

default :	xregion

xregion :		$(THE_OBJS)
		$(CC) $(CFLAGS) $(THE_OBJS) -o xregion  $(XLIBS) -lm

clean:
	$(RM) -rf $(THE_OBJS) xregion

cleanstamp:
	$(RM) -rf *.stamp
