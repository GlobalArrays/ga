LIBRARY_PATH = $(LIB_DISTRIB)\$(LIBRARY)
OBJS=$(OBJ_OPTIMIZE) $(OBJ)
STAMP = WIN32.stamp

$(LIBRARY_PATH): $(INCDIR) $(STAMP) $(OBJDIR) $(LIB_DISTRIB) $(OBJS)
	$(AR) @<<
	$(ARFLAGS) $(OBJS)
<<

$(STAMP): $(HEADERS)
	!copy $** $(INCDIR)
	erase "*.stamp"
	@echo "" > $(STAMP)

"$(LIB_DISTRIB)" :
    if not exist "$(LIB_DISTRIB)/$(NULL)" mkdir "$(LIB_DISTRIB)"

"$(OBJDIR)" :
    if not exist "$(OBJDIR)/$(NULL)" mkdir "$(OBJDIR)"

"$(INCDIR)" :
    if not exist "$(INCDIR)/$(NULL)" mkdir "$(INCDIR)"

clean:
	-@erase /q $(STAMP) *.exe *.ilk *.pdb $(OBJDIR)\*.*  $(LIBRARY_PATH)
	-@if exist "$(OBJDIR)" rmdir "$(OBJDIR)"
	-@if exist "*.stamp" erase /q "*.stamp"
	
