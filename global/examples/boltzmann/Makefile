INCLUDES = -I../../cvs/include
LIBS = -L../../cvs/lib/LAPI
boltz : main.o boltzmann.o setup.o initpar.o properties.o\
        equil.o get_patch.o timestep.o printdat.o vorticity.o
	mpxlf_r main.o boltzmann.o setup.o initpar.o properties.o\
        equil.o get_patch.o timestep.o printdat.o vorticity.o\
        -qextname -O $(LIBS) -lglobal -lma -larmci -ltcgmsg\
        -o boltz
main.o : main.F common
	xlf main.F -qextname -O -c $(INCLUDES)
boltzmann.o : boltzmann.F common
	xlf boltzmann.F -qextname -O -c $(INCLUDES)
setup.o : setup.F common
	xlf setup.F -qextname -O -c $(INCLUDES)
initpar.o : initpar.F common
	xlf initpar.F -qextname -O -c $(INCLUDES)
properties.o: properties.F common
	xlf properties.F -qextname -O -c $(INCLUDES)
equil.o: equil.F common
	xlf equil.F -qextname -O -c $(INCLUDES)
get_patch.o : get_patch.F common
	xlf get_patch.F -qextname -O -c $(INCLUDES)
timestep.o : timestep.F common
	xlf timestep.F -qextname -O -c $(INCLUDES)
printdat.o : printdat.F common
	xlf printdat.F -qextname -O -c $(INCLUDES)
vorticity.o : vorticity.F common
	xlf vorticity.F -qextname -O -c $(INCLUDES)
