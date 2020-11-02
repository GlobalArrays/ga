# Building GA with SICM
  SICM was downloaded from https://github.com/lanl/SICM.git
In order to build following modules were used. Some other combination of compilers and libraries may also work. 

- numactl-2.0.11
- libpfm4-4.10.1
- hwloc-1.11.9
- openmpi-3.1.0
- llvm

With above modules loaded, SICM was configured as follows:

The command::
```
$ ./configure --with-jemalloc="PATH_TO/jemalloc/install" --prefix="/PATH_TO/SICM/install"
```

Then ga was configured using following command:
 
The command::
```
$ ./configure --with-mpi-pr=1 --prefix="/PATH_TO/GA/install" --x-includes="/PATH_TO/SICM/install/include" --x-libraries="/PATH_TO/SICM/install/lib:/PATH_TO/jemalloc/install/lib" CFLAGS="-DUSE_SICM=1 -I/PATH_TO/SICM/install/include"
```
