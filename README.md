# GLOBAL ARRAYS

Travis: [![Build Status](https://travis-ci.org/GlobalArrays/ga.svg?branch=master)](https://travis-ci.org/GlobalArrays/ga)

## DISCLAIMER

This material was prepared as an account of work sponsored by an agency of the United States Government.  Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, MAKES ANY WARRANTY, EXPRESS OR IMPLIED, OR ASSUMES ANY LEGAL LIABILITY OR RESPONSIBILITY FOR THE ACCURACY, COMPLETENESS, OR USEFULNESS OF ANY INFORMATION, APPARATUS, PRODUCT, SOFTWARE, OR PROCESS DISCLOSED, OR REPRESENTS THAT ITS USE WOULD NOT INFRINGE PRIVATELY OWNED RIGHTS.

## ACKNOWLEDGMENT

This software and its documentation were produced with United States Government support under Contract Number DE-AC06-76RLO-1830 awarded by the United States Department of Energy. The United States Government retains a paid-up non-exclusive, irrevocable worldwide license to reproduce, prepare derivative works, perform publicly and display publicly by or for the US Government, including the right to distribute to other US Government contractors.

The primary current source of funding for development of GA is the Exascale Computing Project.
https://exascaleproject.org/

## GETTING STARTED

If the `configure` script is not present, run `./autogen.sh`.  It will install the necessary versions of autoconf, automake, and libtool into an `autotools` subdirectory and then run `autoreconf` automatically to generate the `configure` script.

The command::

    ./configure && make && make install

should compile the static GA library (libga.a) to use MPI two-sided communication primitives and install headers and libraries to /usr/local/include and /usr/local/lib, respectively.

Please refer to the INSTALL file for generic build instructions.  That is a good place to start if you are new to using "configure; make; make install" types of builds.  Detailed instructions are covered later in this file.

## QUESTIONS/HELP/SUPPORT/BUG-REPORT

Please submit issues to our [GitHub issue tracker](https://github.com/GlobalArrays/ga/issues).  We use Google Groups to host [our discussion forum](https://groups.google.com/forum/#!forum/hpctools), or send an email to hpctools@pnl.gov as an alias for that group.

## WHERE IS THE DOCUMENTATION?

The [GA webpage](http://hpc.pnl.gov/globalarrays/) has the most current versions of the Fortran and C documentation and the User's Manual in the HTML format.

## ABOUT THIS SOFTWARE

This directory contains the Global Arrays (GA), Communications Runtime for Exascale (ComEx) run-time library, Aggregate Remote Memory Copy Interface (ARMCI) run-time library, Memory Allocator (MA), parallel I/O libraries (DRA,EAF,SF), TCGMSG, and TCGMSG-MPI packages bundled together. 

Global Arrays is a portable Non-Uniform Memory Access (NUMA) shared-memory programming environment for distributed and shared memory computers. It augments the message-passing model by providing a shared-memory like access to distributed dense arrays. This is also known as the Partitioned Global Address Space (PGAS) model.

ARMCI provides one-sided remote memory operations used by GA.

DRA (Disk Resident Arrays) is a parallel I/O library that maintains dense two-dimensional arrays on disk. 

SF (Shared Files) is a parallel I/O library that allows noncollective I/O to a parallel file.

EAF (Exclusive Access Files) is parallel I/O library that supports I/O to private files.

TCGMSG is a simple, efficient, but obsolete message-passing library.

TCGMSG-MPI is a TCGMSG interface implementation on top of MPI and ARMCI. 

MA is a dynamic memory allocator/manager for Fortran and C programs.

GA++ is a C++ binding for global arrays.

See file 'COPYRIGHT' for copying conditions.
See file 'INSTALL' for compilation and installation instructions (generic).
See file 'CHANGELOG.md' for a list of major changes in the current release.

## HOW TO BUILD THE PACKAGE?

### autotools build

Please refer to the INSTALL file for generic build instructions.  That is a good place to start if you are new to using "configure; make; make install" types of builds.  The following will cover platform-specific considerations as well as the various optional features of GA.  Customizations to the GA build via the configure script are discussed next.

#### Configuration Options

There are many options available when configuring GA.  Although configure can be safely run within this distributions' root folder, we recommend performing an out-of-source (aka VPATH) build.  This will cleanly separate the generated Makefiles and compiled object files and libraries from the source code.  This will allow, for example, one build using MPI two-sided versus another build using OpenIB for the communication layer to use the same source tree e.g.:
```
mkdir bld_mpi_ts && cd bld_mpi_ts && ../configure
mkdir bld_mpi_openib  && cd bld_mpi_openib  && ../configure --with-openib
```
Regardless of your choice to perform a VPATH build, the following should hopefully elucidate the myriad options to configure.  Only the options requiring additional details are documented here. `./configure --help` will certainly list more options in addition to limited documentation.
```
--disable-f77           Disable Fortran code. This used to be the old
                        GA_C_CORE or NOFORT environment variables which
                        enabled the C++ bindings. However, it is severely
                        broken. There are certain cases where Fortran code is
                        required but this will not inhibit the building of the
                        C++ bindings.  In the future we may be able to
                        eliminate the need for the Fortran compiler/linker.
                        Use at your own risk (of missing symbols at link-time.)
--enable-cxx            Build C++ interface. This will require the C++ linker
                        to locate the Fortran libraries (handled
                        automatically) but user C++ code will require the same
                        considerations (C++ linker, Fortran libraries.)
--disable-opt           Don't use hard-coded optimization flags. GA is a
                        highly-optimized piece of software. There are certain
                        optimization levels or flags that are known to break
                        the software. If you experience mysterious faults,
                        consider rebuilding without optimization by using this
                        option.
--enable-peigs          Enable Parallel Eigensystem Solver interface. This
                        will build the stubs required to call into the peigs
                        library (external). 
--enable-checkpoint     Enable checkpointing.  Untested.  For use with old
                        X-based visualization tool.
--enable-profile        Enable profiling. Not sure what this does, sorry.
--enable-trace          Enable tracing. Not sure what this does, sorry.
--enable-underscoring   Force single underscore for all external Fortran
                        symbols. Usually, configure is able to detect the name
                        mangling scheme of the detected Fortran compiler and
                        will default to using what is detected. This includes
                        any variation of zero, one, or two underscores or
                        whether UPPERCASE or lowercase symbols are used. If
                        you want to force a single underscore which was the
                        default of older GA builds, use this option.
                        Otherwise, you can use the FFLAGS environment variable
                        to override the Fortran compiler's or platform's
                        defaults e.g. configure FFLAGS=-fno-underscoring.
--enable-i4             Use 4 bytes for Fortran INTEGER size. Otherwise, the
                        default INTEGER size is set to the results of the C
                        sizeof(void*) operator.
--enable-i8             Use 8 bytes for Fortran INTEGER size. Otherwise, the
                        default INTEGER size is set to the results of the C
                        sizeof(void*) operator.
--enable-shared         Build shared libraries [default=no]. Useful, for
                        example, if you plan on wrapping GA with an
                        interpreted language such as Python. Otherwise, some
                        systems only support static libraries (or vice versa)
                        but static libraries are the default.
```
For most of the external software packages an optional argument is allowed
(represented as ARG below.) **ARG can be omitted** or can be one or more
whitespace-separated directories, linker or preprocessor directives.  For example::
```
    --with-mpi="/path/to/mpi -lmylib -I/mydir"
    --with-mpi=/path/to/mpi/base
    --with-mpi=-lmpich
```
The messaging libraries supported include MPI, TCGMSG, and TCGMSG over MPI.  If you omit their respective `--with-` option, MPI is the default.  GA can be built to work with MPI or TCGMSG. Since the TCGMSG package is small (comparing to portable MPI implementations), compiles fast, it is still bundled with the GA package.
```
--with-mpi=ARG          Select MPI as the messaging library (default). If you
                        omit ARG, we attempt to locate the MPI compiler
                        wrappers. If you supply anything for ARG, we will
                        parse ARG as indicated above.
--with-tcgmsg           Select TCGMSG as the messaging library; if
                        --with-mpi is also specified then TCGMSG over MPI is
                        used.
--with-blas=ARG         Use external BLAS library; attempt to detect
                        sizeof(INTEGER) used to compile BLAS; if not found, an
                        internal BLAS is built
--with-blas4=ARG        Use external BLAS library compiled with
                        sizeof(INTEGER)==4
--with-blas8=ARG        Use external BLAS library compiled with
                        sizeof(INTEGER)==8
--with-lapack=ARG       Use external LAPACK library. If not found, an internal
                        one is built.
--with-scalapack=ARG    Use external ScaLAPACK library.
```

There are some influential environment variables as documented in `configure --help`, however there are a few that are special to GA.
```
- F77_INT_FLAG
  Fortran compiler flag to set the default INTEGER size. We know about certain
  Fortran flags that set the default INTEGER size, but there will certainly be
  some new (or old) ones that we don't know about. If the configure test to
  determine the correct flag fails, please try setting this variable and
  rerunning configure.

- F2C_HIDDEN_STRING_LENGTH_AFTER_ARGS
  If cross compiling, set to either "yes" (default) or "no" (after string).
  For compatibility between Fortran and C, a Fortran subroutine written in C
  that takes a character string must take an additional argument (one per
  character string) indicating the length of the string. This 'hidden'
  argument appears either immediately after the string in the argument list
  or after all other arguments to the function. This is compiler dependent. We
  attempt to detect this behavior automatically, but in the case of
  cross-compiled systems it may be necessary to specify the less usual after
  string convention the gaf2c/testarg program crashes.
```

#### Special Notes for BLAS

BLAS, being a Fortran library, can be compiled with a default INTEGER size of 4 or a promoted INTEGER size of 8.  Experience has shown us that most of the time the default size of INTEGER used is 4.  In some cases, however, you may have an external BLAS library which is using 8-byte INTEGERs.  In order to correctly interface with an external BLAS library, GA must know the size of INTEGER used by the BLAS library.

configure has the following BLAS-related options: `--with-blas`, `--with-blas4`, and `--with-blas8`.  The latter two will force the INTEGER size to 4- or 8-bytes, respectively.  The first option, `--with-blas`, defaults to 4-byte INTEGERS *however* in the two special cases of using ACML or MKL, it is possible to detect 8-byte INTEGERs automatically.  As documented in the ACML manual, if the path to the library has `_int64` then 8-byte INTEGERs are used. As documented in the MKL manual, if the library is `ilp64`, then 8-byte INTEGERs are used.

You may always override `--with-blas` by specifying the INTEGER size using one of the two more specific options.

#### Cross-Compilation Issues

Certain platforms cross-compile from a login node for a compute node, or one might choose to cross-compile for other reasons. Cross-compiling requires the use of the `--host` option to configure which indicates to configure that certain run-time tests should not be executed. See INSTALL for details on use of the `--host` option.

Two of our target platforms are known to require cross-compilation, Cray XT and IBM Blue Gene.

##### Cray XT

It has been noted that configure still succeeds without the use of the --host flag.  If you experience problems without --host, we recommend
```
    configure --host=x86_64-unknown-linux-gnu
```
And if that doesn't work (cross-compilation is not detected) you must then *force* cross-compilation using both `--host` and `--build` together:
```
    configure --host=x86_64-unknown-linux-gnu --build=x86_64-unknown-linux-gnu
```

#### Compiler Selection 

Unless otherwise noted you can try to overwrite the default compiler names detected by configure by defining F77, CC, and CXX for Fortran (77), C, and C++ compilers, respectively.  Or when using the MPI compilers MPIF77, MPICC, and MPICXX for MPI Fortran (77), C, and C++ compilers, respectively:
```
    configure F77=f90 CC=gcc
    configure MPIF77=mpif90 MPICC=mpicc
```
Although you can change the compiler at make-time it will likely fail.  Many platform-specific compiler flags are detected at configure-time based on the compiler selection. If changing compilers, we recommend rerunning configure as above.

#### After Configuration

By this point we assume you have successfully run configure either from the base distribution directory or from a separate build directory (aka VPATH build.)  You are now ready to run 'make'.  You can optionally run parallel make using the "-j" option which significantly speeds up the build.  If using the MPI compiler wrappers, occasionally using "-j" will cause build failures because the MPI compiler wrapper creates a temporary symlink to the mpif.h header.  In that case, you won't be able to use the "-j" option.  Further, the influential environment variables used at configure-time can be overridden at make-time in case problems are encountered.  For example::
```
    ./configure CFLAGS=-Wimplicit
    ...
    make CFLAGS="-Wimplicit -g -O0"
```
One particularly influential make variable is "V" which controls the verbosity of the make output. This variable corresponds to the `--disable-silent-rules/--enable-silent-riles` configure-time option, but we recommend the make-time variable:
```
    make V=0 (configure  --enable-silent-rules)
    make V=1 (configure --disable-silent-rules)
```

#### Test Programs

Running "make checkprogs" will build most test and example programs.  Note that not all tests are built -- some tests depend on certain features being detected or enabled during configure.  These programs are not intented to be examples of good GA coding practices because they often include private headers.  However, they help us debug or time our GA library.

#### Test Suite

Running "make check" will build most test and example programs (See "make checkprogs" notes above) in addition to running the test suite.  The test suite runs both the serial and parallel tests.  The test suite must know how to launch the parallel tests via the MPIEXEC variable.  Please read your MPI flavor's documentation on how to launch, or if using TCGMSG you will use the "parallel" tool.  For example, the following is the command to launch the test suite when compiled with OpenMPI:
```
    make check MPIEXEC="mpiexec -np 4"
```
All tests have a per-test log file containing the output of the test.  So if the test is global/testing/test.x, the log file would be global/testing/test.log.  The output of failed tests is collected in the top-level log summary test-suite.log.

The test suite will recurse into the ComEx directory and run the ComEx test suite first.  If the ComEx test suite fails, the GA test suite will not run (the assumption here is that you should fix bugs in the dependent library first.)  To run only the GA test suite, type "make check-ga" with the appropriate MPIEXEC variable.

#### Performance Tuning

Setting an environment variable MA_USE_ARMCI_MEM forces MA library to use
ARMCI memory, communication via which can be faster on networks like GM, VIA
and InfiniBand.
