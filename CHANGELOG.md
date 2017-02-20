# Change Log
All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

This project follows the [Gitflow Workflow model](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

## [Unreleased]
The Unreleased section will be empty for tagged releases. Unreleased functionality appears in the develop branch.

## 5.1 - 2012-02
-Added 
  - Unified "NGA" prefix for all functions.
  - New profiling interface and weak symbol interposition support.
  - Support for struct data types using the new NGA_Register_type(), NGA_Get_field() and NGA_Put_field() functions.
  - ga-config for 3rd party software to query compilation flags used for GA.
  - Global Arrays in NumPy (GAiN) interface for a NumPy work-alike with a Global Arrays backend.
  - GA_MPI_Comm() and other functions to retrieve MPI_Comm object associated with a GA processor group.
  - MPI-MT (MPI_THREAD_MULTIPLE) port for use when a native port is not available.
  - armci_msg_finalize() to abstract the message passing function required for application termination.
  - Ability for EAF_Open() to use MA memory operations instead of I/O operations.
- Changed
  - ARMCI directory structure.
  - NGA_Add_patch() algorithm to use less memory.
  - tascel to no longer be part of the top-level configure (must be installed separately).
  - Python base module from "ga" to "ga4py" since we now have the submodules ga4py.ga and ga4py.gain.
  - autotools build to use autoconf-2.68 and libtool-2.4.2.
  - ARMCI Fortran sources to use `integer*4` type rather than an integer size compiler flag.
- Fixed
  - Numerous configure and source bugs with our ScaLAPACK integration.
  - Bug in NGA_Matmul_patch().
  - Numerous configure bugs.
  - Numerous Makefile bugs.
  - Support for large files.
- Improved
  - Internal code, reducing the amount of dereferenced pointers.
  - Restriction on calling GA_Initalize() before messaging layer -- GA_Initalize(), ARMCI_Init(), MPI_Init()/tcg_pbegin() can be called in any order.
- Removed
  - Deprecated interconnects Myrinet GM, VIA, Mellanox Verbs API, Quadrics Elan-3, Quadrics Elan-4.
  - Vampir support.
  - KSR and PARAGON support.

## 5.0.3 - 2012-02
- Added
  - Support for Cray compilers.
- Fixed
  - Shared library linking.
  - A few *critical* bugs in GA_Duplicate().
  - Bugs in strided get/put/acc routines.
  - Bugs in GPC support.
  - Numerous compilation warnings.
  - Numerous valgrind warnings.
  - Numerous configure bugs.
  - Numerous Makefile bugs.
  - Numerous bugs in Python interface.
  - Bug in GA_Patch_enum().
  - Bug in TCGMSG-MPI nxtval().
  - Latency reporting in perf.x benchmark.
  - Fortran ordering bug in NGA_Scatter(), NGA_Scatter_acc(), and NGA_Gather().
- Improved
  - BGP configure.
  - TCGMSG-MPI build.
  - Test suite.
  - Numerous inefficiencies in Python interface.

## 5.0.2 - 2011-03
- Added
  - Added support for Sun Studio compilers.
  - Added support for AMD Open64 compilers.
- Changed
  - ARMCI RMW interface now uses void pointer.
  - GA_Patch_enum() now uses void pointer.
- Fixed
  - Bugs in processor groups code.
  - Numerous compilation warnings.
  - Numerous configure bugs.
  - Numerous Makefile bugs.
  - Bug in GA_Unpack().
  - Bug in GA_Dgemm() concerning transpose.
  - Numerous bugs in Python interface.
- Improved
  -  ga_scan_copy() and ga_scan_add().

## 5.0.1 - 2011-01
- Fixed
  - Numerous configure bugs.
  - Numerous Makefile bugs.
  - Numerous bugs in test suite.
  - Atomics bug.
  - Numerous tascel bugs.
  - Bug in single complex matrix multiply.
  - Bug in destruction of mutexes.
  - Bug in process group collectives.
  - Bug in GA_Terminate().
- Improved
  - Configure for NEC and HPUX.

## 5.0 - 2010-11
- Now built using GNU autotools (autoconf,automake,libtool)
- Restricted arrays (see user manual)
- ARMCI runtime enhancements
  - On-demand connection management
  - Improved scalability for fence
- New Python interface
- Task Scheduling Library (tascel)

## 5.0b - 2010-07
- Now built using GNU autotools (autoconf,automake,libtool)

## 4.3 - 2010-05
- Optimized portals port to scale upto 200K procs
- Optimized OpenIB port
- BlueGene/P
- Support for Sparse Data Operations
  - (See GA user manual - Chapter 11 for more details)

## 4.2 - 2009-07
- Support for several new platforms
- Optimized portals port for Cray XT5
- BlueGene/P
- Optimized OpenIB port
- Support for Sparse Data Operations
  - (See GA user manual - Chapter 11 for more details)

## 4.1 - 2008-05
- Support for several new platforms
  - Cray XT4
  - BlueGene/L, BlueGene/P
  - OpenIB network
- Optimized one-sided non-blocking operations
- New networks. i.e. ARMCI_NETWORK
  - OPENIB
  - PORTALS
  - MPI-SPAWN (one-sided communication thru' MPI2 Dynamic Process management and Send/Recv)

## 4.0 - 2006-04
- Support for multilevel parallelism: processor group awareness
- GA_Dgemm matrix multiplication based on SRUMMA
- Support for large arrays (a terabyte Global Array now possible)
- Optimized one-sided non-blocking operations
- Supports various platforms (Crays, IBM SPs, SGI Altix, ...) and interconnects (Myrinet, Quadrics, Infiniband, ...)

[Unreleased]: https://github.com/jeffdaily/parasail/compare/v5.6...develop
