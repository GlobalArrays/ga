# Change Log
The format is based on [Keep a Changelog](http://keepachangelog.com/).

All notable changes to this project will be documented in this file.
This project adheres to [Semantic Versioning](http://semver.org/).

This project follows the [Gitflow Workflow model](https://www.atlassian.com/git/tutorials/comparing-workflows/gitflow-workflow).

## [Unreleased]
The Unreleased section will be empty for tagged releases. Unreleased functionality appears in the develop branch.

## [5.8] - 2020-09-30
- Known Bugs
  - The MPI RMA port remains unreliable for many MPI implementations. Open MPI
    still reports many failures in the test suit. Intel MPI is better but still
    reports several failures. It is recommended to use the latest MPI
    implementations available.
- Added
  - Version function that can be used to report the current version, subversion
    and patch numbers of the current release
  - Overlay option for creating new GAs on top of existing GAs
  - The number of progress ranks per node in the progress ranks runtime is now
    configurable
  - Functions for duplicating process groups and returning a process group that
    only contains the calling process
  - 64-bit versions of block-cyclic data distribution functions to
    C interface
  - Non-blocking test function
  - Read-only property based on caching 
  - GA name can be recovered from handle 
  - Added profiling capabilities to the GA branch that automatically generates
    a log file in the running directory. This can be controlled with GAW_FILE_PREFIX
    environment variable to add a prefix for the log files and the GAW_FMT
    environment variable to create a CSV format or human readable format. The
    default format is human readable.
      - For autotools, add --enable-profile=1 in the configure line
      - For CMake add -DENABLE_PROFILING=ON
- Changed
  - Non-blocking handle management was completely revamped. This simplifies
    implementation and removes some bugs. The number of outstanding non-blocking
    calls was increased to 256
  - Modified internal function that computes rank of processors on the world
    communicator so that it does not use the MPI_Comm_translate_ranks function.
    This function is implemented with a loop that scales as the square of the
    number of processors and is very slow at large processor counts
  - modified internal iterators so that block cyclic data distributions work on
    processor groups
  - Improved CMake build
  - Modified ga_print_distribution so that it works on block-cyclic data
    distributions
- Fixed
  - Fixed a non-blocking error that was showing up in nbtest.x
  

## [5.7] - 2018-03-30
- Known Bugs
  - Some combinations of MPI implementations with the MPI RMA and PR
    ports fail. Recommended to use latest MPI implementations available.
- Added
  - Tiled data layout
  - Read-only property type using replication across SMP nodes
- Changed
  - GA is now thread safe
  - MPI3 implementation based on MPI RMA now uses data types in MPI
    calls by default. This is higher performing but not as reliable as
    using multiple contiguous data transfers. The build can be
    configured to use contiguous transfers if data types are not working
    for your MPI implementation.
  - ComEx MPI-PR now uses MPI data types in strided put and get calls
    by default. To enable the old packed behavior, set the following
    environment variables to 0.
    
    - COMEX_ENABLE_PUT_DATATYPE
    - COMEX_ENABLE_GET_DATATYPE
    
    Additionally, the original packing implementation is faster for smaller
    messages. Two new environment variables control at which point the MPI
    data types are used.
    
    - COMEX_PUT_DATATYPE_THRESHOLD. Default 8192.
    - COMEX_GET_DATATYPE_THRESHOLD. Default 8192.
- Fixed
  - Message sizes exceeding 2GB now work correctly
  - Mirrored Arrays now distributes data across SMP nodes for
    ComEx-based runtimes
  - Matrix multiply works for non-standard data layouts (may not be
    performant)
- Closed Issues
  - \[#48] Message sizes exceeding 2GB may not work correctly

## [5.6.5] - 2018-03-29
- Known Bugs
  - [\#48] Message sizes exceeding 2GB may not work correctly
- Added
  - Environment variables to control internal ComEx MPI-PR settings
    - COMEX_MAX_NB_OUTSTANDING. Default 8.
      The maximum number of concurrent non-blocking operations.
    - COMEX_STATIC_BUFFER_SIZE. Default 2097152 bytes.
      Some ComEx operations require a temporary buffer. Any message larger than this size will dynamically allocate and free a new buffer to hold the larger message.
    - COMEX_EAGER_THRESHOLD. Default -1.
      Small messages can be sent as part of other internal ComEx operations. Recommended to set this to less than or equal to the corresponding MPI eager/rendezvous threshold cutoff.
    - COMEX_ENABLE_PUT_SELF. Default 1 (on). Contiguous put will use memcpy when target is same as originator.
    - COMEX_ENABLE_GET_SELF. Default 1 (on). Contiguous get will use memcpy when target is same as originator.
    - COMEX_ENABLE_ACC_SELF. Default 1 (on). Contiguous acc will use memcpy when target is same as originator.
    - COMEX_ENABLE_PUT_SMP. Default 1 (on). Contiguous put will use memcpy when target is on the same host via shared memory.
    - COMEX_ENABLE_GET_SMP. Default 1 (on). Contiguous get will use memcpy when target is on the same host via shared memory.
    - COMEX_ENABLE_ACC_SMP. Default 1 (on). Contiguous acc will use memcpy when target is on the same host via shared memory.
    - COMEX_ENABLE_PUT_PACKED. Default 1 (on). Strided put will pack the data into a contiguous buffer.
    - COMEX_ENABLE_GET_PACKED. Default 1 (on). Strided get will pack the data into a contiguous buffer.
    - COMEX_ENABLE_ACC_PACKED. Default 1 (on). Strided acc will pack the data into a contiguous buffer.
    - COMEX_ENABLE_PUT_IOV. Default 1 (on). Vector put will pack the data into a contiguous buffer.
    - COMEX_ENABLE_GET_IOV. Default 1 (on). Vector get will pack the data into a contiguous buffer.
    - COMEX_ENABLE_ACC_IOV. Default 1 (on). Vector acc will pack the data into a contiguous buffer.
    - COMEX_MAX_MESSAGE_SIZE. Default INT_MAX. All use of MPI will keep buffers less than this size. Sometimes useful in conjunction with eager thresholds to force all use of MPI below the eager threshold.
  - armci-config and comex-config added
    - --blas_size
    - --use_blas
    - --network_ldflags
    - --network_libs
  - ga-config added
    - --blas_size
    - --scalapack_size
    - --use_blas
    - --use_lapack
    - --use_scalapack
    - --use_peigs
    - --use_elpa
    - --use_elpa_2015
    - --use_elpa_2016
    - --network_ldflags
    - --network_libs
- Changed
  - Removed case statement from install-autotools.sh
- Fixed
  - install-autotools.sh works on FreeBSD
  - patch locally built m4 for OSX High Sierra
- Closed Issues Requests
  - Scalapack with 8-byte integers? [\#93]
  - Please clarify what is "peigs" library [\#96]
  - additional arguments for bin/ga-config describing the presence of Peigs and/or Scalapack interfaces [\#99]
  - additional arguments for bin/ga-config describing the integer size of the Blas library used [\#100]

## [5.6.4] - 2018-03-21
- Known Bugs
  - [\#48] Message sizes exceeding 2GB may not work correctly
- Added
  - armci-config and comex-config scripts to install.
- Changed
  - install-autotools.sh installs all autotools regardless of existing versions
  - configure tests needing mixed C/Fortran code now use C linker
- Fixed
  - Test suite was broken when GA was cross-compiled
  - eliop FreeBSD patch from Debichem
  - Locally installed automake is patched to work with newer perl versions
  - MPI-PR increased limit on number of possible comex_malloc invocations
- Closed Pull Requests
  - \[#92] eliop FreeBSD patch from Debian maintainers of the NWChem Package
- Closed Issues Requests
  - \[#82] Fortran failure on theta
  - \[#88] Automake regex expression broken for Perl versions >=5.26.0
  - \[#89] autogen fails on Mac 10.12
  - \[#90] configure script fails when using clang-4/5 + gfortran 6.3 compilers on Linux
  - \[#95] comex/src-mpi-pr/comex.c:996: _generate_shm_name: Assertion 'snprintf_retval < (int)31' failed

## [5.6.3] - 2017-12-08
- Known Bugs
  - [\#48] Message sizes exceeding 2GB may not work correctly
- Fixed
  - Critical bug, incorrect use of MPI_Comm_split() might prevent startup
    in the following ComEx ports.
    - MPI-PR
    - MPI-PT
    - MPI-MT

## [5.6.2] - 2017-09-29
- Known Bugs
  - [\#48] Message sizes exceeding 2GB may not work correctly
- Fixed
  - Bug in MPI-PT comex_malloc().
  - Revert ARMCI contiguous check due to regression.
  - ELPA updates.
  - ScaLAPACK updates, including case for large matrices.
  - ComEx OFI updates from Intel.
  - Improved configure tests for LAPACK.
  - Improved travis tests.
- Closed Pull Requests
  - [\#87] fix for case for large matrices when nprocs0/(2**I) is always larger than 1

## [5.6.1] - 2017-05-30
- Known Bugs
  - [\#48] Message sizes exceeding 2GB may not work correctly
- Added
  - New ELPA 2015 and 2016 eigensolver interfaces
- Changed
  - autogen.sh unconditionally runs install-autotools.sh
  - install-autotools.sh downloads latest config.guess and config.sub
  - Additional LAPACK symbols are now tested for during configure
- Fixed
  - comex_fence_proc() fixed for MPI-MT, MPI-PT, MPI-PR ports
  - configure --disable-fortran now works again
  - ComEx openib port was missing comex_nbacc symbol
  - Added $(BLAS_LIBS) to libcomex LIBADD to capture BLAS library dependency
  - EISPACK no longer enabled by default; --enable-eispack now works correctly
  - Shared memory name limit on OSX is now followed
  - comex_unlock() race condition
  - install-autotools.sh properly updates $PATH during build
  - install-autotools.sh alternate download location when FTP is blocked
  - patches to generated configure scripts for -lnuma
  - CMake build did not install some fortran headers
  - TravisCI: don't fail when brew install is a no-op
- Closed Pull Requests
  - [\#34] Fix installation of autotools if not present
  - [\#53] new ELPA 2016 eigensolver 2stage interface
  - [\#54] new ELPA 2016 eigensolver interface for the Hotfix/5.6.1 branch
  - [\#55] curl for download when wget not installed
  - [\#58] comex/ofi: max_bytes_in_atomic may not fit in int
- Closed Issues
  - [\#1] Incorporating GAMESS Patch
  - [\#5] Compiler error with --with-ofi
  - [\#9] Adding documentation for GA compilation on Windows
  - [\#25] CMake not building MA fortran wrappers
  - [\#30] Disable Fortran not working
  - [\#33] GA 5.6 release - autotools are downloaded and built even when latest versions exist
  - [\#38] #ifdef ENABLE_EISPACK should be #if ENABLE_EISPACK
  - [\#39] libcomex missing optional BLAS dependency
  - [\#41] develop branch and m4 version on cascade
  - [\#44] Comex OpenIB missing library symbol
  - [\#49] autogen.sh fails when only automake needs to be built
  - [\#50] install-autotools.sh on osx might choke if no timeout tool
  - [\#56] comex_fence_proc() is no-op in MT, PT, PR
  - [\#57] process groups sometimes fail for MPI-PT port

## [5.6] - 2017-04-04
- Added
  - Port for MPI-3 one-sided (--with-mpi3).
  - CMake build.
  - More complete test coverage.
- Changed
  - Initial shared library versioning.
- Fixed
  - Updates to ComEx/OFI provided by Intel.
  - ComEx/MPI-PR added uid and pid to shmem name.
- Closed Pull Requests
  - [\#6]  Comex/OFI: updated initialization of OmniPath provider
  - [\#10] comex/ofi: fixed EP initialization
  - [\#11] COMEX/OFI: added readme file for comex/ofi provider

## 5.5 - 2016-08
- Added
  - Port for libfabric (--with-ofi) via ComEx. This adds native support for Intel Omnipath.
- Fixed
  - Numerous bug fixes.

## 5.4 - 2016-04
- Fixed
  - Numerous bug fixes.
  - Performed license/copyright audit of source files.
- Removed
  - BGML and DCMF ports. Use MPI multithreading instead.

## 5.4b - 2015-05
- Added
  - Port for MPI progress ranks (--with-mpi-pr) via ComEx.
  - Port for MPI multithreading with progress threads (--with-mpi-pt) via ComEx.
  - Port for MPI multithreading (--with-mpi-mt) via ComEx.
- Changed
  - Default network interface from SOCKETS to MPI two-sided (--with-mpi-ts) via ComEx.
- Improved
  - ScaLAPACK and ELPA integration.
- Removed
  - Replaced EISPACK with LAPACK/BLAS.

## 5.3 - 2014-04
- Fixed
  - Bug where incorrect BLAS integer size was used in ComEx.
  - Incompatibilities between this and the 5.2 release with respect to NWChem.
- Testing
  - Validated this software with the NWChem 6.3 sources.

## 5.3b - 2013-12
- Added
  - Port for Portals4 (configure --with-portals4).  When linking to the Portals4 reference implementation, it is highly recommended that the ummunotify driver is installed. Otherwise, memory registration errors and/or dropped messages may occur. This behavior can be verified using the PTL_DEBUG=1 and PTL_LOG_LEVEL=2 Portals4 environment variables.
  - ARMCI profiling to ComEx.
- Changed
  - autotool scripts now latest versions.

## 5.2 - 2012-08
- Added
  - The Communications Runtime for Extreme Scale (ComEx) software to the GA release bundle. ComEx provides the following features to GA:
      - Port for Cray XE and XC (configure --with-dmapp).
      - Port for MPI-1 two-sided (configure --with-mpi-ts).
  - Support for externally linkable ARMCI (configure --with-armci=...)
  - Ability for users to select a different IB device using the ARMCI_OPENIB_DEVICE environment variable.
  - Ability for users to specify upper bound on ARMCI shared memory. Added ARMCI_DEFAULT_SHMMAX_UBOUND which is set and parsed at configure time.
- Changed
  - How users link their applications. You now need "-lga -larmci" since libarmci is possibly an external dependency (see above).
- Fixed
  - Support for Intel/QLogic IB thanks to patches from Dean Luick, Intel.
- Improved
  - BLAS library parsing for ACML and MKL ('-mkl').
  - ScaLAPACK integration thanks to Edo Apra.

## 5.1.1 - 2012-10
- Added
  - A wrapper for fsync to SF library.
  - MA_ACCESS_INDEX_TYPE to ma library.
  - Missing Python C sources.
- Changed
  - Atomic operations.
- Fixed
  - Numerous bugs for compilation on IBM AIX, as well as IBM xl compilers.
  - Many warnings reported by Intel compilers.
  - Integer overflow for indexing large arrays during accumulate.
  - Bug in GA_Zgemm64.
  - Ghosts hanging.
- Removed
  - A few debugging print statements from pario.

## 5.1 - 2012-02
- Added
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

[Unreleased]: https://github.com/GlobalArrays/ga/compare/v5.7...develop
[5.7]: https://github.com/GlobalArrays/ga/compare/v5.6.5...v5.7
[5.6.5]: https://github.com/GlobalArrays/ga/compare/v5.6.4...v5.6.5
[5.6.4]: https://github.com/GlobalArrays/ga/compare/v5.6.3...v5.6.4
[5.6.3]: https://github.com/GlobalArrays/ga/compare/v5.6.2...v5.6.3
[5.6.2]: https://github.com/GlobalArrays/ga/compare/v5.6.1...v5.6.2
[5.6.1]: https://github.com/GlobalArrays/ga/compare/v5.6...v5.6.1
[5.6]: https://github.com/GlobalArrays/ga/releases/tag/v5.6

[\#100]: https://github.com/GlobalArrays/ga/issues/100
[\#99]: https://github.com/GlobalArrays/ga/issues/99
[\#98]: https://github.com/GlobalArrays/ga/issues/98
[\#97]: https://github.com/GlobalArrays/ga/issues/97
[\#96]: https://github.com/GlobalArrays/ga/issues/96
[\#95]: https://github.com/GlobalArrays/ga/issues/95
[\#94]: https://github.com/GlobalArrays/ga/issues/94
[\#93]: https://github.com/GlobalArrays/ga/issues/93
[\#92]: https://github.com/GlobalArrays/ga/pull/92
[\#91]: https://github.com/GlobalArrays/ga/pull/91
[\#90]: https://github.com/GlobalArrays/ga/issues/90
[\#89]: https://github.com/GlobalArrays/ga/issues/89
[\#88]: https://github.com/GlobalArrays/ga/issues/88
[\#87]: https://github.com/GlobalArrays/ga/pull/87
[\#86]: https://github.com/GlobalArrays/ga/issues/86
[\#85]: https://github.com/GlobalArrays/ga/issues/85
[\#84]: https://github.com/GlobalArrays/ga/issues/84
[\#83]: https://github.com/GlobalArrays/ga/issues/83
[\#82]: https://github.com/GlobalArrays/ga/issues/82
[\#81]: https://github.com/GlobalArrays/ga/pull/81
[\#80]: https://github.com/GlobalArrays/ga/pull/80
[\#79]: https://github.com/GlobalArrays/ga/pull/79
[\#78]: https://github.com/GlobalArrays/ga/pull/78
[\#77]: https://github.com/GlobalArrays/ga/pull/77
[\#76]: https://github.com/GlobalArrays/ga/pull/76
[\#75]: https://github.com/GlobalArrays/ga/pull/75
[\#74]: https://github.com/GlobalArrays/ga/pull/74
[\#73]: https://github.com/GlobalArrays/ga/pull/73
[\#72]: https://github.com/GlobalArrays/ga/pull/72
[\#71]: https://github.com/GlobalArrays/ga/pull/71
[\#70]: https://github.com/GlobalArrays/ga/pull/70
[\#69]: https://github.com/GlobalArrays/ga/pull/69
[\#68]: https://github.com/GlobalArrays/ga/pull/68
[\#67]: https://github.com/GlobalArrays/ga/pull/67
[\#66]: https://github.com/GlobalArrays/ga/pull/66
[\#65]: https://github.com/GlobalArrays/ga/pull/65
[\#64]: https://github.com/GlobalArrays/ga/issues/64
[\#63]: https://github.com/GlobalArrays/ga/pull/63
[\#62]: https://github.com/GlobalArrays/ga/pull/62
[\#61]: https://github.com/GlobalArrays/ga/issues/61
[\#60]: https://github.com/GlobalArrays/ga/pull/60
[\#59]: https://github.com/GlobalArrays/ga/pull/59
[\#58]: https://github.com/GlobalArrays/ga/pull/58
[\#57]: https://github.com/GlobalArrays/ga/issues/57
[\#56]: https://github.com/GlobalArrays/ga/issues/56
[\#55]: https://github.com/GlobalArrays/ga/pull/55
[\#54]: https://github.com/GlobalArrays/ga/pull/54
[\#53]: https://github.com/GlobalArrays/ga/pull/53
[\#52]: https://github.com/GlobalArrays/ga/issues/52
[\#51]: https://github.com/GlobalArrays/ga/issues/51
[\#50]: https://github.com/GlobalArrays/ga/issues/50
[\#49]: https://github.com/GlobalArrays/ga/issues/49
[\#48]: https://github.com/GlobalArrays/ga/issues/48
[\#47]: https://github.com/GlobalArrays/ga/issues/47
[\#46]: https://github.com/GlobalArrays/ga/issues/46
[\#45]: https://github.com/GlobalArrays/ga/issues/45
[\#44]: https://github.com/GlobalArrays/ga/issues/44
[\#43]: https://github.com/GlobalArrays/ga/issues/43
[\#42]: https://github.com/GlobalArrays/ga/issues/42
[\#41]: https://github.com/GlobalArrays/ga/issues/41
[\#40]: https://github.com/GlobalArrays/ga/issues/40
[\#39]: https://github.com/GlobalArrays/ga/issues/39
[\#38]: https://github.com/GlobalArrays/ga/issues/38
[\#37]: https://github.com/GlobalArrays/ga/pull/37
[\#36]: https://github.com/GlobalArrays/ga/issues/36
[\#35]: https://github.com/GlobalArrays/ga/issues/35
[\#34]: https://github.com/GlobalArrays/ga/pull/34
[\#33]: https://github.com/GlobalArrays/ga/issues/33
[\#32]: https://github.com/GlobalArrays/ga/issues/32
[\#31]: https://github.com/GlobalArrays/ga/issues/31
[\#30]: https://github.com/GlobalArrays/ga/issues/30
[\#29]: https://github.com/GlobalArrays/ga/issues/29
[\#28]: https://github.com/GlobalArrays/ga/issues/28
[\#27]: https://github.com/GlobalArrays/ga/issues/27
[\#26]: https://github.com/GlobalArrays/ga/issues/26
[\#25]: https://github.com/GlobalArrays/ga/issues/25
[\#24]: https://github.com/GlobalArrays/ga/issues/24
[\#23]: https://github.com/GlobalArrays/ga/issues/23
[\#22]: https://github.com/GlobalArrays/ga/issues/22
[\#21]: https://github.com/GlobalArrays/ga/issues/21
[\#20]: https://github.com/GlobalArrays/ga/issues/20
[\#19]: https://github.com/GlobalArrays/ga/issues/19
[\#18]: https://github.com/GlobalArrays/ga/issues/18
[\#17]: https://github.com/GlobalArrays/ga/issues/17
[\#16]: https://github.com/GlobalArrays/ga/issues/16
[\#15]: https://github.com/GlobalArrays/ga/issues/15
[\#14]: https://github.com/GlobalArrays/ga/issues/14
[\#13]: https://github.com/GlobalArrays/ga/issues/13
[\#12]: https://github.com/GlobalArrays/ga/issues/12
[\#11]: https://github.com/GlobalArrays/ga/pull/11
[\#10]: https://github.com/GlobalArrays/ga/pull/10
[\#9]: https://github.com/GlobalArrays/ga/issues/9
[\#8]: https://github.com/GlobalArrays/ga/issues/8
[\#7]: https://github.com/GlobalArrays/ga/issues/7
[\#6]: https://github.com/GlobalArrays/ga/pull/6
[\#5]: https://github.com/GlobalArrays/ga/issues/5
[\#4]: https://github.com/GlobalArrays/ga/pull/4
[\#3]: https://github.com/GlobalArrays/ga/issues/3
[\#2]: https://github.com/GlobalArrays/ga/issues/2
[\#1]: https://github.com/GlobalArrays/ga/issues/1

