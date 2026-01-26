# Conversation 3: Fixing Autotools Build System for OpenSHMEM

## Date
January 26, 2026

## Problem Statement
After successfully implementing `comex_rmw` and `comex_create_mutexes` in the OpenSHMEM backend, and integrating OpenSHMEM as a network backend in the COMEX build system, we encountered issues with the top-level Global Arrays autotools build when using `--with-oshmem`.

## Issues Encountered

### 1. Compiler Selection Issues
**Problem**: The build system was using `mpicc` instead of `oshcc` after running `./configure --with-oshmem CC=oshcc`.

**Root Cause**: The `GA_PROG_MPICC` macro in `m4/ga_mpicc.m4` was searching for MPI compilers and finding system `mpicc` instead of using the pre-set `CC=oshcc`. When `CC` was set but `MPICC` was unset, the macro would clear `CC` and search for compilers, finding `mpicc`.

**Solution**: Modified `configure.ac` to set both `CC` and `MPICC` when OpenSHMEM is detected:
```bash
CC="$OSHCC"
MPICC="$OSHCC"
CXX="$OSHCXX"
MPICXX="$OSHCXX"
F77="$OSHFC"
MPIF77="$OSHFC"
with_mpi_wrappers=yes
```

This tells the GA build system to treat OpenSHMEM wrappers like MPI wrappers, allowing `GA_PROG_MPICC` to work correctly.

### 2. Type Detection Failures
**Problem**: Configure was incorrectly defining `size_t`, `off_t`, and `pid_t` in `config.h`:
```c
#define size_t unsigned int    // Should be unsigned long or undefined
#define off_t long int         // Should use system definition
#define pid_t __int64          // Windows-specific type on Linux
```

**Root Cause**: The autoconf type detection macros (`AC_TYPE_SIZE_T`, etc.) were running before the compiler environment was properly set up with `oshcc`.

**Solution**: By setting `with_mpi_wrappers=yes` and letting `GA_PROG_MPICC` handle the compiler setup, the type detection ran with the correct compiler environment. Result in `config.h`:
```c
/* #undef size_t */
/* #undef off_t */
/* #undef pid_t */
```

### 3. MPI Library Linking Errors
**Problem**: When running `make checkprogs`, Fortran test programs failed to link with errors like:
```
undefined reference to `mpi_init_'
undefined reference to `mpi_initialized_'
undefined reference to `mpi_finalized_'
undefined reference to `mpi_finalize_'
```

**Root Cause**: The test file `global/testing/ffflush.F` calls MPI functions for initialization and finalization checks. When using OpenSHMEM with `--with-oshmem`, we excluded `GA_MP_LIBS` from the link line (thinking we didn't need MPI), but the OpenSHMEM wrappers weren't automatically providing the MPI Fortran libraries.

**Analysis**: 
- `GA_MP_LIBS` was empty when using `--with-oshmem` because we skipped `GA_MPI_UNWRAP`
- Even after adding `GA_MPI_UNWRAP` for OpenSHMEM, it returned empty because `oshcc` is itself a wrapper
- The OpenSHMEM wrappers from OpenMPI include MPI support, but the MPI Fortran interface library (`libmpi_mpifh`) wasn't being linked automatically

**Solution**: Added explicit MPI library linking when using OpenSHMEM in `Makefile.am`:
```makefile
LDADD += libga.la

# When using OpenSHMEM, add MPI libraries explicitly for test programs
# that call MPI functions (like ffflush.F with mpi_finalize)
if WITH_OSHMEM
LDADD += -lmpi_mpifh -lmpi
endif
```

## Changes Made

### configure.ac
1. Added OpenSHMEM compiler wrapper detection (around line 71):
```m4
AC_ARG_WITH([oshmem],
    [AS_HELP_STRING([--with-oshmem],
        [use OpenSHMEM for COMEX communication layer])],
    [],
    [with_oshmem=no])

AS_IF([test "x$with_oshmem" != xno], [
    AC_MSG_NOTICE([searching for OpenSHMEM compiler wrappers...])
    AC_PATH_PROGS([OSHCC], [oshcc])
    AC_PATH_PROGS([OSHCXX], [oshcxx])
    AC_PATH_PROGS([OSHFC], [oshfort oshfc])
    AS_IF([test "x$OSHCC" = x], [
        AC_MSG_ERROR([--with-oshmem was given but oshcc not found])
    ])
    AC_MSG_NOTICE([found oshcc: $OSHCC])
    AC_MSG_NOTICE([Switching to OpenSHMEM compiler wrappers for GA])
    CC="$OSHCC"
    MPICC="$OSHCC"
    AS_IF([test "x$OSHCXX" != x], [
        CXX="$OSHCXX"
        MPICXX="$OSHCXX"
    ])
    AS_IF([test "x$OSHFC" != x], [
        F77="$OSHFC"
        MPIF77="$OSHFC"
    ])
    with_mpi_wrappers=yes
    AC_MSG_NOTICE([Using CC=$CC (MPICC=$MPICC), CXX=$CXX (MPICXX=$MPICXX), F77=$F77 (MPIF77=$MPIF77)])
])
AM_CONDITIONAL([WITH_OSHMEM], [test "x$with_oshmem" != xno])
```

2. Modified MPI unwrapping logic (around line 136):
```m4
AS_IF([test "x$with_oshmem" = xno], [
    AS_IF([test x$with_mpi_wrappers = xyes],
        [GA_MPI_UNWRAP],
        [GA_ARG_PARSE([with_mpi], [GA_MP_LIBS], [GA_MP_LDFLAGS], [GA_MP_CPPFLAGS])])
    # MPI tests...
], [
    # When using OpenSHMEM, still unwrap to get MPI libs for compatibility
    AC_MSG_NOTICE([Unwrapping OpenSHMEM compiler to get MPI libraries])
    GA_MPI_UNWRAP
])
```

### Makefile.am
1. Removed conditional exclusion of `GA_MP_CPPFLAGS` and `GA_MP_LDFLAGS` (around lines 70-85):
```makefile
AM_CPPFLAGS += $(BLAS_CPPFLAGS)
AM_CPPFLAGS += $(GA_MP_CPPFLAGS)
AM_CPPFLAGS += $(ARMCI_NETWORK_CPPFLAGS)

AM_LDFLAGS += $(BLAS_LDFLAGS)
AM_LDFLAGS += $(GA_MP_LDFLAGS)
AM_LDFLAGS += $(ARMCI_NETWORK_LDFLAGS)
```

2. Added MPI libraries to LDADD when using OpenSHMEM (around line 89):
```makefile
LDADD += libga.la

# When using OpenSHMEM, add MPI libraries explicitly for test programs
# that call MPI functions (like ffflush.F with mpi_finalize)
if WITH_OSHMEM
LDADD += -lmpi_mpifh -lmpi
endif
```

3. Removed conditional exclusion of `GA_MP_LIBS` in libga.la (around line 1691):
```makefile
if ARMCI_NETWORK_ARMCI
libga_la_LIBADD += $(ARMCI_NETWORK_LIBS)
libga_la_LIBADD += $(GA_MP_LIBS)
else
# ...
endif
```

## Build Command
```bash
cd /home/d3g293/ga-oshmem
./autogen.sh
cd build_auto
../configure --enable-i4 --enable-cxx --disable-f77 --with-oshmem \
  --prefix=/home/d3g293/ga-oshmem/build_auto CFLAGS="-g"
make checkprogs
```

## Key Insights

1. **Compiler Wrapper Handling**: GA's build system has sophisticated logic for handling MPI compiler wrappers via `GA_PROG_MPICC`. To integrate OpenSHMEM, we needed to work WITH this system (by setting `with_mpi_wrappers=yes`) rather than trying to bypass it.

2. **Type Detection Dependency**: Autoconf's type detection macros must run with the correct compiler environment. Setting both `CC` and `MPICC` ensures `GA_PROG_MPICC` uses the right compiler before type detection runs.

3. **MPI Compatibility in OpenSHMEM**: OpenMPI's OpenSHMEM implementation includes MPI compatibility, but the Fortran MPI interface library (`libmpi_mpifh`) needs to be explicitly linked when Fortran code calls MPI functions.

4. **Fortran with --disable-f77**: Even with `--disable-f77`, some Fortran test programs are built. The flag disables Fortran *bindings* for GA, but not necessarily Fortran test programs in the test suite.

## Testing Status
- Configure: ✅ Completes successfully with correct compiler and type detection
- Type definitions: ✅ `size_t`, `off_t`, `pid_t` correctly undefined in config.h
- Compiler selection: ✅ Makefile uses `CC = oshcc` 
- Library linking: 🔄 In progress - added `-lmpi_mpifh -lmpi` to resolve Fortran MPI symbols

## Next Steps
1. Complete `make checkprogs` build with MPI library linking fix
2. Run test programs to verify OpenSHMEM runtime behavior
3. Test actual OpenSHMEM operations (RMW, mutexes) in test suite
4. Consider re-enabling Fortran support with proper oshfort configuration

## Related Files
- `configure.ac`: Top-level autoconf configuration
- `Makefile.am`: Top-level automake configuration
- `comex/configure.ac`: COMEX library configuration
- `comex/Makefile.am`: COMEX library build rules
- `comex/m4/comex_network_setup.m4`: Network backend selection
- `m4/ga_mpicc.m4`: MPI C compiler detection macro
- `m4/ga_mpi_unwrap.m4`: MPI compiler unwrapping logic
- `global/testing/ffflush.F`: Test file calling MPI functions
