OpenSHMEM COMEX backend (world-only)

This folder contains a SHMEM-based COMEX backend. Key points:

- Only `COMEX_GROUP_WORLD` is supported. Attempts to create subgroups will fail.
- Accumulate operations (future files) should use get-modify-put under remote locks.
- The backend uses `shmem_malloc` for symmetric allocations and `shmem_put/get(_nbi)` for RMA.

Build integration

To include this backend in the top-level build, add `src-oshmem/Makefile.inc` to `comex/Makefile.am` or equivalent and add checks for SHMEM (e.g., `AC_CHECK_HEADERS([shmem.h openshmem.h])`) in `configure.ac`. Link with the SHMEM library (e.g., `-lshmem`).

Running tests

Use your platform's SHMEM launcher, e.g.:

oshrun -n 4 ./test_put_get

