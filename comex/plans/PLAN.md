# OpenSHMEM COMEX Backend Implementation Plan

This document describes a concrete plan to implement a COMEX backend using OpenSHMEM. It mirrors the design and expectations discussed earlier and is intended to be placed in `comex/src-oshmem` (implementation) while the build integration remains external. All new source files will live under `ga-oshmem/comex/src-oshmem` and no files outside that folder will be modified in this work.

## Summary

Goal: implement a COMEX backend that provides the API declared in `comex/src-common/comex.h` using OpenSHMEM primitives.

Scope: create a new backend under `comex/src-oshmem` containing all source, headers, tests and a `Makefile.inc` (or similar build snippet). No changes will be made outside `comex/src-oshmem` in this task.

Assumptions

- An OpenSHMEM implementation (headers & libraries) will be available at compile and run time.
- Target platforms are 64-bit (pointer collectives use 64-bit collects). If 32-bit is required, adjust collects accordingly.
- Group-level collectives (arbitrary subgroup collectives) are not provided by OpenSHMEM; group semantics will be emulated in software.

Success criteria

- Implemented functions in `comex.c` provide correct semantics for: init/finalize, put/get, acc (typed and fallbacks), malloc/free, nonblocking operations (handles), fences, and mutexes.
- Tests under `comex/src-oshmem/testing` validate basic RMA, atomic, NB, malloc, and mutex semantics.
- Implementation exists entirely under `comex/src-oshmem` and is ready to be wired into the top-level build by adding the new directory to configure/Makefiles (documented, not applied here).

## File layout (to create under `comex/src-oshmem`)

- `comex.c` — Main COMEX API implementation for OpenSHMEM (mapping calls to SHMEM).
- `comex_impl.h` — Backend-local state: `l_state`, macros, helpers.
- `groups.c`, `groups.h` — Group object emulation (rank translation, barriers, group metadata).
- `reg_symm.c`, `reg_symm.h` — Symmetric allocation helpers and pointer-exchange utilities (wrap `shmem_malloc` and collects).
- `mutex.c`, `mutex.h` — Mutex implementation using SHMEM atomic CAS/swap or fallback to spin under a global array.
- `nb.c`, `nb.h` — Non-blocking request table and helpers (handles, wait/test semantics).
- `accumulate.c`, `accumulate.h` — Typed accumulate helpers and fallbacks for complex types.
- `Makefile.inc` — Build snippet to compile the backend sources (must be added to the top-level build in a separate change).
- `testing/` — Small test programs: `test_put_get.c`, `test_acc.c`, `test_nb.c`, `test_mutex.c`, `test_malloc.c`.
- `README.md` — Notes for building and integrating the backend and required configure changes.

All files will be self-contained within `src-oshmem`.

## Function mapping: COMEX -> OpenSHMEM

High-level mapping and notes for key operations:

- Initialization / Finalization
  - `comex_init`, `comex_init_args`, `comex_init_comm`: call `shmem_init()` (or threaded init if necessary), set `l_state.rank = shmem_my_pe()`, `l_state.size = shmem_n_pes()`; init groups and NB table.
  - `comex_finalize`: ensure all outstanding operations complete (use `shmem_quiet()`), free symmetric allocations, finalize group state, call `shmem_finalize()`.

- Put/Get (contiguous)
  - `comex_put(src,dst,bytes,proc,group)`: translate `proc` (group rank -> world PE) then `shmem_putmem(dst, src, bytes, pe)` and call `shmem_quiet()` as needed for local completion semantics.
  - `comex_get(src,dst,bytes,proc,group)`: use `shmem_getmem(src, dst, bytes, pe)` and `shmem_quiet()` as appropriate.

- Strided/Vector operations
  - Implement `comex_puts`, `comex_gets`, `comex_putv`, `comex_getv` as loops calling the contiguous put/get. This mirrors `src-template` behavior and is correct as a first step.

- Non-blocking operations
  - Use `shmem_put_nbi` and `shmem_get_nbi` (or `shmem_put_nbi`/`shmem_get_nbi` variants) when available; otherwise issue the blocking put/get and return a handle that is immediately complete.
  - Maintain an `nb` table: store type, pointers, bytes, target PE, and active flag. `comex_wait(handle)` calls `shmem_quiet()` and marks the entry complete. `comex_test(handle)` will call `shmem_test` if available, else we use `shmem_quiet()` semantics and return completion (document limitation).

- Atomic Accumulate (`comex_acc`, `comex_accs`, `comex_accv`)
  - For integer and long types: prefer `shmem_int_atomic_add`/`shmem_long_atomic_add` when available.
  - For floating-point types: use `shmem_float_atomic_add`/`shmem_double_atomic_add` if provided; otherwise fallback to a get/modify/put under a remote lock.
  - For complex types (COMEX_ACC_CPL, COMEX_ACC_DCP): no native SHMEM atomics; always perform remote get into a local buffer, apply the accumulate locally, then put back under a remote lock.
  - Use `skip_lock` /lock batching similar to MPI backend where multiple ops can be done under a single lock.

- Read-Modify-Write (`comex_rmw`)
  - Map `COMEX_FETCH_AND_ADD`/`COMEX_FETCH_AND_ADD_LONG` to `shmem_atomic_fetch_add` typed variants if available.
  - Map `COMEX_SWAP`/`COMEX_SWAP_LONG` to `shmem_atomic_swap` or implement a CAS-loop fallback with `shmem_atomic_compare_swap`.

- Memory allocation
  - `comex_malloc(ptr_arr, bytes, group)`: use `shmem_malloc(bytes)` on each PE (symmetric allocation), then exchange local pointers among group members. Because SHMEM collectives are global, implement pointer exchange via:
    - If the group == WORLD, use `shmem_fcollect64`/`shmem_allgather` equivalents to exchange pointers in symmetric memory.
    - For subgroups, emulate with pairwise `shmem_putmem` into a symmetric gather buffer on a group leader or perform put/get in a loop using group member list.
  - `comex_free`: `shmem_free(ptr)` locally and perform a barrier/synchronization as required.
  - `comex_malloc_local` / `comex_free_local`: implement via `shmem_malloc` for symmetric behavior (preferred) or `malloc` for strictly local buffers (document difference). Using `shmem_malloc` is safer for RMA.

- Fence / Flush / Barrier
  - `comex_fence_all` / `comex_wait_all`: call `shmem_quiet()` then `shmem_barrier_all()` for global semantics; for per-proc fence call `shmem_quiet()` and check NB table entries targeted to that proc.
  - `comex_barrier(group)`: for WORLD use `shmem_barrier_all()`; for subgroup implement a software barrier using symmetric counters or leader-based gather/spin.

- Mutexes
  - Use symmetric lock array (allocated with `shmem_malloc`). Implement lock acquisition via `shmem_long_atomic_compare_swap` or `shmem_long_atomic_swap` where available and fall back to spin-wait using `shmem_put` + `shmem_quiet()`.
  - Provide exponential/backoff spin to reduce contention.

- Group APIs
  - Implement groups as local structs storing an array of world ranks (PE ids). Provide `translate_ranks`, `group_rank`, `group_size` by consulting the struct.
  - Emulate group collectives where needed by using peer-to-peer put/get loops and leader-based aggregation. Document performance caveats.

## Non-blocking semantics and limitations

- SHMEM non-blocking operations vary between implementations. We'll prefer `shmem_put_nbi`/`shmem_get_nbi`/`shmem_fence`/`shmem_quiet` when available.
- `comex_test` semantics: many SHMEM implementations don't provide an easy per-op completion test. If per-request test is not supported we will implement `comex_test` by checking the NB entry's active flag and calling `shmem_quiet()` if necessary. Document that `comex_test` may effectively trigger progress.

## Groups and collectives caveats

- OpenSHMEM lacks arbitrary subgroup collectives. Emulating subgroup collectives is doable but slower; the options are:
  1. Implement leader-based gather/scatter using symmetric buffers and peer `shmem_putmem` from group members to the leader's symmetric region.
  2. Use world-level collectives and have non-members write sentinel values that members ignore (works for pointer exchanges but may be wasteful).

I recommend implementing option 1 (leader-based) for correctness and clarity.

## Build & integration notes

- Add `src-oshmem/Makefile.inc` containing compilation rules. To include this backend in the repository build you will need to update `comex/Makefile.am` or the top-level `CMakeLists.txt`/`configure.ac` to detect OpenSHMEM and add `src-oshmem` when present. This change is intentionally NOT made here and should be applied separately.
- Configure-time checks to add:
  - Check for `shmem.h` (or `openshmem.h`) and library availability.
  - Check for atomic function availability (e.g., `shmem_atomic_fetch_add` typed variants) and define fallbacks if missing.

Minimum `configure.ac` snippets (to be added by integrator):

- AC_CHECK_HEADERS([shmem.h openshmem.h])
- AC_CHECK_LIB([shmem], [shmem_init], [have_shmem=yes], [have_shmem=no])
- AM_CONDITIONAL([COMEX_WITH_OSHMEM], [test "x$have_shmem" = "xyes"])

Document these in `src-oshmem/README.md`.

## Testing plan

Create tests under `comex/src-oshmem/testing` that exercise:

- Basic RMA: `test_put_get.c` — simple put/get between two PEs, verify correctness.
- Strided/Vector: `test_strided.c` — covers `comex_puts`/`comex_gets` and `putv`/`getv`.
- Accumulate: `test_acc.c` — test integer/float/double/long accumulate and fallbacks.
- NB operations: `test_nb.c` — issue `nbput`, `nbget`, `nbacc`, then `wait`/`test`.
- Malloc/free: `test_malloc.c` — test `comex_malloc` pointer exchange and the ability to RMA to returned pointers.
- Mutex: `test_mutex.c` — concurrent increments under a lock from multiple PEs.

Test runner: use the SHMEM launcher (e.g., `oshrun -n 4 <binary>` or vendor equivalent). Document required run commands in `README.md`.

## Edge cases and mitigation

- Atomic support differences: detect at build time and use lock-based fallback.
- Group collectives: slower emulation is used; document this to users.
- `comex_test` may behave as a progress call on some SHMEMs; document the limitation.
- Pointer size assumptions: use 64-bit collects for pointers; if targeting 32-bit systems, adjust collects.

## Implementation milestones (concrete)

1. Create `src-oshmem` skeleton with `Makefile.inc`, `comex.c`, `comex_impl.h`, and `groups.c/h`.
2. Implement `comex_init` / `comex_finalize` and populate `l_state` using SHMEM APIs.
3. Implement contiguous `comex_put` / `comex_get` and basic `comex_malloc` for WORLD group.
4. Implement NB table & `nbput`/`nbget` with `shmem_put_nbi`/`shmem_get_nbi` and `comex_wait`/`comex_test` (best-effort).
5. Implement atomics for supported datatypes and lock-based fallbacks for others.
6. Implement mutexes and group emulation (including subgroup barriers and pointer-exchange).
7. Add tests and run validation under `oshrun`.
8. Prepare `README.md` describing required `configure.ac`/Makefile changes to enable this backend and provide sample run commands.

## Next steps I can take (if you want me to proceed)

- Create the `comex/src-oshmem` skeleton and implement milestone 1 through 4 (init, put/get, malloc, NB table) entirely inside the new folder. I will also add the test harness and `Makefile.inc`. I will not modify files outside `src-oshmem`.
- Or, if you prefer, I can only provide the exact `configure.ac` / `Makefile.am` patches necessary to wire `src-oshmem` into the project.

---

Created-by: plan generated on October 27, 2025
