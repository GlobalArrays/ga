# MPI Progress Threads (MPI-PT)

These are notes describing the MPI progress threads runtime. These notes are intended to help developers navigate the contents of these files and to locate specific functionality.

This implementation is nearly identical to MPI-PR.  We recommend reading the MPI-PR NOTES.md file for details.  The only difference is that instead of using `MPI_Comm_split()` to reserve a user-level MPI rank for progress, a thread is created for asynch progress.  Read the MPI-PR NOTES.md for details about shared memory and how the progress server works.
