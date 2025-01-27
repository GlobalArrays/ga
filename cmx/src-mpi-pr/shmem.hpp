/* cmx header file */
#ifndef _CMX_P_SHMEM_H
#define _CMX_P_SHMEM_H

#include <sys/mman.h>
#include <errno.h>
#include <string>

#include <mpi.h>

namespace CMX {

class p_Shmem {

public:

/**
 * Simple constructor
 */
p_Shmem();

/**
 * Simple destructor
 */
~p_Shmem();

/**
 * Generate a unique character name for memory segment. This string
 * is unique for each call to the function and is unique across all
 * processes
 * @param[in] rank of calling process
 * @return character string with unique name
 */
char* generateName(const int rank);

/**
 * Map file descriptor to memory segment
 * @param[in] fd file descriptor handle
 * @param[in] size length of allocated segment in bytes
 * @return pointer to allocated segment
 */
void* map(const int fd, const size_t size);

/**
 * Unmap memory segment
 * @param ptr pointer to memory segment
 * @param size length of memory segment
 */
void unmap(void *ptr, size_t size);

/**
 * Create a memory segment based on unique name
 * @param[in] name unique name for segment
 * @param[in] size length of segment in bytes
 * @return pointer to allocated segment
 */
void* create(const char *name, const size_t size);

/**
 * Attach to an existing memory allocation
 * @param[in] name unique name for segment
 * @param[in] size length of segment in bytes
 * @return pointer to allocated segment
 */
void* attach(const char *name, const size_t size);

/**
 * Free a memory segment that was created using create
 * @param[in] name character string name of memory segment
 * @param[in] ptr mapped pointer to memory segment
 * @param[in] size length of memory segment in bytes
 * @return true if no error found
 */
bool free(const char *name, void *ptr, size_t size);

private:

unsigned int p_counter[6];
};

} // namespace CMX
#endif
