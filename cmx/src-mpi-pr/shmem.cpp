#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>

#include "shmem.hpp"

/* This needs to be filled in */
#define CMX_ASSERT(WHAT)

#define SHM_NAME_SIZE 31

namespace CMX {

/**
 * Simple constructor
 */
p_Shmem::p_Shmem()
{
  p_counter[0] = 0;
  p_counter[1] = 0;
  p_counter[2] = 0;
  p_counter[3] = 0;
  p_counter[4] = 0;
  p_counter[5] = 0;
}

/**
 * Simple destructor
 */
p_Shmem::~p_Shmem()
{

}

/**
 * Generate a unique character name for memory segment. This string
 * is unique for each call to the function and is unique across all
 * processes
 * @param[in] rank of calling process
 * @return character string with unique name
 */
char* p_Shmem::generateName(int rank)
{
  int snprintf_retval = 0;
  /* /cmxUUUUUUUUUUPPPPPPPPPPCCCCCCN */
  /* 0000000001111111111222222222233 */
  /* 1234567890123456789012345678901 */
  char *name = NULL;
  static const unsigned int limit = 62;
  static const char letters[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
  static unsigned int counter[6] = {0};

  CMX_ASSERT(rank >= 0);
  name = new char[SHM_NAME_SIZE];
  snprintf_retval = snprintf(name, SHM_NAME_SIZE,
      "/cmx%010u%010u%c%c%c%c%c%c", getuid(), getpid(),
      letters[counter[5]],
      letters[counter[4]],
      letters[counter[3]],
      letters[counter[2]],
      letters[counter[1]],
      letters[counter[0]]);
  CMX_ASSERT(snprintf_retval < (int)SHM_NAME_SIZE);
  name[SHM_NAME_SIZE-1] = '\0';
  ++counter[0];
  if (counter[0] >= limit) { ++counter[1]; counter[0] = 0; }
  if (counter[1] >= limit) { ++counter[2]; counter[1] = 0; }
  if (counter[2] >= limit) { ++counter[3]; counter[2] = 0; }
  if (counter[3] >= limit) { ++counter[4]; counter[3] = 0; }
  if (counter[4] >= limit) { ++counter[5]; counter[4] = 0; }
  if (counter[5] >= limit) {
    printf("generateName: too many names generated");
    CMX_ASSERT(0);
  }

  return name;
}

/**
 * Map file descriptor to memory segment
 * @param[in] fd file descriptor handle
 * @param[in] size length of allocated segment in bytes
 * @return pointer to allocated segment
 */
void* p_Shmem::map(const int fd, const size_t size)
{
  void *memory  = static_cast<void*>(mmap(NULL,
        size, PROT_READ|PROT_WRITE, MAP_SHARED,
        fd, 0));
  if (MAP_FAILED == memory) {
    perror("p_Shmem::map: ");
    // cmx_error("p_Shmem::map ", -1);
  }

  return memory;
}

/**
 * Unmap memory segment
 * @param ptr pointer to memory segment
 * @param size length of memory segment
 */
void p_Shmem::unmap(void *ptr, size_t size)
{
  int retval = 0;
  retval = munmap(ptr, size);
  if (-1 == retval) {
    perror("p_Shmem::unmap: ");
    // cmx_error("p_Shmem::unmap ", -1);
  }
}

/**
 * Create a memory segment based on unique name
 * @param[in] name unique name for segment
 * @param[in] size length of segment in bytes
 * @return pointer to allocated segment
 */
void* p_Shmem::create(const char *name, const size_t size)
{
  void *mapped = NULL;
  int fd = 0;
  int retval = 0;

  /* create shared memory segment */
  fd = shm_open(name, O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
  if (-1 == fd && EEXIST == errno) {
    retval = shm_unlink(name);
    if (-1 == retval) {
      perror("p_Shmem::create: shm_unlink");
      //comex_error("p_Shmem::create: shm_unlink", retval);
    }
  }

  /* try a second time */
  if (-1 == fd) {
    fd = shm_open(name, O_CREAT|O_EXCL|O_RDWR, S_IRUSR|S_IWUSR);
  }

  /* finally report error if needed */
  if (-1 == fd) {
    perror("p_Shmem::create: shm_open");
    //comex_error("p_Shmem::create: shm_open", fd);
  }

  /* set the size of my shared memory object
   * */
  retval = ftruncate(fd, size);
  if (-1 == retval) {
    perror("p_Shmem::create: ftruncate");
    //comex_error("p_Shmem::create: ftruncate", retval);
  }

  /* map into local address space */
  mapped = map(fd, size);

  /* close file descriptor */
  retval = close(fd);
  if (-1 == retval) {
    perror("p_Shmem::create: close");
    //comex_error("p_Shmem::create: close", -1);
  }

  return mapped;
}

/**
 * Attach to an existing memory allocation
 * @param[in] name unique name for segment
 * @param[in] size length of segment in bytes
 * @return pointer to allocated segment
 */
void* p_Shmem::attach(const char *name, const size_t size)
{
  void *mapped = NULL;
  int fd = 0;
  int retval = 0;

  /* attach to shared memory segment */
  fd = shm_open(name, O_RDWR, S_IRUSR|S_IWUSR);
  if (-1 == fd) {
    perror("p_Shmem::attach: shm_open");
    // cmx_error("p_Shmem::attach: shm_open", -1);
  }

  /* map into local address space */
  mapped = map(fd, size);
  /* close file descriptor */
  retval = close(fd);
  if (-1 == retval) {
    perror("p_Shmem::attach: close");
    // cmx_error("p_Shmem::attach: close", -1);
  }

  return mapped;
}

/**
 * Free a memory segment that was created using create
 * @param[in] name character string name of memory segment
 * @param[in] ptr mapped pointer to memory segment
 * @param[in] size length of memory segment in bytes
 * @return true if no error found
 */
bool p_Shmem::free(const char *name, void *ptr, size_t size)
{
  int retval = 0;
  bool ret = true;
  retval = munmap(ptr, size);
  if (retval == -1) {
    perror("p_Shmem::free: munmap");
    ret = false;
  }
  retval = shm_unlink(name);
  if (retval == -1) {
    char buf[128];
    sprintf(buf,"name: (%s) ptr: %p len: %ld p_Shmem::free: shm_unlink",
        name,ptr,size);
    perror(buf);
    ret = false;
  }
  return ret;
}

}
