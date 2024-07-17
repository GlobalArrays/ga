#include <mpi.h>
#include <stdio.h>

int main() {
  char LIB_NAME[MPI_MAX_LIBRARY_VERSION_STRING];
  int result_len;
  MPI_Get_library_version(LIB_NAME, &result_len);
  printf("%s\n",LIB_NAME);
  return 0;
}
