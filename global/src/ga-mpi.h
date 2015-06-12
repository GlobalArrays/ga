#ifndef GA_MPI_H_
#define GA_MPI_H_

#include <mpi.h>

#ifdef __cplusplus
extern "C" {
#endif

extern MPI_Comm GA_MPI_Comm();
extern MPI_Comm GA_MPI_Comm_pgroup(int pgroup);
extern MPI_Comm GA_MPI_Comm_pgroup_default();

#ifdef __cplusplus
}
#endif

#endif /* GA_MPI_H_ */
