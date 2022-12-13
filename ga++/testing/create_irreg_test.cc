#include "armci.h"
#include "ga-mpi.h"
#include "ga.h"
#include "macdecls.h"
#include "mpi.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <numeric>
#include <vector>

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  GA_Initialize();

  int my_rank, nproc;
  MPI_Comm_rank(GA_MPI_Comm(), &my_rank);
  MPI_Comm_size(GA_MPI_Comm(), &nproc);

  if(nproc != 4) {
    if(my_rank == 0) std::cout << "ERROR: Need exactly 4 processes!\n";
    GA_Terminate();
    MPI_Finalize();
    return 0;
  }

  int                  ndims       = 2;
  int64_t              arr_size    = 4;
  int64_t              dims[2]     = {2, 2}; // 2x2 GA
  int64_t              nblock[2]   = {2, 2}; // 2 blocks along each dim
  int64_t              pgrid[2]    = {2, 1}; // total 2 ranks
  int64_t              blk_sz[2]   = {1,1};  // size of individual blocks
  std::vector<int64_t> k_map       = {0, 1, 0, 1};
  int                  proclist[2] = {2, 3}; // restrict to 2 procs

  int g_a = NGA_Create_handle();
  NGA_Set_data64(g_a, ndims, &dims[0], C_DBL);

  GA_Set_restricted(g_a, proclist, 2);
  //NGA_Set_block_cyclic_proc_grid64(g_a, blk_sz, pgrid);
  NGA_Set_tiled_proc_grid64(g_a,blk_sz,pgrid);
  NGA_Set_pgroup(g_a, GA_Pgroup_get_default());
  NGA_Allocate(g_a);

  std::string error_msg = "GA create failed";
  if(!g_a) GA_Error(const_cast<char*>(error_msg.c_str()), arr_size);
  if(my_rank == 0) printf("GA create successful\n");

  std::vector<double>  buf(4);
  std::vector<int64_t> lo = {0, 0};
  std::vector<int64_t> hi = {1, 1};
  std::vector<int64_t> ld = {2};
  printf("p[%d] Calling NGA_Put64\n",my_rank);
  NGA_Put64(g_a, &lo[0], &hi[0], buf.data(), &ld[0]);

  NGA_Destroy(g_a);

  GA_Terminate();
  MPI_Finalize();

  return 0;
}

