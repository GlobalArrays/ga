#include<stdio.h>
#include<stdlib.h>

#include"mpi.h"
#include"ga.h"
#include"macdecls.h"

main(int argc, char **argv)
{
  int g_A, g_B, value = 7, dims[2]={5,10}, block[2]={3,2}, val=5, map[5]={0,2,6,0,2};
  MPI_Init(&argc, &argv);
  GA_Initialize();
  MA_init(C_INT, 1000, 1000);

  //  g_A=NGA_Create(C_INT, 2, dims, "array_A", NULL);
  //GA_Fill(g_A, &val);

  g_B = NGA_Create_irreg(C_INT, 2, dims, "array_B", block, map);

  GA_Fill(g_B, &value);
  GA_Sync();

  //GA_Print(g_A);
  GA_Print(g_B);

  GA_Terminate();
  MPI_Finalize();
  return 0;
}
 
