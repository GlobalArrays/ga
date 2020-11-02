#include "mpi.h"
#include <stdio.h>
#include "ga.h"
#include <algorithm>
#include <cstdlib>
#include <cstring>
#if defined(_OPENMP)
#include "omp.h"
#endif
//#define N 500
#define N 5
int main(int argc, char * argv[])
{
#if defined(_OPENMP)
    int x = N;
    int y = N;
    int return_code = 0;
    int dims[2] = {N,N};
    int lo[2] = {0,0};
    int hi[2] = {N-1,N-1};
    int ld[1] = {N};
    int local_buffer[N][N];
    int local_buffer2[N*N];
    char name[20];
    int rank, ranks;
    int provided;
    int stop, start;
    int writers;
    int handle, thread_count;

    MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);
    GA_Initialize();


    rank = GA_Nodeid();
    ranks = GA_Nnodes();

    if (provided < MPI_THREAD_MULTIPLE && rank == 0) {
      printf("MPI_THREAD_MULTIPLE not provided\n");
    }

    if(argc >= 3) {
        x = atoi(argv[1]);
        y = atoi(argv[2]);
    }

    for(int i =0; i<dims[0]; i++) {
        for(int j =0; j<dims[1]; j++) {
            local_buffer[i][j] = i*dims[1]+j;
            local_buffer2[i*N+j] = i*dims[1]+j;
        }
    }
    
    strcpy(name, "test");
    handle = NGA_Create(C_INT, 2, dims, name, NULL);
    
    thread_count = 4; 
    
    if( char * env_threads = std::getenv("OMP_NUM_THREADS"))
        thread_count = atoi(env_threads);
    else
        omp_set_num_threads(thread_count);

    printf("[%d]Testing %d threads.\n", rank, thread_count);
    
    printf("[%d]Testing write1 from 0.\n", rank);
    if(rank == 0)
        NGA_Put(handle, lo, hi, local_buffer, ld);
    NGA_Sync(); 
    
    
    printf("[%d]Testing read1.\n", rank);
    
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++) {
        int thread_id = omp_get_thread_num();
        int local_buffer_read[dims[0]][dims[1]];
        
        //printf("[%d][%d] Getting.\n", rank, thread_id);
        NGA_Get(handle, lo, hi, local_buffer_read, ld);
        for(int i =0; i<dims[0]; i++) {
            for(int k =0; k<dims[1]; k++) {
                if(local_buffer_read[i][k] != i*dims[1]+k) {
                    return_code = 1;
                    printf("[%d][%d][%d] write1/read1 error %d expected %d\n",
                        rank, thread_id, j, local_buffer_read[i][k],
                        i*dims[1]+k);
                }
            }
        }
    }
    NGA_Sync(); 
    GA_Destroy(handle);
    
    strcpy(name, "test1");
    handle = NGA_Create(C_INT, 2, dims, name, NULL);
    
    for(int i =0; i<dims[0]; i++) {
        for(int j =0; j<dims[1]; j++) {
            local_buffer[i][j] = i*dims[1]+j+1;
        }
    }
    
    writers = dims[0]*dims[1];
    int div = writers/ranks;
    int rem = writers%ranks;
    start = (rank*writers)/ranks;
    if (rank < ranks-1) {
      stop = ((rank+1)*writers)/ranks - 1;
    } else {
      stop = writers-1;
    }

#if 0
    if (rank < rem) {
        start = (div+1)*rank;
        stop = start+(div+1);
    } else {
        start = (div+1)*rem+(rank-rem)*div;
        stop = start+(div);
    }
#endif
    

    printf("[%d]Testing write2 all nodes/threads.\n", rank);
    #pragma omp parallel for
    for(int i =start; i<= stop; i++) {
      int lot[2], j, k;
      j = i%dims[1];
      k = (i-j)/dims[1];
      lot[0] = j;
      lot[1] = k;
      NGA_Put(handle, lot, lot, &local_buffer[j][k], ld);
    }
    NGA_Sync();
    lo[0] = 0;
    lo[1] = 0;
    hi[0] = dims[0]-1;
    hi[1] = dims[1]-1;
    ld[0] = dims[1];
    
    printf("[%d]Testing read2.\n", rank);
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++) {
        int thread_id = omp_get_thread_num();
        int local_buffer_read[dims[0]][dims[1]];
        NGA_Get(handle, lo, hi, local_buffer_read, ld);
        for(int i =0; i<dims[0]; i++) {
            for(int k =0; k<dims[1]; k++) {
                if(local_buffer_read[i][k] != i*dims[1]+k+1) {
                    return_code = 1;
                    printf("[%d][%d][%d] write2/read2 error %d at [%d][%d]"
                        " expected %d\n", rank, thread_id, j,
                        local_buffer_read[i][k], i, k, i*dims[1]+k+1);
                }
            }
        }
    }
    NGA_Sync(); 
    GA_Destroy(handle);
    
    
    strcpy(name, "test2");
    handle = NGA_Create(C_INT, 2, dims, name, NULL);
    
    for(int i =0; i<dims[0]; i++) {
        for(int j =0; j<dims[1]; j++) {
            local_buffer[i][j] = i*dims[0]+j+2;
        }
    }
    
    printf("[%d]Testing nb write1.\n", rank);
    ga_nbhdl_t wait_handle[dims[0]][dims[1]];
    #pragma omp parallel for
    for(int i =start; i<=stop; i++) {
        int thread_id = omp_get_thread_num();
        int j, k, lot[2];
        j = i%dims[1];
        k = (i-j)/dims[1];
        lot[0] = j;
        lot[1] = k;
        NGA_NbPut(handle, lot, lot, &local_buffer[j][k],
            ld, &wait_handle[j][k]);
    }
    #pragma omp parallel for
    for(int i =start; i<= stop; i++) {
        int j, k;
        j = i%dims[1];
        k = (i-j)/dims[1];
        NGA_NbWait(&wait_handle[j][k]);
    }
    NGA_Sync();
    lo[0] = 0;
    lo[1] = 0;
    hi[0] = dims[0]-1;
    hi[1] = dims[1]-1;
    ld[0] = dims[1];
    
    ga_nbhdl_t wait_handle_read[thread_count];
    int * local_buffer_read = new int[thread_count*dims[0]*dims[1]];
    printf("[%d]Testing nb read1.\n", rank);
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++) {
        int thread_id = omp_get_thread_num();
        
        NGA_NbGet(handle, lo, hi, &local_buffer_read[thread_id*dims[0]*dims[1]],
            ld, &wait_handle_read[thread_id]);
#if 0
    }
    // Breaking up the NbGet and the Wait in this way is a more rigorous
    // non-blocking test, but the current implementation does not appear
    // to be capable of handling it
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++) {
        int thread_id = omp_get_thread_num();
#endif
        NGA_NbWait(&wait_handle_read[thread_id]);
        for(int i =0; i<dims[0]; i++) {
            for(int k =0; k<dims[1]; k++) {
                if(local_buffer_read[thread_id*dims[0]*dims[1]+i*dims[1]+k] !=
                    i*dims[0]+k+2) {
                    return_code = 1;
                    printf("[%d][%d][%d] nb write1/read1 error %d at [%d][%d]"
                        " expected %d\n", rank, thread_id, j,
                        local_buffer_read[thread_id*dims[0]*dims[1]+i*dims[1]+k],
                        i, k, i*dims[1]+k+2);
                }
            }
        }
    }
    GA_Destroy(handle);
    int dim_atomic[1] = {1};
    int lohi_atomic[1] = {0};
    ld[0] = 1;
    strcpy(name, "test3");
    handle = NGA_Create(C_INT, 1, dim_atomic, name, NULL);
    dims[0] = thread_count*ranks;
    strcpy(name, "test_correct");
    int handle_correct = NGA_Create(C_INT, 1, dims, name, NULL);
    int * res_array = new int[dims[0]];
    int atomic = 0; 
    
    if(rank == 0)
        NGA_Put(handle, lohi_atomic, lohi_atomic, &atomic, ld);
    
    int correct = thread_count*ranks-1;
    int success = false;
    
    printf("[%d]Testing read inc1.\n", rank);
    //Check if summation is correct at end (do not introduce overhead for other calls)
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++) {
        long val = NGA_Read_inc(handle, lohi_atomic, 1);
        if(correct == val)
            success = 1;
        //printf("inc %d\n", val);
    }
    // Success will equal 1 on only 1 processor
     
    NGA_Sync();
    //Check if we get all results correctly 
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++) {
        long val = NGA_Read_inc(handle, lohi_atomic, 1);
        int ival = (int)val;
        int pos = val-dims[0]; 
        //printf("inc %d %d %d\n", val, dims[0], lo[0]);
        // fill each position in handle_correct with consecutive
        // values between dims[0] and 2*dims[0]-1.
        NGA_Put(handle_correct, &pos, &pos, &ival, ld);
    }
    
    NGA_Sync();

    //Tell everyone inc worked. Currently only one process has success=1
    if(success == 1) {
        printf("Atomic appears correct with outputs\n");
        NGA_Put(handle, lohi_atomic, lohi_atomic, &success, ld);
    }
    
    NGA_Sync();
    NGA_Get(handle, lohi_atomic, lohi_atomic, &success, ld);
    // Every process has success=1
    lo[0]=0;
    hi[0]=dims[0]-1;
    ld[0]=1;
    NGA_Get(handle_correct, lo, hi, res_array, ld);
    NGA_Sync();
    

    if(success != 1) {
        printf("[%d]Error read inc1 failed\n", rank);
        return_code = 1;
    }

    for(int i=0; i<dims[0]; i++) {
        if(res_array[i] != i+dims[0]) {
            printf("[%d]Error atomic inc failed %d %d %d\n", rank, i,
                res_array[i], i+dims[0]);
            return_code = 1;
        }
    }
    GA_Destroy(handle);
    GA_Destroy(handle_correct);
    
    dims[0] = x;
    dims[1] = y;
    strcpy(name, "test3");
    handle = NGA_Create(C_INT, 2, dims, name, NULL);
    
    long scale = 1;
    lo[0] = 0;
    lo[1] = 0;
    hi[0] = dims[0]-1;
    hi[1] = dims[1]-1;
    ld[0] = dims[1];
    
    for(int i =0; i<dims[0]; i++) {
        for(int j =0; j<dims[1]; j++) {
            local_buffer[i][j] = 0;
        }
    }
    
    // Set array to zero
    if(rank == 0)
        NGA_Put(handle, lo, hi, local_buffer, ld);
    NGA_Sync();
    
    for(int i =0; i<dims[0]; i++) {
        for(int j =0; j<dims[1]; j++) {
            local_buffer[i][j] = (i*dims[1]+j);
        }
    }

    printf("[%d]Testing nb acc1.\n", rank);
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++)
    {
        int thread_id = omp_get_thread_num();
        NGA_NbAcc(handle, lo, hi, local_buffer, ld, &scale,
            &wait_handle_read[thread_id]);
        printf("Acc: p[%d] T[%d] handle: %d\n",rank,thread_id,
            wait_handle_read[thread_id]);
    }
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++) {
        int thread_id = omp_get_thread_num();
        printf("Wait: p[%d] T[%d] handle: %d\n",rank,thread_id,
            wait_handle_read[thread_id]);
        NGA_NbWait(&wait_handle_read[thread_id]);
    }
    NGA_Sync();
    
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++) {
        int thread_id = omp_get_thread_num();
        
        NGA_NbGet(handle, lo, hi, &local_buffer_read[thread_id*dims[0]*dims[1]],
            ld, &wait_handle_read[thread_id]);
#if 0
    }
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++) {
        int thread_id = omp_get_thread_num();
#endif
        NGA_NbWait(&wait_handle_read[thread_id]);
        for(int i =0; i<dims[0]; i++) {
            for(int k =0; k<dims[1]; k++) {
                local_buffer[i][j] = rank*dims[0]*dims[1]+(i*dims[0]+j);
                if(local_buffer_read[thread_id*dims[0]*dims[1]+i*dims[1]+k] !=
                    thread_count*(ranks*i*dims[1]+ ranks*k)) {
                    return_code = 1;
                    printf("[%d][%d][%d] nb acc1 error %d at [%d][%d]"
                        " expected %d\n", rank, thread_id, j,
                        local_buffer_read[thread_id*dims[0]*dims[1]+i*dims[1]+k],
                        i, k, thread_count*ranks*i*dims[1]+ thread_count*ranks*k);
                }
            }
        }
    }
    NGA_Sync(); 
    printf("[%d]Testing done %d threads.\n", rank, thread_count);
    
    if(return_code == 0)
        if(rank==0)printf("Success\n\n");

    GA_Destroy(handle);
    GA_Terminate();
    MPI_Finalize();

    return return_code;
#else
    printf("OPENMP Disabled\n");
    return 1;
#endif
}
