#include "mpi.h"
#include <stdio.h>
#include "ga.h"
#include <algorithm>
#include <cstdlib>
#include <math.h>
#if defined(_OPENMP)
#include "omp.h"
#endif

#define DEFAULT_DIM 500

#define MAX_FACTOR 256
/**
 * Factor p processors into 2D processor grid of dimensions px, py
 */
void grid_factor(int p, int *idx, int *idy) {
  int i, j; 
  int ip, ifac, pmax, prime[MAX_FACTOR];
  int fac[MAX_FACTOR];
  int ix, iy, ichk;

  i = 1;
/**
 *   factor p completely
 *   first, find all prime numbers, besides 1, less than or equal to 
 *   the square root of p
 */
  ip = (int)(sqrt((double)p))+1;
  pmax = 0;
  for (i=2; i<=ip; i++) {
    ichk = 1;
    for (j=0; j<pmax; j++) {
      if (i%prime[j] == 0) {
        ichk = 0;
        break;
      }
    }
    if (ichk) {
      pmax = pmax + 1;
      if (pmax > MAX_FACTOR) printf("Overflow in grid_factor\n");
      prime[pmax-1] = i;
    }
  }
/**
 *   find all prime factors of p
 */
  ip = p;
  ifac = 0;
  for (i=0; i<pmax; i++) {
    while(ip%prime[i] == 0) {
      ifac = ifac + 1;
      fac[ifac-1] = prime[i];
      ip = ip/prime[i];
    }
  }
/**
 *  p is prime
 */
  if (ifac==0) {
    ifac++;
    fac[0] = p;
  }
/**
 *    find two factors of p of approximately the
 *    same size
 */
  *idx = 1;
  *idy = 1;
  for (i = ifac-1; i >= 0; i--) {
    ix = *idx;
    iy = *idy;
    if (ix <= iy) {
      *idx = fac[i]*(*idx);
    } else {
      *idy = fac[i]*(*idy);
    }
  }
}
int main(int argc, char * argv[])
{
#if defined(_OPENMP)
    int x = DEFAULT_DIM;
    int y = DEFAULT_DIM;
    int me, nproc;
    int px, py, ipx, ipy;
    int glo[2], ghi[2], gld[2];
    int tx, ty;
    int i,j,icnt;
    int return_code = 0;
    int dims[2];
    int ndimx = 2;
    int thread_count = 4;
    int *local_buffer;
    int ok;
    int one = 1;
    int *ptr;
    int next, nextx, nexty;

    /* Use a different array size if specified in arguments */
    if(argc >= 3)
    {
        x = atoi(argv[1]);
        y = atoi(argv[2]);
    }

    MPI_Init(&argc, &argv);
    GA_Initialize();

    nproc = GA_Nnodes();
    me = GA_Nodeid();

    /* Find processor grid dimensions and processor grid coordinates */
    grid_factor(nproc, &px, &py);
    ipy = me%py;
    ipx = (me-ipy)/py;
    if (me==0) {
      printf("\nTest running of %d processors\n",nproc);
      printf("\n  Array dimension is %d X %d\n",x,y);
      printf("\n  Processor grid is %d X %d\n\n",px,py);
    }

    dims[0] = x;
    dims[1] = y;
    local_buffer = (int*)malloc(x*y*sizeof(int));
  
#if 0
    icnt = 0;
    for(i =0; i<dims[0]; i++) {
      for(j =0; j<dims[1]; j++) {
        local_buffer[i][j] = i*dims[0]+j;
        local_buffer2[i*5+j] = i*dims[0]+j;
      }
    }
#endif
    
    /* Create GA and set all elements to zero */
    int handle = NGA_Create(C_INT, 2, dims, "test", NULL);
    GA_Zero(handle);
    
    if( char * env_threads = std::getenv("OMP_NUM_THREADS"))
        thread_count = atoi(env_threads);
    else
        omp_set_num_threads(thread_count);

    /* Find subblock grid based on the number of threads */
    grid_factor(thread_count, &tx, &ty);

    if (me==0) {
      printf("[%d]Testing %d threads.\n", me, thread_count);

      printf("[%d]Testing write1 from 0.\n", me);
    }

    /* Process 0 fills array with by having separate threads write parts
     * to each remote processor */
    if(me == 0) {
      #pragma omp parallel for
      for(int j = 0; j< thread_count; j++) {
        /* declare variables local to each thread */
        int lo[2], hi[2], tlo[2], thi[2];
        int ld[2];
        int k, m, n;
        int xinc, yinc;
        int itx, ity;
        int offset;
        int *buf;
        int lld;
        ity = j%ty;
        itx = (j-ity)/ty;
        for (k=0; k<nproc; k++) {
          NGA_Distribution(handle, k, lo, hi);
          ld[0] = hi[0]-lo[0]+1;
          ld[1] = hi[1]-lo[1]+1;
          xinc = ld[0]/tx;
          yinc = ld[1]/ty;
          tlo[0] = lo[0]+itx*xinc;
          tlo[1] = lo[1]+ity*yinc;
          /*
printf("j: %d k: %d tlo[0]: %d tlo[1]: %d xinc: %d yinc: %d\n",
    j,k,tlo[0],tlo[1],xinc,yinc);
    */
          offset = tlo[0]*dims[1]+tlo[1];
          buf = local_buffer + offset;
          if (itx<tx-1) thi[0] = tlo[0]+xinc-1;
          else thi[0] = hi[0];
          if (ity<ty-1) thi[1] = tlo[1]+yinc-1;
          else thi[1] = hi[1];
          /*
printf("j: %d k: %d lo[0]: %d hi[0]: %d  lo[1]: %d hi[1]: %d\n",
    j,k,lo[0],hi[0],lo[1],hi[1]);
printf("j: %d k: %d tlo[0]: %d thi[0]: %d  tlo[1]: %d thi[1]: %d\n",
    j,k,tlo[0],thi[0],tlo[1],thi[1]);
    */

          /* Fill a portion of local buffer with correct values */
          for (m=tlo[0]; m<=thi[0]; m++) {
            for (n=tlo[1]; n<=thi[1]; n++) {
              offset = (m-tlo[0])*dims[1] + (n-tlo[1]);
              buf[offset] = m*dims[1]+n;
            }
          }
          lld = dims[1];
          NGA_Put(handle, tlo, thi, buf, &lld);
        }
      }
    }
    /* Sync all processors at end of initialization loop */
    NGA_Sync(); 

    /* Each process determines if it is holding the correct data */
    NGA_Distribution(handle,me,glo,ghi);
    NGA_Access(handle,glo,ghi,&ptr,gld);
    ok = 1;
    icnt = 0;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        if (ptr[icnt] != i*dims[1]+j) {
          ok = 0;
          printf("p[%d] (write1) mismatch at point [%d,%d] actual: %d expected: %d\n",
              me,i,j,ptr[icnt],i*dims[1]+j);
        }
        icnt++;
      }
    }
    NGA_Release(handle,glo,ghi);
    if (me==0 && ok) {
      printf("write1 test OK\n");
    } else if (!ok) {
      printf("write1 test failed on process %d\n",me);
    }

    /* Move data from remote processor to local buffer */
    if (me==0) {
      printf("\n[%d]Testing read1 from 0.\n", me);
    }

    /* Each process copies data from next higher rank to a local buffer.
     * Gets are distributed between different threads */
    next = (me+1)%nproc;
    nexty = next%py;
    nextx = (next-nexty)/py;

    #pragma omp parallel for
    for(int j = 0; j< thread_count; j++) {
      /* declare variables local to each thread */
      int lo[2], hi[2], tlo[2], thi[2];
      int ld[2];
      int k, m, n;
      int xinc, yinc;
      int itx, ity;
      int offset;
      int *buf;
      int lld;
      ity = j%ty;
      itx = (j-ity)/ty;
      NGA_Distribution(handle, next, lo, hi);
      ld[0] = hi[0]-lo[0]+1;
      ld[1] = hi[1]-lo[1]+1;
      xinc = ld[0]/tx;
      yinc = ld[1]/ty;
      tlo[0] = lo[0]+itx*xinc;
      tlo[1] = lo[1]+ity*yinc;
      offset = tlo[0]*dims[1]+tlo[1];
      buf = local_buffer+offset;
      if (itx<tx-1) thi[0] = tlo[0]+xinc-1;
      else thi[0] = hi[0];
      if (ity<ty-1) thi[1] = tlo[1]+yinc-1;
      else thi[1] = hi[1];
      /* fill local portion of buffer with data from next processor */
      lld = dims[1];
      NGA_Get(handle, tlo, thi, buf, &lld);
    }
    /* Sync all processors at end of initialization loop */
    NGA_Sync(); 

    /* Check local buffer for correct values */
    NGA_Distribution(handle,next,glo,ghi);
    ok = 1;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        icnt = i*dims[1]+j;
        if (local_buffer[icnt] != i*dims[1]+j) {
          ok = 0;
          printf("p[%d] (read1) mismatch at point [%d,%d] actual: %d expected: %d\n",
              me,i,j,ptr[icnt],i*dims[1]+j);
        }
      }
    }
    NGA_Release(handle,glo,ghi);
    if (me==0 && ok) {
      printf("read1 test OK\n");
    } else if (!ok) {
      printf("read1 test failed on process %d\n",me);
    }
    
    /* Accumulate data to global array */
    if (me==0) {
      printf("\n[%d]Testing acc1 from 0.\n", me);
    }

    /* Each process accumulates data to the next process. Values in
     * global array are doubled */
#pragma omp parallel for
    for(int j = 0; j< thread_count; j++) {
      /* declare variables local to each thread */
      int lo[2], hi[2], tlo[2], thi[2];
      int ld[2];
      int k, m, n;
      int xinc, yinc;
      int itx, ity;
      int offset;
      int *buf;
      int lld;
      ity = j%ty;
      itx = (j-ity)/ty;
      NGA_Distribution(handle, next, lo, hi);
      ld[0] = hi[0]-lo[0]+1;
      ld[1] = hi[1]-lo[1]+1;
      xinc = ld[0]/tx;
      yinc = ld[1]/ty;
      tlo[0] = lo[0]+itx*xinc;
      tlo[1] = lo[1]+ity*yinc;
      offset = tlo[0]*dims[1]+tlo[1];
      buf = local_buffer + offset;
      if (itx<tx-1) thi[0] = tlo[0]+xinc-1;
      else thi[0] = hi[0];
      if (ity<ty-1) thi[1] = tlo[1]+yinc-1;
      else thi[1] = hi[1];

      /* Fill a portion of local buffer with correct values */
      for (m=tlo[0]; m<=thi[0]; m++) {
        for (n=tlo[1]; n<=thi[1]; n++) {
          offset = (m-tlo[0])*dims[1] + (n-tlo[1]);
          buf[offset] = m*dims[1]+n;
        }
      }
      lld = dims[1];
      NGA_Acc(handle, tlo, thi, buf, &lld, &one);
    }
    /* Sync all processors at end of initialization loop */
    NGA_Sync(); 

    /* Each process determines if it is holding the correct data */
    NGA_Distribution(handle,me,glo,ghi);
    NGA_Access(handle,glo,ghi,&ptr,gld);
    ok = 1;
    icnt = 0;
    for (i=glo[0]; i<=ghi[0]; i++) {
      for (j=glo[1]; j<=ghi[1]; j++) {
        if (ptr[icnt] != 2*(i*dims[1]+j)) {
          ok = 0;
          printf("p[%d] (Acc1) mismatch at point [%d,%d] actual: %d expected: %d\n",
              me,i,j,ptr[icnt],i*dims[1]+j);
        }
        icnt++;
      }
    }
    NGA_Release(handle,glo,ghi);
    if (me==0 && ok) {
      printf("acc1 test OK\n");
    } else if (!ok) {
      printf("acc1 test failed on process %d\n",me);
    }

#if 0
    printf("[%d]Testing read1.\n", rank);
    
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++)
    {
        int thread_id = omp_get_thread_num();
        int local_buffer_read[dims[0]][dims[1]];
        
        //printf("[%d][%d] Getting.\n", rank, thread_id);
        NGA_Get(handle, lo, hi, local_buffer_read, ld);
        NGA_Sync(); 
        for(int i =0; i<dims[0]; i++)
        {
            for(int k =0; k<dims[1]; k++)
            {
                if(local_buffer_read[i][k] != i*dims[1]+k)
                {
                    return_code = 1;
                    printf("[%d][%d][%d] write1/read1 error %d expected %d\n", rank, thread_id, j, local_buffer_read[i][k], i*dims[1]+k);
                }
            }
        }
    }
    
    handle = NGA_Create(C_INT, 2, dims, "test1", NULL);
    
    for(int i =0; i<dims[0]; i++)
    {
        for(int j =0; j<dims[1]; j++)
        {
            local_buffer[i][j] = i*dims[0]+j+1;
        }
    }
    
    int writers = dims[0]*dims[1];
    int div = writers/ranks;
    int rem = writers%ranks;
    int stop, start;
    if (rank < rem)
    {
        start = (div+1)*rank;
        stop = start+(div+1);
    }
    else
    {
        start = (div+1)*rem+(rank-rem)*div;
        stop = start+(div);
    }
    

    printf("[%d]Testing write2 all nodes/threads.\n", rank);
    #pragma omp parallel for
    for(int i =start; i< stop; i++)
    {
        int thread_id = omp_get_thread_num();
        int lo[2] = {i/dims[0], i%dims[0]};
        NGA_Put(handle, lo, lo, &local_buffer[i/dims[0]][i%dims[0]], ld);
    }
    NGA_Sync();
    lo[0] = 0;
    lo[1] = 0;
    hi[0] = dims[0]-1;
    hi[1] = dims[1]-1;
    ld[0] = dims[1];
    
    printf("[%d]Testing read2.\n", rank);
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++)
    {
        int thread_id = omp_get_thread_num();
        int local_buffer_read[dims[0]][dims[1]];
        
        NGA_Get(handle, lo, hi, local_buffer_read, ld);
        NGA_Sync(); 
        for(int i =0; i<dims[0]; i++)
        {
            for(int k =0; k<dims[1]; k++)
            {
                if(local_buffer_read[i][k] != i*dims[1]+k+1)
                {
                    return_code = 1;
                    printf("[%d][%d][%d] write2/read2 error %d at [%d][%d] expected %d\n", rank, thread_id, j, local_buffer_read[i][k], i, k, i*dims[1]+k+1);
                }
            }
        }
    }
    
    
    handle = NGA_Create(C_INT, 2, dims, "test2", NULL);
    
    for(int i =0; i<dims[0]; i++)
    {
        for(int j =0; j<dims[1]; j++)
        {
            local_buffer[i][j] = i*dims[0]+j+2;
        }
    }
    
    printf("[%d]Testing nb write1.\n", rank);
    ga_nbhdl_t wait_handle[dims[0]][dims[1]];
    #pragma omp parallel for
    for(int i =start; i< stop; i++)
    {
        int thread_id = omp_get_thread_num();
        int lo[2] = {i/dims[0], i%dims[0]};
        NGA_NbPut(handle, lo, lo, &local_buffer[i/dims[0]][i%dims[0]], ld, &wait_handle[i/dims[0]][i%dims[0]]);
    }
    #pragma omp parallel for
    for(int i =start; i< stop; i++)
        NGA_NbWait(&wait_handle[i/dims[0]][i%dims[0]]);
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
    for(int j =0; j< thread_count; j++)
    {
        int thread_id = omp_get_thread_num();
        
        NGA_NbGet(handle, lo, hi, &local_buffer_read[thread_id*dims[0]*dims[1]], ld, &wait_handle_read[thread_id]);
    }
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++)
    {
        int thread_id = omp_get_thread_num();
        NGA_NbWait(&wait_handle_read[thread_id]);
        for(int i =0; i<dims[0]; i++)
        {
            for(int k =0; k<dims[1]; k++)
            {
                if(local_buffer_read[thread_id*dims[0]*dims[1]+i*dims[1]+k] != i*dims[1]+k+2)
                {
                    return_code = 1;
                    printf("[%d][%d][%d] nb write1/read1 error %d at [%d][%d] expected %d\n", rank, thread_id, j, local_buffer_read[thread_id*dims[0]*dims[1]+i*dims[1]+k], i, k, i*dims[1]+k+2);
                }
            }
        }
    }
    
    int dim_atomic[1] = {1};
    int lohi_atomic[1] = {0};
    ld[0] = 1;
    handle = NGA_Create(C_INT, 1, dim_atomic, "test3", NULL);
    dims[0] = thread_count*ranks;
    int handle_correct = NGA_Create(C_INT, 1, dims, "test_correct", NULL);
    int * res_array = new int[dims[0]];
    int atomic = 0; 
    
    if(rank == 0)
        NGA_Put(handle, lohi_atomic, lohi_atomic, &atomic, ld);
    
    int correct = thread_count*ranks-1;
    int success = false;
    
    printf("[%d]Testing read inc1.\n", rank);
    //Check if summation is correct at end (do not introduce overhead for other calls)
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++)
    {
        long val = NGA_Read_inc(handle, lohi_atomic, 1);
        if(correct == val)
            success = 1;
        //printf("inc %d\n", val);
    }
     
    NGA_Sync();
    //Check if we get all results correctly 
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++)
    {
        long val = NGA_Read_inc(handle, lohi_atomic, 1);
        int pos = val-dims[0]; 
        //printf("inc %d %d %d\n", val, dims[0], lo[0]);
        NGA_Put(handle_correct, &pos, &pos, &val, ld);
    }
    
    NGA_Sync();

    //Tell everyone inc worked
    if(success == 1)
    {
        printf("Atomic appears correct with outputs\n");
        NGA_Put(handle, lohi_atomic, lohi_atomic, &success, ld);
    }
    
    NGA_Sync();
    NGA_Get(handle, lohi_atomic, lohi_atomic, &success, ld);
    lo[0]=0;
    hi[0]=dims[0]-1;
    ld[0]=1;
    NGA_Get(handle_correct, lo, hi, res_array, ld);
    NGA_Sync();
    

    if(success != 1)
    {
        printf("[%d]Error read inc1 failed\n", rank);
        return_code = 1;
    }

    for(int i=0; i<dims[0]; i++)
    {
        if(res_array[i] != i+dims[0])
        {
            printf("[%d]Error atomic inc failed %d %d %d\n", rank, i, res_array[i], i+dims[0]);
            return_code = 1;
        }
    }
    
    dims[0] = x;
    dims[1] = y;
    handle = NGA_Create(C_INT, 2, dims, "test3", NULL);
    
    long scale = 1;
    lo[0] = 0;
    lo[1] = 0;
    hi[0] = dims[0]-1;
    hi[1] = dims[1]-1;
    ld[0] = dims[1];
    
    for(int i =0; i<dims[0]; i++)
    {
        for(int j =0; j<dims[1]; j++)
        {
            local_buffer[i][j] = 0;
        }
    }
    
    if(rank == 0)
        NGA_Put(handle, lo, hi, local_buffer, ld);
    NGA_Sync();
    
    for(int i =0; i<dims[0]; i++)
    {
        for(int j =0; j<dims[1]; j++)
        {
            local_buffer[i][j] = (i*dims[1]+j);
        }
    }

    printf("[%d]Testing nb acc1.\n", rank);
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++)
    {
        int thread_id = omp_get_thread_num();
        NGA_NbAcc(handle, lo, hi, local_buffer, ld, &scale, &wait_handle_read[thread_id]);
    }
    
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++)
        NGA_NbWait(&wait_handle_read[j]);
    NGA_Sync();
    
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++)
    {
        int thread_id = omp_get_thread_num();
        
        NGA_NbGet(handle, lo, hi, &local_buffer_read[thread_id*dims[0]*dims[1]], ld, &wait_handle_read[thread_id]);
    }
    #pragma omp parallel for
    for(int j =0; j< thread_count; j++)
    {
        int thread_id = omp_get_thread_num();
        NGA_NbWait(&wait_handle_read[thread_id]);
        NGA_Sync(); 
        for(int i =0; i<dims[0]; i++)
        {
            for(int k =0; k<dims[1]; k++)
            {
                local_buffer[i][j] = rank*dims[0]*dims[1]+(i*dims[0]+j);
                if(local_buffer_read[thread_id*dims[0]*dims[1]+i*dims[1]+k] != thread_count*(ranks*i*dims[1]+ ranks*k))
                {
                    return_code = 1;
                    printf("[%d][%d][%d] nb acc1 error %d at [%d][%d] expected %d\n", rank, thread_id, j, local_buffer_read[thread_id*dims[0]*dims[1]+i*dims[1]+k], i, k, thread_count*ranks*i*dims[1]+ thread_count*ranks*k);
                }
            }
        }
    }
    
    printf("[%d]Testing done %d threads.\n", rank, thread_count);
    
    if(return_code == 0)
        if(rank==0)printf("Success\n\n");


#endif
    GA_Destroy(handle);
    free(local_buffer);

    GA_Terminate();
    MPI_Finalize();

    return return_code;
#else
    printf("OPENMP Disabled\n");
    return 1;
#endif
}

