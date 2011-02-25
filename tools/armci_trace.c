
/***************************************************************
 *    WARNING! DO NOT EDIT! THIS CODE IS AUTO-GENERATED        *
 **************************************************************/


#include <stdio.h>
#include "armci.h"

#include "parmci.h"
    /*
       Functions not handled: set(['ARMCI_Memget', 'ARMCI_Memat'])
     */


int ARMCI_AccV(int op, void *scale, armci_giov_t * darr, int len, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_AccV(op, scale, darr, len, proc);
    printf("%lf,ARMCI_AccV,(%d;%d;%d)\n", tstamp, len, (int) darr, proc);
    return rval;
}


void ARMCI_Barrier()
{

    double tstamp = MPI_Wtime();
    PARMCI_Barrier();
    printf("%lf,ARMCI_Barrier,\n", tstamp);
}


int ARMCI_AccS(int optype, void *scale, void *src_ptr, int *src_stride_arr,
	       void *dst_ptr, int *dst_stride_arr, int *count,
	       int stride_levels, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval =
	PARMCI_AccS(optype, scale, src_ptr, src_stride_arr, dst_ptr,
		    dst_stride_arr, count, stride_levels, proc);
    printf("%lf,ARMCI_AccS,(%d;%d;%d)\n", tstamp, stride_levels,
	   (int) count, proc);
    return rval;
}


void ARMCI_Finalize()
{

    double tstamp = MPI_Wtime();
    PARMCI_Finalize();
    printf("%lf,ARMCI_Finalize,\n", tstamp);
}


int ARMCI_NbPut(void *src, void *dst, int bytes, int proc,
		armci_hdl_t * nb_handle)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_NbPut(src, dst, bytes, proc, nb_handle);
    printf("%lf,ARMCI_NbPut,(%d;%d)\n", tstamp, bytes, proc);
    return rval;
}


int ARMCI_GetValueInt(void *src, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_GetValueInt(src, proc);
    printf("%lf,ARMCI_GetValueInt,(%p,%d)\n", tstamp, src, proc);
    return rval;
}


int ARMCI_Put_flag(void *src, void *dst, int bytes, int *f, int v,
		   int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Put_flag(src, dst, bytes, f, v, proc);
    printf("%lf,ARMCI_Put_flag,(%d;%d;%d)\n", tstamp, bytes, proc, v);
    return rval;
}


int ARMCI_NbGetS(void *src_ptr, int *src_stride_arr, void *dst_ptr,
		 int *dst_stride_arr, int *count, int stride_levels,
		 int proc, armci_hdl_t * nb_handle)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval =
	PARMCI_NbGetS(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
		      count, stride_levels, proc, nb_handle);
    printf("%lf,ARMCI_NbGetS,(%d;%d;%d)\n", tstamp, stride_levels,
	   (int) count, proc);
    return rval;
}


void *ARMCI_Malloc_local(armci_size_t bytes)
{
    void *rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Malloc_local(bytes);
    printf("%lf,ARMCI_Malloc_local,(%d)\n", tstamp, bytes);
    return rval;
}


int ARMCI_Free_local(void *ptr)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Free_local(ptr);
    printf("%lf,ARMCI_Free_local,(%p)\n", tstamp, ptr);
    return rval;
}


int ARMCI_Get(void *src, void *dst, int bytes, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Get(src, dst, bytes, proc);
    printf("%lf,ARMCI_Get,(%d;%d)\n", tstamp, bytes, proc);
    return rval;
}


int ARMCI_Put(void *src, void *dst, int bytes, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Put(src, dst, bytes, proc);
    printf("%lf,ARMCI_Put,(%d;%d)\n", tstamp, bytes, proc);
    return rval;
}


int ARMCI_Destroy_mutexes()
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Destroy_mutexes();
    printf("%lf,ARMCI_Destroy_mutexes,\n", tstamp);
    return rval;
}


int ARMCI_GetS(void *src_ptr, int *src_stride_arr, void *dst_ptr,
	       int *dst_stride_arr, int *count, int stride_levels,
	       int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval =
	PARMCI_GetS(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
		    count, stride_levels, proc);
    printf("%lf,ARMCI_GetS,(%d;%d;%d)\n", tstamp, stride_levels,
	   (int) count, proc);
    return rval;
}


int ARMCI_NbAccV(int op, void *scale, armci_giov_t * darr, int len,
		 int proc, armci_hdl_t * nb_handle)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_NbAccV(op, scale, darr, len, proc, nb_handle);
    printf("%lf,ARMCI_NbAccV,(%d;%d;%d)\n", tstamp, len, (int) darr, proc);
    return rval;
}


float ARMCI_GetValueFloat(void *src, int proc)
{
    float rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_GetValueFloat(src, proc);
    printf("%lf,ARMCI_GetValueFloat,(%p,%d)\n", tstamp, src, proc);
    return rval;
}


int ARMCI_Malloc(void **ptr_arr, armci_size_t bytes)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Malloc(ptr_arr, bytes);
    printf("%lf,ARMCI_Malloc,(%d)\n", tstamp, bytes);
    return rval;
}


int ARMCI_NbAccS(int optype, void *scale, void *src_ptr,
		 int *src_stride_arr, void *dst_ptr, int *dst_stride_arr,
		 int *count, int stride_levels, int proc,
		 armci_hdl_t * nb_handle)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval =
	PARMCI_NbAccS(optype, scale, src_ptr, src_stride_arr, dst_ptr,
		      dst_stride_arr, count, stride_levels, proc,
		      nb_handle);
    printf("%lf,ARMCI_NbAccS,(%d;%d;%d)\n", tstamp, stride_levels,
	   (int) count, proc);
    return rval;
}


int ARMCI_PutS(void *src_ptr, int *src_stride_arr, void *dst_ptr,
	       int *dst_stride_arr, int *count, int stride_levels,
	       int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval =
	PARMCI_PutS(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
		    count, stride_levels, proc);
    printf("%lf,ARMCI_PutS,(%d;%d;%d)\n", tstamp, stride_levels,
	   (int) count, proc);
    return rval;
}


void *ARMCI_Memat(armci_meminfo_t * meminfo, int memflg)
{
    void *rval;
    rval = PARMCI_Memat(meminfo, memflg);
    return rval;
}


int ARMCI_PutV(armci_giov_t * darr, int len, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_PutV(darr, len, proc);
    printf("%lf,ARMCI_PutV,(%d;%d;%d)\n", tstamp, len, (int) darr, proc);
    return rval;
}


int ARMCI_Free(void *ptr)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Free(ptr);
    printf("%lf,ARMCI_Free,(%p)\n", tstamp, ptr);
    return rval;
}


int ARMCI_Init_args(int *argc, char ***argv)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Init_args(argc, argv);
    printf("%lf,ARMCI_Init_args,\n", tstamp);
    return rval;
}


int ARMCI_PutValueInt(int src, void *dst, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    double tstamp = MPI_Wtime();
    rval = PARMCI_PutValueInt(src, dst, proc);
    printf("%lf,ARMCI_PutValueInt,(%ld,%p,%d)\n", tstamp, (long) src, dst,
	   proc);

    printf("%lf,ARMCI_PutValueInt,(%ld,%p,%d)\n", tstamp, (long) src, dst,
	   proc);
    return rval;
}


void ARMCI_Memget(size_t bytes, armci_meminfo_t * meminfo, int memflg)
{
    PARMCI_Memget(bytes, meminfo, memflg);
}


void ARMCI_AllFence()
{

    double tstamp = MPI_Wtime();
    PARMCI_AllFence();
    printf("%lf,ARMCI_AllFence,\n", tstamp);
}


int ARMCI_NbPutV(armci_giov_t * darr, int len, int proc,
		 armci_hdl_t * nb_handle)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_NbPutV(darr, len, proc, nb_handle);
    printf("%lf,ARMCI_NbPutV,(%d;%d;%d)\n", tstamp, len, (int) darr, proc);
    return rval;
}


int ARMCI_PutValueDouble(double src, void *dst, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_PutValueDouble(src, dst, proc);
    printf("%lf,ARMCI_PutValueDouble,(%lf,%p,%d)\n", tstamp, (double) src,
	   dst, proc);
    return rval;
}


int ARMCI_GetV(armci_giov_t * darr, int len, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_GetV(darr, len, proc);
    printf("%lf,ARMCI_GetV,(%d;%d;%d)\n", tstamp, len, (int) darr, proc);
    return rval;
}


int ARMCI_Test(armci_hdl_t * nb_handle)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Test(nb_handle);
    printf("%lf,ARMCI_Test,(%p)\n", tstamp, nb_handle);
    return rval;
}


void ARMCI_Unlock(int mutex, int proc)
{

    double tstamp = MPI_Wtime();
    PARMCI_Unlock(mutex, proc);
    printf("%lf,ARMCI_Unlock,(%d;%d)\n", tstamp, mutex, proc);
}


void ARMCI_Fence(int proc)
{

    double tstamp = MPI_Wtime();
    PARMCI_Fence(proc);
    printf("%lf,ARMCI_Fence,(%d)\n", tstamp, proc);
}


int ARMCI_Create_mutexes(int num)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Create_mutexes(num);
    printf("%lf,ARMCI_Create_mutexes,(%d)\n", tstamp, num);
    return rval;
}


int ARMCI_PutS_flag(void *src_ptr, int *src_stride_arr, void *dst_ptr,
		    int *dst_stride_arr, int *count, int stride_levels,
		    int *flag, int val, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval =
	PARMCI_PutS_flag(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
			 count, stride_levels, flag, val, proc);
    printf("%lf,ARMCI_PutS_flag,(%d;%d;%d,%d)\n", tstamp, stride_levels,
	   (int) count, proc, val);
    return rval;
}


int ARMCI_WaitProc(int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_WaitProc(proc);
    printf("%lf,ARMCI_WaitProc,(%d)\n", tstamp, proc);
    return rval;
}


void ARMCI_Lock(int mutex, int proc)
{

    double tstamp = MPI_Wtime();
    PARMCI_Lock(mutex, proc);
    printf("%lf,ARMCI_Lock,(%d;%d)\n", tstamp, mutex, proc);
}


double ARMCI_GetValueDouble(void *src, int proc)
{
    double rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_GetValueDouble(src, proc);
    printf("%lf,ARMCI_GetValueDouble,(%p,%d)\n", tstamp, src, proc);
    return rval;
}


int ARMCI_NbGetV(armci_giov_t * darr, int len, int proc,
		 armci_hdl_t * nb_handle)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_NbGetV(darr, len, proc, nb_handle);
    printf("%lf,ARMCI_NbGetV,(%d;%d;%d)\n", tstamp, len, (int) darr, proc);
    return rval;
}


int ARMCI_Rmw(int op, int *ploc, int *prem, int extra, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Rmw(op, ploc, prem, extra, proc);
    printf("%lf,ARMCI_Rmw,(%d,%d,%d)\n", tstamp, op, extra, proc);
    return rval;
}


int ARMCI_Init()
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Init();
    printf("%lf,ARMCI_Init,\n", tstamp);
    return rval;
}


int ARMCI_WaitAll()
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_WaitAll();
    printf("%lf,ARMCI_WaitAll,\n", tstamp);
    return rval;
}


int ARMCI_NbGet(void *src, void *dst, int bytes, int proc,
		armci_hdl_t * nb_handle)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_NbGet(src, dst, bytes, proc, nb_handle);
    printf("%lf,ARMCI_NbGet,(%d;%d)\n", tstamp, bytes, proc);
    return rval;
}


int ARMCI_PutValueFloat(float src, void *dst, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_PutValueFloat(src, dst, proc);
    printf("%lf,ARMCI_PutValueFloat,(%lf,%p,%d)\n", tstamp, (double) src,
	   dst, proc);
    return rval;
}


int ARMCI_NbPutS(void *src_ptr, int *src_stride_arr, void *dst_ptr,
		 int *dst_stride_arr, int *count, int stride_levels,
		 int proc, armci_hdl_t * nb_handle)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval =
	PARMCI_NbPutS(src_ptr, src_stride_arr, dst_ptr, dst_stride_arr,
		      count, stride_levels, proc, nb_handle);
    printf("%lf,ARMCI_NbPutS,(%d;%d;%d)\n", tstamp, stride_levels,
	   (int) count, proc);
    return rval;
}


int ARMCI_PutS_flag_dir(void *src_ptr, int *src_stride_arr, void *dst_ptr,
			int *dst_stride_arr, int *count, int stride_levels,
			int *flag, int val, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval =
	PARMCI_PutS_flag_dir(src_ptr, src_stride_arr, dst_ptr,
			     dst_stride_arr, count, stride_levels, flag,
			     val, proc);
    printf("%lf,ARMCI_PutS_flag_dir,(%d;%d;%d,%d)\n", tstamp,
	   stride_levels, (int) count, proc, val);
    return rval;
}


int ARMCI_PutValueLong(long src, void *dst, int proc)
{
    int rval;
    double tstamp = MPI_Wtime();
    double tstamp = MPI_Wtime();
    rval = PARMCI_PutValueLong(src, dst, proc);
    printf("%lf,ARMCI_PutValueLong,(%ld,%p,%d)\n", tstamp, (long) src, dst,
	   proc);

    printf("%lf,ARMCI_PutValueLong,(%ld,%p,%d)\n", tstamp, (long) src, dst,
	   proc);
    return rval;
}


int ARMCI_Wait(armci_hdl_t * nb_handle)
{
    int rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_Wait(nb_handle);
    printf("%lf,ARMCI_Wait,(%p)\n", tstamp, nb_handle);
    return rval;
}


long ARMCI_GetValueLong(void *src, int proc)
{
    long rval;
    double tstamp = MPI_Wtime();
    rval = PARMCI_GetValueLong(src, proc);
    printf("%lf,ARMCI_GetValueLong,(%p,%d)\n", tstamp, src, proc);
    return rval;
}
