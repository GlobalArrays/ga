#ifndef _DEV_UTILS_H
#define _DEV_UTILS_H

#include "dev_mem_handle.h"
#include "comex.h"

#if defined(ENABLE_DEVICE)


#if defined(ENABLE_CUDA)
extern int numDevices();
extern void setDevice(int id);
extern void mallocDevice(void **buf, int size);
extern void freeDevice(void *buf);
extern void copyToDevice(void *devptr, void *hostptr, int bytes);
extern void copyToHost(void *hostptr, void *devptr, int bytes);
extern void copyDevToDev(void *dst, void *src, int bytes);
extern void copyPeerToPeer(void *dstptr, int dstID, void *srcptr, int srcID, int bytes);
extern void deviceMemset(void *ptr, int val, size_t bytes);
extern int isHostPointer(void *ptr);
extern int getDeviceID(void *ptr);
extern void deviceAddInt(int *ptr, const int inc);
extern void deviceAddLong(long *ptr, const long inc);
extern int deviceGetMemHandle(devMemHandle_t *handle, void *memory);
extern int deviceOpenMemHandle(void **memory, devMemHandle_t handle);
extern int deviceCloseMemHandle(void *memory);

extern void deviceIaxpy(int *dst, const int *src, const int *scale, int n);
extern void deviceLaxpy(long *dst, const long *src, const long *scale, int n);

extern int _comex_dev_flag;
extern int _comex_dev_id;

// #elif defined(ENABLE_HIP)
// #include "dev_utils_hip.hpp"
#endif

/**
 * Skip the intense macro usage and just do this the old-fashion way
 * @param op type of operation including data type
 * @param bytes number of bytes
 * @param dst destination buffer (on device)
 * @param src source buffer
 * @param scale factor for scaling value before accumalation
 */
static inline void _acc_dev(
        const int op,
        const int bytes,
        void * const restrict dst,
        const void * const restrict src,
        const void * const restrict scale)
{
    typedef struct {
      double real;
      double imag;
    } DoubleComplexDev;

    typedef struct {
      float real;
      float imag;
    } SingleComplexDev;

#if defined(ENABLE_CUDA)
  cublasHandle_t handle;
  if (op == COMEX_ACC_DBL) {
    const int n = bytes/sizeof(double);
    cublasCreate(&handle);
    cublasDaxpy(handle,n,scale,src,1,dst,1);
    cublasDestroy(handle);
  } else if (op == COMEX_ACC_FLT) {
    const int n = bytes/sizeof(float);
    cublasCreate(&handle);
    cublasSaxpy(handle,n,scale,src,1,dst,1);
    cublasDestroy(handle);
  } else if (op == COMEX_ACC_INT) {
    const int n = bytes/sizeof(int);
    deviceIaxpy(dst, src, scale, n);
  } else if (op == COMEX_ACC_LNG) {
    const int n = bytes/sizeof(long);
    deviceLaxpy(dst, src, scale, n);
  } else if (op == COMEX_ACC_DCP) {
    const int n = bytes/sizeof(DoubleComplexDev);
    cublasCreate(&handle);
    cublasZaxpy(handle,n,scale,src,1,dst,1);
    cublasDestroy(handle);
  } else if (op == COMEX_ACC_CPL) {
    const int n = bytes/sizeof(SingleComplexDev);
    cublasCreate(&handle);
    cublasCaxpy(handle,n,scale,src,1,dst,1);
    cublasDestroy(handle);
  } else {
  }
#elif defined(ENABLE_HIP)
  rocblas_handle handle;
  if (op == COMEX_ACC_DBL) {
    const int n = bytes/sizeof(double);
    rocblasCreate(&handle);
    rocblasDaxpy(handle,n,scale,src,1,dst,1);
    rocblasDestroy(handle);
  } else if (op == COMEX_ACC_FLT) {
    const int n = bytes/sizeof(float);
    rocblasCreate(&handle);
    rocblasSaxpy(handle,n,scale,src,1,dst,1);
    rocblasDestroy(handle);
  } else if (op == COMEX_ACC_INT) {
    const int n = bytes/sizeof(int);
    deviceIaxpy(dst, src, scale, n);
  } else if (op == COMEX_ACC_LNG) {
    const int n = bytes/sizeof(long);
    deviceLaxpy(dst, src, scale, n);
  } else if (op == COMEX_ACC_DCP) {
    const int n = bytes/sizeof(DoubleComplexDev);
    rocblasCreate(&handle);
    rocblasZaxpy(handle,n,scale,src,1,dst,1);
    rocblasDestroy(handle);
  } else if (op == COMEX_ACC_CPL) {
    const int n = bytes/sizeof(SingleComplexDev);
    rocblasCreate(&handle);
    rocblasCaxpy(handle,n,scale,src,1,dst,1);
    rocblasDestroy(handle);
  } else {
  }
#endif
}

#endif //ENABLE_DEVICE
#endif /*_DEV_UTILS_H */
