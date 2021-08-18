#ifndef _DEV_UTILS_H
#define _DEV_UTILS_H

#include "dev_mem_handle.h"

#ifdef ENABLE_DEVICE
extern int numDevices();
extern void setDevice(int id);
extern void mallocDevice(void **buf, int size);
extern void freeDevice(void *buf);
extern void copyToDevice(void *hostptr, void *devptr, int bytes);
extern void copyToHost(void *hostptr, void *devptr, int bytes);
extern void copyDevToDev(void *src, void *dst, int bytes);
extern void deviceMemset(void *ptr, int val, size_t bytes);
extern int isHostPointer(void *ptr);
extern void deviceAddInt(int *ptr, const int inc);
extern void deviceAddLong(long *ptr, const long inc);
extern int deviceGetMemHandle(devMemHandle_t *handle, void *memory);
extern int deviceOpenMemHandle(void **memory, devMemHandle_t handle);
extern int deviceCloseMemHandle(void *memory);

extern int _comex_dev_flag;
extern int _comex_dev_id;
#endif

#endif /*_DEV_UTILS_H */
