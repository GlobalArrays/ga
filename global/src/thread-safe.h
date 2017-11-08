#ifndef _GA_THREAD_SAFE_H_
#define _GA_THREAD_SAFE_H_

#define THREAD_SAFE 1

#if defined(PTHREADS) && defined(THREAD_SAFE)

#ifdef __cplusplus
extern "C" {
#endif

extern void GA_Internal_Threadsafe_Lock();
extern void GA_Internal_Threadsafe_Unlock();

#ifdef __cplusplus
}
#endif

#else

#define GA_Internal_Threadsafe_Lock()
#define GA_Internal_Threadsafe_Unlock()

#endif

#endif /* _GA_THREAD_SAFE_H_ */
