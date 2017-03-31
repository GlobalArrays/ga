#if HAVE_CONFIG_H
#   include "config.h"
#endif

#define THREAD_SAFE 1

#if defined(PTHREADS) && defined(THREAD_SAFE)

void GA_Internal_Threadsafe_Lock();
void GA_Internal_Threadsafe_Unlock();

#else

#define GA_Internal_Threadsafe_Lock()
#define GA_Internal_Threadsafe_Unlock()

#endif

