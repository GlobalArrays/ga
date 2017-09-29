/*
 *  * Copyright (c) 2016 Intel Corporation.  All rights reserved.
 */

#include <stddef.h>

#include "comex.h"
#include "comex_impl.h"

#include "env.h"
#include "log.h"

#if defined(__APPLE__) && defined(__MACH__)
#define DEFAULT_OFI_LIB "libfabric.dylib"
#else
#define DEFAULT_OFI_LIB "libfabric.so"
#endif

#define IS_SPACE(c)     ((c==0x20 || c==0x09 || c==0x0a || c==0x0b || c==0x0c || c==0x0d) ? 8 : 0)

env_data_t env_data = { ERROR,     /* log_level */
                        1,         /* native_atomics */
                        et_target, /* emulation_type */
                        1,         /* progress_thread */
                        8,         /* cq_entries_count */
                        0,         /* force_sync */
                        NULL,      /* provider */
                        NULL       /* library_path */ };

void parse_env_vars()
{
    env_to_int("COMEX_OFI_LOG_LEVEL", &(env_data.log_level));
    env_to_int("COMEX_OFI_NATIVE_ATOMICS", &(env_data.native_atomics));
    env_to_int("COMEX_OFI_ATOMICS_EMULATION_TYPE", &(env_data.emulation_type));
    env_to_int("COMEX_OFI_PROGRESS_THREAD", &(env_data.progress_thread));
    env_to_int("COMEX_OFI_CQ_ENTRIES_COUNT", &(env_data.cq_entries_count));
    env_to_int("COMEX_OFI_FORCE_SYNC", &(env_data.force_sync));
    env_data.provider = getenv("COMEX_OFI_PROVIDER");
    env_data.library_path = getenv("COMEX_OFI_LIBRARY");

    if (l_state.proc == 0)
    {
        COMEX_OFI_LOG(INFO, "COMEX_OFI_LOG_LEVEL: %d", env_data.log_level);
        COMEX_OFI_LOG(INFO, "COMEX_OFI_NATIVE_ATOMICS: %d", env_data.native_atomics);
        COMEX_OFI_LOG(INFO, "COMEX_OFI_ATOMICS_EMULATION_TYPE: %d", env_data.emulation_type);
        COMEX_OFI_LOG(INFO, "COMEX_OFI_PROGRESS_THREAD: %d", env_data.progress_thread);
        COMEX_OFI_LOG(INFO, "COMEX_OFI_CQ_ENTRIES_COUNT: %d", env_data.cq_entries_count);
        COMEX_OFI_LOG(INFO, "COMEX_OFI_FORCE_SYNC: %d", env_data.force_sync);
        COMEX_OFI_LOG(INFO, "COMEX_OFI_PROVIDER: %s", env_data.provider);
        COMEX_OFI_LOG(INFO, "COMEX_OFI_LIBRARY: %s", env_data.library_path);
    }

    assert(env_data.log_level >= ERROR && env_data.log_level <= TRACE);
    assert(env_data.native_atomics == 0 || env_data.native_atomics == 1);
    assert(env_data.emulation_type == et_origin || env_data.emulation_type == et_target);
    assert(env_data.progress_thread == 0 || env_data.progress_thread == 1);
    assert(env_data.cq_entries_count >= 1 && env_data.cq_entries_count <= 1024);
    assert(env_data.force_sync == 0 || env_data.force_sync == 1);

    if (env_data.native_atomics == 0 && env_data.emulation_type == et_target)
    {
        env_data.progress_thread = 1;
        if (l_state.proc == 0)
            COMEX_OFI_LOG(INFO, "enable progress thread to handle preposted requests in target atomics emulation");
    }

    if (!env_data.library_path || *env_data.library_path == '\0')
        env_data.library_path = DEFAULT_OFI_LIB;
}

int env_to_int(const char* env_name, int* val)
{
    const char* val_ptr;

    val_ptr = getenv(env_name);
    if (val_ptr)
    {
        const char* p;
        int sign = 1, value = 0;
        p = val_ptr;
        while (*p && IS_SPACE(*p)) p++;
        if (*p == '-')
        {
            p++;
            sign = -1;
        }
        if (*p == '+') p++;

        while (*p && isdigit(*p))
            value = 10 * value + (*p++ - '0');

        if (*p)
        {
            COMEX_OFI_LOG(ERROR, "invalid character %c in %s", *p, env_name);
            return -1;
        }
        *val = sign * value;
        return 1;
    }
    return 0;
}
