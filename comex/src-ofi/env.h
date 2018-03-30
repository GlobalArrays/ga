/*
 *  * Copyright (c) 2016 Intel Corporation.  All rights reserved.
 */

#ifndef ENV_H_
#define ENV_H_

#define CACHELINE_SIZE 64

typedef struct env_data_t
{
    int   log_level;
    int   native_atomics;
    int   emulation_type;
    int   progress_thread;
    int   cq_entries_count;
    int   force_sync;
    char* provider;
    char* library_path;
} env_data_t __attribute__ ((aligned (CACHELINE_SIZE)));

extern env_data_t env_data;

int env_to_int(const char* env_name, int* value);
void parse_env_vars();

#endif /* ENV_H_ */
