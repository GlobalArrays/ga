/*
 *  * Copyright (c) 2016 Intel Corporation.  All rights reserved.
 */

#ifndef _COMEX_OFI_H_
#define _COMEX_OFI_H_

#include "env.h"
#include "log.h"

#define STR_ERROR(tbl, ret) CALL_TABLE_FUNCTION(tbl, fi_strerror(ret))
#define OFI_EP_NAME_MAX_LENGTH (512) /* We use constant ep name length */

#define RMA_PROVIDER_CAPS        (FI_RMA | TAGGED_PROVIDER_CAPS)
#define TAGGED_PROVIDER_CAPS     (FI_TAGGED)
#define ATOMICS_PROVIDER_CAPS    (FI_ATOMICS | TAGGED_PROVIDER_CAPS)
#define DESIRED_PROVIDER_CAPS    (RMA_PROVIDER_CAPS | TAGGED_PROVIDER_CAPS | ATOMICS_PROVIDER_CAPS)
#define MR_ACCESS_PERMISSIONS    (FI_READ | FI_WRITE | FI_REMOTE_READ | FI_REMOTE_WRITE)
#define EP_COMPLETIONS_TO_REPORT (FI_TRANSMIT | FI_RECV)

#define OFI_VERSION FI_VERSION(1, 1)
/*#define OFI_VERSION FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION)*/

#define ep_tagged   ep_rma
#define peer_tagged peer_rma

typedef struct peer_t
{
    int       proc; /* world proc */
    fi_addr_t fi_addr;
} peer_t;

typedef struct ofi_ep_t
{
    struct fid_fabric *fabric;   /* fabric object    */
    struct fid_domain *domain;   /* domain object    */
    struct fi_info    *provider; /* provider into */
    struct fid_ep     *endpoint; /* endpoint object  */
    struct fid_cq     *cq;       /* completion queue */
    struct fid_av     *av;       /* address vector   */
    peer_t*           peers;
    enum fi_mr_mode   mr_mode;
} ofi_ep_t;

typedef struct ofi_data_t
{
    ofi_ep_t          ep_rma;
    ofi_ep_t          ep_atomics;
    int               msg_prefix_size;
    int               rma_iov_limit;
    ssize_t           max_bytes_in_atomic[COMEX_DTYPES_COUNT]; /* bytes in one atomic operation per comex datatype */
    int               max_buffered_send;
    uint64_t          mr_counter;
} ofi_data_t;
extern ofi_data_t ofi_data;

#ifndef GA_OFI_STATIC_LINK

#define LOAD_TABLE_FUNCTION(table, fname)                            \
    do {                                                             \
        (table)->fname = dlsym((table)->handle, #fname);             \
        if ((table)->fname == NULL)                                  \
        {                                                            \
            COMEX_OFI_LOG(WARN, "Can't load function %s, error=%s",  \
                          #fname, dlerror());                        \
            dlclose((table)->handle);                                \
            goto fn_fail;                                            \
        }                                                            \
    } while (0)

#define CALL_TABLE_FUNCTION(table, call) (table)->call

#define fi_allocinfo_p() CALL_TABLE_FUNCTION(&ld_table, fi_dupinfo(0))

typedef int             (*fi_getinfo_t)(uint32_t version, const char *node, const char *service,
                          uint64_t flags, struct fi_info *hints, struct fi_info **info);
typedef void            (*fi_freeinfo_t)(struct fi_info *info);
typedef struct fi_info* (*fi_dupinfo_t)(const struct fi_info *info);
typedef int             (*fi_fabric_t)(struct fi_fabric_attr *attr, struct fid_fabric **fabric, void *context);
typedef const char*     (*fi_strerror_t)(int errnum);
typedef char*           (*fi_tostr_t)(const void *data, enum fi_type datatype);

typedef struct fi_loadable_methods
{
    void * handle;

    fi_getinfo_t  fi_getinfo;
    fi_freeinfo_t fi_freeinfo;
    fi_dupinfo_t  fi_dupinfo;
    fi_fabric_t   fi_fabric;
    fi_strerror_t fi_strerror;
    fi_tostr_t    fi_tostr;
} fi_loadable_methods_t;
extern fi_loadable_methods_t ld_table;

static inline int load_ofi(fi_loadable_methods_t* table)
{
    ld_table.handle = dlopen(env_data.library_path, RTLD_NOW);
    if (!ld_table.handle)
    {
        COMEX_OFI_LOG(ERROR, "cannot load default ofi library %s, error=%s", env_data.library_path, dlerror());
        goto fn_fail;
    }

    LOAD_TABLE_FUNCTION(&ld_table, fi_getinfo);
    LOAD_TABLE_FUNCTION(&ld_table, fi_freeinfo);
    LOAD_TABLE_FUNCTION(&ld_table, fi_dupinfo);
    LOAD_TABLE_FUNCTION(&ld_table, fi_fabric);
    LOAD_TABLE_FUNCTION(&ld_table, fi_strerror);
    LOAD_TABLE_FUNCTION(&ld_table, fi_tostr);

fn_success:
    return COMEX_SUCCESS;
fn_fail:
    return COMEX_FAILURE;
}

static inline int unload_ofi(fi_loadable_methods_t* table)
{
    if(ld_table.handle)
        dlclose(ld_table.handle);
    return COMEX_SUCCESS;
}

#else /* GA_OFI_STATIC_LINK */
#define LOAD_TABLE_FUNCTION(table, fname)
#define CALL_TABLE_FUNCTION(table, call) call
#define fi_allocinfo_p(table) fi_allocinfo()
#define load_ofi(table) COMEX_SUCCESS
#define unload_ofi(table) COMEX_SUCCESS
#endif /* GA_OFI_STATIC_LINK */

#endif /* _COMEX_OFI_H_ */
