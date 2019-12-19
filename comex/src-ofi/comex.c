/*
 *  * Copyright (c) 2016 Intel Corporation.  All rights reserved.
 */

#if HAVE_CONFIG_H
#   include "config.h"
#endif

/* C and/or system headers */
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <string.h>
#include <strings.h>
#include <unistd.h>
#include <dlfcn.h>
#include <stdarg.h>
#include <complex.h>
#include <pthread.h>
#include <sys/uio.h>

/* 3rd party headers */
#include <mpi.h>
#include "rdma/fi_domain.h"
#include "rdma/fi_endpoint.h"
#include "rdma/fi_cm.h"
#include "rdma/fi_rma.h"
#include "rdma/fi_atomic.h"
#include "rdma/fi_errno.h"
#include "rdma/fi_tagged.h"

/* our headers */
#include "comex.h"
#include "comex_impl.h"
#include "datatype.h"
#include "env.h"
#include "groups.h"
#include "log.h"
#include "mutex.h"
#include "ofi.h"

/*#define ATOMIC_NATIVE_ONLY*/

#ifndef ATOMIC_NATIVE_ONLY
#  define USE_ATOMIC_EMULATION
#endif /* ATOMIC_NO_EMULATION */

#define WAIT_COMPLETION_AND_RESET(_request)  \
do {                                         \
    while ((_request)->state != rs_complete) \
    {                                        \
        poll(0);                             \
        PAUSE();                             \
    }                                        \
    reset_request((_request));               \
} while (0)

#define IS_TARGET_ATOMICS_EMULATION() ((env_data.native_atomics == 0 && env_data.emulation_type == et_target) ? 1 : 0)

#ifndef GA_OFI_STATIC_LINK
fi_loadable_methods_t ld_table = {0};
#endif /* GA_OFI_STATIC_LINK */

/* exported state */
local_state l_state;
ofi_data_t ofi_data;

struct request_cache_t* request_cache = 0;

/* static state */
static int  initialized = 0;  /* for comex_initialized(), 0=false */
static char skip_lock = 0;    /* don't acquire or release lock */

/* static function declarations */
static inline int wait_request(request_t* request);
static void acquire_remote_lock(int proc);
static void release_remote_lock(int proc);

static int finalize_ofi();
static int create_mutexes(mutex_t** mtx, int num);
static int destroy_mutexes(mutex_t* mutex);
static inline int destroy_all_windows();

struct fi_cq_attr cq_attr = { 0 };
struct fi_av_attr av_attr = { 0 };

local_window_t* local_wnd     = 0;
ofi_window_t*   ofi_wnd       = 0;
ofi_window_t*   ofi_wnd_cache = 0; /* last accessed window */
mutex_t*        global_mutex  = 0; /* public mutex */
mutex_t*        local_mutex   = 0; /* local mutex */

static int* am_mutex_locked = 0;
static int* am_mutex_waiter = 0;
static uint32_t reply_tag = 0;
#define GETTAG() (++reply_tag)

static fastlock_t mutex_lock;
static fastlock_t acc_lock;
static fastlock_t poll_lock;
static int poll(int* items_processed);
static pthread_t tid = 0;

static  int comex_acc_native(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group);

static  int comex_accs_native(
        int datatype, void* scale,
        void* src_ptr, int* src_stride_ar,
        void* dst_ptr, int* dst_stride_ar,
        int* count, int stride_levels,
        int proc, comex_group_t group);

static  int comex_accv_native(
        int datatype, void* scale,
        comex_giov_t* iov, int iov_len,
        int proc, comex_group_t group);

static  int comex_nbacc_native(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group,
        comex_request_t* handle);

static  int comex_nbaccs_native(
        int datatype, void* scale,
        void* src, int* src_stride,
        void* dst, int* dst_stride,
        int* count, int stride_levels,
        int proc, comex_group_t group,
        comex_request_t* handle);

static  int comex_nbaccv_native(
        int datatype, void* scale,
        comex_giov_t* iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* handle);

#ifdef USE_ATOMIC_EMULATION
static  int comex_acc_emu(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group);

static  int comex_accs_emu(
        int datatype, void* scale,
        void* src_ptr, int* src_stride_ar,
        void* dst_ptr, int* dst_stride_ar,
        int* count, int stride_levels,
        int proc, comex_group_t group);

static  int comex_accv_emu(
        int datatype, void* scale,
        comex_giov_t* iov, int iov_len,
        int proc, comex_group_t group);

static  int comex_nbacc_emu(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group,
        comex_request_t* handle);

static  int comex_nbaccs_emu(
        int datatype, void* scale,
        void* src, int* src_stride,
        void* dst, int* dst_stride,
        int* count, int stride_levels,
        int proc, comex_group_t group,
        comex_request_t *handle);

static  int comex_nbaccv_emu(
        int datatype, void* scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* handle);


typedef int (comex_acc_t)(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group);

typedef int (comex_accs_t)(
        int datatype, void* scale,
        void* src_ptr, int* src_stride_ar,
        void* dst_ptr, int* dst_stride_ar,
        int* count, int stride_levels,
        int proc, comex_group_t group);

typedef int (comex_accv_t)(
        int datatype, void* scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group);

typedef int (comex_nbacc_t)(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group,
        comex_request_t *handle);

typedef int (comex_nbaccs_t)(
        int datatype, void* scale,
        void* src, int* src_stride,
        void* dst, int* dst_stride,
        int* count, int stride_levels,
        int proc, comex_group_t group,
        comex_request_t *handle);

typedef int (comex_nbaccv_t)(
        int datatype, void* scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* handle);

static comex_acc_t*    comex_acc_f    = comex_acc_emu;
static comex_accs_t*   comex_accs_f   = comex_accs_emu;
static comex_accv_t*   comex_accv_f   = comex_accv_emu;
static comex_nbacc_t*  comex_nbacc_f  = comex_nbacc_emu;
static comex_nbaccs_t* comex_nbaccs_f = comex_nbaccs_emu;
static comex_nbaccv_t* comex_nbaccv_f = comex_nbaccv_emu;
#else /* USE_ATOMIC_EMULATION */
#define comex_acc_f    comex_acc_native
#define comex_accs_f   comex_accs_native
#define comex_accv_f   comex_accv_native
#define comex_nbacc_f  comex_nbacc_native
#define comex_nbaccs_f comex_nbaccs_native
#define comex_nbaccv_f comex_nbaccv_native
#endif /* USE_ATOMIC_EMULATION */

#define mr_regv(iov, count, mrs, op_type)             \
do                                                    \
{                                                     \
    struct fi_context ctx;                            \
    int i;                                            \
    for (i = 0; i < count; i++)                       \
    {                                                 \
        COMEX_CHKANDJUMP(mr_reg(iov[i].iov_base,      \
                               iov[i].iov_len,        \
                               MR_ACCESS_PERMISSIONS, \
                               &(mrs)[i],             \
                               &ctx,                  \
                               op_type),              \
                               "fi_mr_reg error:");   \
    }                                                 \
} while(0)

static inline ofi_ep_t* get_ep(op_type_t op_type)
{
    assert(op_type == ot_rma || op_type == ot_atomic);
    return (op_type == ot_rma) ? &(ofi_data.ep_rma) : &(ofi_data.ep_atomics);
}

static inline int mr_reg(const void* buf, size_t len,
                         uint64_t access, struct fid_mr** mr,
                         void* context, op_type_t op_type)
{
    ofi_ep_t* ep = get_ep(op_type);

    uint64_t key = 0;
    if (FI_MR_SCALABLE == ep->mr_mode)
    {
        /**
         * the key should be specified explicitly in case of scalable mr
         * use simple counter to get unique value
         */
        key = __sync_fetch_and_add(&ofi_data.mr_counter, 1);
    }

    OFI_RETRY(fi_mr_reg(ep->domain, /* In:  domain object */
                        buf,        /* In:  lower memory address */
                        len,        /* In:  length */
                        access,     /* In:  access rights */
                        0ULL,       /* In:  offset (not used) */
                        key,        /* In:  requested key */
                        0ULL,       /* In:  flags */
                        mr,         /* Out: memregion object */
                        context),   /* In:  context */
                        "fi_mr_reg error:");
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static inline uint64_t get_remote_addr(uint64_t base_addr, void* ptr, op_type_t op_type)
{
    assert(base_addr <= (uint64_t)ptr);
    ofi_ep_t* ep = get_ep(op_type);

    if (FI_MR_SCALABLE == ep->mr_mode)
    {
        /* returns offset inside of memory region */
        return ((uint64_t)ptr - base_addr);
    }
    else
        return (uint64_t)ptr;
}

static inline int mr_unreg(struct fid* mr)
{
    OFI_CHKANDJUMP(fi_close(mr), "failed to unregister memory");
    return COMEX_SUCCESS;
fn_fail:
    return COMEX_FAILURE;
}

static inline void req_dtor(request_t* request)
{
    assert(request);
    assert(request->state == rs_complete);
    if (request->mr_count && request->mrs)
    {
        int i;
        for (i = 0; i < request->mr_count; i++)
            if (request->mrs[i])
                COMEX_CHKANDJUMP(mr_unreg(&request->mrs[i]->fid), "failed to unregister memory");
        free(request->mrs);
    }

    if (request->mr_single)
        COMEX_CHKANDJUMP(mr_unreg(&request->mr_single->fid), "failed to unregister memory");

    if (request->data && !(request->flags & rf_no_free_data))
        free(request->data);

fn_fail:
    return;
}

static inline int dual_provider()
{
    return ofi_data.ep_rma.provider != ofi_data.ep_atomics.provider;
}

void complete_acc(request_t* request)
{
    assert(request);
    if (request->data)
    {
        acc_data_t* data = (acc_data_t*)request->data;

        if (data->middle)
            free(data->middle);

        struct fi_msg_atomic* msg = (struct fi_msg_atomic*)data->msg;
        if (msg)
        {
            if (msg->msg_iov) free((void*)msg->msg_iov);
            if (msg->rma_iov) free((void*)msg->rma_iov);
            free(msg);
        }

        /* unlock remote host
         * TODO: is it required? per-element atomic is provided by OFI
         */

        /*if (data->proc != PROC_NONE)
            release_remote_lock(data->proc);*/
        free(data);
        request->data = 0;
    }
}

void complete_getput(request_t* request)
{
    assert(request);
    if (request->data)
    {
        struct fi_msg_rma* msg = (struct fi_msg_rma*)request->data;

        assert(msg->context);
        if (msg && msg->context)
        {
            if (msg->msg_iov) free((void*)msg->msg_iov);
            if (msg->rma_iov) free((void*)msg->rma_iov);
        }

        free(request->data);
        request->data = 0;
    }
}

static int exchange_with_all(void* chunk, int len, comex_group_t group, void** result)
{
    void* buf = 0;
    EXPR_CHKANDJUMP((chunk && len && result), "incorrect arguments");

    comex_igroup_t* igroup = comex_get_igroup_from_group(group);
    EXPR_CHKANDJUMP(igroup, "failed to lookup group");

    int group_size = 0;
    COMEX_CHKANDJUMP(comex_group_size(group, &group_size), "failed to get group size");

    buf = malloc(len * group_size);
    EXPR_CHKANDJUMP(buf, "failed to allocate buffer");

    MPI_CHKANDJUMP(MPI_Allgather(chunk, len, MPI_BYTE, buf, len, MPI_BYTE, igroup->comm),
                   "failed to perform MPI_Allgather");

    *result = buf;

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    if (buf) free(buf);
    return COMEX_FAILURE;
}

static int connect_all(ofi_ep_t* ep)
{
    int ret = COMEX_SUCCESS;

    size_t name_len = OFI_EP_NAME_MAX_LENGTH;
    char name[OFI_EP_NAME_MAX_LENGTH];

    OFI_CALL(ret, fi_getname(&ep->endpoint->fid, &name, &name_len));
    OFI_CHKANDJUMP(ret, "fi_getname:");

    /* exchange OFI end-points adresses */

    /* check if all addresses have same length */
    typedef struct name_length_t
    {
        int proc;
        int length;
    } name_length_t;

    typedef struct proc_name_t
    {
        int  proc;
        char name[];
    } proc_name_t;

    int group_size = l_state.size;
    int i;
    name_length_t* lengths = 0;
    char* names = 0;
    proc_name_t* my_name = 0;
    name_length_t my_name_len = {l_state.proc, name_len};
    COMEX_CHKANDJUMP(exchange_with_all(&my_name_len, sizeof(my_name_len),
                                       COMEX_GROUP_WORLD, (void**)&lengths),
                                       "failed to exchange name_len");

    for (i = 0; i < group_size; i++)
    {
        EXPR_CHKANDJUMP(lengths[i].length == my_name_len.length,
                        "proc %d has incorrect name length: %d (expected %d)",
                        lengths[i].proc, lengths[i].length, my_name_len.length);
    }

    /* ok, all name lengths are equal. let's publish it */
    int sizeof_procname = sizeof(proc_name_t) + my_name_len.length;
    my_name = malloc(sizeof_procname);
    EXPR_CHKANDJUMP(my_name, "failed to allocate data");

    my_name->proc = my_name_len.proc;
    memcpy(my_name->name, name, my_name_len.length);

    COMEX_CHKANDJUMP(exchange_with_all(my_name, sizeof_procname, COMEX_GROUP_WORLD, (void**)&names),
                                       "failed to exchange proc name");

    ep->peers = malloc(sizeof(*(ep->peers)) * group_size);
    EXPR_CHKANDJUMP(ep->peers, "failed to allocate peer's data");
    memset(ep->peers, 0, sizeof(*(ep->peers)) * group_size);

    /* fill ofi_data.peers array: use proc as index in array */
    for (i = 0; i < group_size; i++)
    {
        proc_name_t* proc_name = (proc_name_t*)(names + (i * sizeof_procname));
        assert(proc_name->proc < group_size);
        peer_t* peer = ep->peers + proc_name->proc;
        peer->proc = proc_name->proc;
        struct fi_context av_context;
        int ret;
        OFI_CALL(ret, fi_av_insert(ep->av,
                                   proc_name->name,
                                   1,
                                   &peer->fi_addr,
                                   0,
                                   &av_context));
        OFI_CHKANDJUMP(ret, "failed to fi_av_insert:");
    }

fn_success:
    if (lengths) free(lengths);
    if (names) free(names);
    return COMEX_SUCCESS;

fn_fail:
    if (my_name) free(my_name);
    if (lengths) free(lengths);
    if (names) free(names);
    return COMEX_FAILURE;
}

void tune_ofi_provider()
{
    #define VAR_NAME_LEN 32
    #define PSM_COUNT 2
    #define PREFIX_COUNT 2

    char var_name[VAR_NAME_LEN] = {0};
    const char * psms[PSM_COUNT] = {"PSM", "PSM2"};
    const char * prefixes[PREFIX_COUNT] = {"OFI", "FI"};

    int prefix_idx;
    for (prefix_idx = 0; prefix_idx < PREFIX_COUNT; prefix_idx++)
    {
        int psm_idx = 0;
        for (; psm_idx < PSM_COUNT; psm_idx++)
        {
            snprintf(var_name, VAR_NAME_LEN, "%s_%s_NAME_SERVER", prefixes[prefix_idx], psms[psm_idx]);
            setenv(var_name, "0", 1);

            snprintf(var_name, VAR_NAME_LEN, "%s_%s_TAGGED_RMA", prefixes[prefix_idx], psms[psm_idx]);
            setenv(var_name, "0", 0);
        }
    }

    char var[256];
    sprintf(var, "%d", l_state.local_size);
    setenv("MPI_LOCALNRANKS", var, 0);
    sprintf(var, "%d", l_state.local_proc);
    setenv("MPI_LOCALRANKID", var, 0);

    setenv("IPATH_NO_CPUAFFINITY", "1", 0);
    setenv("HFI_NO_CPUAFFINITY", "1", 0);
}

static int init_ep(struct fi_info* hints, ofi_ep_t* ep, int suppress_fail)
{
    memset(ep, 0, sizeof(*ep)); /* clean ep data */

    struct fi_info* provider = 0;
    int ret;
    OFI_CALL(ret, CALL_TABLE_FUNCTION(&ld_table,
                                      fi_getinfo(OFI_VERSION,
                                                 NULL,
                                                 NULL,
                                                 0ULL,
                                                 hints,
                                                 &provider)));
    if (ret < 0 && suppress_fail)
        goto fn_fail;

    OFI_CHKANDJUMP(ret, "fi_getinfo:");
    EXPR_CHKANDJUMP(provider, "no provider found with desired capabilities");

    COMEX_OFI_LOG(INFO, "using provider '%s.%d'",
                  provider->fabric_attr->prov_name,
                  provider->fabric_attr->prov_version);

    ep->mr_mode = provider->domain_attr->mr_mode;
    COMEX_OFI_LOG(INFO, "prov_name: %s, mr_mode %d", provider->fabric_attr->prov_name, ep->mr_mode);
    EXPR_CHKANDJUMP(((ep->mr_mode == FI_MR_BASIC) || (ep->mr_mode == FI_MR_SCALABLE)), "unsupported mr mode");

    /* ---------------------------- */
    /* Open fabric                  */
    /* ---------------------------- */
    OFI_CALL(ret, CALL_TABLE_FUNCTION(&ld_table, fi_fabric(provider->fabric_attr, &ep->fabric, NULL)));
    if (ret < 0)
    {
        if (!suppress_fail) OFI_CHKANDJUMP(ret, "fi_fabric('%s')", provider->fabric_attr->prov_name);
        else goto fn_fail;
    }

    /* ---------------------------- */
    /* Open domain                  */
    /* ---------------------------- */
    OFI_CALL(ret, fi_domain(ep->fabric, provider, &ep->domain, NULL));
    if (ret < 0)
    {
        if(!suppress_fail) OFI_CHKANDJUMP(ret, "fi_domain");
        else goto fn_fail_dom;
    }

    /* ----------------------------- */
    /* Open endpoint                 */
    /* ----------------------------- */
    OFI_CALL(ret, fi_endpoint(ep->domain, provider, &ep->endpoint, NULL));
    if (ret < 0)
    {
        if (!suppress_fail) OFI_CHKANDJUMP(ret, "fi_endpoint");
        else goto fn_fail_ep;
    }

    /* -------------------------------- */
    /* Open Completion Queue            */
    /* -------------------------------- */
    memset(&cq_attr, 0, sizeof(cq_attr));
    cq_attr.format = FI_CQ_FORMAT_TAGGED;

    OFI_CHKANDJUMP(fi_cq_open(ep->domain, &cq_attr, &ep->cq, NULL), "fi_cq_open:");

    /* -------------------------------- */
    /* Open Address Vector              */
    /* -------------------------------- */
    memset(&av_attr, 0, sizeof(av_attr));
    av_attr.type = FI_AV_MAP;

    OFI_CHKANDJUMP(fi_av_open(ep->domain, &av_attr, &ep->av, NULL), "fi_av_open:");

    /* -------------------------------- */
    /* Bind Endpoint to both the CQ     */
    /* and to the AV                    */
    /* -------------------------------- */
    OFI_CHKANDJUMP(fi_ep_bind(ep->endpoint, (fid_t)ep->cq, EP_COMPLETIONS_TO_REPORT), "fi_bind EP-CQ:");
    OFI_CHKANDJUMP(fi_ep_bind(ep->endpoint, (fid_t)ep->av, 0), "fi_bind EP-AV:");

    /* -------------------------------- */
    /* Enable EP                        */
    /* -------------------------------- */
    OFI_CHKANDJUMP(fi_enable(ep->endpoint), "fi_enable:");

    COMEX_CHKANDJUMP(connect_all(ep), "connect_all error");

    ep->provider = provider;

fn_success:
    return COMEX_SUCCESS;

fn_fail_ep:
    fi_close(&ep->domain->fid);
fn_fail_dom:
    fi_close(&ep->fabric->fid);
fn_fail:
    if (provider)
        OFI_VCALL(CALL_TABLE_FUNCTION(&ld_table, fi_freeinfo(provider)));
    return COMEX_FAILURE;
}

#define PREPOST_ATOMICS_COUNT 64
static ofi_atomics_t atomics_headers[PREPOST_ATOMICS_COUNT];
static request_t atomics_requests[PREPOST_ATOMICS_COUNT];
static int completed_atomics_count = 0;

static volatile int progress_thread_complete = 0;
static pthread_t progress_thread = 0;

static void* progress_thread_func(void* __data)
{
    while (!progress_thread_complete)
    {
        poll(0);
        PAUSE();
    }
    return 0;
}

#define PREPOST_ATOMICS()                                          \
do                                                                 \
{                                                                  \
    completed_atomics_count = 0;                                   \
    int req_idx;                                                   \
    for (req_idx = 0; req_idx < PREPOST_ATOMICS_COUNT; req_idx++)  \
    {                                                              \
        request_t* req = &atomics_requests[req_idx];               \
        ofi_atomics_t* header = &atomics_headers[req_idx];         \
        init_request(req);                                         \
        req->data = header;                                        \
        req->cmpl = target_emulated_atomics_completion;            \
        OFI_RETRY(fi_trecv(ofi_data.ep_tagged.endpoint,            \
                           header,                                 \
                           sizeof(*header),                        \
                           0,                                      \
                           FI_ADDR_UNSPEC,                         \
                           ATOMICS_PROTO_TAGMASK,                  \
                           ATOMICS_PROTO_IGNOREMASK,               \
                           req),                                   \
                           "fi_trecv: failed to prepost request"); \
    }                                                              \
} while(0)

#define ADD(_dst, _src, _len, type)         \
do                                          \
{                                           \
    int i;                                  \
    type* dst = (type*)_dst;                \
    type* src = (type*)_src;                \
    int cnt = (_len) / sizeof(type);        \
    fastlock_acquire(&acc_lock);            \
    for (i = 0; i < cnt; i++, dst++, src++) \
        *dst += *src;                       \
    fastlock_release(&acc_lock);            \
} while(0)

static void chunk_acc_completion(request_t* request)
{
    ofi_atomics_t* header = request->data;
    assert(header);
    EXPR_CHKANDJUMP(header, "incorrect header");

    switch (header->proto.op)
    {
        case COMEX_ACC_DBL:
            ADD(header->acc.addr, header->acc.data, header->acc.posted, double);
            break;
        case COMEX_ACC_FLT:
            ADD(header->acc.addr, header->acc.data, header->acc.posted, float);
            break;
        case COMEX_ACC_INT:
            ADD(header->acc.addr, header->acc.data, header->acc.posted, int);
            break;
        case COMEX_ACC_LNG:
            ADD(header->acc.addr, header->acc.data, header->acc.posted, long);
            break;
        case COMEX_ACC_DCP:
            ADD(header->acc.addr, header->acc.data, header->acc.posted, double complex);
            break;
        case COMEX_ACC_CPL:
            ADD(header->acc.addr, header->acc.data, header->acc.posted, float complex);
            break;
        default:
            assert(0);
            break;
    }
fn_fail:
    return;
}

static void full_acc_completion(request_t* request)
{
    ofi_atomics_t* header = request->data;
    assert(header);
    EXPR_CHKANDJUMP(header, "incorrect header");
    int v;
    OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                         &v,
                         sizeof(v),
                         ofi_data.ep_tagged.peers[header->proto.proc].fi_addr,
                         ATOMICS_ACC_CMPL_TAGMASK | header->proto.tag),
                         "fi_tinject: failed");
fn_fail:
    return;
}

static void target_emulated_atomics_completion(request_t* request)
{
    int proc = PROC_NONE;

    assert(request);
    ofi_atomics_t header = *(ofi_atomics_t*)request->data;
    request->state = rs_complete;

    if (__sync_add_and_fetch(&completed_atomics_count, 1) == PREPOST_ATOMICS_COUNT)
    {
        /* re-post atomics requests */
        PREPOST_ATOMICS();
    }

    switch (header.proto.op)
    {
        case COMEX_FETCH_AND_ADD:
        {
            int v = __sync_fetch_and_add((int*)header.rmw.addr, header.rmw.extra);
            OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                                 &v,
                                 sizeof(v),
                                 ofi_data.ep_tagged.peers[header.proto.proc].fi_addr,
                                 ATOMICS_DATA_TAGMASK | header.proto.tag),
                                 "fi_tinject: failed");
        }
        break;
        case COMEX_FETCH_AND_ADD_LONG:
        {
            uint64_t v = __sync_fetch_and_add((uint64_t*)header.rmw.addr, (uint64_t)header.rmw.extra);
            OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                                 &v,
                                 sizeof(v),
                                 ofi_data.ep_tagged.peers[header.proto.proc].fi_addr,
                                 ATOMICS_DATA_TAGMASK | header.proto.tag),
                                 "fi_tinject: failed");
        }
        break;
        case COMEX_SWAP:
        {
            int v = __sync_lock_test_and_set((int*)header.rmw.addr, (int)header.rmw.src);
            OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                                 &v,
                                 sizeof(v),
                                 ofi_data.ep_tagged.peers[header.proto.proc].fi_addr,
                                 ATOMICS_DATA_TAGMASK | header.proto.tag),
                                 "fi_tinject: failed");
        }
        break;
        case COMEX_SWAP_LONG:
        {
            uint64_t v = __sync_lock_test_and_set((uint64_t*)header.rmw.addr, header.rmw.src);
            OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                                 &v,
                                 sizeof(v),
                                 ofi_data.ep_tagged.peers[header.proto.proc].fi_addr,
                                 ATOMICS_DATA_TAGMASK | header.proto.tag),
                                 "fi_tinject: failed");
        }
        break;
        case COMEX_ACC_DBL:
        case COMEX_ACC_FLT:
        case COMEX_ACC_INT:
        case COMEX_ACC_LNG:
        case COMEX_ACC_DCP:
        case COMEX_ACC_CPL:
        {
            int i;
            request_t* parent = 0;
            size_t chunk = (size_t)header.acc.len + sizeof(header);
            size_t total = (size_t)header.acc.count * chunk + sizeof(header);
            char* buffer = 0;

            parent = alloc_request();
            parent->dtor = req_dtor;
            parent->cmpl = full_acc_completion;
            parent->data = (char*)malloc(total);
            EXPR_CHKANDJUMP(parent->data, "failed to allocate data");
            parent->flags |= rf_auto_free;
            increment_request_cnt(parent);

            /* after all accs are completed - reply packet will be sent.
             * save data for reply packet into buffer. */
            ofi_atomics_t* reply_header = parent->data;
            *reply_header = header;
            buffer = reply_header->acc.data;

            if (header.acc.len > ofi_data.max_buffered_send && buffer && 0)
            {
                /* allocate region to receive data */
                struct fi_context context;
                COMEX_CHKANDJUMP(mr_reg(buffer, total, MR_ACCESS_PERMISSIONS,
                                        &parent->mr_single, &context, ot_rma),
                                        "fi_mr_reg failed:");
            }

            for (i = 0; i < header.acc.count; i++)
            {
                request_t* request = alloc_request();
                request->dtor = req_dtor;
                request->flags |= rf_no_free_data; /* data will be removed by parent */
                set_parent_request(parent, request);
                request->data = buffer;
                buffer += chunk;
                request->cmpl = chunk_acc_completion;
                OFI_RETRY(fi_trecv(ofi_data.ep_tagged.endpoint,
                                   request->data,
                                   chunk,
                                   parent->mr_single ? fi_mr_desc(parent->mr_single) : 0,
                                   ofi_data.ep_tagged.peers[header.proto.proc].fi_addr,
                                   ATOMICS_ACC_DATA_TAGMASK | header.proto.tag, 0,
                                   request),
                                   "fi_trecv: failed to prepost request");
            }
            assert(parent->state == rs_progress);
            decrement_request_cnt(parent);
        }
        break;
        case OFI_MUTEX_AM_LOCK:
        {
            assert(am_mutex_locked);
            assert(am_mutex_waiter);
            fastlock_acquire(&mutex_lock);

            if (am_mutex_locked[header.mutex.num] == PROC_NONE)
            {
                /* mutex is not locked */
                proc = header.proto.proc;
                am_mutex_locked[header.mutex.num] = header.proto.proc;
            }
            else
            {
                /* mutex is locked. add rank to waiters list */
                int idx = am_mutex_locked[header.mutex.num];
                do
                {
                    if (am_mutex_waiter[idx] == PROC_NONE)
                    {
                        am_mutex_waiter[idx] = header.proto.proc;
                        idx = PROC_NONE;
                    }
                    else idx = am_mutex_waiter[idx];
                } while (idx != PROC_NONE);
            }

            fastlock_release(&mutex_lock);
            if (proc != PROC_NONE)
            {
                int v = 0;
                OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                                     &v,
                                     sizeof(v),
                                     ofi_data.ep_tagged.peers[header.proto.proc].fi_addr,
                                     ATOMICS_MUTEX_TAGMASK | header.proto.proc),
                                     "fi_tinject: failed");
            }
        }
        break;
        case OFI_MUTEX_AM_UNLOCK:
        {
            assert(am_mutex_locked);
            assert(am_mutex_waiter);
            fastlock_acquire(&mutex_lock);
            assert(am_mutex_locked[header.mutex.num] == header.proto.proc);
            am_mutex_locked[header.mutex.num] = am_mutex_waiter[header.proto.proc];
            am_mutex_waiter[header.proto.proc] = PROC_NONE;
            proc = am_mutex_locked[header.mutex.num];
            fastlock_release(&mutex_lock);
            if (proc != PROC_NONE) /* notify new owner of mutex */
            {
                int v = 0;
                OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                                     &v,
                                     sizeof(v),
                                     ofi_data.ep_tagged.peers[proc].fi_addr,
                                     ATOMICS_MUTEX_TAGMASK | proc),
                                     "fi_tinject: failed");
            }
        }
        break;
        default:
            COMEX_OFI_LOG(ERROR, "incorrect atomic operation type: %d", header.proto.op);
            break;
    }
fn_success:
    return;

fn_fail:
    return;
}

static int init_ofi()
{
    OFI_LOCK_INIT();

    parse_env_vars();

    if (load_ofi(&ld_table) != COMEX_SUCCESS)
        goto fn_fail;

    struct fi_info* hints_saw = fi_allocinfo_p();
    hints_saw->mode                          = FI_CONTEXT;
    hints_saw->ep_attr->type                 = FI_EP_RDM;   /* Reliable datagram */
    hints_saw->caps                          = DESIRED_PROVIDER_CAPS;
    hints_saw->domain_attr->threading        = FI_THREAD_ENDPOINT;
    hints_saw->domain_attr->control_progress = FI_PROGRESS_AUTO;
    hints_saw->domain_attr->data_progress    = FI_PROGRESS_AUTO;
    hints_saw->fabric_attr->prov_name        = env_data.provider ? strdup(env_data.provider) : 0;

    struct fi_info* hints_remcon = CALL_TABLE_FUNCTION(&ld_table, fi_dupinfo(hints_saw));

    hints_saw->tx_attr->op_flags            |= FI_COMPLETION | FI_TRANSMIT_COMPLETE;
    hints_saw->tx_attr->msg_order           |= FI_ORDER_SAW;
    hints_saw->rx_attr->msg_order           |= FI_ORDER_SAW;

    hints_remcon->tx_attr->op_flags         |= FI_COMPLETION | FI_DELIVERY_COMPLETE;

    /* ------------------------------------------------------------------------ */
    /* Set default settings before any ofi-provider is inited                   */
    /* (before any call of fi_getinfo and fi_fabric)                            */
    /* ------------------------------------------------------------------------ */
    tune_ofi_provider();

    /* first try to initialize requested endpoint with all desired capabilities */

    int init_done;

    /* first try to get provider where delivery complete supported */
    init_done = (init_ep(hints_remcon, &ofi_data.ep_rma, 1) == COMEX_SUCCESS);

    if (init_done)
        ofi_data.ep_atomics = ofi_data.ep_rma;
    else
    {
        init_done = (init_ep(hints_saw, &ofi_data.ep_rma, 1) == COMEX_SUCCESS);
        if (init_done)
        {
            /* great!!! we got provider with all required caps */
            ofi_data.ep_atomics = ofi_data.ep_rma;
        }
    }

    if (!init_done)
    {
        /* ok, try to use different providers for RMA & atomics */
        hints_saw->caps = RMA_PROVIDER_CAPS;
        hints_remcon->caps = RMA_PROVIDER_CAPS;

        init_done = (init_ep(hints_remcon, &ofi_data.ep_rma, 1) == COMEX_SUCCESS);
        if (!init_done)
            init_done = (init_ep(hints_saw, &ofi_data.ep_rma, 1) == COMEX_SUCCESS);
        EXPR_CHKANDJUMP(init_done, "failed to create endpoint");

        if (IS_TARGET_ATOMICS_EMULATION())
        {
            /* in case when emulated atomics are performed on target side use only p2p api */
            ofi_data.ep_atomics = ofi_data.ep_rma;
        }
        else /* native or emulated on origin side (acc over rma, rmw over atomic) */
        {
            hints_remcon->caps = ATOMICS_PROVIDER_CAPS;
            hints_saw->caps = ATOMICS_PROVIDER_CAPS;
            hints_remcon->fabric_attr->prov_name = 0;
            hints_saw->fabric_attr->prov_name = 0;

            init_done = (init_ep(hints_remcon, &ofi_data.ep_atomics, 0) == COMEX_SUCCESS);
            if (!init_done)
                init_done = (init_ep(hints_saw, &ofi_data.ep_atomics, 0) == COMEX_SUCCESS);
            EXPR_CHKANDJUMP(init_done, "failed to create endpoint");
        }
    }

    CALL_TABLE_FUNCTION(&ld_table, fi_freeinfo(hints_saw));
    CALL_TABLE_FUNCTION(&ld_table, fi_freeinfo(hints_remcon));

    /* ----------------------------- */
    /* Get provider limitations      */
    /* ----------------------------- */
    if (ofi_data.ep_tagged.provider->mode & FI_MSG_PREFIX && ofi_data.ep_tagged.provider->ep_attr)
        ofi_data.msg_prefix_size = ofi_data.ep_tagged.provider->ep_attr->msg_prefix_size;
    if (ofi_data.ep_rma.provider->tx_attr)
    {
        ofi_data.rma_iov_limit = ofi_data.ep_rma.provider->tx_attr->rma_iov_limit;
        ofi_data.max_buffered_send = ofi_data.ep_rma.provider->tx_attr->inject_size;
    }

    int comex_dtype = 0;
    enum fi_datatype ofi_dtype = FI_DATATYPE_LAST;
    size_t max_elems_in_atomic = 0;

#ifdef USE_ATOMIC_EMULATION
    if (env_data.native_atomics)
    {
        comex_acc_f    = comex_acc_native;
        comex_accs_f   = comex_accs_native;
        comex_accv_f   = comex_accv_native;
        comex_nbacc_f  = comex_nbacc_native;
        comex_nbaccs_f = comex_nbaccs_native;
        comex_nbaccv_f = comex_nbaccv_native;
    }
    else
    {
        comex_acc_f    = comex_acc_emu;
        comex_accs_f   = comex_accs_emu;
        comex_accv_f   = comex_accv_emu;
        comex_nbacc_f  = comex_nbacc_emu;
        comex_nbaccs_f = comex_nbaccs_emu;
        comex_nbaccv_f = comex_nbaccv_emu;
    }
#endif /* USE_ATOMIC_EMULATION */

    if (env_data.native_atomics)
    {
        for (comex_dtype = COMEX_ACC_INT; comex_dtype <= COMEX_ACC_LNG; comex_dtype++)
        {
            ofi_dtype = GET_FI_DTYPE(comex_dtype);
            EXPR_CHKANDJUMP(ofi_dtype != -1, "datatype is not supported: %d", comex_dtype);
            int ret;
            OFI_CALL(ret, fi_atomicvalid(ofi_data.ep_atomics.endpoint, ofi_dtype, FI_SUM, &max_elems_in_atomic));

            if (ret < 0)
                ofi_data.max_bytes_in_atomic[COMEX_DTYPE_IDX(comex_dtype)] = -1;
            else
            {
                size_t comex_dtype_size = 0;
                COMEX_DTYPE_SIZEOF(comex_dtype, comex_dtype_size);
                ofi_data.max_bytes_in_atomic[COMEX_DTYPE_IDX(comex_dtype)] = max_elems_in_atomic * comex_dtype_size;
            }
        }
    }

    if (l_state.proc == 0)
    {
        COMEX_OFI_LOG(INFO, "rma_ep: %s", ofi_data.ep_rma.provider->fabric_attr->prov_name);
        COMEX_OFI_LOG(INFO, "atomics_ep: %s", ofi_data.ep_atomics.provider->fabric_attr->prov_name);
        COMEX_OFI_LOG(INFO, "msg_prefix_size: %d", ofi_data.msg_prefix_size);
        COMEX_OFI_LOG(INFO, "rma_iov_limit: %d", ofi_data.rma_iov_limit);
        COMEX_OFI_LOG(INFO, "max_buffered_send: %d", ofi_data.max_buffered_send);

        if (env_data.native_atomics)
        {
            for (comex_dtype = COMEX_ACC_INT; comex_dtype <= COMEX_ACC_LNG; comex_dtype++)
                COMEX_OFI_LOG(INFO, "max_bytes_in_atomic: datatype %d, bytes %zd",
                              comex_dtype, ofi_data.max_bytes_in_atomic[COMEX_DTYPE_IDX(comex_dtype)]);
        }
    }

    if (env_data.progress_thread)
    {
        tid = pthread_self();
        pthread_create(&progress_thread, 0, progress_thread_func, 0);
    }

    /* prepost atomics requests */
    if (IS_TARGET_ATOMICS_EMULATION())
        PREPOST_ATOMICS();

    ofi_data.mr_counter = 0;

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    finalize_ofi();
    return COMEX_FAILURE;
}

#define CQ_CHKANDJUMP(cq, ret)                                               \
do                                                                           \
{                                                                            \
    if (ret < 0 && ret != -FI_EAGAIN)                                        \
    {                                                                        \
        struct fi_cq_err_entry error;                                        \
        COMEX_OFI_LOG(INFO, "cq_read: error available");                     \
        int err = fi_cq_readerr(cq, (void*)&error, 0);                       \
        if (err < 0)                                                         \
        {                                                                    \
            COMEX_OFI_LOG(INFO, "cq_read_err: can't retrieve error... (%ld)",\
                          ret);                                              \
            goto fn_fail;                                                    \
        }                                                                    \
        COMEX_OFI_LOG(INFO, "cq_read_err: error is %d (ret=%ld)",            \
                      error.err, ret);                                       \
        goto fn_fail;                                                        \
    }                                                                        \
} while(0)

#define CQ_READ(cq)                                                 \
  do                                                                \
  {                                                                 \
      if (OFI_TRYLOCK())                                            \
      {                                                             \
          locked = 1;                                               \
          ret = fi_cq_read(cq, entries, env_data.cq_entries_count); \
          CQ_CHKANDJUMP(cq, ret);                                   \
          OFI_UNLOCK();                                             \
          locked = 0;                                               \
      }                                                             \
  } while (0)

static int poll(int* items_processed)
{
    ssize_t ret = 0;
    int locked = 0;
    struct fi_cq_tagged_entry entries[env_data.cq_entries_count];
    memset(entries, 0, sizeof(entries));

    CQ_READ(ofi_data.ep_rma.cq);
    if (ret <= 0 && dual_provider())
        CQ_READ(ofi_data.ep_atomics.cq);

    if (items_processed)
        *items_processed = 0;

    if (ret > 0)
    {
        int idx;
        for (idx = 0; idx < ret; idx++)
        {
            request_t* request = (request_t*)entries[idx].op_context;
            if (request && request->magic == REQUEST_MAGIC)
                complete_request(request);
        }

        if (items_processed)
            *items_processed = (int)ret;
    }
    else if (ret == -FI_EAGAIN) {}
    else if (ret < 0) { assert(0); /* should not be here */ }

fn_success:
    return COMEX_SUCCESS;
fn_fail:
    if (locked) OFI_UNLOCK();
    return COMEX_FAILURE;
}

int comex_init()
{
    int status;

    if (initialized) return 0;
    initialized = 1;

    /* Assert MPI has been initialized */
    int init_flag;
    status = MPI_Initialized(&init_flag);
    assert(MPI_SUCCESS == status);
    assert(init_flag);

    /* Duplicate the World Communicator */
    status = MPI_Comm_dup(MPI_COMM_WORLD, &(l_state.world_comm));
    assert(MPI_SUCCESS == status);
    assert(l_state.world_comm);

    /* My Proc Number */
    status = MPI_Comm_rank(l_state.world_comm, &(l_state.proc));
    assert(MPI_SUCCESS == status);

    /* World Size */
    status = MPI_Comm_size(l_state.world_comm, &(l_state.size));
    assert(MPI_SUCCESS == status);

    /* Evaluate local-proc information: local comm & rank */
    MPI_Comm local_comm;
    status = MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local_comm);
    assert(MPI_SUCCESS == status);
    assert(local_comm);

    /* Local Proc Number */
    status = MPI_Comm_rank(local_comm, &(l_state.local_proc));
    assert(MPI_SUCCESS == status);

    /* Local Size */
    status = MPI_Comm_size(local_comm, &(l_state.local_size));
    assert(MPI_SUCCESS == status);
    MPI_Comm_free(&local_comm);

    /* groups */
    comex_group_init();

    /* OFI initialization */
    COMEX_CHKANDJUMP(init_ofi(), "failed to init ofi");

    request_cache = create_request_cache();

    int ret = create_mutexes(&local_mutex, 1);
    if (ret == COMEX_SUCCESS)
    {
        assert(local_mutex);
        local_mutex->tagmask = LOCAL_MUTEX_TAGMASK;
    }

    /* Synch - Sanity Check */
    MPI_Barrier(l_state.world_comm);

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_init_args(int* argc, char*** argv)
{
    int init_flag;

    MPI_Initialized(&init_flag);
    if (!init_flag)
        MPI_CHKANDJUMP(MPI_Init(argc, argv), "failed to init mpi");

    COMEX_CHKANDJUMP(comex_init(), "failed to init comex");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_initialized()
{
    return initialized;
}

void comex_error(char *msg, int code)
{
    COMEX_OFI_LOG(ERROR, "[%d] Received an Error in Communication: (%d) %s",
                  l_state.proc, code, msg);
    MPI_Abort(l_state.world_comm, code);
}

static int lookup_window(void* ptr, int size, int proc, comex_group_t group, ofi_window_t** res)
{
    VALIDATE_GROUP_AND_PROC(group, proc);
    int world_proc = PROC_NONE;
    comex_group_translate_world(group, proc, &world_proc);
    EXPR_CHKANDJUMP((world_proc != PROC_NONE), "invalid world proc");

    uint64_t uptr = (uint64_t)ptr;
    ofi_window_t* wnd = ofi_wnd;

#define CHECK_WND(_wnd, _proc, _ptr, _size) \
    (_wnd->world_proc == _proc && _wnd->ptr <= _ptr && _wnd->ptr + _wnd->size >= _ptr + _size)

    /* first look in cache */
    if (ofi_wnd_cache && CHECK_WND(ofi_wnd_cache, world_proc, uptr, size))
    {
        *res = ofi_wnd_cache;
        return COMEX_SUCCESS;
    }

    /* else traverse list of windows */
    while (wnd)
    {
        assert(wnd->local);
        if (CHECK_WND(wnd, world_proc, uptr, size))
        {
            *res = ofi_wnd_cache = wnd;
            return COMEX_SUCCESS;
        }
        else if (wnd->world_proc == proc && wnd->ptr <= (uint64_t)ptr && wnd->ptr + (uint64_t)wnd->size > (uint64_t)ptr)
        {
            COMEX_OFI_LOG(INFO, "WARNING: found candidate window: missing %d bytes tail (length: %lu, expected: %d:%d)",
                          (int)(((long)ptr + size) - ((long)wnd->ptr + wnd->size)), wnd->size, (int)((long)ptr - (long)wnd->ptr), size);
        }
        wnd = wnd->next;
    }

fn_fail:
    return COMEX_FAILURE;
}

int comex_put(
        void* src, void* dst, int bytes,
        int proc, comex_group_t group)
{
    comex_request_t handle;
    COMEX_CHKANDJUMP(comex_nbput(src, dst, bytes, proc, group, &handle),
                     "failed to perform comex_nbput");
    COMEX_CHKANDJUMP(comex_wait(&handle),
                     "failed to perform comex_wait");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_get(
        void* src, void* dst, int bytes,
        int proc, comex_group_t group)
{
    comex_request_t handle;
    COMEX_CHKANDJUMP(comex_nbget(src, dst, bytes, proc, group, &handle),
                     "failed to perform comex_get");
    COMEX_CHKANDJUMP(comex_wait(&handle),
                     "failed to perform comex_wait");

fn_success:
    return COMEX_SUCCESS;
fn_fail:
    return COMEX_FAILURE;
}

static int comex_acc_native(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group)
{
    comex_request_t handle;
    COMEX_CHKANDJUMP(comex_nbacc_native(datatype, scale, src_ptr, dst_ptr, bytes, proc, group, &handle),
                     "failed to perform comex_nbacc");
    COMEX_CHKANDJUMP(comex_wait(&handle),
                     "failed to perform comex_wait");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static int list_strides(
        int* src_stride_ar,
        int* dst_stride_ar,
        int* count, int stride_levels,
        int (*callback)(size_t src_idx, size_t dst_idx, void* data), void* data)
{
    int i, j;
    size_t src_idx, dst_idx;  /* index offset of current block position to ptr */
    size_t n1dim;  /* number of 1 dim block */
    size_t src_bvalue[7], src_bunit[7];
    size_t dst_bvalue[7], dst_bunit[7];

    assert(callback);

    /* number of n-element of the first dimension */
    n1dim = 1;
    for (i = 1; i <= stride_levels; i++)
    {
        n1dim *= count[i];
    }

    /*assert(n1dim < 0x100000000);*/

    /* calculate the destination indices */
    src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;
    dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

    for (i = 2; i <= stride_levels; i++)
    {
        src_bvalue[i] = 0;
        dst_bvalue[i] = 0;
        src_bunit[i] = src_bunit[i - 1] * count[i - 1];
        dst_bunit[i] = dst_bunit[i - 1] * count[i - 1];
    }

    for (i = 0; i < n1dim; i++)
    {
        src_idx = 0;
        for (j = 1; j <= stride_levels; j++)
        {
            src_idx += src_bvalue[j] * src_stride_ar[j - 1];
            if ((i + 1) % src_bunit[j] == 0)
            {
                src_bvalue[j]++;
            }
            if (src_bvalue[j] > (count[j] - 1))
            {
                src_bvalue[j] = 0;
            }
        }

        dst_idx = 0;

        for (j = 1; j <= stride_levels; j++)
        {
            dst_idx += dst_bvalue[j] * dst_stride_ar[j - 1];
            if ((i + 1) % dst_bunit[j] == 0)
            {
                dst_bvalue[j]++;
            }
            if (dst_bvalue[j] > (count[j] - 1))
            {
                dst_bvalue[j] = 0;
            }
        }

        COMEX_CHKANDJUMP(callback(src_idx, dst_idx, data),
                         "failed to perform callback");
    }

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_puts(
        void* src_ptr, int* src_stride_ar,
        void* dst_ptr, int* dst_stride_ar,
        int* count, int stride_levels,
        int proc, comex_group_t group)
{
    comex_request_t handle;
    COMEX_CHKANDJUMP(comex_nbputs(
                     src_ptr, src_stride_ar,
                     dst_ptr, dst_stride_ar,
                     count, stride_levels,
                     proc, group, &handle),
                     "failed to perform comex_nbputs");
    COMEX_CHKANDJUMP(comex_wait(&handle),
                     "failed to perform comex_wait");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_gets(
        void* src_ptr, int* src_stride_ar,
        void* dst_ptr, int* dst_stride_ar,
        int* count, int stride_levels,
        int proc, comex_group_t group)
{
    comex_request_t handle;
    COMEX_CHKANDJUMP(comex_nbgets(
                     src_ptr, src_stride_ar,
                     dst_ptr, dst_stride_ar,
                     count, stride_levels,
                     proc, group, &handle),
                     "failed to perform comex_nbgets");
    COMEX_CHKANDJUMP(comex_wait(&handle),
                     "failed to perform comex_wait");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static int comex_accs_native(
        int datatype, void* scale,
        void* src_ptr, int* src_stride_ar,
        void* dst_ptr, int* dst_stride_ar,
        int* count, int stride_levels,
        int proc, comex_group_t group)
{
    comex_request_t handle;
    COMEX_CHKANDJUMP(comex_nbaccs_native(datatype, scale,
                     src_ptr, src_stride_ar,
                     dst_ptr, dst_stride_ar,
                     count, stride_levels,
                     proc, group, &handle),
                     "failed to perform comex_nbaccs");
    COMEX_CHKANDJUMP(comex_wait(&handle),
                     "failed to perform comex_wait");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_putv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group)
{
    comex_request_t handle;
    COMEX_CHKANDJUMP(comex_nbputv(iov, iov_len, proc, group, &handle),
                     "failed to perform comex_nbputv");
    COMEX_CHKANDJUMP(comex_wait(&handle),
                     "failed to perform comex_wait");
fn_success:
    return COMEX_SUCCESS;
    
fn_fail:
    return COMEX_FAILURE;
}

int comex_getv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group)
{
    comex_request_t handle;
    COMEX_CHKANDJUMP(comex_nbgetv(iov, iov_len, proc, group, &handle),
                     "failed to perform comex_nbgetv");
    COMEX_CHKANDJUMP(comex_wait(&handle),
                     "failed to perform comex_wait");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static int comex_accv_native(
        int datatype, void* scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group)
{
    comex_request_t handle;
    COMEX_CHKANDJUMP(comex_nbaccv_native(datatype, scale,
                     iov, iov_len,
                     proc, group,
                     &handle),
                     "failed to perform comex_nbaccv");
    COMEX_CHKANDJUMP(comex_wait(&handle),
                     "failed to perform comex_wait");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_fence_all(comex_group_t group)
{
    COMEX_CHKANDJUMP(comex_wait_all(group),
                     "failed to perform comex_wait_all");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_fence_proc(int proc, comex_group_t group)
{
    COMEX_CHKANDJUMP(comex_wait_all(group),
                     "failed to perform comex_wait_all");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

/* comex_barrier is comex_fence_all + MPI_Barrier */
int comex_barrier(comex_group_t group)
{
    MPI_Comm comm;

    COMEX_CHKANDJUMP(comex_fence_all(group), "failed to fence all");
    COMEX_CHKANDJUMP(comex_group_comm(group, &comm), "failed to get group comm");
    MPI_CHKANDJUMP(MPI_Barrier(comm), "failed to perform MPI_Barrier");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

void* comex_malloc_local(size_t size)
{
    return malloc(size);
}

int comex_free_local(void* ptr)
{
    assert(ptr);
    if (ptr) free(ptr);
    return COMEX_SUCCESS;
}

static int finalize_ep(ofi_ep_t* ep)
{
    int ret;

    /* close OFI devices */
    if (ep->endpoint)
    {
        OFI_CHKANDJUMP(fi_close((struct fid*)ep->endpoint), "fi_close endpoint:");
        ep->endpoint = 0;
    }

    if (ep->av)
    {
        OFI_CHKANDJUMP(fi_close((struct fid*)ep->av), "fi_close address vector:");
        ep->av = 0;
    }

    if (ep->cq)
    {
        OFI_CHKANDJUMP(fi_close((struct fid*)ep->cq), "fi_close completion queue:");
        ep->cq = 0;
    }

    if (ep->domain)
    {
        OFI_CALL(ret, fi_close((struct fid*)ep->domain));
        if (ret == -FI_EBUSY)
        {
            COMEX_OFI_LOG(WARN, "domain is busy");
            COMEX_CHKANDJUMP(destroy_all_windows(), "failed to destroy all windows");
            OFI_CALL(ret, fi_close((struct fid*)ep->domain));
        }
        OFI_CHKANDJUMP(ret, "fi_close domain:");
        ep->domain = 0;
    }

    if (ep->fabric)
    {
        OFI_CHKANDJUMP(fi_close((struct fid*)ep->fabric), "fi_close fabric:");
        ep->fabric = 0;
    }

    if (ep->peers)
    {
        free(ep->peers);
        ep->peers = 0;
    }

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static int finalize_ofi()
{
    int ret = 0;

    if (progress_thread)
    {
        progress_thread_complete = 1;
        pthread_join(progress_thread, 0);
    }

    if (IS_TARGET_ATOMICS_EMULATION())
    {
        int req_idx;
        for (req_idx = 0; req_idx < PREPOST_ATOMICS_COUNT; req_idx++)
        {
            request_t* req = &atomics_requests[req_idx];
            if (req->state == rs_complete) continue;

            assert(req->state == rs_progress);
            OFI_CHKANDJUMP(fi_cancel((fid_t)ofi_data.ep_tagged.endpoint, &atomics_requests[req_idx]), "fi_cancel failed");
            struct fi_cq_err_entry err;
            fi_cq_readerr(ofi_data.ep_tagged.cq, &err, 0);
        }
    }

    if (dual_provider())
        COMEX_CHKANDJUMP(finalize_ep(&ofi_data.ep_atomics), "failed to finalize ep_atomics");

    COMEX_CHKANDJUMP(finalize_ep(&ofi_data.ep_rma), "failed to finalize ep_rma");

    OFI_LOCK_DESTROY();

    unload_ofi(&ld_table);

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_finalize()
{
    /* it's okay to call multiple times -- extra calls are no-ops */
    if (!initialized) return COMEX_SUCCESS;

    initialized = 0;

    /* Make sure that all outstanding operations are done */
    COMEX_CHKANDJUMP(comex_wait_all(COMEX_GROUP_WORLD), "failed to wait all");
    COMEX_CHKANDJUMP(destroy_mutexes(local_mutex), "failed to destroy local mutex");
    COMEX_CHKANDJUMP(finalize_ofi(), "failed to finalize ofi");

    MPI_CHKANDJUMP(MPI_Barrier(l_state.world_comm), "failed to perform MPI_Barrier");

    /* groups */
    comex_group_finalize();

    // destroy the communicators
    MPI_CHKANDJUMP(MPI_Comm_free(&l_state.world_comm), "failed to perform MPI_Comm_free");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_wait_proc(int proc, comex_group_t group)
{
    request_cache_t* cache = request_cache;
    while (cache)
    {
        int i;
        if (group != COMEX_GROUP_WORLD)
        {
            /* wait for requests for specific group */
            for (i = 0; i < sizeofa(cache->request); i++)
            {
                request_t* request = cache->request + i;
                /* check for proc/group on every iteration because
                 * it may be changed during poll */
                while (!(request->flags & rf_no_group_wait) &&
                      request->proc == proc && request->group == group &&
                      request->state == rs_progress)
                {
                    COMEX_CHKANDJUMP(poll(0), "failed to poll");
                    PAUSE();
                }
            }
        }
        else
        {
            /* wait all */
            for (i = 0; i < sizeofa(cache->request); i++)
            {
                request_t* request = cache->request + i;
                while (request->proc == proc && request->state == rs_progress)
                {
                    COMEX_CHKANDJUMP(poll(0), "failed to poll");
                    PAUSE();
                }
            }
        }
        cache = cache->next;
    }

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static inline int wait_request(request_t* request)
{
    assert(request);
    assert(!request->cmpl);
    assert(request->state == rs_progress || request->state == rs_complete);
    assert(!(request->flags & rf_auto_free));

    if (request->state == rs_complete)
    {
        free_request(request);
        return COMEX_SUCCESS;
    }

    while (request->state == rs_progress)
    {
        COMEX_CHKANDJUMP(poll(0), "failed to poll");
        PAUSE();
    }

    free_request(request);

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_wait(comex_request_t* handle)
{
    assert(handle);
    if (*handle == HANDLE_UNDEFINED)
        return COMEX_SUCCESS;

    request_t* request = lookup_request(*handle);
    if (!request || request->state == rs_none)
        return COMEX_FAILURE;

    return wait_request(request);

fn_success:
    return COMEX_SUCCESS;
fn_fail:
    return COMEX_FAILURE;
}

/* *status == 1 - request is in progress */
int comex_test(comex_request_t* handle, int* status)
{
    assert(handle);
    assert(status);

    *status = 0;

    request_t* request = lookup_request(*handle);
    if (!request || request->state == rs_none)
        return COMEX_FAILURE;

    int items_processed = 1;

    /* process all CQ items in queue till request in 'progress' state
     * or queue is not empty (items_processed is not 0) */
    while (request->state == rs_progress && items_processed)
        COMEX_CHKANDJUMP(poll(&items_processed), "failed to poll");

    *status = (request->state == rs_progress);

fn_success:
    return COMEX_SUCCESS;
fn_fail:
    return COMEX_FAILURE;
}

int comex_wait_all(comex_group_t group)
{
    request_cache_t* cache = request_cache;
    while (cache)
    {
        int i;
        if (group != COMEX_GROUP_WORLD)
        {
            /* wait for requests for specific group */
            for (i = 0; i < sizeofa(cache->request); i++)
            {
                request_t* request = cache->request + i;
                while (!(request->flags & rf_no_group_wait) &&
                      request->group == group && request->state == rs_progress)
                {
                    COMEX_CHKANDJUMP(poll(0), "failed to poll");
                    PAUSE();
                }
            }
        }
        else
        {
            /* wait all */
            for (i = 0; i < sizeofa(cache->request); i++)
            {
                request_t* request = cache->request + i;
                while (!(request->flags & rf_no_group_wait) &&
                      request->state == rs_progress)
                {
                    COMEX_CHKANDJUMP(poll(0), "failed to poll");
                    PAUSE();
                }
            }
        }
        cache = cache->next;
    }

fn_success:
    return COMEX_SUCCESS;
fn_fail:
    return COMEX_FAILURE;
}

#define SCALE(type)                                                  \
do                                                                   \
{                                                                    \
    type* ptr = 0;                                                   \
    for (ptr = middle, i = 0; i < iov_len; i++)                      \
        for (j = 0; j < iov[i].count; j++)                           \
            for (k = 0; k < iov[i].bytes / sizeof(*ptr); k++, ptr++) \
                *ptr = ((type*)iov[i].src[j])[k] * *(type*)scale;    \
} while (0)

static int iov_acc(int datatype, void*  scale,
                   int proc, comex_group_t group,
                   comex_giov_t *iov, int iov_len,
                   comex_request_t * handle)
{
    VALIDATE_GROUP_AND_PROC(group, proc);

    if (handle) *handle = HANDLE_UNDEFINED;

    request_t* parent_req = alloc_request();
    EXPR_CHKANDJUMP(parent_req, "failed to allocate parent request");

    if (!handle) parent_req->flags |= rf_auto_free;
    else
    {
        /* create user's request. we can't use parent request because it
           can't have completion callback */
        request_t* user = alloc_request();
        user->group = group;
        user->proc = proc;
        *handle = user->index;
        set_parent_request(user, parent_req);
    }

    int ofi_iov_count = 0;
    int ofi_msg_count = 0;

    int bytes_in_msg = 0; // current count of bytes in atomic msg
    int iov_in_msg = 0;   // current count of iov in atomic msg

    int middle_len = 0; /* common number of bytes to process */
    void* middle = 0; /* buffer to store scaled values (in case if scale is not 1) */
    struct fi_msg_atomic* msg = 0;

    int iov_idx;
    int iov_column;

    acc_data_t* acc_data = 0;

    ssize_t max_bytes_in_atomic = ofi_data.max_bytes_in_atomic[COMEX_DTYPE_IDX(datatype)];
    EXPR_CHKANDJUMP(max_bytes_in_atomic > 0, "datatype is not supported: %d", datatype);

    /* calculate common count of iov elements & common length of data to send */
    for (iov_idx = 0; iov_idx < iov_len; iov_idx++)
    {
        /* count of vector elements to process */
        ofi_iov_count += iov[iov_idx].count * ((iov[iov_idx].bytes / max_bytes_in_atomic) + 1);
        middle_len += iov[iov_idx].count * iov[iov_idx].bytes;
    }

    /*
     * Calculate count of atomic msg which enough to pack all comex iov
     */


     /* First rough option - in worst case we will have one iov in atomic msg & number of msg's will be equal number of iov*/
     //ofi_msg_count = ofi_iov_count;


     /* Second more accurate option - in worst case we will have only one not fully filled (in terms of iov elements) atomic msg per comex iov */
    /*iov_in_msg = 0;
    for (iov_idx = 0; iov_idx < iov_len; iov_idx++)
    {
        // initiate counts of msg and iov in msg for worst case
        int msg_per_comex_iov = iov[iov_idx].count * ((iov[iov_idx].bytes / max_bytes_in_atomic) + 1);
        iov_in_msg = 1;

        // bytes in iov-column less than atomic msg can hold and there is enough space for more than 1 iov-column
        if (max_bytes_in_atomic / iov[iov_idx].bytes > 1)
        {
            iov_in_msg = max_bytes_in_atomic / iov[iov_idx].bytes;
            iov_in_msg = min(iov_in_msg, ofi_data.rma_iov_limit);
        } 

        if (iov_in_msg > 1)
            msg_per_comex_iov = iov[iov_idx].count / iov_in_msg + 1;
        ofi_msg_count += msg_per_comex_iov;
    }*/


    /* Third the most accurate option - to calculate msg count we repeat iov repacking algorithm */
    bytes_in_msg = 0;
    iov_in_msg = 0;
    for (iov_idx = 0; iov_idx < iov_len; iov_idx++)
    {
        for (iov_column = 0; iov_column < iov[iov_idx].count; iov_column++)
        {
            ssize_t bytes_to_send = iov[iov_idx].bytes;
            for (; bytes_to_send > 0;
                bytes_to_send -= max_bytes_in_atomic)
            {
                int bytes_in_chunk = min(max_bytes_in_atomic, bytes_to_send);
                if (iov_in_msg >= ofi_data.rma_iov_limit || bytes_in_msg + bytes_in_chunk > max_bytes_in_atomic)
                {
                    iov_in_msg = 0;
                    bytes_in_msg = 0;
                    ofi_msg_count++;
                }
                bytes_in_msg += bytes_in_chunk;
                iov_in_msg++;
            }
        }
    }
    ofi_msg_count++;

    /* no data to process? just exit */
    if (!middle_len) goto fn_success;

    acc_data = malloc(sizeof(*acc_data));
    EXPR_CHKANDJUMP(acc_data, "failed to allocate acc_data");
    memset(acc_data, 0, sizeof(*acc_data));
    acc_data->proc = PROC_NONE;

    parent_req->data = acc_data;
    parent_req->cmpl = complete_acc;

    if (!scale_is_1(datatype, scale))
    {
        /* create local scaled buffer */
        int i;
        int j;
        int k;

        acc_data->middle = middle = malloc(middle_len);
        switch (datatype)
        {
            case COMEX_ACC_INT:
                SCALE(int);
                break;
            case COMEX_ACC_DBL:
                SCALE(double);
                break;
            case COMEX_ACC_FLT:
                SCALE(float);
                break;
            case COMEX_ACC_LNG:
                SCALE(long);
                break;
            case COMEX_ACC_DCP:
                assert(sizeof(DoubleComplex) == sizeof(double complex));
                SCALE(double complex);
                break;
            case COMEX_ACC_CPL:
                assert(sizeof(SingleComplex) == sizeof(float complex));
                SCALE(float complex);
                break;
            default:
                COMEX_OFI_LOG(WARN, "iov_acc: incorrect data type: %d", datatype);
                return 1;
        }
    }

    enum fi_datatype fi_dtype = GET_FI_DTYPE(datatype);
    EXPR_CHKANDJUMP(fi_dtype >= 0, "incorrect fi_datatype: %d", datatype);
    int datasize;
    COMEX_DTYPE_SIZEOF(datatype, datasize);

    int msg_len = sizeof(*msg) * ofi_msg_count;
    assert(msg_len);

    acc_data->msg = msg = malloc(msg_len);
    EXPR_CHKANDJUMP(msg, "failed to allocate atomic messages");
    memset(msg, 0, msg_len);

    msg->msg_iov = malloc(sizeof(*msg->msg_iov) * ofi_iov_count);
    EXPR_CHKANDJUMP(msg->msg_iov, "failed to allocate msg iov");

    msg->rma_iov = malloc(sizeof(*msg->rma_iov) * ofi_iov_count);
    EXPR_CHKANDJUMP(msg->rma_iov, "failed to allocate rma iov");

    struct fi_msg_atomic* m = 0;
    struct fi_ioc*     ioc     = (struct fi_ioc*)msg->msg_iov;
    struct fi_rma_ioc* rma_ioc = (struct fi_rma_ioc*)msg->rma_iov;
    char* sbuf = middle;

    /* lock remote host *
     * TODO: is it required? per-element atomic is provided by OFI */
    //acquire_remote_lock(proc);

    acc_data->proc = proc;

    bytes_in_msg = 0;
    iov_in_msg = 0;
    int msg_num = 0;

    /* lock request - deny to complete */
    increment_request_cnt(parent_req);

    for (iov_idx = 0; iov_idx < iov_len; iov_idx++)
    {
        for (iov_column = 0; iov_column < iov[iov_idx].count; iov_column++)
        {
            ofi_window_t* wnd = 0;
            COMEX_CHKANDJUMP(lookup_window(iov[iov_idx].dst[iov_column], iov[iov_idx].bytes, proc, group, &wnd),
                             "failed to lookup window");
            assert(wnd);

            ssize_t bytes_to_send = iov[iov_idx].bytes;
            char* src = sbuf ? sbuf : iov[iov_idx].src[iov_column];
            char* dst = iov[iov_idx].dst[iov_column];

            for (; bytes_to_send > 0;
                ioc++, rma_ioc++,
                bytes_to_send -= max_bytes_in_atomic, src += max_bytes_in_atomic, dst += max_bytes_in_atomic)
            {
                int bytes_in_chunk = min(max_bytes_in_atomic, bytes_to_send);

                /*
                 * OFI atomic msg has 2 limits: count of iov and count of bytes in all iovs.
                 * If msg is close to one of limits - pass it to OFI & clean pointer
                 */
                if (m && (iov_in_msg >= ofi_data.rma_iov_limit || bytes_in_msg + bytes_in_chunk > max_bytes_in_atomic))
                {
                    OFI_RETRY(fi_atomicmsg(ofi_data.ep_atomics.endpoint, m, 0), "failed to fi_atomicmsg:");
                    m = 0;
                    bytes_in_msg = 0;
                    iov_in_msg = 0;
                }

                if (!m) /* if msg is NULL (not created or cleaned above) - just create it */
                {
                    request_t* child_req = alloc_request();
                    EXPR_CHKANDJUMP(child_req, "failed to allocate child request");
                    set_parent_request(parent_req, child_req);
                    child_req->proc = proc;

                    assert(msg_num < ofi_msg_count);
                    m = msg + msg_num;
                    msg_num++;
                    m->datatype = fi_dtype;
                    m->op = FI_SUM;
                    m->context = child_req;
                    m->msg_iov = ioc;
                    m->rma_iov = rma_ioc;
                    m->addr = wnd->peer_atomics->fi_addr;
                }

                /* add new chunk into IOV */
                bytes_in_msg += bytes_in_chunk;
                iov_in_msg++;

                assert(m->iov_count == m->rma_iov_count);
                ioc->count = rma_ioc->count = (bytes_in_chunk % datasize) ?
                                              (bytes_in_chunk / datasize + 1) :
                                              (bytes_in_chunk / datasize);
                rma_ioc->addr = get_remote_addr(wnd->ptr, dst, ot_atomic);
                ioc->addr = src;
                rma_ioc->key = wnd->key_atomics;
                m->iov_count = m->rma_iov_count = iov_in_msg;
            }

            if (sbuf) sbuf += iov[iov_idx].bytes;
        }
    }

    /* send tail of data */
    if (m) OFI_RETRY(fi_atomicmsg(ofi_data.ep_atomics.endpoint, m, 0), "failed to fi_atomicmsg:");

    decrement_request_cnt(parent_req);

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    if (parent_req)
    {
        complete_request(parent_req);
        free_request(parent_req);
    }
    return COMEX_FAILURE;
}

int nb_getput(
        void* src, void* dst, int bytes,
        int proc, comex_group_t group,
        comex_request_t *handle, int is_get_op)
{
    VALIDATE_GROUP_AND_PROC(group, proc);
    if (handle) *handle = HANDLE_UNDEFINED;

    request_t* request = alloc_request();
    EXPR_CHKANDJUMP(request, "failed to allocate request");

    ofi_window_t* wnd = 0;
    COMEX_CHKANDJUMP(lookup_window(is_get_op ? src : dst, bytes, proc, group, &wnd),
                     "failed to lookup window");
    assert(wnd);

    request->group = group;
    request->proc = proc;

    if (bytes > ofi_data.max_buffered_send)
    {
        /* we have to register buffer */
        struct fi_context context;
        request->dtor = req_dtor;
        COMEX_CHKANDJUMP(mr_reg(is_get_op ? dst : src, bytes, MR_ACCESS_PERMISSIONS,
                                &request->mr_single, &context, ot_rma),
                                "fi_mr_reg failed:");

        if (!is_get_op)
            OFI_RETRY(fi_write(ofi_data.ep_rma.endpoint,
                               src,
                               bytes,
                               fi_mr_desc(request->mr_single),
                               wnd->peer_rma->fi_addr,
                               get_remote_addr(wnd->ptr, dst, ot_rma),
                               wnd->key_rma,
                               request),
                               "fi_write error:");
    }

    if (is_get_op)
        OFI_RETRY(fi_read(ofi_data.ep_rma.endpoint,
                          dst,
                          bytes,
                          request->mr_single ? fi_mr_desc(request->mr_single) : 0,
                          wnd->peer_rma->fi_addr,
                          get_remote_addr(wnd->ptr, src, ot_rma),
                          wnd->key_rma,
                          request),
                          "fi_read error:");
    else if (bytes <= ofi_data.max_buffered_send)
    {
        OFI_RETRY(fi_inject_write(ofi_data.ep_rma.endpoint,
                                  src,
                                  bytes,
                                  wnd->peer_rma->fi_addr,
                                  get_remote_addr(wnd->ptr, dst, ot_rma),
                                  wnd->key_rma),
                                  "fi_inject_write error:");
        complete_request(request);
        /*OFI_RETRY(fi_write(ofi_data.ep_rma.endpoint,
                             src,
                             bytes,
                             0,
                             wnd->peer_rma->fi_addr,
                             get_remote_addr(wnd->ptr, dst, ot_rma),
                             wnd->key_rma,
                             request),
                             "fi_write error:");
        */
    }

    if (handle) *handle = (comex_request_t)request->index;

fn_success:
    if (env_data.force_sync)
    {
        if (request) wait_request(request);
        if (handle)  *handle = HANDLE_UNDEFINED;
    }
    return COMEX_SUCCESS;

fn_fail:
    if (request)
        free_request(request);
    return COMEX_FAILURE;
}

int comex_nbput(
        void* src, void* dst, int bytes,
        int proc, comex_group_t group,
        comex_request_t *handle)
{
    COMEX_CHKANDJUMP(nb_getput(src, dst, bytes, proc, group, handle, 0),
                     "failed to perform nb_getput");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_nbget(
        void* src, void* dst, int bytes,
        int proc, comex_group_t group,
        comex_request_t *handle)
{
    COMEX_CHKANDJUMP(nb_getput(src, dst, bytes, proc, group, handle, 1),
                     "failed to perform nb_getput");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static int comex_nbacc_native(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group,
        comex_request_t *handle)
{
    comex_giov_t iov;
    iov.bytes = bytes;
    iov.count = 1;
    iov.src = &src_ptr;
    iov.dst = &dst_ptr;

    COMEX_CHKANDJUMP(iov_acc(datatype, scale, proc, group, &iov, 1, handle),
                     "failed to perform iov_acc");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

#define PUT_MSG(endpoint, msg)                                      \
do                                                                  \
{                                                                   \
    OFI_RETRY(fi_writemsg(endpoint, msg, 0), "fi_writemsg error:"); \
} while (0)

#define GET_MSG(endpoint, msg)                                    \
do                                                                \
{                                                                 \
    OFI_RETRY(fi_readmsg(endpoint, msg, 0), "fi_readmsg error:"); \
} while (0)

#define REG_LOCAL_MR(msg, request, op_type)                                               \
do                                                                                        \
{                                                                                         \
    ofi_ep_t* ep = get_ep(op_type);                                                       \
    if (FI_MR_SCALABLE == ep->mr_mode)                                                    \
    {                                                                                     \
        /* skip registration of local buffers in case of scalable mr */                   \
        break;                                                                            \
    }                                                                                     \
    (request)->mrs = (struct fid_mr**)malloc((msg)->iov_count * sizeof(*(request)->mrs)); \
    EXPR_CHKANDJUMP((request)->mrs, "failed to allocate memory");                         \
    mr_regv((msg)->msg_iov, (msg)->iov_count, (request)->mrs, ot_rma);                    \
    (request)->mr_count = (msg)->iov_count;                                               \
    (request)->dtor = req_dtor;                                                           \
                                                                                          \
    (msg)->desc = (void**)malloc(sizeof(*(msg)->desc) * (msg)->iov_count);                \
    EXPR_CHKANDJUMP((msg)->desc, "failed to register memory:");                           \
    int i;                                                                                \
    OFI_LOCK();                                                                           \
    for(i = 0; i < (msg)->iov_count; i++)                                                 \
        (msg)->desc[i] = fi_mr_desc((request)->mrs[i]);                                   \
    OFI_UNLOCK();                                                                         \
} while(0)

#define FREE_DESC(msg) free((msg)->desc)

static inline int getputs_stride(size_t src_idx, size_t dst_idx, void* data)
{
    assert(data);
    assert(src_idx >= 0);
    assert(dst_idx >= 0);

    strided_context_t* context = (strided_context_t*)data;
    struct fi_msg_rma* msg = context->msg;

    if (!msg)
    {
        context->msg = msg = malloc(sizeof(*msg));
        EXPR_CHKANDJUMP(msg, "failed to allocate rma message");
        memset(msg, 0, sizeof(*msg));

        msg->msg_iov = malloc(ofi_data.rma_iov_limit * sizeof(*msg->msg_iov));
        EXPR_CHKANDJUMP(msg->msg_iov, "failed to allocate msg iov");

        msg->rma_iov = malloc(ofi_data.rma_iov_limit * sizeof(*msg->rma_iov));
        EXPR_CHKANDJUMP(msg->rma_iov, "failed to allocate rma iov");

        request_t* child_req = alloc_request();
        EXPR_CHKANDJUMP(child_req, "failed to allocate child request");
        set_parent_request(context->request, child_req);
        msg->context = child_req;
        child_req->data = msg;
        child_req->cmpl = complete_getput;
    }

    assert(msg);
    struct iovec* iovelem = (struct iovec*)msg->msg_iov + msg->iov_count;
    struct fi_rma_iov* rmaelem = (struct fi_rma_iov*)msg->rma_iov + msg->rma_iov_count;

    iovelem->iov_base = context->is_get_op ? (char*)context->dst + dst_idx : (char*)context->src + src_idx;
    void* raw_remote_addr = (context->is_get_op ? (char*)context->src + src_idx : (char*)context->dst + dst_idx);
    rmaelem->addr = get_remote_addr(context->wnd->ptr, raw_remote_addr, ot_rma);

    iovelem->iov_len = context->count[0];
    rmaelem->len = context->count[0];

    assert((uint64_t)raw_remote_addr >= context->wnd->ptr);
    assert(rmaelem->len > 0);
    assert((uint64_t)raw_remote_addr + rmaelem->len <= context->wnd->ptr + context->wnd->size);

    rmaelem->key = context->wnd->key_rma;
    msg->addr = context->wnd->peer_rma->fi_addr;

    msg->iov_count++;
    msg->rma_iov_count++;
    assert(msg->iov_count == msg->rma_iov_count);

    if (msg->iov_count >= ofi_data.rma_iov_limit)
    {
        REG_LOCAL_MR(msg, (request_t*)msg->context, ot_rma);

        if (context->is_get_op)
            GET_MSG(ofi_data.ep_rma.endpoint, msg);
        else
            PUT_MSG(ofi_data.ep_rma.endpoint, msg);
        FREE_DESC(context->msg);
        context->msg = 0;
    }

fn_success:
    return COMEX_SUCCESS;
fn_fail:
    if (msg)
    {
        /* in case if context defined - then msg will be destroyed */
        /* by request completion callback */
        if (msg->context)
        {
            complete_request((request_t*)msg->context);
            free_request((request_t*)msg->context);
        }
        else
        {
            if (msg->msg_iov) free((void*)msg->msg_iov);
            if (msg->rma_iov) free((void*)msg->rma_iov);
            free(msg);
        }
    }

    return COMEX_FAILURE;
}

/*
 * src - pointer to source continuous buffer
 * src_stride - array with count of bytes for each dimension (src_stride[0] - for row, src_stride[1] - for matrix, ...)
 * dst, dst_stride - same as for src
 * count - count of elements to send for each dimension (count[0] - bytes! to send in first dimension, count[1] - elems! to send in second dimension, ...)
 * stride_levels - count of dimensions -1 (0 for row, 1 for matrix, ...)
 */
static int nb_getputs(
        void* src, int* src_stride,
        void* dst, int* dst_stride,
        int* count, int stride_levels,
        int proc, comex_group_t group,
        comex_request_t *handle, int is_get_op)
{
    VALIDATE_GROUP_AND_PROC(group, proc);
    if (handle) *handle = HANDLE_UNDEFINED;

    ofi_window_t* wnd = 0;
    int total = count[0]; /* total count of bytes to process */

    if (stride_levels > 0)
    {
        int i;
        for (i = 1; i <= stride_levels; i++)
            total *= count[i];

        int * stride = (is_get_op) ? src_stride : dst_stride;
    }

    if (total == 0) goto fn_success;

    request_t* parent_req = alloc_request();
    EXPR_CHKANDJUMP(parent_req, "failed to allocate parent request");

    parent_req->group = group;
    parent_req->proc = proc;

    if (lookup_window(is_get_op ? src : dst, count[0], proc, group, &wnd) != COMEX_SUCCESS)
    {
        COMEX_OFI_LOG(INFO, "Failed to lookup window: length: %d", count[0]);
        int i;
        for (i = 0; i <= stride_levels; i++)
            COMEX_OFI_LOG(DEBUG, " %d: count = %d, stride: %d\n", i, count[i], ((is_get_op) ? src_stride : dst_stride)[i]);
    }
    assert(wnd);

    strided_context_t context = {.src = src, .dst = dst, .count = count,
        .proc = proc, .group = group, .request = parent_req, .ops = 0,
        .is_get_op = is_get_op, .msg = 0, .wnd = wnd};

    /* lock parent_req - deny to complete */
    increment_request_cnt(parent_req);

    COMEX_CHKANDJUMP(list_strides(src_stride, dst_stride, count, stride_levels, getputs_stride, &context),
                     "failed to list strides");

    if (context.msg) /* not all operation are scheduled */
    {
        struct fi_msg_rma* msg = context.msg;
        REG_LOCAL_MR(msg, (request_t*)msg->context, ot_rma);
        if (is_get_op)
            GET_MSG(ofi_data.ep_rma.endpoint, msg);
        else
            PUT_MSG(ofi_data.ep_rma.endpoint, msg);
        FREE_DESC(msg);
    }

    if (!handle)
        parent_req->flags |= rf_auto_free;
    else
        *handle = parent_req->index;

    decrement_request_cnt(parent_req);

    /*if(!parent_req->child_cnt)*/
    /*{*/
        /*printf("no requests\n");*/
        /*goto fn_fail;*/
    /*}*/

fn_success:
    if (env_data.force_sync)
    {
        if (parent_req) wait_request(parent_req);
        if (handle) *handle = HANDLE_UNDEFINED;
    }
    return COMEX_SUCCESS;

fn_fail:
    if (parent_req) free_request(parent_req);
    return COMEX_FAILURE;
}

int comex_nbputs(
        void* src, int* src_stride,
        void* dst, int* dst_stride,
        int* count, int stride_levels,
        int proc, comex_group_t group,
        comex_request_t *handle)
{
    COMEX_CHKANDJUMP(nb_getputs(src, src_stride, dst, dst_stride, count, stride_levels,
                     proc, group, handle, 0), "failed to perform nb_getputs");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_nbgets(
        void* src, int* src_stride,
        void* dst, int* dst_stride,
        int* count, int stride_levels,
        int proc, comex_group_t group,
        comex_request_t *handle)
{
    COMEX_CHKANDJUMP(nb_getputs(src, src_stride, dst, dst_stride, count, stride_levels,
                     proc, group, handle, 1), "failed to perform nb_getputs");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int add_to_iov_callback(size_t src_idx, size_t dst_idx, void* data)
{
    assert(data);
    strided_context_t* ctx = (strided_context_t*)data;

    int iov_idx = ctx->cur_iov_idx;
    EXPR_CHKANDJUMP(iov_idx < ctx->iov_len, "incorrect iov_idx");

    ctx->src_array[iov_idx] = ((char*)ctx->src + src_idx);
    ctx->dst_array[iov_idx] = ((char*)ctx->dst + dst_idx);
    ctx->iov[iov_idx].bytes = ctx->count[0];
    ctx->iov[iov_idx].count = 1;
    ctx->iov[iov_idx].src = &(ctx->src_array[iov_idx]);
    ctx->iov[iov_idx].dst = &(ctx->dst_array[iov_idx]);
    ctx->cur_iov_idx++;

fn_success:
    return COMEX_SUCCESS;
fn_fail:
    return COMEX_FAILURE;
}

static int comex_nbaccs_native(
        int datatype, void* scale,
        void* src, int* src_stride,
        void* dst, int* dst_stride,
        int* count, int stride_levels,
        int proc, comex_group_t group,
        comex_request_t *handle)
{

    int ret = COMEX_SUCCESS;

    int iov_len = 1;
    int i;

    for (i = 1; i <= stride_levels; i++)
        iov_len *= count[i];
    EXPR_CHKANDJUMP(iov_len, "incorrect iov_len");

    comex_giov_t* iov = malloc(sizeof(comex_giov_t) * iov_len);
    EXPR_CHKANDJUMP(iov, "failed to allocate iov array");
    memset(iov, 0, sizeof(*iov));

    void** src_array = malloc(sizeof(void*) * iov_len);
    EXPR_CHKANDJUMP(src_array, "failed to allocate src array");

    void** dst_array = malloc(sizeof(void*) * iov_len);
    EXPR_CHKANDJUMP(dst_array, "failed to allocate dst array");

    strided_context_t context = {
                                    .datatype = datatype,
                                    .scale = scale,
                                    .src = src,
                                    .dst = dst,
                                    .count = count,
                                    .proc = proc,
                                    .group = group,
                                    .ops = 0,
                                    .cur_iov_idx = 0,
                                    .src_array = src_array,
                                    .dst_array = dst_array,
                                    .iov = iov,
                                    .iov_len = iov_len
                                };

    COMEX_CHKANDJUMP(list_strides(src_stride, dst_stride, count, stride_levels, add_to_iov_callback, &context),
                    "failed to list strides");

    COMEX_CHKANDJUMP(iov_acc(datatype, scale, proc, group, iov, iov_len, handle),
                     "failed to perform iov_acc");

fn_success:
    ret = COMEX_SUCCESS;
    goto fn_clean;

fn_fail:
    ret = COMEX_FAILURE;
    goto fn_clean;

fn_clean:
    if (iov) free(iov);
    if (src_array) free(src_array);
    if (dst_array) free(dst_array);

    return ret;
}

static int nb_getputv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* handle, int is_get_op)
{
    int i;
    int j;
    int count = 0;
    struct fi_msg_rma* msg = 0;

    VALIDATE_GROUP_AND_PROC(group, proc);
    if (handle) *handle = HANDLE_UNDEFINED;

    /* calculate common count of iov elements */
    for (i = 0; i < iov_len; i++)
        count += iov[i].count;

    /* no data to process? just exit */
    if (!count) goto fn_success;

    request_t* parent_req = alloc_request();
    EXPR_CHKANDJUMP(parent_req, "failed to allocate parent request");

    parent_req->group = group;
    parent_req->proc = proc;

    /* lock request - deny to complete */
    increment_request_cnt(parent_req);

    /* create non-blocking requests */
    for (i = 0; i < iov_len; i++)
    {
        for (j = 0; j < iov[i].count; j++)
        {
            if (!msg)
            {
                msg = malloc(sizeof(*msg));
                EXPR_CHKANDJUMP(msg, "failed to allocate iov message");
                memset(msg, 0, sizeof(*msg));

                msg->msg_iov = malloc(ofi_data.rma_iov_limit * sizeof(*msg->msg_iov));
                EXPR_CHKANDJUMP(msg->msg_iov, "failed to allocate msg iov");

                msg->rma_iov = malloc(ofi_data.rma_iov_limit * sizeof(*msg->rma_iov));
                EXPR_CHKANDJUMP(msg->rma_iov, "failed to allocate rma iov");

                request_t* child_req = alloc_request();
                EXPR_CHKANDJUMP(child_req, "failed to allocate child request");
                set_parent_request(parent_req, child_req);

                msg->context = child_req;
                child_req->data = msg;
                child_req->cmpl = complete_getput;
            }

            ofi_window_t* wnd = 0;
            void* raw_remote_addr = (is_get_op ? iov[i].src[j] : iov[i].dst[j]);
            int iov_len = iov[i].bytes;
            COMEX_CHKANDJUMP(lookup_window(raw_remote_addr, iov_len, proc, group, &wnd),
                             "failed to lookup window");
            assert(wnd);

            assert(msg);
            struct iovec* iovelem = (struct iovec*)msg->msg_iov + msg->iov_count;
            struct fi_rma_iov* rmaelem = (struct fi_rma_iov*)msg->rma_iov + msg->rma_iov_count;

            iovelem->iov_base = is_get_op ? iov[i].dst[j] : iov[i].src[j];
            rmaelem->addr = get_remote_addr(wnd->ptr, raw_remote_addr, ot_rma);

            iovelem->iov_len = iov_len;
            rmaelem->len = iov_len;

            rmaelem->key = wnd->key_rma;
            msg->addr = wnd->peer_rma->fi_addr;

            assert((uint64_t)raw_remote_addr >= wnd->ptr);
            assert(rmaelem->len > 0);
            assert((uint64_t)raw_remote_addr + rmaelem->len <= wnd->ptr + wnd->size);

            msg->iov_count++;
            msg->rma_iov_count++;
            assert(msg->iov_count == msg->rma_iov_count);

            if (msg->iov_count >= ofi_data.rma_iov_limit)
            {
                REG_LOCAL_MR(msg, (request_t*)msg->context, ot_rma);

                if (is_get_op)
                    GET_MSG(ofi_data.ep_rma.endpoint, msg);
                else
                    PUT_MSG(ofi_data.ep_rma.endpoint, msg);
                FREE_DESC(msg);
                msg = 0;
            }
        }
    }

    if (msg)
    {
        REG_LOCAL_MR(msg, (request_t*)msg->context, ot_rma);
        if (is_get_op)
            GET_MSG(ofi_data.ep_rma.endpoint, msg);
        else
            PUT_MSG(ofi_data.ep_rma.endpoint, msg);
        FREE_DESC(msg);
        msg = 0;
    }

    if(!handle)
        parent_req->flags |= rf_auto_free;
    else
        *handle = parent_req->index;

    decrement_request_cnt(parent_req);

fn_success:
    if (env_data.force_sync)
    {
        if (parent_req) wait_request(parent_req);
        if (handle) *handle = HANDLE_UNDEFINED;
    }
    return COMEX_SUCCESS;

fn_fail:
    if (msg)
    {
        /* in case if context defined - them msg will be destroyed */
        /* by request completion callback */
        if (msg->context)
        {
            complete_request((request_t*)msg->context);
            free_request((request_t*)msg->context);
        }
        else
        {
            if (msg->msg_iov) free((void*)msg->msg_iov);
            if (msg->rma_iov) free((void*)msg->rma_iov);
            free(msg);
        }
    }

    if (parent_req) free_request(parent_req);
    return COMEX_FAILURE;
}

int comex_nbputv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* handle)
{
    COMEX_CHKANDJUMP(nb_getputv(iov, iov_len, proc, group, handle, 0),
                     "failed to perform nb_getputv");
fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_nbgetv(
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* handle)
{
    COMEX_CHKANDJUMP(nb_getputv(iov, iov_len, proc, group, handle, 1),
                     "failed to perform nb_getputv");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static int comex_nbaccv_native(
        int datatype, void* scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* handle)
{
    COMEX_CHKANDJUMP(iov_acc(datatype, scale, proc, group, iov, iov_len, handle),
                     "failed to perform iov_acc");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_acc(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group)
{
    return comex_acc_f(datatype, scale, src_ptr, dst_ptr, bytes, proc, group);
}

int comex_accs(
        int datatype, void* scale,
        void* src_ptr, int* src_stride_ar,
        void* dst_ptr, int* dst_stride_ar,
        int* count, int stride_levels,
        int proc, comex_group_t group)
{
    return comex_accs_f(datatype, scale,
            src_ptr, src_stride_ar,
            dst_ptr, dst_stride_ar,
            count, stride_levels, proc, group);
}

int comex_accv(
        int datatype, void* scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group)
{
    return comex_accv_f(datatype, scale, iov, iov_len, proc, group);
}

int comex_nbacc(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group,
        comex_request_t *handle)
{
    return comex_nbacc_f(datatype, scale, src_ptr, dst_ptr, bytes, proc, group, handle);
}

int comex_nbaccs(
        int datatype, void* scale,
        void* src, int* src_stride,
        void* dst, int* dst_stride,
        int* count, int stride_levels,
        int proc, comex_group_t group,
        comex_request_t *handle)
{
    return comex_nbaccs_f(datatype, scale,
            src, src_stride,
            dst, dst_stride,
            count, stride_levels, proc, group, handle);
}

int comex_nbaccv(
        int datatype, void* scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* handle)
{
    return comex_nbaccv_f(datatype, scale, iov, iov_len, proc, group, handle);
}

/*
 * ploc - pointer to fetch data (and in case of SWAP also pointer to src)
   prem - pointer to dst
 */
int comex_rmw(
        int op, void* ploc, void* prem, int extra,
        int proc, comex_group_t group)
{
    request_t request;
    init_request(&request);
    ofi_window_t* window;

    COMEX_CHKANDJUMP(lookup_window(prem, sizeof(extra), proc, group, &window),
                     "failed to lookup window");
    assert(window);

    if (IS_TARGET_ATOMICS_EMULATION())
    {
        ofi_atomics_t header = {
                                   .rmw =
                                   {
                                       .proto.proc = l_state.proc,
                                       .proto.op = op,
                                       .proto.tag = GETTAG(),
                                       .src = (op == COMEX_SWAP) ? *(int*)ploc : (op == COMEX_SWAP_LONG) ? *(uint64_t*)ploc : 0,
                                       .extra = extra,
                                       .addr = (uint64_t)prem
                                   }
                               };

        OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                             &header,
                             sizeof(header),
                             window->peer_tagged->fi_addr,
                             ATOMICS_PROTO_TAGMASK),
                             "failed to send tagged:");

        OFI_RETRY(fi_trecv(ofi_data.ep_tagged.endpoint,
                           ploc,
                           (op == COMEX_FETCH_AND_ADD || op == COMEX_SWAP) ? sizeof(int) : sizeof(uint64_t),
                           0,
                           window->peer_tagged->fi_addr,
                           ATOMICS_DATA_TAGMASK | header.proto.tag,
                           0,
                           &request),
                           "failed to send tagged:");

        COMEX_CHKANDJUMP(wait_request(&request), "failed to wait request");
    }
    else
    {
        union atomics_param_t
        {
            int int_value;
            long long_value;
        } param;

        switch (op)
        {
            case COMEX_FETCH_AND_ADD:
            {
                OFI_RETRY(fi_fetch_atomic(ofi_data.ep_atomics.endpoint,
                                          &extra,
                                          1,
                                          0,
                                          ploc,
                                          0,
                                          window->peer_atomics->fi_addr,
                                          get_remote_addr(window->ptr, prem, ot_atomic),
                                          window->key_atomics,
                                          FI_INT32,
                                          FI_SUM,
                                          &request),
                                          "fi_fetch_atomic error:");
            }
            break;
            case COMEX_FETCH_AND_ADD_LONG:
            {
                param.long_value = (long)extra;
                OFI_RETRY(fi_fetch_atomic(ofi_data.ep_atomics.endpoint,
                                          &(param.long_value),
                                          1,
                                          0,
                                          ploc,
                                          0,
                                          window->peer_atomics->fi_addr,
                                          get_remote_addr(window->ptr, prem, ot_atomic),
                                          window->key_atomics,
                                          FI_INT64,
                                          FI_SUM,
                                          &request),
                                          "fi_fetch_atomic error:");
            }
            break;
            case COMEX_SWAP:
            {
                param.int_value = *(int*)ploc;
                OFI_RETRY(fi_fetch_atomic(ofi_data.ep_atomics.endpoint,
                                          &(param.int_value),
                                          1,
                                          0,
                                          ploc,
                                          0,
                                          window->peer_atomics->fi_addr,
                                          get_remote_addr(window->ptr, prem, ot_atomic),
                                          window->key_atomics,
                                          FI_INT32,
                                          FI_ATOMIC_WRITE,
                                          &request),
                                          "fi_fetch_atomic error:");
            }
            break;
            case COMEX_SWAP_LONG:
            {
                param.long_value = *(long*)ploc;
                OFI_RETRY(fi_fetch_atomic(ofi_data.ep_atomics.endpoint,
                                          &(param.long_value),
                                          1,
                                          0,
                                          ploc,
                                          0,
                                          window->peer_atomics->fi_addr,
                                          get_remote_addr(window->ptr, prem, ot_atomic),
                                          window->key_atomics,
                                          FI_INT64,
                                          FI_ATOMIC_WRITE,
                                          &request),
                                          "fi_fetch_atomic error");
            }
            break;
            default:
                EXPR_CHKANDJUMP(0, "incorrect rmw operation type");
                break;
        }
        COMEX_CHKANDJUMP(wait_request(&request), "failed to wait request");
    }

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

/* Mutex Operations */
static int create_mutexes(mutex_t** mtx, int num)
{
    typedef struct mutex_cnt_t
    {
        int proc;
        int count;
    } mutex_cnt_t;

    mutex_cnt_t* global = 0;
    mcs_mutex_t* unsorted = 0;
    mutex_t*     mutex = 0;

    assert(!*mtx);

    /* calculate total number of mutexes */
    mutex_cnt_t info = {.proc = l_state.proc, .count = num};

    COMEX_CHKANDJUMP(exchange_with_all(&info, sizeof(info), COMEX_GROUP_WORLD, (void**)&global),
                     "failed to publish mutex info");

    int group_size = l_state.size;
    int i;
    int count = 0;
    for (i = 0; i < group_size; i++)
        count += global[i].count;

    mutex = malloc(sizeof(*mutex));
    EXPR_CHKANDJUMP(mutex, "failed to allocate mutex object");
    memset(mutex, 0, sizeof(*mutex));

    mutex->elem_offset = malloc(group_size * sizeof(*mutex->elem_offset));
    mutex->mcs_mutex = malloc(group_size * sizeof(*mutex->mcs_mutex));

    EXPR_CHKANDJUMP(mutex->elem_offset && mutex->mcs_mutex, "failed to allocate mutex object data");
    memset(mutex->mcs_mutex, 0, group_size * sizeof(*mutex->mcs_mutex));

    /* allocate data for local mutex & register it in OFI mr */
    /* total number of elements to allocate is count + num:
     *   count - total number of mutexes,
     *   num   - number of mutexes on current proc.
     * allocated 'count' elements for elem + 'num' elements for
     * tail.
     * add data is allocated as solid array */
    mcs_mutex_t local_mutex = {
                                  .proc = l_state.proc,
                                  .count = num,
                                  .key = 0,
                                  .tail = 0,
                                  .elem = 0
                              };
    /*mcs_mutex_t local_mutex = {.proc = l_state.proc, .count = num, .key = 0, .tail = 0, .elem_idx = 0};*/
    size_t buflen = (count + num) * sizeof(*local_mutex.tail);
    mutex->data = malloc(buflen);
    local_mutex.tail = mutex->data;
    local_mutex.elem = local_mutex.tail + num;
    /*local_mutex.elem_idx = num;*/

    EXPR_CHKANDJUMP(mutex->data, "failed to allocate mutex object shared data");

    /* set default values to PROC_NONE (0 could not be used because 0 is valid proc value) */
    for (i = 0; i < count + num; i++)
        local_mutex.tail[i] = PROC_NONE;

    struct fi_context context;
    COMEX_CHKANDJUMP(mr_reg(mutex->data, buflen, MR_ACCESS_PERMISSIONS,
                            &mutex->mr, &context, ot_atomic),
                            "failed to register memory:");

    OFI_CALL(local_mutex.key, fi_mr_key(mutex->mr));
    /*MUTEX_MR_KEY(mutex->mr, local_mutex.key);*/

    /* data allocated, let's publish it to other procs */
    COMEX_CHKANDJUMP(exchange_with_all(&local_mutex, sizeof(local_mutex), COMEX_GROUP_WORLD, (void**)&unsorted),
                     "failed to publish mutex data");

    /* sort data by proc for better performance */
    for (i = 0; i < group_size; i++)
    {
        assert(unsorted[i].proc < group_size);
        mutex->mcs_mutex[unsorted[i].proc] = unsorted[i];
    }

    /* ok, now calculate offset of elem item for every proc */
    int offset = 0;
    for (i = 0; i < group_size; i++)
    {
        mutex->elem_offset[i] = offset;
        offset += mutex->mcs_mutex[i].count;
    }

    *mtx = mutex;

fn_success:
    if (global) free(global);
    if (unsorted) free(unsorted);
    return COMEX_SUCCESS;

fn_fail:
    if (global) free(global);
    if (mutex)
    {
        if (mutex->mr) OFI_VCALL(fi_close((struct fid*)(mutex->mr)));
        if (mutex->elem_offset) free(mutex->elem_offset);
        if (mutex->mcs_mutex) free(mutex->mcs_mutex);
        free(mutex);
        mutex = 0;
    }
    if (unsorted) free(unsorted);
    return COMEX_FAILURE;
}

static int destroy_mutexes(mutex_t* mutex)
{
    assert(mutex);

    COMEX_CHKANDJUMP(comex_barrier(COMEX_GROUP_WORLD), "failed to barrier");

    COMEX_CHKANDJUMP(mr_unreg(&(mutex->mr->fid)), "fi_close mutex memory region:");

    free(mutex->data);
    free(mutex->elem_offset);
    free(mutex->mcs_mutex);
    free(mutex);
    mutex = 0;

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static int lock_mutex(mutex_t* mutex, int mtx, int proc)
{
    assert(mutex);
    assert(mtx < mutex->mcs_mutex[proc].count);

    int my_proc = l_state.proc;
    int elem_index = ELEM_INDEX(mutex, proc, mtx);

    void* buf = 0;

    /* reset 'elem' element of current proc
     * using direct access because writing to this value is not
     * concurrent with remote write */
    mutex->mcs_mutex[my_proc].elem[elem_index] = PROC_NONE;
    /*MUTEX_ELEM(mutex->mcs_mutex[my_proc], elem_index) = PROC_NONE;*/

    request_t request;
    init_request(&request);

    /* trying to lock mutex */
    int prev = PROC_NONE;
    OFI_RETRY(fi_fetch_atomic(ofi_data.ep_atomics.endpoint,
                              &my_proc,
                              1,
                              0,
                              &prev,
                              0,
                              ofi_data.ep_atomics.peers[proc].fi_addr,
                              get_remote_addr((uint64_t)mutex->mcs_mutex[proc].tail,
                                              mutex->mcs_mutex[proc].tail + mtx,
                                              ot_atomic),
                              mutex->mcs_mutex[proc].key,
                              FI_INT32,
                              FI_ATOMIC_WRITE,
                              &request),
                              "failed to process atomic ops:");
    WAIT_COMPLETION_AND_RESET(&request);
    /*MUTEX_FOP_ATOMIC(mutex, proc, &my_proc, &prev, mtx, MUTEX_OP_WRITE);*/

    if (prev != PROC_NONE)
    {
        /* mutex was locked by another proc. write to prev's 'elem' object current
           proc & wait for notification from it */
        OFI_RETRY(fi_atomic(ofi_data.ep_atomics.endpoint,
                            &my_proc,
                            1,
                            0,
                            ofi_data.ep_atomics.peers[prev].fi_addr,
                            get_remote_addr((uint64_t)mutex->mcs_mutex[prev].tail,
                                            mutex->mcs_mutex[prev].elem + elem_index,
                                            ot_atomic),
                            mutex->mcs_mutex[prev].key,
                            FI_INT32,
                            FI_ATOMIC_WRITE,
                            &request),
                            "failed to process atomic ops:");
        WAIT_COMPLETION_AND_RESET(&request);

        int _buf;
        buf = ofi_data.msg_prefix_size ? malloc(sizeof(int) + ofi_data.msg_prefix_size) : &_buf;
        OFI_RETRY(fi_trecv(ofi_data.ep_tagged.endpoint,
                           buf,
                           sizeof(int) + ofi_data.msg_prefix_size,
                           0,
                           ofi_data.ep_tagged.peers[prev].fi_addr,
                           mutex->tagmask | mtx,
                           0,
                           &request),
                           "failed to receive tagged:");
        WAIT_COMPLETION_AND_RESET(&request);

        if (ofi_data.msg_prefix_size) free(buf);
    }

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    if (buf) free(buf);
    return COMEX_FAILURE;
}


static int unlock_mutex(mutex_t* mutex, int mtx, int proc)
{
    assert(mutex);
    assert(mtx < mutex->mcs_mutex[proc].count);

    int elem_index = ELEM_INDEX(mutex, proc, mtx);
    int my_proc = l_state.proc;

    request_t request;
    init_request(&request);

    /* trying to lock mutex */
    int next = PROC_NONE;
    /* read local 'next' value using atomic to prevent condition races:
     * same value may be processed by remote host at same time */
    OFI_RETRY(fi_fetch_atomic(ofi_data.ep_atomics.endpoint,
                              &my_proc,
                              1,
                              0,
                              &next,
                              0,
                              ofi_data.ep_atomics.peers[my_proc].fi_addr,
                              get_remote_addr((uint64_t)mutex->mcs_mutex[my_proc].tail,
                                              mutex->mcs_mutex[my_proc].elem + elem_index,
                                              ot_atomic),
                              mutex->mcs_mutex[my_proc].key,
                              FI_INT32,
                              FI_ATOMIC_READ,
                              &request),
                              "failed to process atomic ops:");
    WAIT_COMPLETION_AND_RESET(&request);
    /*MUTEX_READ(mutex, my_proc, &next, MUTEX_ELEM_IDX(mutex->mcs_mutex[my_proc], elem_index));*/

    if (next == PROC_NONE)
    {
        /* check if somebody is waiting for mutex unlock */
        int no_proc = PROC_NONE;
        int tail = PROC_NONE;
        OFI_RETRY(fi_compare_atomic(ofi_data.ep_atomics.endpoint,
                                    &no_proc,
                                    1,
                                    0,
                                    &my_proc,
                                    0,
                                    &tail,
                                    0,
                                    ofi_data.ep_atomics.peers[proc].fi_addr,
                                    get_remote_addr((uint64_t)mutex->mcs_mutex[proc].tail,
                                                    mutex->mcs_mutex[proc].tail + mtx,
                                                    ot_atomic),
                                    mutex->mcs_mutex[proc].key,
                                    FI_INT32,
                                    FI_CSWAP,
                                    &request),
                                    "failed to process atomic ops:");
        WAIT_COMPLETION_AND_RESET(&request);
        /*MUTEX_CSWAP(mutex, proc, &no_proc, &my_proc, &tail, mtx);*/

        if (tail != my_proc)
        {
            /*printf("[%d] mutex tailed by: %d\n", l_state.proc, tail);*/
            /*printf("[%d] reading from %d -> %d\n", l_state.proc, my_proc, MUTEX_ELEM_IDX(mutex->mcs_mutex[my_proc], elem_index));*/
            while (next == PROC_NONE)
            {
                OFI_RETRY(fi_fetch_atomic(ofi_data.ep_atomics.endpoint,
                                          &my_proc,
                                          1,
                                          0,
                                          &next,
                                          0,
                                          ofi_data.ep_atomics.peers[my_proc].fi_addr,
                                          get_remote_addr((uint64_t)mutex->mcs_mutex[my_proc].tail,
                                                          mutex->mcs_mutex[my_proc].elem + elem_index,
                                                          ot_atomic),
                                          mutex->mcs_mutex[my_proc].key,
                                          FI_INT32,
                                          FI_ATOMIC_READ,
                                          &request),
                                          "failed to process atomic ops:");
                WAIT_COMPLETION_AND_RESET(&request);
                /*MUTEX_READ(mutex, my_proc, &next, MUTEX_ELEM_IDX(mutex->mcs_mutex[my_proc], elem_index));*/
            }
            /*printf("[%d] mutex next to: %d\n", l_state.proc, next);*/
        }
    }

    if (next != PROC_NONE)
    {
        int _data;
        void* data = ofi_data.msg_prefix_size ? malloc(sizeof(int) + ofi_data.msg_prefix_size) : &_data;
        OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                             data,
                             sizeof(int) + ofi_data.msg_prefix_size,
                             ofi_data.ep_tagged.peers[next].fi_addr,
                             mutex->tagmask | mtx),
                             "failed to send tagged:");
        if (ofi_data.msg_prefix_size) free(data);
        /* do not wait for operation completed, use dtor callback to clean buffer instead */
    }

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

/* AM mutex is mutex implemented on holder side (active messages)
 * used same queue for wait mutexes, assuming that rank could not
 * wait for 2 different mutexes at same time */
static int create_am_mutexes(int num)
{
    int i;

    am_mutex_locked = (int*)malloc(num * sizeof(*am_mutex_locked));
    EXPR_CHKANDJUMP(am_mutex_locked, "failed to allocate mutex");
    am_mutex_waiter = (int*)malloc(l_state.size * sizeof(*am_mutex_waiter));
    EXPR_CHKANDJUMP(am_mutex_waiter, "failed to allocate mutex");

    for(i = 0; i < num; i++)
        am_mutex_locked[i] = PROC_NONE;
    for(i = 0; i < l_state.size; i++)
        am_mutex_waiter[i] = PROC_NONE;

    comex_barrier(COMEX_GROUP_WORLD);

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static int destroy_am_mutexes()
{
    EXPR_CHKANDJUMP(am_mutex_locked, "mutex is not allocated");
    EXPR_CHKANDJUMP(am_mutex_waiter, "mutex is not allocated");

    comex_barrier(COMEX_GROUP_WORLD);

    free(am_mutex_locked);
    free(am_mutex_waiter);

    am_mutex_locked = am_mutex_waiter = 0;

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static int lock_am_mutex(int num, int proc)
{
    request_t request;
    init_request(&request);

    ofi_atomics_t header = {
                               .mutex =
                               {
                                   .proto.proc = l_state.proc,
                                   .proto.op = OFI_MUTEX_AM_LOCK,
                                   .num = num
                               }
                           };

    OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                         &header,
                         sizeof(header),
                         ofi_data.ep_tagged.peers[proc].fi_addr,
                         ATOMICS_PROTO_TAGMASK),
                         "failed to send tagged:");

    int v;
    OFI_RETRY(fi_trecv(ofi_data.ep_tagged.endpoint,
                       &v,
                       sizeof(v),
                       0,
                       ofi_data.ep_tagged.peers[proc].fi_addr,
                       ATOMICS_MUTEX_TAGMASK | l_state.proc,
                       0,
                       &request),
                       "failed to send tagged:");

    COMEX_CHKANDJUMP(wait_request(&request), "failed to wait request");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static int unlock_am_mutex(int num, int proc)
{
    ofi_atomics_t header = {
                               .mutex =
                               {
                                   .proto.proc = l_state.proc,
                                   .proto.op = OFI_MUTEX_AM_UNLOCK,
                                   .num = num
                               }
                           };

    OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                         &header,
                         sizeof(header),
                         ofi_data.ep_tagged.peers[proc].fi_addr,
                         ATOMICS_PROTO_TAGMASK),
                         "failed to send tagged:");

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_lock(int mtx, int proc)
{
    if (IS_TARGET_ATOMICS_EMULATION())
        return lock_am_mutex(mtx, proc);
    else
        return lock_mutex(global_mutex, mtx, proc);
}

int comex_unlock(int mtx, int proc)
{
    if (IS_TARGET_ATOMICS_EMULATION())
        return unlock_am_mutex(mtx, proc);
    else
        return unlock_mutex(global_mutex, mtx, proc);
}

int comex_create_mutexes(int num)
{
    if (IS_TARGET_ATOMICS_EMULATION())
        COMEX_CHKANDJUMP(create_am_mutexes(num), "failed to create mutexes");
    else
    {
        COMEX_CHKANDJUMP(create_mutexes(&global_mutex, num),
                         "failed to create mutexes");
        assert(global_mutex);
        global_mutex->tagmask = MUTEX_TAGMASK;
    }

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_destroy_mutexes()
{
    if (IS_TARGET_ATOMICS_EMULATION())
        COMEX_CHKANDJUMP(destroy_am_mutexes(), "failed to destroy mutexes");
    else
    {
        COMEX_CHKANDJUMP(destroy_mutexes(global_mutex),
                         "failed to destroy mutexes");
        global_mutex = 0;
    }

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_malloc(void* ptrs[], size_t size, comex_group_t group)
{
    typedef struct wnd_data_t
    {
        uint64_t key_rma;
        uint64_t key_atomics;
        uint64_t ptr;
        size_t   size;
        int      local_proc; /* proc in current group 'group' */
    } wnd_data_t;

    wnd_data_t* all_wnd_data = 0;
    local_window_t* l_wnd = malloc(sizeof(*l_wnd));
    EXPR_CHKANDJUMP(l_wnd, "failed to allocate local window");
    memset(l_wnd, 0, sizeof(*l_wnd));

    if (size)
    {
        l_wnd->ptr = comex_malloc_local(size);
        EXPR_CHKANDJUMP(l_wnd->ptr, "failed to allocate local window buffer");

        struct fi_context context;
        COMEX_CHKANDJUMP(mr_reg(l_wnd->ptr, size, MR_ACCESS_PERMISSIONS,
                                &l_wnd->mr_rma, &context, ot_rma),
                                "failed to register memory:");

        if (dual_provider())
        {
            COMEX_CHKANDJUMP(mr_reg(l_wnd->ptr, size, MR_ACCESS_PERMISSIONS,
                                    &l_wnd->mr_atomics, &context, ot_atomic),
                                    "failed to register memory:");
        }
        else
            l_wnd->mr_atomics = l_wnd->mr_rma;
    }

    wnd_data_t my_wnd_data = {
                                 .ptr = (uint64_t)l_wnd->ptr,
                                 .size = size,
                                 .local_proc = 0
                             };
    if (size)
    {
        OFI_CALL(my_wnd_data.key_rma, fi_mr_key(l_wnd->mr_rma));
        OFI_CALL(my_wnd_data.key_atomics, fi_mr_key(l_wnd->mr_atomics));
    }
    comex_group_rank(group, &my_wnd_data.local_proc);

    COMEX_CHKANDJUMP(exchange_with_all(&my_wnd_data, sizeof(wnd_data_t), group, (void**)&all_wnd_data),
                     "failed to exchange memory regions");

    int group_size = 0;
    comex_group_size(group, &group_size);

     /* ok, we have locally created window, also we have info about all other windows in group*/
     /* let's create appropriate structures */

    /* create local window structure */
    INSERT_TO_LIST(local_wnd, l_wnd);

    /* create remote windows structures */
    int wnd_idx;
    for (wnd_idx = 0; wnd_idx < group_size; wnd_idx++)
    {
        wnd_data_t* wnd_data = all_wnd_data + wnd_idx;
        int world_proc = PROC_NONE;
        comex_group_translate_world(group, wnd_data->local_proc, &world_proc);
        EXPR_CHKANDJUMP((world_proc != PROC_NONE), "invalid world proc");

        ofi_window_t* remote_wnd = malloc(sizeof(*remote_wnd));
        EXPR_CHKANDJUMP(remote_wnd, "failed to allocate data for remote window");

        remote_wnd->local = l_wnd;
        remote_wnd->peer_rma = ofi_data.ep_rma.peers + world_proc;
        remote_wnd->peer_atomics = ofi_data.ep_atomics.peers + world_proc;
        remote_wnd->world_proc = world_proc;
        remote_wnd->key_rma = wnd_data->key_rma;
        remote_wnd->key_atomics = wnd_data->key_atomics;
        remote_wnd->ptr = wnd_data->ptr;
        remote_wnd->size = wnd_data->size;
        INSERT_TO_LIST(ofi_wnd, remote_wnd);

        ptrs[wnd_data->local_proc] = (void*)wnd_data->ptr;
    }

    comex_barrier(group);

fn_success:
    if (all_wnd_data) free(all_wnd_data);
    return COMEX_SUCCESS;

fn_fail:
    if (l_wnd)
    {
        REMOVE_FROM_LIST(local_wnd, l_wnd, local_window_t);
        if (l_wnd->ptr) free(l_wnd->ptr);
        if (l_wnd->mr_rma) OFI_VCALL(fi_close((struct fid*)l_wnd->mr_rma));
        if (l_wnd->mr_atomics) OFI_VCALL(fi_close((struct fid*)l_wnd->mr_atomics));
        free(l_wnd);
    }
    if (all_wnd_data) free(all_wnd_data);
    return COMEX_FAILURE;
}

int comex_malloc_mem_dev(void *ptrs[], size_t size, comex_group_t group,
        const char* device)
{
    return comex_malloc(ptrs,size,group);
}

int comex_free(void* ptr, comex_group_t group)
{
    COMEX_CHKANDJUMP(comex_barrier(group), "failed to barrier");

    local_window_t* local = local_wnd;
    while (local)
    {
        if (ptr == local->ptr) break;
        local = local->next;
    }

    EXPR_CHKANDJUMP(local, "failed to locate local window");

    ofi_window_t* ofi = ofi_wnd;
    while (ofi)
    {
        if (ofi->local == local)
        {
            ofi_window_t* to_remove = ofi;
            ofi = ofi->next;
            REMOVE_FROM_LIST(ofi_wnd, to_remove, ofi_window_t);

            if (to_remove == ofi_wnd_cache)
                ofi_wnd_cache = 0;

            free(to_remove);
        }
        else
            ofi = ofi->next;
    }

    if (ptr)
    {
        int ret;
        COMEX_CHKANDJUMP(mr_unreg(&local->mr_rma->fid), "fi_close memory region:");
        if (dual_provider())
            COMEX_CHKANDJUMP(mr_unreg(&local->mr_atomics->fid), "fi_close memory region:");

        comex_free_local(ptr);
        REMOVE_FROM_LIST(local_wnd, local, local_window_t);
        free(local);
    }

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

int comex_free_dev(void *ptr, comex_group_t group)
{
    return comex_free(ptr, group);
}

static inline int destroy_all_windows()
{
    local_window_t* local = local_wnd;
    while (local)
    {
        COMEX_CHKANDJUMP(comex_free(local->ptr, 0),
                         "failed to free window");
        local = local->next;
    }

fn_success:
    return COMEX_SUCCESS;

fn_fail:
    return COMEX_FAILURE;
}

static void acquire_remote_lock(int proc)
{
    assert(!IS_TARGET_ATOMICS_EMULATION());
    lock_mutex(local_mutex, 0, proc);
}

static void release_remote_lock(int proc)
{
    assert(!IS_TARGET_ATOMICS_EMULATION());
    unlock_mutex(local_mutex, 0, proc);
}

#ifdef USE_ATOMIC_EMULATION

#define CALC(_dst, _src, _scale, len, type) \
do                                          \
{                                           \
    int i;                                  \
    int cnt = len / sizeof(type);           \
    type* src = (type*)(_src);              \
    type* dst = (type*)(_dst);              \
    type  scl = *(type*)(_scale);           \
    for (i = 0; i < cnt; i++, src++, dst++) \
        *dst = (*src) * scl;                \
} while(0)

static inline void acc_emu(
        int datatype, int count, void* get_buf,
        void* src_ptr, size_t src_idx, void* scale);

static int comex_nbaccs_emu(
        int datatype, void* scale,
        void* src_ptr, int* src_stride_ar,
        void* dst_ptr, int* dst_stride_ar,
        int* count, int stride_levels,
        int proc, comex_group_t group,
        comex_request_t* comex_request)
{
    assert(env_data.native_atomics == 0);
    assert(env_data.emulation_type == et_origin || env_data.emulation_type == et_target);

    int i, j;
    size_t src_idx, dst_idx;  /* index offset of current block position to ptr */
    int n1dim;  /* number of 1 dim block */
    size_t src_bvalue[7], src_bunit[7];
    size_t dst_bvalue[7], dst_bunit[7];
    void* get_buf = 0;
    int result = COMEX_SUCCESS;
    int world_proc = PROC_NONE;

    void* desc = 0;
    int chunk = 0;

    request_t* request = 0;

    if (comex_request) *comex_request = HANDLE_UNDEFINED;

    /* number of n-element of the first dimension */
    n1dim = 1;
    for (i = 1; i <= stride_levels; i++)
        n1dim *= count[i];

    ofi_atomics_t header = {
                               .acc =
                               {
                                   .proto.proc = l_state.proc,
                                   .proto.op = datatype,
                                   .proto.tag = GETTAG(),
                                   .count = n1dim,
                                   .len = count[0],
                                   .posted = 0
                               }
                           };

    /* calculate the destination indices */
    src_bvalue[0] = 0; src_bvalue[1] = 0; src_bunit[0] = 1; src_bunit[1] = 1;
    dst_bvalue[0] = 0; dst_bvalue[1] = 0; dst_bunit[0] = 1; dst_bunit[1] = 1;

    for (i = 2; i <= stride_levels; i++)
    {
        src_bvalue[i] = 0;
        dst_bvalue[i] = 0;
        src_bunit[i] = src_bunit[i - 1] * count[i - 1];
        dst_bunit[i] = dst_bunit[i - 1] * count[i - 1];
    }

    if (env_data.emulation_type == et_origin && 0 == skip_lock)
    {
        // grab the atomics lock
        acquire_remote_lock(proc);
    }

    if (env_data.emulation_type == et_origin)
    {
        /* use common buffer to acc */
        get_buf = (char *)malloc(sizeof(char) * count[0]);
        EXPR_CHKANDJUMP(get_buf, "failed to allocate memory\n");
    }
    else if (env_data.emulation_type == et_target)
    {
        /* use p2p way for atomics: send header to allow pre-post request */
        VALIDATE_GROUP_AND_PROC(group, proc);
        comex_group_translate_world(group, proc, &world_proc);
        EXPR_CHKANDJUMP((world_proc != PROC_NONE), "invalid world proc");

        OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                             &header,
                             sizeof(header),
                             ofi_data.ep_tagged.peers[world_proc].fi_addr,
                             ATOMICS_PROTO_TAGMASK),
                             "failed to send tagged:");

        /* allocate buffer for all acc data  & register it */
        chunk = sizeof(ofi_atomics_t) + (sizeof(char) * count[0]);
        int total = chunk * n1dim;

        request = alloc_request();
        EXPR_CHKANDJUMP(request, "failed to allocate request");

        request->group = group;
        request->proc = proc;

        request->dtor = req_dtor;
        increment_request_cnt(request);
        if (comex_request) *comex_request = request->index;
        else request->flags |= rf_auto_free;

        request_t* complete = alloc_request();
        EXPR_CHKANDJUMP(complete, "failed to allocate request");
        set_parent_request(request, complete);
        OFI_RETRY(fi_trecv(ofi_data.ep_tagged.endpoint,
                           &complete->inplace,
                           sizeof(complete->inplace),
                           0,
                           ofi_data.ep_tagged.peers[world_proc].fi_addr,
                           ATOMICS_ACC_CMPL_TAGMASK | header.proto.tag,
                           0,
                           complete),
                           "failed to send tagged:");

        if (chunk > ofi_data.max_buffered_send)
        {
            /* chunk used in comaring because in case if chunk is small data is sent by inject */
            get_buf = (char *)malloc(total);
            request->data = get_buf;
        }
        else
        {
            /* if chunk is small - use same buffer for all packets (use inject call) */
            get_buf = (char *)malloc(chunk);
        }
        EXPR_CHKANDJUMP(get_buf, "failed to allocate memory\n");
    }

    for (i = 0; i < n1dim; i++)
    {
        src_idx = 0;
        for (j = 1; j <= stride_levels; j++)
        {
            src_idx += src_bvalue[j] * src_stride_ar[j - 1];
            if ((i + 1) % src_bunit[j] == 0)
            {
                src_bvalue[j]++;
            }
            if (src_bvalue[j] > (count[j] - 1))
            {
                src_bvalue[j] = 0;
            }
        }

        dst_idx = 0;

        for(j = 1; j <= stride_levels; j++)
        {
            dst_idx += dst_bvalue[j] * dst_stride_ar[j - 1];
            if ((i + 1) % dst_bunit[j] == 0)
            {
                dst_bvalue[j]++;
            }
            if(dst_bvalue[j] > (count[j] - 1))
            {
                dst_bvalue[j] = 0;
            }
        }

        if (env_data.emulation_type == et_origin)
        {
            // Get the remote data in a temp buffer
            COMEX_CHKANDJUMP(comex_get((char *)dst_ptr + dst_idx, get_buf, count[0], proc, group),
                    "comex_accs_emu: failed to get data");

            // Local accumulate
            acc_emu(datatype, count[0], get_buf, src_ptr, src_idx, scale);

            // Write back to remote data
            COMEX_CHKANDJUMP(comex_put(get_buf, (char *)dst_ptr + dst_idx, count[0], proc, group),
                    "comex_accs_emu: failed to put data");
        }
        else if (env_data.emulation_type == et_target)
        {
            /* use p2p way for atomics: send header to allow pre-post request */
            ofi_atomics_t _hdr = {
                                     .acc =
                                     {
                                         .proto.proc = l_state.proc,
                                         .proto.op = datatype,
                                         .proto.tag = header.proto.tag,
                                         .len = count[0],
                                         .posted = count[0],
                                         .addr = (uint64_t)((char *)dst_ptr + dst_idx),
                                         .count = 0
                                     }
                                 };
            ofi_atomics_t* hdr = get_buf;
            *hdr = _hdr;

            switch (datatype)
            {
                case COMEX_ACC_DBL:
                    CALC(hdr->acc.data, (char*)src_ptr + src_idx, scale, count[0], double);
                    break;
                case COMEX_ACC_FLT:
                    CALC(hdr->acc.data, (char*)src_ptr + src_idx, scale, count[0], float);
                    break;
                case COMEX_ACC_INT:
                    CALC(hdr->acc.data, (char*)src_ptr + src_idx, scale, count[0], int);
                    break;
                case COMEX_ACC_LNG:
                    CALC(hdr->acc.data, (char*)src_ptr + src_idx, scale, count[0], long);
                    break;
                case COMEX_ACC_DCP:
                    CALC(hdr->acc.data, (char*)src_ptr + src_idx, scale, count[0], double complex);
                    break;
                case COMEX_ACC_CPL:
                    CALC(hdr->acc.data, (char*)src_ptr + src_idx, scale, count[0], float complex);
                    break;
                default:
                    EXPR_CHKANDJUMP(0, "incorrect datatype: %d\n", datatype);
                    break;
            }

            if (chunk > ofi_data.max_buffered_send)
            {
                request_t* child = alloc_request();
                EXPR_CHKANDJUMP(child, "failed to allocate request");
                set_parent_request(request, child);

                OFI_RETRY(fi_tsend(ofi_data.ep_tagged.endpoint,
                                   hdr,
                                   chunk,
                                   desc,
                                   ofi_data.ep_tagged.peers[world_proc].fi_addr,
                                   ATOMICS_ACC_DATA_TAGMASK | header.proto.tag,
                                   child),
                                   "failed to send tagged:");
                get_buf += chunk;
            }
            else
            {
                OFI_RETRY(fi_tinject(ofi_data.ep_tagged.endpoint,
                                     hdr,
                                     chunk,
                                     ofi_data.ep_tagged.peers[world_proc].fi_addr,
                                     ATOMICS_ACC_DATA_TAGMASK | header.proto.tag),
                                     "failed to send tagged:");
            }
        }
    }

    if (request)
    {
        decrement_request_cnt(request);
        if(request->data) get_buf = 0; /* buffer will be removed by request */
    }

    if (env_data.emulation_type == et_origin && 0 == skip_lock)
    {
        // ungrab the lock
        release_remote_lock(proc);
    }

fn_exit:
    if (get_buf) free(get_buf);
    if (env_data.force_sync)
    {
        if (request) wait_request(request);
        if (comex_request) *comex_request = HANDLE_UNDEFINED;
    }
    return result;

fn_fail:
    result = COMEX_FAILURE;
    goto fn_exit;

}

static int comex_acc_emu(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group)
{
    return comex_accs_emu(
            datatype, scale,
            src_ptr, NULL,
            dst_ptr, NULL,
            &bytes, 0,
            proc, group);
}

static int comex_nbacc_emu(
        int datatype, void* scale,
        void* src_ptr, void* dst_ptr, int bytes,
        int proc, comex_group_t group,
        comex_request_t *hdl)
{
    return comex_nbaccs_emu(
            datatype, scale,
            src_ptr, NULL,
            dst_ptr, NULL,
            &bytes, 0,
            proc, group, hdl);
}

int comex_accs_emu(
        int datatype, void* scale,
        void* src, int* src_stride,
        void* dst, int* dst_stride,
        int* count, int stride_levels,
        int proc, comex_group_t group)
{
    comex_request_t hdl;
    COMEX_CHKANDJUMP(comex_nbaccs_emu(datatype, scale,
            src, src_stride, dst, dst_stride,
            count, stride_levels, proc, group, &hdl), "failed");
    return comex_wait(&hdl);

fn_fail:
    return COMEX_FAILURE;
}

int comex_accv_emu(
        int datatype, void* scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group)
{
    comex_request_t hdl;
    COMEX_CHKANDJUMP(comex_nbaccv_emu(datatype, scale, iov, iov_len, proc, group, &hdl), "failed");
    return comex_wait(&hdl);

fn_fail:
    return COMEX_FAILURE;
}

static int comex_nbaccv_emu(
        int datatype, void* scale,
        comex_giov_t *iov, int iov_len,
        int proc, comex_group_t group,
        comex_request_t* hdl)
{
    int i;
    int n;
    int result = COMEX_SUCCESS;
    comex_request_t* reqs = 0;

    if (hdl) *hdl = HANDLE_UNDEFINED;

    if (env_data.emulation_type == et_origin)
    {
        skip_lock = 1;
        acquire_remote_lock(proc);
    }

    int count = 0;
    for (i = 0; i < iov_len; ++i)
        count += iov[i].count;

    reqs = (comex_request_t*)malloc(count * sizeof(*reqs));
    EXPR_CHKANDJUMP(reqs, "failed to llocate data");

    for (i = 0, n = 0; i < iov_len; ++i)
    {
        int j;
        void** src = iov[i].src;
        void** dst = iov[i].dst;
        int bytes = iov[i].bytes;
        int limit = iov[i].count;
        for (j = 0; j < limit; ++j, n++)
        {
            comex_request_t h = HANDLE_UNDEFINED;
            COMEX_CHKANDJUMP(comex_nbacc_emu(datatype, scale, src[j], dst[j], bytes, proc, group, &reqs[n]),
                             "comex_accv_emu: failed to acc");
        }
    }

    for (i = 0; i < count; i++)
        comex_wait(&reqs[i]);

fn_exit:
    if (reqs) free(reqs);
    if (env_data.emulation_type == et_origin)
    {
        skip_lock = 0;
        release_remote_lock(proc);
    }
    return result;
fn_fail:
    result = COMEX_FAILURE;
    goto fn_exit;

}

static inline void acc_emu(
        int datatype, int count, void* get_buf,
        void* src_ptr, size_t src_idx, void* scale)
{
#define EQ_ONE_REG(A) ((A) == 1.0)
#define EQ_ONE_CPL(A) ((A).real == 1.0 && (A).imag == 0.0)
#define IADD_REG(A,B) (A) += (B)
#define IADD_CPL(A,B) (A).real += (B).real; (A).imag += (B).imag
#define IADD_SCALE_REG(A,B,C) (A) += (B) * (C)
#define IADD_SCALE_CPL(A,B,C) (A).real += ((B).real*(C).real) - ((B).imag*(C).imag);\
                              (A).imag += ((B).real*(C).imag) + ((B).imag*(C).real);

#define ACC(WHICH, COMEX_TYPE, C_TYPE)                                 \
    if (datatype == COMEX_TYPE)                                        \
    {                                                                  \
        int m;                                                         \
        int m_lim = count / sizeof(C_TYPE);                            \
        C_TYPE* iterator = (C_TYPE*)get_buf;                           \
        C_TYPE* value = (C_TYPE*)((char*)src_ptr + src_idx);           \
        C_TYPE calc_scale = *(C_TYPE*)scale;                           \
        if (EQ_ONE_##WHICH(calc_scale))                                \
        {                                                              \
            for (m = 0 ; m < m_lim; ++m)                               \
            {                                                          \
                IADD_##WHICH(iterator[m], value[m]);                   \
            }                                                          \
        }                                                              \
        else                                                           \
        {                                                              \
            for (m = 0 ; m < m_lim; ++m)                               \
            {                                                          \
                IADD_SCALE_##WHICH(iterator[m], value[m], calc_scale); \
            }                                                          \
        }                                                              \
    }                                                                  \
    else

    ACC(REG, COMEX_ACC_DBL, double)
    ACC(REG, COMEX_ACC_FLT, float)
    ACC(REG, COMEX_ACC_INT, int)
    ACC(REG, COMEX_ACC_LNG, long)
    ACC(CPL, COMEX_ACC_DCP, DoubleComplex)
    ACC(CPL, COMEX_ACC_CPL, SingleComplex)
    {
        assert(0);
    }
#undef ACC
#undef EQ_ONE_REG
#undef EQ_ONE_CPL
#undef IADD_REG
#undef IADD_CPL
#undef IADD_SCALE_REG
#undef IADD_SCALE_CPL
}

#endif /* USE_ATOMIC_EMULATION */

