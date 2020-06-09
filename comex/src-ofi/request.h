/*
 *  * Copyright (c) 2016 Intel Corporation.  All rights reserved.
 */

#ifndef REQUEST_H_
#define REQUEST_H_

#include <assert.h>
#include "comex.h"
#include "comex_impl.h"

#define sizeofa(x) (sizeof(x) / sizeof(*(x)))

#define REQUEST_MAGIC 0xB4BAE6A2
#define REQUEST_CACHE_SIZE 1024
#define PROC_NONE (-1)

#define ZERO_REQUEST(r)        \
    do                         \
    {                          \
        (r)->group = 0;        \
        (r)->proc = PROC_NONE; \
        (r)->flags = rf_none;  \
        (r)->dtor = 0;         \
        (r)->cmpl = 0;         \
        (r)->parent = 0;       \
        (r)->child_cnt = 0;    \
        (r)->data = 0;         \
        (r)->mrs = 0;          \
        (r)->mr_count = 0;     \
        (r)->mr_single = 0;    \
    } while (0)

extern struct request_cache_t* request_cache;

typedef enum request_flag_t
{
    rf_none          = 0,
    rf_auto_free     = 1 << 0,
    rf_no_group_wait = 1 << 1,
    rf_no_free_data  = 1 << 2
} request_flag_t;

typedef enum request_state
{
    rs_none,     /* message object/slot is empty */
    rs_progress, /* slot is in progress */
    rs_complete  /* slot is ready to be read */
} request_state;

typedef struct request_t
{
    /* linked list of received messages */
    struct fi_context context;  /* OFI context */
    int               magic;    /* magic */
    int               index;    /* request index */
    request_state     state;    /* writing rs_progress value to this value must be atomic */
    comex_group_t     group;    /* group to wait */
    int               proc;     /* proc to wait */
    int               flags;    /* some useful flags */
    void              (*dtor)(struct request_t* request); /* destructor callback */
    void              (*cmpl)(struct request_t* request); /* compete state callback */
    struct request_t* parent;   /* pointer to parent request */
    int               child_cnt; /* number of child requests connected here */
    void*             data;     /* request specific data */
    int               inplace;  /* used as buffer to store trecv */

    int               mr_count;
    struct fid_mr**   mrs;
    struct fid_mr*    mr_single; /* some optimization - used when there is single buffer */
} request_t;

typedef struct request_cache_t
{
    request_t request[REQUEST_CACHE_SIZE];
    int index; /* index of first request in current cache (used to find request by index) */
    struct request_cache_t* next;
} request_cache_t;

static inline struct request_cache_t* create_request_cache()
{
    request_cache_t* cache = malloc(sizeof(*cache));
    if (cache)
    {
        cache->next = 0;
        cache->index = 0;
        int i;
        for (i = 0; i < sizeofa(cache->request); i++)
        {
            cache->request[i].state = rs_none;
            cache->request[i].magic = REQUEST_MAGIC;
        }
        return cache;
    }
    else
        return 0;
}

static inline void free_request_cache()
{
    struct request_cache_t* cache = request_cache;
    while (cache)
    {
        struct request_cache_t* c = cache->next;
        free(cache);
        cache = c;
    }
    request_cache = 0;
}

static inline void reset_request(struct request_t* req)
{
    assert(req);
    assert(req->state == rs_complete);
    if (req)
    {
        if (req->dtor)
            req->dtor(req);
        req->state = rs_progress;
        ZERO_REQUEST(req);
    }
}

static inline void free_request(struct request_t* req)
{
    assert(req);
    assert(req->state == rs_complete);
    if (req)
    {
        if (req->dtor)
            req->dtor(req);
        ZERO_REQUEST(req);
        req->state = rs_none;
    }
}

/* find/create empty request slot. function is thread safe due to used
 * atomic primitives to locate/update entries */
static inline struct request_t* alloc_request()
{
    assert(request_cache);
    int i;
    request_t* request = 0;
    /* try to find free request (state == rs_none) */
    struct request_cache_t* cache = request_cache;
    while (cache)
    {
        for (i = 0; i < sizeofa(cache->request); i++)
            if (__sync_bool_compare_and_swap(&cache->request[i].state, rs_none, rs_progress))
            {
                /* update request index */
                cache->request[i].index = cache->index + i;
                ZERO_REQUEST(&cache->request[i]);
                request = cache->request + i; /* ok, one of entries has rs_none state, catched it & return */
                goto fn_exit;
            }
        /* no entries in current cache element, try next one... */
        cache = cache->next;
    }

    /* ok, no free entries... no problemo - create one more element for request cache,
     * catch there one element and append new cache into tail of cache list */
    cache = create_request_cache();
    assert(cache);
    if (cache)
    {
        cache->request[0].state = rs_progress;
        /* add new cache entry into list */
        struct request_cache_t* c = request_cache;
        /* here is trick: __sync_val_compare_and_swap updates c->next only in case if there
         * was zero value, and returns original value. in case if c->next element is not zero
         * then just jump to next element */
        for (i = 0; c; i++)
        {
            /* set index value to 'current' count */
            cache->index = (i + 1) * sizeofa(c->request);
            c = __sync_val_compare_and_swap(&c->next, 0, cache);
        }
        cache->request->index = cache->index;
        ZERO_REQUEST(cache->request);
        request = cache->request;
        goto fn_exit;
    }

fn_exit:
    return request;
}

/* initialize request on stack */
static inline void init_request(request_t* request)
{
    request->magic = REQUEST_MAGIC;
    request->state = rs_progress;
    ZERO_REQUEST(request);
}

/* lookup request by index: list all cache arrays till
 * index is inside of cache array */
static inline request_t* lookup_request(int index)
{
    request_cache_t* cache = request_cache;
    while (cache)
    {
        if (index >= cache->index && index < cache->index + sizeofa(cache->request))
            return cache->request + (index - cache->index);
        cache = cache->next;
    }

    return 0;
}

static inline void complete_request(request_t* request);

static inline void increment_request_cnt(request_t* request)
{
    assert(request);
    assert(request->state == rs_progress);
    __sync_fetch_and_add(&request->child_cnt, 1);
}

static inline int decrement_request_cnt(request_t* request)
{
    assert(request);
    assert(request->state == rs_progress);
    int ret = __sync_sub_and_fetch(&request->child_cnt, 1);
    if (!ret)
    {
        complete_request(request);
    }
    return ret;
}

/* setup request to follow another request */
static inline void set_parent_request(request_t* parent, request_t* child)
{
    assert(parent);
    assert(child);

    child->flags |= rf_auto_free; /* child must be removed automatically */
    increment_request_cnt(parent);
    child->parent = parent;
}

static inline void complete_request(request_t* request)
{
    assert(request);
    assert(request->state == rs_progress);

    request_t _r = *request;

    if (__sync_lock_test_and_set(&request->state, rs_complete) == rs_progress)
    {
        if(_r.cmpl)
            _r.cmpl(request);

        if (_r.parent)
            decrement_request_cnt(_r.parent);

        /* auto-free request should not be waited */
        if (_r.flags & rf_auto_free)
            free_request(request);
    }
}

#endif /* REQUEST_H_ */
