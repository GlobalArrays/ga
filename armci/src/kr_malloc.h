#ifndef KR_MALLOC_H /* K&R malloc */
#define KR_MALLOC_H


typedef struct malloc_context {
  size_t usize;                 /* unit size in bytes */
  size_t nalloc;                /* No. of units of length ALIGNMENT */
  size_t max_nalloc;            /* Maximum  no. of units that can get */
  void * (*alloc_fptr)();       /* function pointer to memory alloc routine */
  size_t total;                 /* Amount request from system in units */
  long nchunk;                  /* No. of chunks of system memory */
  long inuse;                   /* Amount in use in units */
  long maxuse;                  /* Maximum value of inuse */
  long nfrags;                  /* No. of fragments divided into */
  long nmcalls;                 /* No. of calls to _armci_alloc() */
  long nfcalls;                 /* No. of calls to memfree */
} context_t;

extern void kr_malloc_init(size_t usize, /* unit size in bytes */
			   size_t nalloc,
			   size_t max_nalloc,
			   void * (*alloc_fptr)(), /* memory alloc routine */
			   int debug,
			   context_t *ctx);

/*
  Returns data aligned on a quad boundary. Even if the request
  size is zero it returns a non-zero pointer.
*/
extern char *kr_malloc(size_t size, context_t *ctx);

/*
  Frees memory allocated by kr_malloc(). Ignores NULL pointers
  but must not be called twice for the same pointer or called
  with non-memalloc'ed pointers
*/
extern void  kr_free(char *ptr, context_t *ctx);

/*
  Print to standard output the usage statistics ... a wrapper
  for kr_malloc_stats();
*/
extern void kr_malloc_print_stats(context_t *ctx);

extern void kr_malloc_verify(context_t *ctx);

#endif
