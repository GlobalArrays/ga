#ifndef KR_MALLOC_H /* K&R malloc */
#define KR_MALLOC_H

#ifdef CRAY
#define LOG_ALIGN 6
#elif defined(KSR)
#define LOG_ALIGN 7
#else
#define LOG_ALIGN 6
#endif
 
#define ALIGNMENT (1 << LOG_ALIGN)
 
union header{
  struct {
    unsigned valid1;            /* Token to check if is not overwritten */
    union header *ptr;          /* next block if on free list */
    size_t size;                /* size of this block*/
    unsigned valid2;            /* Another token acting as a guard */
  } s;
  char align[ALIGNMENT];        /* Align to ALIGNMENT byte boundary */
};
 
typedef union header Header;

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
  Header base;                  /* empty list to get started */
  Header *freep;                /* start of free list */
  Header *usedp;                /* start of used list */
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
