/* $Id: kr_malloc.c,v 1.4 2003-07-30 23:29:21 d3h325 Exp $ */
#include <stdio.h>
#include "kr_malloc.h"

#define DEBUG 0

/* Storage allocator basically copied from ANSI K&R and corrupted */

extern char *armci_allocate(); /* Used to get memory from the system */
extern void armci_die();

/**
 * DEFAULT_NALLOC: No. of units of length ALIGNMENT to get in every 
 * request to the system for memory (8MB/64 => 128*1024units)
 * DEFAULT_MAX_NALLOC: Maximum number of units that can get i.e.128MB 
 * (if unit size=64bytes, then max units=128MB/64 = 2*1024*1024)
 */
#define DEFAULT_NALLOC       (128*1024)  
#define DEFAULT_NALLOC_ALIGN 1024  
#define DEFAULT_MAX_NALLOC   (1024*1024*2) 

/* mutual exclusion defs go here */
#define  LOCKIT 
#define  UNLOCKIT 

static int do_verify = 0;	/* Flag for automatic heap verification */

#define VALID1  0xaaaaaaaa	/* For validity check on headers */
#define VALID2  0x55555555

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
    unsigned valid1;		/* Token to check if is not overwritten */
    union header *ptr;		/* next block if on free list */
    size_t size;		/* size of this block*/
    unsigned valid2;		/* Another token acting as a guard */
  } s;
  char align[ALIGNMENT];	/* Align to ALIGNMENT byte boundary */
};

typedef union header Header;

static Header base;		/* empty list to get started */
static Header *freep = NULL;	/* start of free list */
static Header *usedp = NULL;	/* start of used list */

static void kr_error(char *s, unsigned long i, context_t *ctx) {
char string[256];
    sprintf(string,"kr_malloc: %s %ld(0x%lx)\n", s, i, i);
#if 0
    kr_malloc_print_stats(ctx);
#endif
    armci_die(string, i);
}

static Header *morecore(size_t nu, context_t *ctx) {
    char *cp;
    Header *up;

#if DEBUG
    (void) printf("morecore 1: Getting %ld more units of length %d nalloc=%d\n",
		  (long)nu, sizeof(Header),ctx->nalloc);
#endif

    (void) fflush(stdout);
    if (ctx->total >= ctx->max_nalloc)
      return (Header *) NULL;   /* Enforce upper limit on core usage */

#if 1
    /* 07/03 ctx->nalloc is now the minimum # units we ask from OS */
    nu = DEFAULT_NALLOC_ALIGN*((nu-1)/DEFAULT_NALLOC_ALIGN+1);
    if(nu < ctx->nalloc) nu = ctx->nalloc; 
#else
    nu = ctx->nalloc*((nu-1)/ctx->nalloc+1); /* nu must by a multiplicity of nalloc */
#endif

#if DEBUG
    (void) printf("morecore: Getting %ld more units of length %d\n",
		  (long)nu, sizeof(Header));
    (void) fflush(stdout);
#endif
    
    if ((cp =(char *)(*ctx->alloc_fptr)((size_t)nu * sizeof(Header))) == (char *)NULL)
      return (Header *) NULL;
    
    ctx->total += nu;   /* Have just got nu more units */
    ctx->nchunk++;      /* One more chunk */
    ctx->nfrags++;      /* Currently one more frag */
    ctx->inuse += nu;   /* Inuse will be decremented by kr_free */
    
    up = (Header *) cp;
    up->s.size = nu;
    up->s.valid1 = VALID1;
    up->s.valid2 = VALID2;
    
    /* Insert into linked list of blocks in use so that kr_free works
       ... for debug only */
    up->s.ptr = usedp;
    usedp = up;
    
    kr_free((char *)(up+1), ctx);  /* Try to join into the free list */
    return freep;
}

void kr_malloc_init(size_t usize, /* unit size in bytes */
		    size_t nalloc,
		    size_t max_nalloc,
		    void * (*alloc_fptr)(), /* memory alloc routine */
		    int debug,
		    context_t *ctx) {
    int scale;

    if(usize <= 0) usize = sizeof(Header);
    
    scale = usize>>LOG_ALIGN;
    if(scale<1)fprintf(stderr,"Error: kr_malloc_init !!!\n");
    
    if(nalloc==0) nalloc = DEFAULT_NALLOC;
    if(max_nalloc==0) max_nalloc = DEFAULT_MAX_NALLOC;

    ctx->usize      = sizeof(Header);
    ctx->nalloc     = nalloc * scale;
    ctx->max_nalloc = max_nalloc * scale;
    ctx->alloc_fptr = alloc_fptr;

    do_verify = debug;
}


char *kr_malloc(size_t nbytes, context_t *ctx) {
    Header *p, *prevp;
    size_t nunits;
    char *return_ptr;
    
    LOCKIT;
    
    /* If first time in need to initialize the free list */ 
    
    if ((prevp = freep) == NULL) { 
      
      if (sizeof(Header) != ALIGNMENT)
	kr_error("Alignment is not valid", (unsigned long) ALIGNMENT, ctx);
      
      ctx->total  = 0;  /* Initialize statistics */
      ctx->nchunk = 0;
      ctx->inuse  = 0;
      ctx->nfrags = 0;
      ctx->maxuse = 0;
      ctx->nmcalls= 0;
      ctx->nfcalls= 0;
      
      base.s.ptr = freep = prevp = &base;  /* Initialize linked list */
      base.s.size = 0;
      base.s.valid1 = VALID1;
      base.s.valid2 = VALID2;
    }
    
    ctx->nmcalls++;
    
    if (do_verify)
      kr_malloc_verify(ctx);
    
    /* Rather than divide make the alignment a known power of 2 */
    
    nunits = ((nbytes + sizeof(Header) - 1)>>LOG_ALIGN) + 1;
    
    for (p=prevp->s.ptr; ; prevp = p, p = p->s.ptr) {
      
      if (p->s.size >= nunits) {	/* Big enuf */
	if (p->s.size == nunits)	/* exact fit */
	  prevp->s.ptr = p->s.ptr;
	else {			/* allocate tail end */
	  p->s.size -= nunits;
	  p += p->s.size;
	  p->s.size = nunits;
	  p->s.valid1 = VALID1;
	  p->s.valid2 = VALID2;
	  ctx->nfrags++;  /* Have just increased the fragmentation */
	}
	
	/* Insert into linked list of blocks in use ... for debug only */
	p->s.ptr = usedp;
	usedp = p;
	
	ctx->inuse += nunits;  /* Record usage */
	if (ctx->inuse > ctx->maxuse)
	  ctx->maxuse = ctx->inuse;
	freep = prevp;
	return_ptr = (char *) (p+1);
	break;
      }
      
      if (p == freep)	{	/* wrapped around the free list */
	if ((p = morecore(nunits, ctx)) == (Header *) NULL) {
	  return_ptr = (char *) NULL;
	  break;
	}
      }
    }
    
    UNLOCKIT;

    return return_ptr;
    
}


void kr_free(char *ap, context_t *ctx) {
    Header *bp, *p, **up;
    
    LOCKIT;
    
    ctx->nfcalls++;
    
    
    if (do_verify)
      kr_malloc_verify(ctx);
    
    /* only do something if pointer is not NULL */
    
    if ( ap ) {
      
      bp = (Header *) ap - 1;  /* Point to block header */
      
      if (bp->s.valid1 != VALID1 || bp->s.valid2 != VALID2)
	kr_error("kr_free: pointer not from kr_malloc", 
		 (unsigned long) ap, ctx);
      
      ctx->inuse -= bp->s.size; /* Decrement memory ctx->usage */
      
      /* Extract the block from the used linked list
	 ... for debug only */
      
      for (up=&usedp; ; up = &((*up)->s.ptr)) {
	if (!*up)
	  kr_error("kr_free: block not found in used list\n", 
		   (unsigned long) ap, ctx);
	if (*up == bp) {
	  *up = bp->s.ptr;
	  break;
	}
      }
      
      /* Join the memory back into the free linked list */
      
      for (p=freep; !(bp > p && bp < p->s.ptr); p = p->s.ptr)
	if (p >= p->s.ptr && (bp > p || bp < p->s.ptr))
	  break; /* Freed block at start or end of arena */
      
      if (bp + bp->s.size == p->s.ptr) {/* join to upper neighbour */
	bp->s.size += p->s.ptr->s.size;
	bp->s.ptr = p->s.ptr->s.ptr;
	ctx->nfrags--;                 /* Lost a fragment */
      } else
	bp->s.ptr = p->s.ptr;
      
      if (p + p->s.size == bp) { /* Join to lower neighbour */
	p->s.size += bp->s.size;
	p->s.ptr = bp->s.ptr;
	ctx->nfrags--;          /* Lost a fragment */
      } else
	p->s.ptr = bp;
      
      freep = p;
      
    } /* end if on ap */
    
    UNLOCKIT;
}

/*
  Print to standard output the usage statistics.
*/
void kr_malloc_print_stats(context_t *ctx) {
    fflush(stderr);
    printf("\nkr_malloc statistics\n-------------------\n\n");
    
    printf("Total memory from system ... %ld bytes\n", 
	   (long)(ctx->total*ctx->usize));
    printf("Current memory usage ....... %ld bytes\n", 
	   (long)(ctx->inuse*ctx->usize));
    printf("Maximum memory usage ....... %ld bytes\n", 
	   (long)(ctx->maxuse*ctx->usize));
    printf("No. chunks from system ..... %ld\n", ctx->nchunk);
    printf("No. of fragments ........... %ld\n", ctx->nfrags);
    printf("No. of calls to kr_malloc ... %ld\n", ctx->nmcalls);
    printf("No. of calls to kr_free ..... %ld\n", ctx->nfcalls);
    printf("\n");
    
    fflush(stdout);
}

/*
  Currently assumes that are working in a single region.
*/
void kr_malloc_verify(context_t *ctx) {
    Header *p;
    
    LOCKIT;
    
    if ( freep ) {
      
      /* Check the used list */
      
      for (p=usedp; p; p=p->s.ptr) {
	if (p->s.valid1 != VALID1 || p->s.valid2 != VALID2)
	  kr_error("invalid header on usedlist", 
		   (unsigned long) p->s.valid1, ctx);
	
	if (p->s.size > ctx->total)
	  kr_error("invalid size in header on usedlist", 
		   (unsigned long) p->s.size, ctx);
      }
      
      /* Check the free list */
      
      p = base.s.ptr;
      while (p != &base) {
	if (p->s.valid1 != VALID1 || p->s.valid2 != VALID2)
	  kr_error("invalid header on freelist", 
		   (unsigned long) p->s.valid1, ctx);
	
	if (p->s.size > ctx->total)
	  kr_error("invalid size in header on freelist", 
		   (unsigned long) p->s.size, ctx);
	
	p = p->s.ptr;
      }
    } /* end if */
    
    UNLOCKIT;
}

/**
issues:
1. do usage statistics only if debug/DEBUG is enabled 
*/
