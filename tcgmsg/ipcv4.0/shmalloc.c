
/*
  Memory management routines from ANSI K&R C, modified to manage
  a single block of shared memory.
  Have stripped out all the usage monitoring to keep it simple.

  The usage model is as follows:

  1) A single process initializes shared memory management with

     InitSharedMalloc(char *memory, unsigned nbytes)

     memory ... points to a region of shared memory size nbytes

     InitSharedMalloc() internally creates a semaphore to
     co-ordinate access to its data structures.

  2) The process then forks its children and can use SharedMalloc()
     and SharedFree() as one would use malloc/free.
*/
  
#define LOG_ALIGN 4
#define ALIGNMENT (1 << LOG_ALIGN)

/* ALIGNMENT is assumed below to be bigger than sizeof(Header *), 
   so do not reduce LOG_ALIGN below 4 */

union header{
  struct {
    union header *ptr;  /* next block if on free list */
    unsigned size;      /* size of this block*/
  } s;
  char align[ALIGNMENT]; /* Align to ALIGNMENT byte boundary */
};

typedef union header Header;

static Header **freep;         /* pointer to pointer to start of free list */
static int semid=-1, semnum=0; /* Id and number of semaphore */

void InitSharedMalloc(memory, nbytes)
     char *memory;
     unsigned nbytes;
/*
  memory points to a region of shared memory nbytes long.
  initialize the data structures needed to manage this memory
*/
{
  int nunits = nbytes >> LOG_ALIGN;
  Header *region = (Header *) memory;

  /* Quick check that things are OK */

  if (ALIGNMENT != sizeof(Header) || ALIGNMENT < sizeof(Header *))
    p4_error("InitSharedMalloc: Alignment is wrong", ALIGNMENT);

  if (!region)
    p4_error("InitSharedMalloc: Passed null pointer",0);

  if (nunits < 2)
    p4_error("InitSharedMalloc: Initial region is ridiculously small", 
	     (int) nbytes);

  /* Shared memory region is structured as follows

     1) (Header *) freep ... free list pointer
     2) padding up to alignment boundary
     3) First header of free list */

  freep = (Header **) region;       /* Free space pointer in first block  */
  (region+1)->s.ptr = *freep = region+1;   /* Data in rest */
  (region+1)->s.size = nunits-1;           /* One header consumed already */

  semid = SemSetCreate(1,0);  /* Make semaphore for access */
}

char *SharedMalloc(nbytes)
     unsigned nbytes;
{
  Header *p, *prevp;
  char *address = (char *) NULL;
  unsigned nunits;
  
  /* Force entire routine to be single threaded */
  SemWait(semid, semnum);
  
  nunits = ((nbytes + sizeof(Header) - 1)>>LOG_ALIGN) + 1;
  
  prevp = *freep;
  for (p=prevp->s.ptr; ; prevp = p, p = p->s.ptr) {
    if (p->s.size >= nunits) {	/* Big enuf */
      if (p->s.size == nunits)	/* exact fit */
        prevp->s.ptr = p->s.ptr;
      else {			/* allocate tail end */
	p->s.size -= nunits;
	p += p->s.size;
	p->s.size = nunits;
      }
      *freep = prevp;
      address = (char *) (p+1);
      break;
    }
    if (p == *freep) {  /* wrapped around the free list ... no fit found */
      address = (char *) NULL;
      break;
    }
  }
  
  /* End critical region */
  SemPost(semid, semnum);
  
  return address;
}

void SharedFree(ap)
     char *ap;
{
  Header *bp, *p;
  
  /* Begin critical region */
  SemWait(semid, semnum);
  
  if (!ap)
    return;  /* Do nothing with NULL pointers */
  
  bp = (Header *) ap - 1;  /* Point to block header */
  
  for (p = *freep; !(bp > p && bp < p->s.ptr); p = p->s.ptr)
    if (p >= p->s.ptr && (bp > p || bp < p->s.ptr))
      break; /* Freed block at start of end of arena */
  
  if (bp + bp->s.size == p->s.ptr) {/* join to upper neighbour */
    bp->s.size += p->s.ptr->s.size;
    bp->s.ptr = p->s.ptr->s.ptr;
  } else
    bp->s.ptr = p->s.ptr;
  
  if (p + p->s.size == bp) { /* Join to lower neighbour */
    p->s.size += bp->s.size;
    p->s.ptr = bp->s.ptr;
  } else
    p->s.ptr = bp;
  
  *freep = p;
  
  /* End critical region */
  SemPost(semid, semnum);
}
