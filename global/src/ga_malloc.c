
/** GA Memory Allocation Routines: uses either MA or external allocator */

#include "globalp.h"
#define GA_MAXMEM_AVAIL 786432  
#define CHECK 0

static void * (*ga_ext_alloc)();
static void (*ga_ext_free)();
short int ga_usesMA = 1; 

void GA_Register_stack_memory(void * (*ext_alloc)(), void (*ext_free)()) {
    if(ext_alloc == NULL || ext_free == NULL)
      ga_error("GA_Register_stack_memory():Invalid pointer(s) passed\n", 0);
    ga_ext_alloc = ext_alloc; ga_ext_free  = ext_free; ga_usesMA=0;
}

void* FATR ga_malloc(Integer nelem, int type, char *name) {
    void *ptr;  
    unsigned long addr;
    Integer handle, adjust=0, bytes, item_size=GAsizeofM(ga_type_f2c(type));

    nelem += sizeof(double)/item_size+1; /* extra space for storing handle */
    
    if(ga_usesMA) { /* Uses Memory Allocator (MA) */
       if(MA_push_stack(type,nelem,name,&handle)) MA_get_pointer(handle,&ptr);
       else ga_error("ga_malloc: MA_push_stack failed",0);
    }
    else { /* else, using external memory allocator */
       bytes = nelem*item_size + item_size; /* extra-memory for alignment */
       addr  = (unsigned long)(*ga_ext_alloc)((size_t)bytes, item_size, name);
       /* Address Alignment: align the memory to that boundary */
       adjust = (Integer) (addr%item_size);
       if(adjust != 0) { adjust=item_size-adjust; addr+=adjust; }
       ptr = (void *)addr; handle = adjust;
    }

    if(ptr == NULL) ga_error("ga_malloc failed", 0L);
    *((Integer*)ptr)= handle;/* store handle/adjustment-value in this buffer */
    ptr = ((double*)ptr)+ 1;  /*needs sizeof(double)>=sizeof(Integer) */
    
    return ptr;
}

void FATR ga_free(void *ptr) {
    Integer handle= *( (Integer*) (-1 + (double*)ptr)); /* retreive handle */
    if(ga_usesMA) {
      if(!MA_pop_stack(handle)) ga_error("ga_free: MA_pop_stack failed",0);}
    else {/*make sure to free original(before address alignment) pointer*/
      ptr = (-1 + (double*)ptr); (*ga_ext_free)((char *)ptr-handle); }
}

Integer ga_memory_avail(Integer datatype) {
    if(ga_usesMA)  return MA_inquire_avail(datatype);
    else return GA_MAXMEM_AVAIL;
}
