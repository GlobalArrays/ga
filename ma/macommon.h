#ifndef _macommon_h
#define _macommon_h

#define MA_FALSE	0
#define MA_TRUE		1

#define MA_DEFAULT_SPACE	(-1)

#define MA_NAMESIZE	32

#define MT_BASE		1000

#define MT_C_CHAR	(MT_BASE + 0)
#define MT_C_INT	(MT_BASE + 1)
#define MT_C_LONGINT	(MT_BASE + 2)
#define MT_C_FLOAT	(MT_BASE + 3)
#define MT_C_DBL	(MT_BASE + 4)
#define MT_C_LDBL	(MT_BASE + 5)
#define MT_C_SCPL	(MT_BASE + 6)
#define MT_C_DCPL	(MT_BASE + 7)
#define MT_C_LDCPL	(MT_BASE + 8)

#define MT_F_BYTE	(MT_BASE + 9)
#define MT_F_INT	(MT_BASE + 10)
#define MT_F_LOG	(MT_BASE + 11)
#define MT_F_REAL	(MT_BASE + 12)
#define MT_F_DBL	(MT_BASE + 13)
#define MT_F_SCPL	(MT_BASE + 14)
#define MT_F_DCPL	(MT_BASE + 15)

#define MT_FIRST	MT_C_CHAR
#define MT_LAST		MT_F_DCPL
#define MT_NUMTYPES	(MT_LAST - MT_FIRST + 1)

#if defined(_CRAY) || defined(WIN32) || defined(HITACHI)
#define f2c_alloc_get_				F2C_ALLOC_GET
#define f2c_allocate_heap_			F2C_ALLOCATE_HEAP
#define f2c_chop_stack_				F2C_CHOP_STACK
#define f2c_free_heap_				F2C_FREE_HEAP
#define f2c_free_heap_piece_			F2C_FREE_HEAP_PIECE
#define f2c_get_index_				F2C_GET_INDEX
#define f2c_get_next_memhandle_			F2C_GET_NEXT_MEMHANDLE
#define f2c_get_numalign_                       F2C_GET_NUMALIGN
#define f2c_inform_base_			F2C_INFORM_BASE
#define f2c_inform_base_fcd_			F2C_INFORM_BASE_FCD
#define f2c_init_				F2C_INIT
#define f2c_init_memhandle_iterator_		F2C_INIT_MEMHANDLE_ITERATOR
#define f2c_initialized_			F2C_INITIALIZED
#define f2c_inquire_avail_			F2C_INQUIRE_AVAIL
#define f2c_inquire_heap_			F2C_INQUIRE_HEAP
#define f2c_inquire_stack_			F2C_INQUIRE_STACK
#define f2c_pop_stack_				F2C_POP_STACK
#define f2c_print_stats_			F2C_PRINT_STATS
#define f2c_push_get_				F2C_PUSH_GET
#define f2c_push_stack_				F2C_PUSH_STACK
#define f2c_set_auto_verify_			F2C_SET_AUTO_VERIFY
#define f2c_set_error_print_			F2C_SET_ERROR_PRINT
#define f2c_set_hard_fail_			F2C_SET_HARD_FAIL
#define f2c_set_numalign_                       F2C_SET_NUMALIGN
#define f2c_sizeof_				F2C_SIZEOF
#define f2c_sizeof_overhead_			F2C_SIZEOF_OVERHEAD
#define f2c_summarize_allocated_blocks_		F2C_SUMMARIZE_ALLOCATED_BLOCKS
#define f2c_trace_                              F2C_TRACE
#define f2c_verify_allocator_stuff_		F2C_VERIFY_ALLOCATOR_STUFF
#define ma_set_sizes_				MA_SET_SIZES



#elif defined(F2C2_)

#define f2c_alloc_get_                  f2c_alloc_get__
#define f2c_allocate_heap_              f2c_allocate_heap__
#define f2c_chop_stack_                 f2c_chop_stack__
#define f2c_free_heap_                  f2c_free_heap__
#define f2c_free_heap_piece_            f2c_free_heap_piece__
#define f2c_get_index_                  f2c_get_index__
#define f2c_get_next_memhandle_         f2c_get_next_memhandle__
#define f2c_get_numalign_               f2c_get_numalign__
#define f2c_inform_base_                f2c_inform_base__
#define f2c_inform_base_fcd_            f2c_inform_base_fcd__
#define f2c_init_                       f2c_init__
#define f2c_init_memhandle_iterator_    f2c_init_memhandle_iterator__
#define f2c_initialized_                f2c_initialized__
#define f2c_inquire_avail_              f2c_inquire_avail__
#define f2c_inquire_heap_               f2c_inquire_heap__
#define f2c_inquire_stack_              f2c_inquire_stack__
#define f2c_pop_stack_                  f2c_pop_stack__
#define f2c_print_stats_                f2c_print_stats__
#define f2c_push_get_                   f2c_push_get__
#define f2c_push_stack_                 f2c_push_stack__
#define f2c_set_auto_verify_            f2c_set_auto_verify__
#define f2c_set_error_print_            f2c_set_error_print__
#define f2c_set_hard_fail_              f2c_set_hard_fail__
#define f2c_set_numalign_               f2c_set_numalign__
#define f2c_sizeof_                     f2c_sizeof__
#define f2c_sizeof_overhead_            f2c_sizeof_overhead__
#define f2c_summarize_allocated_blocks_ f2c_summarize_allocated_blocks__
#define f2c_trace_                      f2c_trace__
#define f2c_verify_allocator_stuff_     f2c_verify_allocator_stuff__
#define ma_set_sizes_                   ma_set_sizes__

#endif


#endif
