#ifndef _f2c_h
#define _f2c_h

#endif /* _f2c_h */

	integer f2c_alloc_get
	integer f2c_allocate_heap
	integer f2c_chop_stack
	integer f2c_free_heap
	integer f2c_get_index
	integer f2c_get_next_memhandle
	integer f2c_inform_base
#ifdef _CRAY
	integer f2c_inform_base_fcd
#endif /* _CRAY */
	integer f2c_init
	integer f2c_init_memhandle_iterator
	integer f2c_inquire_avail
	integer f2c_inquire_heap
	integer f2c_inquire_stack
	integer f2c_pop_stack

	integer f2c_push_get
	integer f2c_push_stack
	integer f2c_set_auto_verify
	integer f2c_set_error_print
	integer f2c_set_hard_fail
	integer f2c_sizeof
	integer f2c_sizeof_overhead

	integer f2c_verify_allocator_stuff

	external f2c_alloc_get
	external f2c_allocate_heap
	external f2c_chop_stack
	external f2c_free_heap
	external f2c_get_index
	external f2c_get_next_memhandle
	external f2c_inform_base
#ifdef _CRAY
	external f2c_inform_base_fcd
#endif /* _CRAY */
	external f2c_init
	external f2c_init_memhandle_iterator
	external f2c_inquire_avail
	external f2c_inquire_heap
	external f2c_inquire_stack
	external f2c_pop_stack
	external f2c_print_stats
	external f2c_push_get
	external f2c_push_stack
	external f2c_set_auto_verify
	external f2c_set_error_print
	external f2c_set_hard_fail
	external f2c_sizeof
	external f2c_sizeof_overhead
	external f2c_summarize_allocated_blocks
	external f2c_verify_allocator_stuff
