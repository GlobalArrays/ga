#ifndef _ACC_H_
#define _ACC_H_

#if defined(AIX)
#    define I_ACCUMULATE_2D	i_accumulate_2d_u
#    define D_ACCUMULATE_2D	d_accumulate_2d_u
#    define C_ACCUMULATE_2D	c_accumulate_2d_u
#    define Z_ACCUMULATE_2D	z_accumulate_2d_u
#    define F_ACCUMULATE_2D	f_accumulate_2d_u
#elif !defined(CRAY) && !defined(WIN32)
#    define I_ACCUMULATE_2D	i_accumulate_2d_u_
#    define D_ACCUMULATE_2D	d_accumulate_2d_u_
#    define C_ACCUMULATE_2D	c_accumulate_2d_u_
#    define Z_ACCUMULATE_2D	z_accumulate_2d_u_
#    define F_ACCUMULATE_2D	f_accumulate_2d_u_
#endif

void I_ACCUMULATE_2D(); 
void D_ACCUMULATE_2D(); 
void C_ACCUMULATE_2D(); 
void Z_ACCUMULATE_2D(); 
void F_ACCUMULATE_2D();

extern void armci_acc_2D(int op, void* scale, int proc, void *src_ptr, void *dst_ptr, int bytes, 
		  int cols, int src_stride, int dst_stride); 

#endif
