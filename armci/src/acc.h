#ifndef _ACC_H_
#define _ACC_H_

/* on 64-bit platforms use integer*8 version */
#if defined(SGI64) || defined(SGITFP) || defined(DECOSF)
#   define  L_ACCUMULATE_2D I8_ACCUMULATE_2D
#else
#   define  L_ACCUMULATE_2D I_ACCUMULATE_2D
#endif

#if defined(AIX)
#    define I_ACCUMULATE_2D	i_accumulate_2d_u
#    define I8_ACCUMULATE_2D    i8_accumulate_2d_u
#    define D_ACCUMULATE_2D	d_accumulate_2d_u
#    define C_ACCUMULATE_2D	c_accumulate_2d_u
#    define Z_ACCUMULATE_2D	z_accumulate_2d_u
#    define F_ACCUMULATE_2D	f_accumulate_2d_u
#elif !defined(CRAY) && !defined(WIN32)
#    define I_ACCUMULATE_2D     i_accumulate_2d_u_
#    define I8_ACCUMULATE_2D    i8_accumulate_2d_u_
#    define D_ACCUMULATE_2D     d_accumulate_2d_u_
#    define C_ACCUMULATE_2D     c_accumulate_2d_u_
#    define Z_ACCUMULATE_2D     z_accumulate_2d_u_
#    define F_ACCUMULATE_2D     f_accumulate_2d_u_
#endif

#ifdef CRAY
#define  D_ACCUMULATE_2D daxpy_2d_
#endif

#ifdef FUJITSU
#undef D_ACCUMULATE_2D
#define D_ACCUMULATE_2D daxpy_2d_
#endif

void FATR I_ACCUMULATE_2D(void*, int*, int*, void*, int*, void*, int*); 
void FATR L_ACCUMULATE_2D(void*, int*, int*, void*, int*, void*, int*); 
void FATR D_ACCUMULATE_2D(void*, int*, int*, void*, int*, void*, int*); 
void FATR C_ACCUMULATE_2D(void*, int*, int*, void*, int*, void*, int*); 
void FATR Z_ACCUMULATE_2D(void*, int*, int*, void*, int*, void*, int*); 
void FATR F_ACCUMULATE_2D(void*, int*, int*, void*, int*, void*, int*);

extern void armci_acc_2D(int op, void* scale, int proc, void *src_ptr, void *dst_ptr, int bytes, 
		  int cols, int src_stride, int dst_stride, int lockit); 

#endif
