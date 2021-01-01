//#define BLIS_PARAM_MACRO_DEFS_H
//#define BLIS_OBJ_MACRO_DEFS_H
//#define BLIS_MISC_MACRO_DEFS_H
//#define BLIS_SCAL2RIS_MXN_H
//#define BLIS_SCALRIS_MXN_UPLO_H
//#define BLIS_ABSQR2_H
//#define BLIS_ABVAL2S_H
//#define BLIS_ADDS_H
//#define BLIS_ADDJS_H
//#define BLIS_ADD3S_H
//#define BLIS_AXPBYS_H
//#define BLIS_AXPBYJS_H
//#define BLIS_AXPYS_H
//#define BLIS_AXPYJS_H
//#define BLIS_AXMYS_H
//#define BLIS_CONJS_H
//#define BLIS_COPYS_H
//#define BLIS_COPYJS_H
//#define BLIS_COPYCJS_H
//#define BLIS_COPYNZS_H
//#define BLIS_COPYJNZS_H
//#define BLIS_ADDS_MXN_H
#include <blis/blis.h>

//blis_version.c:(.text+0x304): undefined reference to `round'
void* bli_thrcomm_bcast( dim_t inside_id, void* to_send, thrcomm_t* comm ) { }
void  bli_thrcomm_barrier( dim_t thread_id, thrcomm_t* comm ){ }
void bli_thread_range_sub
     (
       thrinfo_t* thread,
       dim_t      n,
       dim_t      bf,
       bool       handle_edge_low,
       dim_t*     start,
       dim_t*     end
     ){ }

BLIS_EXPORT_BLIS int bli_pthread_mutex_init
     (
       bli_pthread_mutex_t*           mutex,
       const bli_pthread_mutexattr_t* attr
     ){ }

BLIS_EXPORT_BLIS int bli_pthread_mutex_destroy
     (
       bli_pthread_mutex_t* mutex
     ){ }

BLIS_EXPORT_BLIS int bli_pthread_mutex_lock
     (
       bli_pthread_mutex_t* mutex
     ){ }

BLIS_EXPORT_BLIS int bli_pthread_mutex_trylock
     (
       bli_pthread_mutex_t* mutex
     ){ }

BLIS_EXPORT_BLIS int bli_pthread_mutex_unlock
     (
       bli_pthread_mutex_t* mutex
     ) { }

void bli_abort(){ }
void bli_obj_scalar_detach
     (
       obj_t* a,
       obj_t* alpha
     ){ }
bool bli_obj_imag_is_zero( obj_t* a ){ }
double round( double x) {}

int main() {
  int blis_int_size = BLIS_BLAS_INT_TYPE_SIZE;
  if( blis_int_size == 32 ) return 0;
  else                      return 1;
}
