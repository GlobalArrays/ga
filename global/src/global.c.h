/* file globalp.c.h */

#include "types.f2c.h"


#ifdef FALSE
#undef FALSE
#endif
#ifdef TRUE 
#undef TRUE
#endif
#define FALSE (logical) 0
#define TRUE  (logical) 1

void     ga_symmetrize_(), ga_print_(),        ga_distribution_(),
         ga_dgop_(),       ga_check_handle(),  ga_sync_(),
         ga_copy_(),       ga_inquire(),       ga_release_(),
         ga_access_(),     ga_brdcst_(),       ga_distribution_(),
         ga_put_(),        ga_get_(),          ga_zero_(),
         ga_dscatter_(),   ga_gather_(),       ga_acc_(),   
         ga_dgemm_(),      ga_diag_(),         ga_initialize_(),
         ga_diag_reuse_(), ga_inquire_name(),  ga_release_update_(),   
         ga_copy_patch_(), ga_print_patch_(),  ga_matmul_patch_(),
	 ga_dadd_patch_(), ga_dscal_patch_(),  ga_dfill_patch_(), 
	 ga_ifill_patch_(),ga_summarize_();

Integer  ga_nnodes_(),     ga_nodeid_(),       ga_read_inc_(),
         ga_verify_handle_();

logical  ga_create_(),     ga_create_irreg_(), ga_destroy_() ; 

DoublePrecision ga_ddot_(), ga_ddot_patch_();

void     ga_error();

#define     GA_TYPE_REQ 32760 - 1
#define     GA_TYPE_GET 32760 - 2
#define     GA_TYPE_SYN 32760 - 3
#define     GA_TYPE_PUT 32760 - 4
#define     GA_TYPE_ACC 32760 - 5
#define     GA_TYPE_GSM 32760 - 6
#define     GA_TYPE_ACK 32760 - 7
#define     GA_TYPE_ADD 32760 - 8
#define     GA_TYPE_DCV 32760 - 9
#define     GA_TYPE_DCI 32760 - 10
#define     GA_TYPE_DCJ 32760 - 11
#define     GA_TYPE_DSC 32760 - 12
#define     GA_TYPE_RDI 32760 - 13
#define     GA_TYPE_GOP 32760 - 29
#define     GA_TYPE_BRD 32760 - 30

#define     GA_OP_GET 1          /* Get				*/
#define     GA_OP_END 2          /* Terminate			*/
#define     GA_OP_CRE 3          /* Create			*/
#define     GA_OP_PUT 4          /* Put				*/
#define     GA_OP_ACC 5          /* Accumulate			*/
#define     GA_OP_DES 6          /* Destroy			*/
#define     GA_OP_ZER 7          /* Zero			*/
#define     GA_OP_DDT 8          /* Double precision dot product*/
#define     GA_OP_DST 10         /* Double precision scatter	*/
#define     GA_OP_DGT 11         /* Double precision gather	*/
#define     GA_OP_DSC 12         /* Double precision scale	*/
#define     GA_OP_COP 13         /* Copy			*/
#define     GA_OP_ADD 14         /* Double precision add	*/
#define     GA_OP_RDI 15         /* Integer read and increment	*/


#ifdef GA_TRACE
static Integer     op_code;
#endif
