#define MAX_ARRAYS  40             /* max number of global arrays */

#ifndef MAX_NPROC                  /* default max number of processors  */
#   ifdef PARAGON
#     define MAX_NPROC    1024
#   elif defined(DELTA)
#     define MAX_NPROC     512
#   elif defined(SP1)
#     define MAX_NPROC     400
#   elif defined(CRAY_T3D)
#     define MAX_NPROC     256
#   elif defined(KSR)
#     define MAX_NPROC      80
#   else
#     define MAX_NPROC      128     /* default for everything else */
#   endif
#endif

/* types/tags of messages used internally by GA */
#define     GA_MSG_OFFSET 32000
#define     GA_TYPE_REQ   GA_MSG_OFFSET + 1
#define     GA_TYPE_GET   GA_MSG_OFFSET + 2
#define     GA_TYPE_PUT   GA_MSG_OFFSET + 3
#define     GA_TYPE_ACC   GA_MSG_OFFSET + 4
#define     GA_TYPE_GSM   GA_MSG_OFFSET + 5
#define     GA_TYPE_ACK   GA_MSG_OFFSET + 6
#define     GA_TYPE_ADD   GA_MSG_OFFSET + 7
#define     GA_TYPE_DCV   GA_MSG_OFFSET + 8
#define     GA_TYPE_DCI   GA_MSG_OFFSET + 9
#define     GA_TYPE_DCJ   GA_MSG_OFFSET + 10
#define     GA_TYPE_DSC   GA_MSG_OFFSET + 11
#define     GA_TYPE_RDI   GA_MSG_OFFSET + 12
#define     GA_TYPE_DGT   GA_MSG_OFFSET + 13
#define     GA_TYPE_SYN   GA_MSG_OFFSET + 14
#define     GA_TYPE_GOP   GA_MSG_OFFSET + 15
#define     GA_TYPE_BRD   GA_MSG_OFFSET + 16
#define     GA_TYPE_MAS   GA_MSG_OFFSET + 20

/* GA operation ids */
#define     GA_OP_GET 1          /* Get                         */
#define     GA_OP_END 2          /* Terminate                   */
#define     GA_OP_CRE 3          /* Create                      */
#define     GA_OP_PUT 4          /* Put                         */
#define     GA_OP_ACC 5          /* Accumulate                  */
#define     GA_OP_DES 6          /* Destroy                     */
#define     GA_OP_DUP 7          /* Duplicate                   */
#define     GA_OP_ZER 8          /* Zero                        */
#define     GA_OP_DDT 9          /* Double precision dot product*/
#define     GA_OP_DST 10         /* Double precision scatter    */
#define     GA_OP_DGT 11         /* Double precision gather     */
#define     GA_OP_DSC 12         /* Double precision scale      */
#define     GA_OP_COP 13         /* Copy                        */
#define     GA_OP_ADD 14         /* Double precision add        */
#define     GA_OP_RDI 15         /* Integer read and increment  */
#define     GA_OP_ACK 16         /* acknowledgment              */


#ifdef GA_TRACE
  static Integer     op_code;
#endif


#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a)   (((a) >= 0) ? (a) : (-(a)))

#define GAsizeofM(type)  ( (type)==MT_F_DBL? sizeof(DoublePrecision): \
                           (type)==MT_F_INT? sizeof(Integer): 0)

#define NAME_STACK_LEN 10
#define PAGE_SIZE  4096

extern char *GA_name_stack[NAME_STACK_LEN];    /* stack for names of GA ops */ 
extern int  GA_stack_size;
#define  GA_PUSH_NAME(name) (GA_name_stack[GA_stack_size++] = (name)) 
#define  GA_POP_NAME        (GA_stack_size--)

#if defined(__STDC__) || defined(__cplusplus)
# define ARGS_(s) s
#else
# define ARGS_(s) ()
#endif

extern void f2cstring    ARGS_((char*, Integer, char*, Integer));
extern void c2fstring    ARGS_(( char*, char*, Integer));
extern void ga_clean_resources ARGS_(( void));


#undef ARGS_
