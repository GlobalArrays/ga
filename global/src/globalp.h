
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
#define     GA_TYPE_DGT 32760 - 14
#define     GA_TYPE_GOP 32760 - 29
#define     GA_TYPE_BRD 32760 - 30

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


#ifdef GA_TRACE
  static Integer     op_code;
#endif


#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))
#define ABS(a) (((a) >= 0) ? (a) : (-(a)))

#if !defined(NX) && defined(__STDC__) || defined(__cplusplus)
# define ARGS_(s) s
#else
# define ARGS_(s) ()
#endif

extern void f2cstring    ARGS_((char*, Integer, char*, Integer));
extern void c2fstring    ARGS_(( char*, char*, Integer));

#undef ARGS_
