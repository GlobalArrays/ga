/* Configuration header file for GA 
 *
 * The following INTERNAL GA parameters can be customized:
 *   - max number of arrays
 *   - range of message passing tag/type values
 *   - length of message buffer(s) 
 *   - max number of processors
 *   - disabling MA use  
 *
 */

#ifndef _CONFIG_H
#define _CONFIG_H 


/* max number of global arrays */
#define MAX_ARRAYS  256             

/* max number of mutexes */
#define MAX_MUTEXES 32768

/* there are 20 message-passing tags/types numbered from GA_MSG_OFFSET up */
#define  GA_MSG_OFFSET 32000

/* length (in bytes) for send and receive buffers to handle remote requests */
/* this is obsolete in GA 3.0 and higher  left for OLD_GA builds */
#if defined(NX) || defined(SP1) || defined(SP)
#   ifdef IWAY
#      define MSG_BUF_SIZE    129000
#   else
#      define MSG_BUF_SIZE    122840
#   endif
#elif defined(LAPI)
#   define MSG_BUF_SIZE      131072
#elif defined(SYSV)
#   define MSG_BUF_SIZE      262152
#elif defined(CRAY)
#   define MSG_BUF_SIZE      1024
#else
#   define MSG_BUF_SIZE      4*4096
#endif

/* max number of dimensions  */
#define MAXDIM  7

/* uncomment the following line to overwrite default max number of processors */
/*#define MAX_NPROC 128*/


/* uncoment the following line to never use MA (Memory Allocator) for
 * storing data in global arrays (not temporary buffers!)  */
/*#define AVOID_MA_STORAGE */

#endif
