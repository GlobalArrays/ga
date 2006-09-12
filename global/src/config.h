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
#define MAX_ARRAYS 32768            

/* max number of mutexes */
#define MAX_MUTEXES 32768

/* there are 20 message-passing tags/types numbered from GA_MSG_OFFSET up */
#define  GA_MSG_OFFSET 32000

/* max number of dimensions
 * Now set in global.h and global.fh so users can access the value.
 * We set the macro used internally from the global.h value.
 */
#define MAXDIM  GA_MAX_DIM

/* uncomment the following line to overwrite default max number of processors */
/*#define MAX_NPROC 128*/


/* uncoment the following line to never use MA (Memory Allocator) for
 * storing data in global arrays (not temporary buffers!)  */
#define AVOID_MA_STORAGE
 
#endif
