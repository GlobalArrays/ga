/*********************** "private" include file for DRA *****************/
 
#include "elio.h"
#include "macdecls.h"
#include <string.h>
#define MAXDIM 7

#ifdef FALSE
#undef FALSE
#endif
#ifdef TRUE
#undef TRUE
#endif
#ifdef CRAY_YMP
#include <fortran.h>
#define FALSE _btol(0)
#define TRUE  _btol(1)
#else
#define FALSE (logical) 0
#define TRUE  (logical) 1
#endif


/************************** common constants ***********************************/
#define DRA_OFFSET     5000                    /* DRA handle offset            */
#define DRA_BRD_TYPE  30000                    /* msg type for DRA broadcast   */
#define DRA_GOP_TYPE  30001                    /* msg type for DRA sum         */
#define DRA_MAX_NAME     72                    /* max length of array name     */
#define DRA_MAX_FNAME   248                    /* max length of metafile name  */


/************************* common data structures **************************/
typedef struct{                               /* stores basic DRA info */
        Integer ndim;                         /* dimension of array */
        Integer dims[MAXDIM];                 /* array dimensions */
        Integer chunk[MAXDIM];                /* data layout chunking */
        Integer layout;                       /* date layout type */
        int type;                         /* data type */
        int mode;                         /* file/array access permissions */
        char    name[DRA_MAX_NAME+8];         /* array name */
        char    fname[DRA_MAX_FNAME+8];       /* metafile name */
        Integer actv;                         /* is array active ? */ 
        Integer indep;                        /* shared/independent files ? */
        Fd_t      fd;                         /* ELIO meta-file descriptor */
        int algorithm;                        /* open file system algorithm */
        int nnodes;                           /* number of IO nodes */
} disk_array_t;

#define MAX_ALGN  1                /* max # aligned subsections   */ 
#define MAX_UNLG  (2*(MAXDIM-1))   /* max # unaligned subsections */

typedef struct{                   /* object describing DRA/GA section */
        Integer handle;
        Integer ndim;
        Integer lo[MAXDIM];
        Integer hi[MAXDIM];
}section_t;


typedef struct{                  /* structure stores arguments for callback f */
        int op;
        int transp;
        Integer ld[MAXDIM];
        section_t gs_a;
        section_t ds_a;
        section_t ds_chunk;
}args_t;


typedef struct{                   /* stores info associated with DRA request */
        Integer  d_a;             /* disk array handle */
        int num_pending;          /* number of pending  asynch. I/O ops */ 
        Integer list_algn[MAX_ALGN][2*MAXDIM]; /* coordinates of aligned subsection */
        Integer list_unlgn[MAX_UNLG][2*MAXDIM];/*coordinates of unaligned subsections*/
        Integer list_cover[MAX_UNLG][2*MAXDIM];/* coordinates of "cover" subsections */
        int        ibuf;
        int        nu;            
        int        na;
        int        callback;      /* callback status flag ON/OFF */
}request_t;

typedef struct{
  args_t args;
  Integer    req;
  io_request_t id;
  char *buffer;
} buffer_t;


extern disk_array_t *DRA;
extern logical dra_debug_flag;


/**************************** common macros ********************************/
#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))

#define dai_error ga_error

extern int dai_read_param(char* filename, Integer d_a);
extern void dai_write_param(char* filename, Integer d_a);
extern void dai_delete_param(char* filename, Integer d_a);
extern int dai_file_config(char* filename);
extern logical dai_section_intersect(section_t sref, section_t* sadj);
extern int  drai_get_num_serv(void);
