/*********************** "private" include file for DRA *****************/
 
#include "elio.h"
#include <string.h>


/************************** common constants *******************************/
#define DRA_OFFSET     5000                    /* DRA handle offset          */
#define DRA_BRD_TYPE  30000                    /* msg type for DRA broadcast */
#define DRA_MAX_NAME     80                      /* max length of array name  */
#define DRA_MAX_FNAME   200                    /* max length of array name  */


/************************* common data structures **************************/
typedef struct{                               /* stores basic DRA info */
        Integer dim1, dim2;
        Integer chunk1, chunk2;
        Integer type;
        Integer layout;
        Integer actv;
        Integer mode;
        Fd_t      fd;
        char    name[DRA_MAX_NAME+8];
        char    fname[DRA_MAX_FNAME+8];
} disk_array_t;

#define MAX_ALGN  1                /* max # aligned subsections   */ 
#define MAX_UNLG  2               /* max # unaligned subsections */

typedef struct{                   /* object describing DRA/GA section */
        Integer handle;
        Integer ilo;
        Integer ihi;
        Integer jlo;
        Integer jhi;
}section_t;


typedef struct{                  /* structure stores arguments for callback f */
        int op;
        int transp;
        Integer ld;
        section_t gs_a;
        section_t ds_a;
        section_t ds_chunk;
}args_t;


typedef struct{                   /* stores info associated with DRA request */
        Integer  d_a;             /* disk array handle */
        io_request_t  id;         /* low level asynch. I/O  op. id */
        int num_pending;          /* number of pending  asynch. I/O ops */ 
        Integer list_algn[MAX_ALGN][4]; /* coordinates of aligned subsection */
        Integer list_unlgn[MAX_UNLG][4];/*coordinates of unaligned subsections*/
        Integer list_cover[MAX_UNLG][4];/* coordinates of "cover" subsections */
        int        nu;            
        int        na;
        int        callback;      /* callback status flag ON/OFF */
        args_t     args;          /* arguments to callback function */
}request_t;


extern disk_array_t *DRA;


/**************************** common macros ********************************/
#define MAX(a,b) (((a) >= (b)) ? (a) : (b))
#define MIN(a,b) (((a) <= (b)) ? (a) : (b))

#define dai_error ga_error

extern void dai_read_param(char* filename, Integer d_a);
extern void dai_write_param(char* filename, Integer d_a);
extern void dai_delete_param(char* filename, Integer d_a);
extern logical dai_section_intersect(section_t sref, section_t* sadj);
extern Integer MA_push_get (Integer, Integer, char*, Integer*, Integer*);
extern Integer MA_pop_stack (Integer);

