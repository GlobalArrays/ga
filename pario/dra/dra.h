/******************* header file for Disk Arrays *****************/

/* used to be file.modes.h */
#include "chemio.h"
#define  DRA_RW ELIO_RW
#define  DRA_R  ELIO_R
#define  DRA_W  ELIO_W




#define  DRA_REQ_INVALID -333

#define DRA_create            dra_create
#define NDRA_create           ndra_create
#define DRA_open              dra_open
#define DRA_inquire           dra_inquire
#define NDRA_inquire          ndra_inquire
#define DRA_flick             dra_flick

#define DRA_init              dra_init_
#define DRA_close             dra_close_
#define DRA_delete            dra_delete_
#define DRA_read              dra_read_
#define NDRA_read             ndra_read_
#define DRA_read_section      dra_read_section_
#define NDRA_read_section     ndra_read_section_
#define DRA_write             dra_write_
#define NDRA_write            ndra_write_
#define DRA_write_section     dra_write_section_
#define NDRA_write_section    ndra_write_section_
#define DRA_probe             dra_probe_
#define DRA_set_debug         dra_set_debug_
#define DRA_wait              dra_wait_
#define DRA_terminate         dra_terminate_

#if defined(CRAY) || defined(WIN32)
#  define dra_create_         DRA_CREATE
#  define ndra_create_        NDRA_CREATE
#  define dra_open_           DRA_OPEN
#  define dra_inquire_        DRA_INQUIRE
#  define ndra_inquire_       NDRA_INQUIRE
#  define dra_init_           DRA_INIT
#  define dra_close_          DRA_CLOSE
#  define dra_delete_         DRA_DELETE
#  define dra_read_           DRA_READ
#  define ndra_read_          NDRA_READ
#  define dra_read_section_   DRA_READ_SECTION
#  define ndra_read_section_  NDRA_READ_SECTION
#  define dra_write_          DRA_WRITE
#  define ndra_write_         NDRA_WRITE
#  define dra_write_section_  DRA_WRITE_SECTION
#  define ndra_write_section_ NDRA_WRITE_SECTION
#  define dra_probe_          DRA_PROBE
#  define dra_set_debug_      DRA_SET_DEBUG
#  define dra_wait_           DRA_WAIT
#  define dra_terminate_      DRA_TERMINATE
#  define dra_flick_          DRA_FLICK
#endif




#if defined(__STDC__) || defined(__cplusplus) || defined(WIN32)
# define _ARGS_(s) s
#else
# define _ARGS_(s) ()
#endif
#ifdef __cplusplus
extern "C" {
#endif
extern Integer FATR DRA_init           _ARGS_((Integer *max_arrays,\
                                         DoublePrecision *max_array_size,\
                                         DoublePrecision *tot_disk_space,\
                                         DoublePrecision *max_memory)); 
extern Integer DRA_create         _ARGS_((Integer *type,\
                                         Integer *dim1,\
                                         Integer *dim2,\
                                         char    *name,\
                                         char    *filename,\
                                         Integer *mode,\
                                         Integer *block1,\
                                         Integer *block2,\
                                         Integer *d_a));
extern Integer NDRA_create         _ARGS_((Integer *type,\
                                         Integer *ndim,\
                                         Integer dims[],\
                                         char    *name,\
                                         char    *filename,\
                                         Integer *mode,\
                                         Integer reqdims[],\
                                         Integer *d_a));
extern Integer DRA_open           _ARGS_((char *filename,\
                                         Integer *mode, 
                                         Integer *d_a )); 
extern Integer DRA_inquire        _ARGS_((Integer *d_a,\
                                         Integer *type,\
                                         Integer *dim1,\
                                         Integer *dim2,\
                                         char    *name,\
                                         char    *filename));  
extern Integer NDRA_inquire       _ARGS_((Integer *d_a,\
                                         Integer *type,\
                                         Integer *ndim,\
                                         Integer dims[],\
                                         char    *name,\
                                         char    *filename));  
extern Integer FATR DRA_close          _ARGS_((Integer *d_a));
extern Integer FATR DRA_delete         _ARGS_((Integer *d_a)); 
extern Integer FATR DRA_write          _ARGS_((Integer *g_a,\
                                         Integer *d_a,\
                                         Integer *request));
extern Integer FATR NDRA_write         _ARGS_((Integer *g_a,\
                                         Integer *d_a,\
                                         Integer *request));
extern Integer FATR DRA_read           _ARGS_((Integer *g_a,\
                                         Integer *d_a,\
                                         Integer *request));
extern Integer FATR NDRA_read          _ARGS_((Integer *g_a,\
                                         Integer *d_a,\
                                         Integer *request));
extern Integer FATR DRA_write_section  _ARGS_((logical *transp, 
                                         Integer *g_a, 
                                         Integer *gilo,
                                         Integer *gihi,
                                         Integer *gjlo,
                                         Integer *gjhi,
                                         Integer *d_a,
                                         Integer *dilo,
                                         Integer *dihi,
                                         Integer *djlo,
                                         Integer *djhi,
                                         Integer *request));
extern Integer FATR NDRA_write_section _ARGS_((logical *transp, 
                                         Integer *g_a, 
                                         Integer glo[],
                                         Integer ghi[],
                                         Integer *d_a,
                                         Integer dlo[],
                                         Integer dhi[],
                                         Integer *request));
extern Integer FATR DRA_read_section   _ARGS_((logical *transp, 
                                         Integer *g_a,  
                                         Integer *gilo,
                                         Integer *gihi,
                                         Integer *gjlo,
                                         Integer *gjhi,
                                         Integer *d_a,
                                         Integer *dilo,
                                         Integer *dihi,
                                         Integer *djlo,
                                         Integer *djhi,
                                         Integer *request));
extern Integer FATR NDRA_read_section  _ARGS_((logical *transp, 
                                         Integer *g_a,  
                                         Integer glo[],
                                         Integer ghi[],
                                         Integer *d_a,
                                         Integer dlo[],
                                         Integer dhi[],
                                         Integer *request));
extern Integer FATR DRA_probe          _ARGS_((Integer *request, Integer *status));
extern void FATR DRA_set_debug         _ARGS_((logical *flag));
extern Integer FATR DRA_wait           _ARGS_((Integer *request));
extern Integer FATR DRA_terminate      _ARGS_(());
extern void    DRA_flick               _ARGS_(());

/* C-interface prototypes */

extern int NDRA_Create(       int type,
                              int ndim,
                              int dims[],
                              char *name,
                              char* filename,
                              int mode,
                              int reqdims[],
                              int *d_a);

extern int NDRA_Inquire(      int d_a,
                              int *type,
                              int *ndim,
                              int dims[],
                              char *name,
                              char* filename);

extern int NDRA_Write(        int g_a,
                              int d_a,
                              int *request);

extern int NDRA_Read(         int g_a,
                              int d_a,
                              int *request);

extern int NDRA_Write_section(logical transp,
                              int g_a,
                              int glo[],
                              int ghi[],
                              int d_a,
                              int dlo[],
                              int dhi[],
                              int *request);

extern int NDRA_Read_section( logical transp,
                              int g_a,
                              int glo[],
                              int ghi[],
                              int d_a,
                              int dlo[],
                              int dhi[],
                              int *request);

extern int DRA_Init(          int max_arrays,
                              double max_array_size,
                              double total_disk_space,
                              double max_memory);

extern int DRA_Terminate();

extern int DRA_Open(          char* filename,
                              int mode,
                              int *d_a);

extern int DRA_Probe(         int request,
                              int *compl_status);

extern void DRA_Set_debug(    logical flag);

extern int DRA_Wait(          int request);

extern int DRA_Delete(        int d_a);

extern int DRA_Close(         int d_a);

extern void DRA_Flick();

#ifdef __cplusplus
	   }
#endif

#undef _ARGS_
