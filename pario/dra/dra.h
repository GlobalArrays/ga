/******************* header file for Disk Arrays *****************/

#include "file.modes.h"
#define  DRA_REQ_INVALID -333

#define DRA_create           dra_create
#define DRA_open             dra_open
#define DRA_inquire          dra_inquire

#define DRA_init             dra_init_
#define DRA_close            dra_close_
#define DRA_delete           dra_delete_
#define DRA_read             dra_read_
#define DRA_read_section     dra_read_section_
#define DRA_write            dra_write_
#define DRA_write_section    dra_write_section_
#define DRA_probe            dra_probe_
#define DRA_wait             dra_wait_
#define DRA_terminate        dra_terminate_
#define DRA_flick            dra_flick

#ifdef CRAY_T3D
#  define dra_create_        DRA_CREATE
#  define dra_open_          DRA_OPEN
#  define dra_inquire_       DRA_INQUIRE
#  define dra_init_          DRA_INIT
#  define dra_close_         DRA_CLOSE
#  define dra_delete_        DRA_DELETE
#  define dra_read_          DRA_READ
#  define dra_read_section_  DRA_READ_SECTION
#  define dra_write_         DRA_WRITE
#  define dra_write_section_ DRA_WRITE_SECTION
#  define dra_probe_         DRA_PROBE
#  define dra_wait_          DRA_WAIT
#  define dra_terminate_     DRA_TERMINATE
#  define dra_flick_         DRA_FLICK
#endif




#if defined(__STDC__) || defined(__cplusplus)
# define _ARGS_(s) s
#else
# define _ARGS_(s) ()
#endif

extern Integer DRA_init           _ARGS_((Integer *max_arrays,\
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
extern Integer DRA_open           _ARGS_((char *filename,\
                                         Integer *mode, 
                                         Integer *d_a )); 
extern Integer DRA_inquire        _ARGS_((Integer *d_a,\
                                         Integer *type,\
                                         Integer *dim1,\
                                         Integer *dim2,\
                                         char    *name,\
                                         char    *filename));  
extern Integer DRA_close          _ARGS_((Integer *d_a));
extern Integer DRA_delete         _ARGS_((Integer *d_a)); 
extern Integer DRA_write          _ARGS_((Integer *g_a,\
                                         Integer *d_a,\
                                         Integer *request));
extern Integer DRA_read           _ARGS_((Integer *g_a,\
                                         Integer *d_a,\
                                         Integer *request));
extern Integer DRA_write_section  _ARGS_((logical *transp, 
                                         Integer *g_a, 
                                         Integer *d_a,
                                         Integer *gilo,
                                         Integer *gihi,
                                         Integer *gjlo,
                                         Integer *gjhi,
                                         Integer *dilo,
                                         Integer *dihi,
                                         Integer *djlo,
                                         Integer *djhi,
                                         Integer *request));
extern Integer DRA_read_section   _ARGS_((logical *transp, 
                                         Integer *g_a,  
                                         Integer *d_a,
                                         Integer *gilo,
                                         Integer *gihi,
                                         Integer *gjlo,
                                         Integer *gjhi,
                                         Integer *dilo,
                                         Integer *dihi,
                                         Integer *djlo,
                                         Integer *djhi,
                                         Integer *request));
extern Integer DRA_probe          _ARGS_((Integer *request, Integer *status));
extern Integer DRA_wait           _ARGS_((Integer *request));
extern Integer DRA_terminate      _ARGS_(());
extern void    DRA_flick          _ARGS_(());

#undef _ARGS_
