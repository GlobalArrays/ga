/******************* header file for Disk Arrays *****************/

#ifndef _DRA_H_
#define _DRA_H_
/* used to be file.modes.h */
#include "chemio.h"
typedef long dra_size_t;
#define  DRA_RW ELIO_RW
#define  DRA_R  ELIO_R
#define  DRA_W  ELIO_W


#if defined(CRAY) && defined(__crayx1)
#undef CRAY
#endif


#define  DRA_REQ_INVALID -333

#if defined(CRAY) || defined(WIN32)
#  define dra_create_             DRA_CREATE
#  define ndra_create_            NDRA_CREATE
#  define ndra_create_config_     NDRA_CREATE_CONFIG
#  define dra_open_               DRA_OPEN
#  define dra_inquire_            DRA_INQUIRE
#  define ndra_inquire_           NDRA_INQUIRE
#  define dra_init_               DRA_INIT
#  define dra_close_              DRA_CLOSE
#  define dra_delete_             DRA_DELETE
#  define dra_read_               DRA_READ
#  define ndra_read_              NDRA_READ
#  define dra_read_section_       DRA_READ_SECTION
#  define ndra_read_section_      NDRA_READ_SECTION
#  define dra_write_              DRA_WRITE
#  define ndra_write_             NDRA_WRITE
#  define dra_write_section_      DRA_WRITE_SECTION
#  define ndra_write_section_     NDRA_WRITE_SECTION
#  define dra_probe_              DRA_PROBE
#  define dra_set_debug_          DRA_SET_DEBUG
#  define dra_print_internals_    DRA_PRINT_INTERNALS
#  define dra_set_default_config_ DRA_SET_DEFAULT_CONFIG
#  define dra_wait_               DRA_WAIT
#  define dra_terminate_          DRA_TERMINATE
#  define dra_flick_              DRA_FLICK

#elif defined(F2C2_)

#  define dra_create_             dra_create__         
#  define ndra_create_            ndra_create__        
#  define ndra_create_config_     ndra_create_config__        
#  define dra_open_               dra_open__           
#  define dra_inquire_            dra_inquire__        
#  define ndra_inquire_           ndra_inquire__       
#  define dra_init_               dra_init__           
#  define dra_close_              dra_close__          
#  define dra_delete_             dra_delete__         
#  define dra_read_               dra_read__           
#  define ndra_read_              ndra_read__          
#  define dra_read_section_       dra_read_section__   
#  define ndra_read_section_      ndra_read_section__  
#  define dra_write_              dra_write__          
#  define ndra_write_             ndra_write__         
#  define dra_write_section_      dra_write_section__  
#  define ndra_write_section_     ndra_write_section__ 
#  define dra_probe_              dra_probe__          
#  define dra_set_debug_          dra_set_debug__      
#  define dra_print_internals_    dra_print_internals__      
#  define dra_set_default_config_ dra_set_default_config__      
#  define dra_wait_               dra_wait__           
#  define dra_terminate_          dra_terminate__      
#  define dra_flick_              dra_flick__          

#endif

#ifdef __cplusplus
extern "C" {
#endif

/* C-interface prototypes */

extern int NDRA_Create(       int type,
                              int ndim,
                              dra_size_t dims[],
                              char *name,
                              char* filename,
                              int mode,
                              dra_size_t reqdims[],
                              int *d_a);

extern int NDRA_Inquire(      int d_a,
                              int *type,
                              int *ndim,
                              dra_size_t dims[],
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
                              dra_size_t dlo[],
                              dra_size_t dhi[],
                              int *request);

extern int NDRA_Read_section( logical transp,
                              int g_a,
                              int glo[],
                              int ghi[],
                              int d_a,
                              dra_size_t dlo[],
                              dra_size_t dhi[],
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

extern void DRA_Print_internals(    int d_a);

extern void DRA_Set_default_config(    int numfiles, int numioprocs);

extern int DRA_Wait(          int request);

extern int DRA_Delete(        int d_a);

extern int DRA_Close(         int d_a);

extern void DRA_Flick();

#ifdef __cplusplus
       }	   
#endif

#undef _ARGS_

#endif

