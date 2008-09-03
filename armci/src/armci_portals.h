#ifndef PORTALS_H
#define PORTALS_H

/* portals header file */

#include <portals/portals3.h>
#include <catamount/cnos_mpi_os.h>

#define NUM_COMP_DSCR 16

#define ARMCI_PORTALS_PTL_NUMBER 37

#define HAS_RDMA_GET

/*corresponds to num of different armci mem regions*/
#define MAX_MEM_REGIONS 64

#define ARMCI_NET_ERRTOSTR(__ARMCI_ERC_) ptl_err_str[__ARMCI_ERC_]

typedef enum op {
        ARMCI_PORTALS_PUT,
        ARMCI_PORTALS_NBPUT,
        ARMCI_PORTALS_GET, 
        ARMCI_PORTALS_NBGET, 
        ARMCI_PORTALS_ACC,
        ARMCI_PORTALS_NBACC,
        ARMCI_PORTALS_GETPUT,
        ARMCI_PORTALS_NBGETPUT
} armci_portals_optype;

typedef struct armci_portals_desc{
       int active;
       int tag;
       int dest_id;
       armci_portals_optype type;
       ptl_md_t mem_dsc;
       ptl_handle_md_t mem_dsc_hndl;
}comp_desc;


#define NB_CMPL_T comp_desc*

#define ARMCI_NB_WAIT(_cntr) if(_cntr){\
        int rc;\
        if(nb_handle->tag==_cntr->tag)\
          rc = armci_client_complete(NULL,nb_handle->proc,nb_handle->tag,_cntr);\
} else{\
printf("\n%d:wait null ctr\n",armci_me);}


/* structure of computing process */
typedef struct {
  ptl_pt_index_t ptl;
  ptl_process_id_t  rank;
  ptl_handle_ni_t   ni_h; 
  ptl_handle_eq_t   eq_h;
  int               outstanding_puts;
  int               outstanding_gets;
  cnos_nidpid_map_t *ptl_pe_procid_map;  
  int               free_comp_desc_index;
}armci_portals_proc_t;

typedef struct {
  ptl_match_bits_t  mb;
  ptl_md_t          md;
  ptl_handle_me_t   me_h;
  ptl_handle_md_t   md_h;
}armci_portals_serv_mem_t;

typedef struct {
  ptl_process_id_t  rank;
  ptl_handle_eq_t   eq_h;
  int               reg_count;
  int               outstanding_puts;
  int               outstanding_gets;
  armci_portals_serv_mem_t meminfo[MAX_MEM_REGIONS];
}armci_portals_serv_t;

extern void print_mem_desc_table(void);
extern int armci_init_portals(void);
extern void armci_fini_portals(void);
extern int armci_post_descriptor(ptl_md_t *md); 
extern int armci_prepost_descriptor(void* start, long bytes);
extern ptl_size_t armci_get_offset(ptl_md_t md, void *ptr,int proc);
extern int armci_get_md(void * start, size_t bytes , ptl_md_t * md, ptl_match_bits_t * mb);
extern void armci_portals_put(int,void *,void *,size_t,void *,int );
extern void armci_portals_get(int,void *,void *,size_t,void *,int );
extern void comp_desc_init();
extern int armci_client_complete(ptl_event_kind_t *evt,int proc_id, int nb_tag ,comp_desc * cdesc);

#endif /* PORTALS_H */
