/* portals header file */
#include <portals3.h>
#include P3_NAL
#include <p3rt/p3rt.h>
#include <p3api/debug.h>
#define MAX_OUT 50
#define MAX_ENT 64
#define MAX_PREPOST 1

typedef enum op {
        ARMCI_PORTALS_PUT,
        ARMCI_PORTALS_NBPUT,
        ARMCI_PORTALS_GET, 
        ARMCI_PORTALS_NBGET, 
        ARMCI_PORTALS_ACC
} armci_portals_optype;

/* array of memory segments and corresponding memory descriptors */
typedef struct md_table{
        int id;
        void * start;
        void * end;
        int bytes;       
        ptl_md_t md;
        ptl_match_bits_t mb;
} md_table_entry_t;


typedef struct desc{
       int active;
       unsigned int tag;      
       int dest_id;
      armci_portals_optype type;
}comp_desc; 

#define NB_CMPL_T int

/* structure of computing process */
typedef struct {
  int armci_rank;  /* if different from portals_rank */      
  int rank;       /* my rank*/
  int size;       /* size of the group */
  ptl_handle_me_t me_h[64];
  ptl_handle_me_t md_h[64];

  ptl_handle_eq_t eq_h;
  ptl_handle_ni_t ni_h; 
  ptl_pt_index_t ptl;
  int outstanding_puts;
  int outstanding_gets;
  int outstanding_accs;
  void * buffers; /* ptr to head of buffer */
  int num_match_entries;
} armci_portals_proc_t;



extern int armci_init_portals(void);
extern void armci_fini_portals(void);
extern int armci_post_descriptor(ptl_md_t md); 
extern int armci_prepost_descriptor(void* start, long bytes);
extern int armci_get_offset(ptl_md_t md, void *ptr);
extern int armci_get_md(void * start, int bytes , ptl_md_t * md, ptl_match_bits_t * mb);
extern int armci_portals_put(ptl_handle_md_t md_h,ptl_process_id_t dest_id,int bytes,int mb,int local_offset, int remote_offset,int ack );
extern comp_desc * get_free_comp_desc();
extern int armci_portals_direct_send(void *src, void* dst, int bytes, int proc, int nbtag, NB_CMPL_T *cmpl_info);
extern int armci_portals_direct_get(void *src, void *dst, int bytes, int proc, int nbtag, NB_CMPL_T *cmpl_info);
extern int armci_portals_complete(int nbtag, NB_CMPL_T *cmpl_info);
extern void comp_desc_init();
extern int armci_client_complete(ptl_event_kind_t *evt,int proc_id, int nb_tag ,comp_desc * cdesc,int b_tag);
