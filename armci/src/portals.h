/* portals header file */

extern int armci_init_portals(void);
extern void armci_fini_portals(void);
int armci_portals_direct_send(void *src, void* dst, int bytes, int proc, armci_hdl_t * nb);





