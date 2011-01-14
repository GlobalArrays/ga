#ifndef GPPAPI_H
#define GPPAPI_H

#include "typesf2c.h"

/* Routines from gpbase.c */

extern logical pgp_allocate(Integer *g_p);
extern void    pgp_assign_element(Integer *g_p, Integer *subscript,
                                  void *ptr);
extern Integer pgp_create_handle();
extern logical pgp_destroy(Integer *g_p);
extern void    pgp_initialize();
extern void    pgp_set_chunk(Integer *g_p, Integer *chunk);
extern void    pgp_set_dimensions(Integer *g_p, Integer *ndim,
                                  Integer *dims);
extern void    pgp_terminate();


/* Routines from gponesided.c */
extern void    pgp_get(Integer g_p, Integer *lo, Integer *hi, void *buf,
                       void **buf_ptr, Integer *ld, void *buf_size, Integer *ld_sz, 
                       Integer *size, Integer intsize);
extern void    pgp_get_size(Integer g_p, Integer *lo, Integer *hi,
                            Integer *size, Integer *intsize);
#endif  /* GPPAPI_H */
