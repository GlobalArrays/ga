#ifndef GPPAPI_H
#define GPPAPI_H

#include "typesf2c.h"

/* Routines from gpbase.c */

extern void pgp_initialize();
extern void pgp_terminate();

extern Integer pgp_create_handle();
extern void pgp_set_dimensions(Integer *g_p, Integer *ndim,
                               Integer *dims);
extern void pgp_set_chunk(Integer *g_p, Integer *chunk);
extern logical pgp_allocate(Integer *g_p);
extern logical pgp_destroy(Integer *g_p);

extern void pgp_assign_element(Integer *g_p, Integer *subscript,
                               void *ptr);

#endif  /* GPPAPI_H */
