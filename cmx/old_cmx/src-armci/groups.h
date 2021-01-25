#ifndef _GROUPS_H
#define _GROUPS_H
/*
#if HAVE_CONFIG_H
#   include "config.h"
#endif
*/

#include <stdlib.h>
#include <assert.h>

#include "armci.h"
#include "message.h"
#include "cmx.h"

typedef struct _armci_handle_link {
  struct _armci_handle_link *next;
  struct _armci_handle_link *prev;
  cmx_handle_t *handle;
} armci_handle_link_t;

typedef struct _armci_group_link {
  struct _armci_group_link *next;
  ARMCI_Group id;
  armci_handle_link_t *handle_list;
  cmx_group_t group;
} armci_igroup_t;

extern armci_igroup_t* iarm_get_igroup_from_group(ARMCI_Group id);
extern void armci_group_init();
extern void armci_group_finalize();
extern cmx_group_t armci_get_cmx_group(ARMCI_Group id);
extern armci_igroup_t* armci_get_igroup_from_group(ARMCI_Group id);
#endif
