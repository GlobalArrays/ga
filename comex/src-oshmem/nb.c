#include "nb.h"
#include "comex_impl.h"
#include <stdlib.h>
#include <string.h>

static comex_nb_entry_t *nb_table = NULL;

int comex_nb_reserve() {
    if (!nb_table) {
        nb_table = (comex_nb_entry_t*)malloc(sizeof(comex_nb_entry_t)*COMEX_MAX_NB_OUTSTANDING);
        if (!nb_table) return -1;
        memset(nb_table,0,sizeof(comex_nb_entry_t)*COMEX_MAX_NB_OUTSTANDING);
    }
    for (int i=0;i<COMEX_MAX_NB_OUTSTANDING;i++) {
        if (!nb_table[i].active) {
            nb_table[i].active = 1;
            return i;
        }
    }
    return -1;
}

void comex_nb_release(int idx) {
    if (!nb_table) return;
    if (idx<0 || idx>=COMEX_MAX_NB_OUTSTANDING) return;
    nb_table[idx].active = 0;
}

comex_nb_entry_t* comex_nb_get_entry(int idx) {
    if (!nb_table) return NULL;
    if (idx<0 || idx>=COMEX_MAX_NB_OUTSTANDING) return NULL;
    return &nb_table[idx];
}
