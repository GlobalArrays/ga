/* $Id:  */

#ifndef HASH_COMMON_H
#define HASH_COMMON_H

#define HASHMAP_CREATE  101
#define HASHMAP_DESTROY 102
#define HASHMAP_INSERT  103
#define HASHMAP_PRINT   104
#define HASHMAP_GET     105
#define HASHMAP_REHASH  106

typedef struct hash_hdr {
      int hash_op;
      void*buf;
      size_t bufsize;
}hash_hdr_t;

#include <string>
using std::string;

//#include "UnicodeString.h"
#ifdef USE_MAP
#  include <map>
typedef std::map<string, int> VocabIntMap;
#else
#  include "hash_map.h"
using STL_HASHMAP_NAMESPACE::hash_map;
using STL_HASHMAP_NAMESPACE::hash;

struct hashStr {
  size_t operator()(const string& str) const {
    hash<const char*> H;
    return H(str.c_str());
  }
};

typedef hash_map< string, int, hashStr> VocabIntMap;
#endif


/* ARMCI is impartial to message-passing libs - we handle them with MP macros
 * */
#if defined(TCGMSG)
#   include <sndrcv.h>
    long tcg_tag =30000;
#   define MP_BARRIER()      SYNCH_(&tcg_tag)
#   define MP_MYID(pid)      *(pid)   = (int)NODEID_()
#   define MP_PROCS(pproc)   *(pproc) = (int)NNODES_()
#else
#   include <mpi.h>
#   define MP_BARRIER()      MPI_Barrier(MPI_COMM_WORLD)
#   define MP_MYID(pid)      MPI_Comm_rank(MPI_COMM_WORLD, (pid))
#   define MP_PROCS(pproc)   MPI_Comm_size(MPI_COMM_WORLD, (pproc));
#endif

extern unsigned long armci_djb2_hash(unsigned char *str, int total_procs);

#endif // HASH_COMMON_H
