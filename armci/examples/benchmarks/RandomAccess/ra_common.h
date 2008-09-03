/*$ID:$*/
#define LONG_IS_64BITS
#ifdef LONG_IS_64BITS
#define POLY 0x0000000000000007UL
#define PERIOD 1317624576693539401L
/*typedef unsigned long u64Int;
typedef long s64Int;*/
#define FSTR64 "%ld"
#define FSTRU64 "%lu"
#define ZERO64B 0L
#else
#define POLY 0x0000000000000007ULL
#define PERIOD 1317624576693539401LL
typedef unsigned long long u64Int;
typedef long long s64Int;
#define FSTR64 "%lld"
#define FSTRU64 "%llu"
#define ZERO64B 0LL
#endif

/* Macros for timing */
#define CPUSEC() (HPL_timer_cputime())
#define RTSEC() (MPI_Wtime())

#define MAX_TOTAL_PENDING_UPDATES 1024
#define LOCAL_BUFFER_SIZE MAX_TOTAL_PENDING_UPDATES
#define MAX_OUTSTANDING_HANDLES 64
extern u64Int **HPCC_Table;

