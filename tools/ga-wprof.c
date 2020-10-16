#include "./ga-wprof.h"

ga_weak_profile_t gaw_global_stats[WPROF_TOTAL];
ga_weak_profile_t gaw_global_min_stats[WPROF_TOTAL];
ga_weak_profile_t gaw_global_max_stats[WPROF_TOTAL];
ga_weak_profile_t gaw_local_stats[WPROF_TOTAL];

char ga_weak_symbols[WPROF_TOTAL+1][80] = {
  "PNGA_ABS_VALUE",
  "PNGA_ABS_VALUE_PATCH",
  "PNGA_ACC",
  "PNGA_ACCESS_BLOCK_GRID_IDX",
  "PNGA_ACCESS_BLOCK_GRID_PTR",
  "PNGA_ACCESS_BLOCK_IDX",
  "PNGA_ACCESS_BLOCK_PTR",
  "PNGA_ACCESS_BLOCK_SEGMENT_IDX",
  "PNGA_ACCESS_BLOCK_SEGMENT_PTR",
  "PNGA_ACCESS_GHOST_ELEMENT",
  "PNGA_ACCESS_GHOST_ELEMENT_PTR",
  "PNGA_ACCESS_GHOST_PTR",
  "PNGA_ACCESS_GHOSTS",
  "PNGA_ACCESS_IDX",
  "PNGA_ACCESS_PTR",
  "PNGA_ADD",
  "PNGA_ADD_CONSTANT",
  "PNGA_ADD_CONSTANT_PATCH",
  "PNGA_ADD_DIAGONAL",
  "PNGA_ADD_PATCH",
  "PNGA_ALLOCATE",
  "PNGA_BIN_INDEX",
  "PNGA_BIN_SORTER",
  "PNGA_BRDCST",
  "PNGA_CHECK_HANDLE",
  "PNGA_CLUSTER_NNODES",
  "PNGA_CLUSTER_NODEID",
  "PNGA_CLUSTER_NPROCS",
  "PNGA_CLUSTER_PROC_NODEID",
  "PNGA_CLUSTER_PROCID",
  "PNGA_COMP_PATCH",
  "PNGA_COMPARE_DISTR",
  "PNGA_COPY",
  "PNGA_COPY_PATCH",
  "PNGA_COPY_PATCH_DP",
  "PNGA_CREATE",
  "PNGA_CREATE_BIN_RANGE",
  "PNGA_CREATE_CONFIG",
  "PNGA_CREATE_GHOSTS",
  "PNGA_CREATE_GHOSTS_CONFIG",
  "PNGA_CREATE_GHOSTS_IRREG",
  "PNGA_CREATE_GHOSTS_IRREG_CONFIG",
  "PNGA_CREATE_HANDLE",
  "PNGA_CREATE_IRREG",
  "PNGA_CREATE_IRREG_CONFIG",
  "PNGA_CREATE_CREATE_MUTEXES",
  "PNGA_DDOT_PATCH_DP",
  "PNGA_DEREGISTER_TYPE",
  "PNGA_DESTROY",
  "PNGA_DESTROY_MUTEXES",
  "PNGA_DIAG",
  "PNGA_DIAG_REUSE",
  "PNGA_DIAG_SEQ",
  "PNGA_DIAG_STD",
  "PNGA_DIAG_STD_SEQ",
  "PNGA_DISTRIBUTION",
  "PNGA_DOT",
  "PNGA_DOT_PATCH",
  "PNGA_DUPLICATE",
  "PNGA_ELEM_DIVIDE",
  "PNGA_ELEM_DIVIDE_PATCH",
  "PNGA_ELEM_MAXIMUM",
  "PNGA_ELEM_MAXIMUM_PATCH",
  "PNGA_ELEM_MINIMUM",
  "PNGA_ELEM_MINIMUM_PATCH",
  "PNGA_ELEM_MULTIPLY",
  "PNGA_ELEM_MULTIPLY_PATCH",
  "PNGA_ELEM_STEP_DIVIDE_PATCH",
  "PNGA_ELEM_STEPB_DIVIDE_PATCH",
  "PNGA_ERROR",
  "PNGA_FENCE",
  "PNGA_FILL",
  "PNGA_FILL_PATCH",
  "PNGA_GATHER",
  "PNGA_GATHER2D",
  "PNGA_GET",
  "PNGA_GET_BLOCK_INFO",
  "PNGA_GET_DEBUG",
  "PNGA_GET_DIAG",
  "PNGA_GET_DIMENSION",
  "PNGA_GET_FIELD",
  "PNGA_GET_GHOST_BLOCK",
  "PNGA_GET_PGROUP",
  "PNGA_GET_PGROUP_SIZE",
  "PNGA_GET_PROC_GRID",
  "PNGA_GET_PROC_INDEX",
  "PNGA_GHOST_BARRIER",
  "PNGA_GOP",
  "PNGA_HAS_GHOSTS",
  "PNGA_INIT_FENCE",
  "PNGA_INITIALIZE",
  "PNGA_INITIALIZE_LTD",
  "PNGA_INQUIRE",
  "PNGA_INQUIRE_MEMORY",
  "PNGA_INQUIRE_NAME",
  "PNGA_INQUIRE_TYPE",
  "PNGA_IS_MIRRORED",
  "PNGA_LIST_NODEID",
  "PNGA_LLT_SOLVE",
  "PNGA_LOCATE",
  "PNGA_LOCATE_NNODES",
  "PNGA_LOCATE_NUM_BLOCKS",
  "PNGA_LOCATE_REGION",
  "PNGA_LOCK",
  "PNGA_LU_SOLVE",
  "PNGA_LU_SOLVE_ALT",
  "PNGA_LU_SOLVE_SEQ",
  "PNGA_MASK_SYNC",
  "PNGA_MATMUL",
  "PNGA_MATMUL_MIRRORED",
  "PNGA_MATMUL_PATCH",
  "PNGA_MEDIAN",
  "PNGA_MEDIAN_PATCH",
  "PNGA_MEMORY_AVAIL",
  "PNGA_MEMORY_AVAIL_TYPE",
  "PNGA_MEMORY_LIMITED",
  "PNGA_MERGE_DISTR_PATCH",
  "PNGA_MERGE_MIRRORED",
  "PNGA_MSG_BRDCST",
  "PNGA_MSG_PGROUP_SYNC",
  "PNGA_MSG_SYNC",
  "PNGA_NBACC",
  "PNGA_NBGET",
  "PNGA_NBGET_FIELD",
  "PNGA_NBGET_GHOST_DIR",
  "PNGA_NBLOCK",
  "PNGA_NBPUT",
  "PNGA_NBPUT_FIELD",
  "PNGA_NBTEST",
  "PNGA_NBWAIT",
  "PNGA_NDIM",
  "PNGA_NNODES",
  "PNGA_NODEID",
  "PNGA_NORM1",
  "PNGA_NORM_INFINITY",
  "PNGA_OVERLAY",
  "PNGA_PACK",
  "PNGA_PATCH_ENUM",
  "PNGA_PATCH_INTERSECT",
  "PNGA_PERIODIC",
  "PNGA_PGROUP_ABSOLUTE_ID",
  "PNGA_PGROUP_BRDCST",
  "PNGA_PGROUP_CREATE",
  "PNGA_PGROUP_DESTROY",
  "PNGA_PGROUP_GET_DEFAULT",
  "PNGA_PGROUP_GET_MIRROR",
  "PNGA_PGROUP_GET_WORLD",
  "PNGA_PGROUP_GOP",
  "PNGA_PGROUP_NNODES",
  "PNGA_PGROUP_NODEID",
  "PNGA_PGROUP_SET_DEFAULT",
  "PNGA_PGROUP_SPLIT",
  "PNGA_PGROUP_SPLIT_IRREF",
  "PNGA_PGROUP_SYNC",
  "PNGA_PRINT",
  "PNGA_PRINT_DISTRIBUTION",
  "PNGA_PRINT_FILE",
  "PNGA_PRINT_PATCH",
  "PNGA_PRINT_PATCH2D",
  "PNGA_PRINT_PATCH_FILE",
  "PNGA_PRINT_PATCH_FILE2D",
  "PNGA_PRINT_STATS",
  "PNGA_PROC_TOPOLOGY",
  "PNGA_PUT",
  "PNGA_PUT_FIELD",
  "PNGA_RANDOMIZE",
  "PNGA_READ_INC",
  "PNGA_RECIP",
  "PNGA_RECIP_PATCH",
  "PNGA_REGISTER_TYPE",
  "PNGA_RELEASE",
  "PNGA_RELEASE_BLOCK",
  "PNGA_RELEASE_BLOCK_GRID",
  "PNGA_RELEASE_BLOCK_SEGMENT",
  "PNGA_RELEASE_GHOST_ELEMENT",
  "PNGA_RELEASE_GHOSTS",
  "PNGA_RELEASE_UPDATE",
  "PNGA_RELEASE_UDPATE_BLOCK",
  "PNGA_RELEASE_UDPATE_BLOCK_GRID",
  "PNGA_RELEASE_UDPATE_BLOCK_SEGMENT",
  "PNGA_RELEASE_UDPATE_GHOST_ELEMENT",
  "PNGA_RELEASE_UDPATE_GHOSTS",
  "PNGA_SCALE",
  "PNGA_SCALE_COLS",
  "PNGA_SCALE_PATCH",
  "PNGA_SCALE_ROWS",
  "PNGA_SCAN_ADD",
  "PNGA_SCAN_COPY",
  "PNGA_SCATTER",
  "PNGA_SCATTER2D",
  "PNGA_SCATTER_ACC",
  "PNGA_SCATTER_ACC2D",
  "PNGA_SELECT_ELEM",
  "PNGA_SET_ARRAY_NAME",
  "PNGA_SET_BLOCK_CYCLIC",
  "PNGA_SET_BLOCK_CYCLIC_PROC_GRID",
  "PNGA_SET_CHUNK",
  "PNGA_SET_DATA",
  "PNGA_SET_DEBUG",
  "PNGA_SET_DIAGONAL",
  "PNGA_SET_GHOST_CORNER_FLAG",
  "PNGA_SET_GHOST_INFO",
  "PNGA_SET_GHOSTS",
  "PNGA_SET_IRREG_DISTR",
  "PNGA_SET_IRREG_FLAG",
  "PNGA_SET_MEMORY_LIMIT",
  "PNGA_SET_PGROUP",
  "PNGA_SET_RESTRICTED",
  "PNGA_SET_RESTRCITED_RANGE",
  "PNGA_SET_UPDATE4_INFO",
  "PNGA_SET_UPDATE5_INFO",
  "PNGA_SHIFT_DIAGONAL",
  "PNGA_SOLVE",
  "PNGA_SPD_INVERT",
  "PNGA_STEP_BOUND_INFO",
  "PNGA_STEP_BOUND_INFO_PATCH",
  "PNGA_STEP_MASK_PATCH",
  "PNGA_STEP_MAX",
  "PNGA_STEP_MAX_PATCH",
  "PNGA_STRIDED_ACC",
  "PNGA_STRIDED_GET",
  "PNGA_STRIDED_PUT",
  "PNGA_SUMMARIZE",
  "PNGA_SYMMETRIZE",
  "PNGA_SYNC",
  "PNGA_TERMINATE",
  "PNGA_TIMER",
  "PNGA_TOTAL_BLOCKS",
  "PNGA_TRANSPOSE",
  "PNGA_TYPE_C2F",
  "PNGA_TYPE_F2C",
  "PNGA_UNLOCK",
  "PNGA_UNPACK",
  "PNGA_UPDATE1_GHOSTS",
  "PNGA_UPDATE2_GHOSTS",
  "PNGA_UPDATE3_GHOSTS",
  "PNGA_UPDATED44_GHOSTS",
  "PNGA_UPDATE4_GHOSTS",
  "PNGA_UPDATE55_GHOSTS",
  "PNGA_UPDATE5_GHOSTS",
  "PNGA_UPDATE6_GHOSTS",
  "PNGA_UPDATE7_GHOSTS",
  "PNGA_UDPATE_GHOST_DIR",
  "PNGA_UPDATE_GHOSTS",
  "PNGA_USES_MA",
  "PNGA_USES_PROC_GRID",
  "PNGA_VALID_HANDLE",
  "PNGA_VERIFY_HANDLE",
  "PNGA_WTIME",
  "PNGA_ZERO",
  "PNGA_ZERO_DIAGONAL",
  "PNGA_ZERO_PATCH",
  "WPROF_TOTAL"
};

#include <string.h>

#include <time.h>
#include <sys/time.h>
#include <alloca.h>
#define NANOSECS 1000000000ULL

uint64_t I_Wtime(void)
{
    struct timespec res;
    clock_gettime(CLOCK_REALTIME, &res);
    uint64_t timeRes = res.tv_sec*NANOSECS+res.tv_nsec;
    return timeRes;
}

enum FMT logs = HUMAN_FMT;
FILE *filePre = NULL;
FILE *globalFilePre = NULL;
int gaw_depth = 1;
int enable_local = 0;

int init_ga_prof_struct(int me, int nproc){
   int i;
   for(i = 0; i < WPROF_TOTAL; ++i){
      strncpy(gaw_global_stats[i].name, ga_weak_symbols[i], 80);
      strncpy(gaw_local_stats[i].name, ga_weak_symbols[i], 80);
   
      gaw_global_stats[i].count = 0;
      gaw_global_stats[i].time = 0;
      gaw_global_stats[i].total_bytes = 0;

      gaw_local_stats[i].count = 0;
      gaw_local_stats[i].time = 0;
      gaw_local_stats[i].total_bytes = 0;
   }
   char *filePrefix = getenv("GAW_FILE_PREFIX");
   char *strLogs = getenv("GAW_FMT");
   char *Gdepth = getenv("GAW_DEPTH");
   char *Glocal = getenv("GAW_ENABLE_LOCAL");
   if(Glocal){
      enable_local = 1;
   }
   if(!filePrefix){
      filePrefix = alloca(1024);
      strncpy(filePrefix, "logs_pre", 1024);      
   }    
   char fname[1024];
   if(enable_local){
      sprintf(fname, "%s-%d-%d.txt", filePrefix, me, nproc);
      filePre = fopen(fname, "wb+");
      if(!filePre){
         printf("Error in openning logs files: %s\n", fname);
         exit(11);
      }
      else{
         printf("Log File openned: %s\n", fname);
      }
   }
   if(me == 0){
      sprintf(fname, "%s-GLOBAL.txt", filePrefix);
      globalFilePre = fopen(fname, "wb+");
      if(!globalFilePre){
         printf("Error in openning logs files: %s\n", fname);
         exit(11);
      }
      else{
         printf("Log File openned: %s\n", fname);
      }      
   }
   if(!strLogs){
       /* Assume that the default is HUMAN readable */
   }
   else{
      int fm = atoi(strLogs); /* zero for CSV and one for HUMAN READABLE */
      switch(fm){
         case HUMAN_FMT:
            logs = HUMAN_FMT;
            break;
         case CSV_FMT:
             logs = CSV_FMT;
             break;
         default:
            logs = HUMAN_FMT;
            break;
      }   
   }

}

pthread_mutex_t gprof_lock =  PTHREAD_MUTEX_INITIALIZER;

int update_local_entry(enum WPROF_GA e, uint64_t tme, uint64_t bytes){
   if(e >= WPROF_TOTAL){
      return -1;
   }
   pthread_mutex_lock(&gprof_lock);
   gaw_local_stats[e].count++;
   gaw_local_stats[e].time += tme;
   gaw_local_stats[e].total_bytes += bytes;
   pthread_mutex_unlock(&gprof_lock);
   return 0;
}

int update_global_entry(enum WPROF_GA e, MPI_Comm comm){
   if(e >= WPROF_TOTAL){
      return -1;
   }
   MPI_Reduce(&gaw_local_stats[e].count, &gaw_global_stats[e].count, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
   MPI_Reduce(&gaw_local_stats[e].time, &gaw_global_stats[e].time, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);
   MPI_Reduce(&gaw_local_stats[e].total_bytes, &gaw_global_stats[e].total_bytes, 1, MPI_LONG_LONG, MPI_SUM, 0, comm);

   MPI_Reduce(&gaw_local_stats[e].count, &gaw_global_min_stats[e].count, 1, MPI_LONG_LONG, MPI_MIN, 0, comm);
   MPI_Reduce(&gaw_local_stats[e].time, &gaw_global_min_stats[e].time, 1, MPI_LONG_LONG, MPI_MIN, 0, comm);
   MPI_Reduce(&gaw_local_stats[e].total_bytes, &gaw_global_min_stats[e].total_bytes, 1, MPI_LONG_LONG, MPI_MIN, 0, comm);

   MPI_Reduce(&gaw_local_stats[e].count, &gaw_global_max_stats[e].count, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);
   MPI_Reduce(&gaw_local_stats[e].time, &gaw_global_max_stats[e].time, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);
   MPI_Reduce(&gaw_local_stats[e].total_bytes, &gaw_global_max_stats[e].total_bytes, 1, MPI_LONG_LONG, MPI_MAX, 0, comm);   

   return 0;
}

int print_ga_prof_stats(enum FMT f, FILE *fp, MPI_Comm comm){
   int i;
   int me, nproc;
   MPI_Comm_rank(MPI_COMM_WORLD, &me);
   MPI_Comm_size(MPI_COMM_WORLD, &nproc);
   /* TODO: change the arguments, they are not longer needed */
   f = logs;
   
   if(enable_local){
      fp = filePre;
      fprintf(fp, "Local stats\n");
      switch(f)
      {
         case CSV_FMT:
            fprintf(fp, "Func Name, Node, Count, Time, Bytes\n");
            break;
         case HUMAN_FMT:
            break;
      }
      for (i = 0; i < WPROF_TOTAL; ++i){
         if(gaw_local_stats[i].count){
            switch(f){
               case CSV_FMT:  
                  fprintf(fp, "%s, %d, %"PRIu64", %"PRIu64", %"PRIu64"\n",
                      gaw_local_stats[i].name, me, 
                      gaw_local_stats[i].count, gaw_local_stats[i].time, gaw_local_stats[i].total_bytes);
                  break;
               case HUMAN_FMT:
                  fprintf(fp, "[%d] Function Name: %s(%d)\n", me, gaw_local_stats[i].name, i);
                  fprintf(fp, "[%d]\tCount: %"PRIu64"\n", me, gaw_local_stats[i].count);
                  fprintf(fp, "[%d]\tCumm Time: %"PRIu64"\n", me, gaw_local_stats[i].time);
                  fprintf(fp, "[%d]\tTotal Bytes: %"PRIu64"\n", me, gaw_local_stats[i].total_bytes);
                  break;
            }
         }
      }
   }
   MPI_Barrier(comm);
   fp = globalFilePre;
   
   if(me == 0){
      fprintf(fp, "Global Stats\n");     
   switch(f)
   {
      case CSV_FMT:
         fprintf(fp, "Func Name, Node, Count, Avg Time, Bytes, min Count, min Time, min Bytes, max Count, max Time, max Bytes\n");
         break;
      case HUMAN_FMT:
         break;
   }
   for (i = 0; i < WPROF_TOTAL; ++i){
      if(gaw_local_stats[i].count){
         switch(f){
            case CSV_FMT:  
               fprintf(fp, "%s, %"PRIu64", %.3lf, %"PRIu64", %"PRIu64", %"PRIu64", %"PRIu64", %"PRIu64", %"PRIu64", %"PRIu64"\n",
                      gaw_global_stats[i].name, 
                      gaw_global_stats[i].count, (double)(gaw_global_stats[i].time/(double)(nproc)), gaw_global_stats[i].total_bytes,
                      gaw_global_min_stats[i].count, gaw_global_min_stats[i].time, gaw_global_min_stats[i].total_bytes,
                      gaw_global_max_stats[i].count, gaw_global_max_stats[i].time, gaw_global_max_stats[i].total_bytes);
               break;
            case HUMAN_FMT:
               fprintf(fp, "Function Name: %s\n", gaw_global_stats[i].name);
               fprintf(fp, "\tCount: %"PRIu64"\n", gaw_global_stats[i].count);
               fprintf(fp, "\tAverage Time: %.3lf\n", (double)(gaw_global_stats[i].time/(double)(nproc)));
               fprintf(fp, "\tTotal Bytes: %"PRIu64"\n", gaw_global_stats[i].total_bytes);
               fprintf(fp, "\tMin Count: %"PRIu64"\n", gaw_global_min_stats[i].count);
               fprintf(fp, "\tMin Time: %"PRIu64"\n", gaw_global_min_stats[i].time);
               fprintf(fp, "\tMin Bytes: %"PRIu64"\n", gaw_global_min_stats[i].total_bytes);
               fprintf(fp, "\tMax Count: %"PRIu64"\n", gaw_global_max_stats[i].count);
               fprintf(fp, "\tMax Time: %"PRIu64"\n", gaw_global_max_stats[i].time);
               fprintf(fp, "\tMax Bytes: %"PRIu64"\n", gaw_global_max_stats[i].total_bytes);                              
               break;
         }
      }
   }   
   }
   if(enable_local) fclose(filePre);
   if(me == 0) fclose(globalFilePre);   
}
