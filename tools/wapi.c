
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "papi.h"
#include "typesf2c.h"

static FILE *fplog=NULL;
int me, nproc;

static void log_init() {
    PMPI_Barrier(MPI_COMM_WORLD);
    PMPI_Comm_rank(MPI_COMM_WORLD, &me);
    PMPI_Comm_size(MPI_COMM_WORLD, &nproc);
    /* create files to write trace data */
    char *profile_dir;
    char *file_name;
    struct stat f_stat;

    profile_dir = getenv("PNGA_PROFILE_DIR");
    if (0 == me) {
        if (!profile_dir) {
            pnga_error("You need to set PNGA_PROFILE_DIR env var", 1);
        }
        fprintf(stderr, "PNGA_PROFILE_DIR=%s\n", profile_dir);
        if (-1 == stat(profile_dir, &f_stat)) {
            perror("stat");
            fprintf(stderr, "Cannot successfully stat to PNGA_PROFILE_DIR.\n");
            fprintf(stderr, "Check %s profile dir\n", profile_dir);
            pnga_error("aborting", 1);
        }
    }
    PMPI_Barrier(MPI_COMM_WORLD);
    /* TODO finish per-process trace file */
}



void wnga_abs_value(Integer g_a)
{
    printf("%lf,pnga_abs_value,(%ld)\n",MPI_Wtime(),g_a);
    pnga_abs_value(g_a);
    printf("%lf,/pnga_abs_value,(%ld)\n",MPI_Wtime(),g_a);

}


void wnga_abs_value_patch(Integer g_a, Integer *lo, Integer *hi)
{
    printf("%lf,pnga_abs_value_patch,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);
    pnga_abs_value_patch(g_a, lo, hi);
    printf("%lf,/pnga_abs_value_patch,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);

}


void wnga_acc(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha)
{
    printf("%lf,pnga_acc,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld,alpha);
    pnga_acc(g_a, lo, hi, buf, ld, alpha);
    printf("%lf,/pnga_acc,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld,alpha);

}


void wnga_access_block_grid_idx(Integer g_a, Integer *subscript, AccessIndex *index, Integer *ld)
{
    printf("%lf,pnga_access_block_grid_idx,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,subscript,index,ld);
    pnga_access_block_grid_idx(g_a, subscript, index, ld);
    printf("%lf,/pnga_access_block_grid_idx,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,subscript,index,ld);

}


void wnga_access_block_grid_ptr(Integer g_a, Integer *index, void *ptr, Integer *ld)
{
    printf("%lf,pnga_access_block_grid_ptr,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,index,ptr,ld);
    pnga_access_block_grid_ptr(g_a, index, ptr, ld);
    printf("%lf,/pnga_access_block_grid_ptr,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,index,ptr,ld);

}


void wnga_access_block_idx(Integer g_a, Integer idx, AccessIndex *index, Integer *ld)
{
    printf("%lf,pnga_access_block_idx,(%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,idx,index,ld);
    pnga_access_block_idx(g_a, idx, index, ld);
    printf("%lf,/pnga_access_block_idx,(%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,idx,index,ld);

}


void wnga_access_block_ptr(Integer g_a, Integer idx, void *ptr, Integer *ld)
{
    printf("%lf,pnga_access_block_ptr,(%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,idx,ptr,ld);
    pnga_access_block_ptr(g_a, idx, ptr, ld);
    printf("%lf,/pnga_access_block_ptr,(%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,idx,ptr,ld);

}


void wnga_access_block_segment_idx(Integer g_a, Integer proc, AccessIndex *index, Integer *len)
{
    printf("%lf,pnga_access_block_segment_idx,(%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,proc,index,len);
    pnga_access_block_segment_idx(g_a, proc, index, len);
    printf("%lf,/pnga_access_block_segment_idx,(%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,proc,index,len);

}


void wnga_access_block_segment_ptr(Integer g_a, Integer proc, void *ptr, Integer *len)
{
    printf("%lf,pnga_access_block_segment_ptr,(%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,proc,ptr,len);
    pnga_access_block_segment_ptr(g_a, proc, ptr, len);
    printf("%lf,/pnga_access_block_segment_ptr,(%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,proc,ptr,len);

}


void wnga_access_ghost_element(Integer g_a, AccessIndex *index, Integer subscript[], Integer ld[])
{
    printf("%lf,pnga_access_ghost_element,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,index,subscript,ld);
    pnga_access_ghost_element(g_a, index, subscript, ld);
    printf("%lf,/pnga_access_ghost_element,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,index,subscript,ld);

}


void wnga_access_ghost_element_ptr(Integer g_a, void *ptr, Integer subscript[], Integer ld[])
{
    printf("%lf,pnga_access_ghost_element_ptr,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,ptr,subscript,ld);
    pnga_access_ghost_element_ptr(g_a, ptr, subscript, ld);
    printf("%lf,/pnga_access_ghost_element_ptr,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,ptr,subscript,ld);

}


void wnga_access_ghost_ptr(Integer g_a, Integer dims[], void *ptr, Integer ld[])
{
    printf("%lf,pnga_access_ghost_ptr,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,dims,ptr,ld);
    pnga_access_ghost_ptr(g_a, dims, ptr, ld);
    printf("%lf,/pnga_access_ghost_ptr,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,dims,ptr,ld);

}


void wnga_access_ghosts(Integer g_a, Integer dims[], AccessIndex *index, Integer ld[])
{
    printf("%lf,pnga_access_ghosts,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,dims,index,ld);
    pnga_access_ghosts(g_a, dims, index, ld);
    printf("%lf,/pnga_access_ghosts,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,dims,index,ld);

}


void wnga_access_idx(Integer g_a, Integer *lo, Integer *hi, AccessIndex *index, Integer *ld)
{
    printf("%lf,pnga_access_idx,(%ld;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,index,ld);
    pnga_access_idx(g_a, lo, hi, index, ld);
    printf("%lf,/pnga_access_idx,(%ld;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,index,ld);

}


void wnga_access_ptr(Integer g_a, Integer *lo, Integer *hi, void *ptr, Integer *ld)
{
    printf("%lf,pnga_access_ptr,(%ld;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,ptr,ld);
    pnga_access_ptr(g_a, lo, hi, ptr, ld);
    printf("%lf,/pnga_access_ptr,(%ld;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,ptr,ld);

}


void wnga_add(void *alpha, Integer g_a, void *beta, Integer g_b, Integer g_c)
{
    printf("%lf,pnga_add,(%p;%ld;%p;%ld;%ld)\n",MPI_Wtime(),alpha,g_a,beta,g_b,g_c);
    pnga_add(alpha, g_a, beta, g_b, g_c);
    printf("%lf,/pnga_add,(%p;%ld;%p;%ld;%ld)\n",MPI_Wtime(),alpha,g_a,beta,g_b,g_c);

}


void wnga_add_constant(Integer g_a, void *alpha)
{
    printf("%lf,pnga_add_constant,(%ld;%p)\n",MPI_Wtime(),g_a,alpha);
    pnga_add_constant(g_a, alpha);
    printf("%lf,/pnga_add_constant,(%ld;%p)\n",MPI_Wtime(),g_a,alpha);

}


void wnga_add_constant_patch(Integer g_a, Integer *lo, Integer *hi, void *alpha)
{
    printf("%lf,pnga_add_constant_patch,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,alpha);
    pnga_add_constant_patch(g_a, lo, hi, alpha);
    printf("%lf,/pnga_add_constant_patch,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,alpha);

}


void wnga_add_diagonal(Integer g_a, Integer g_v)
{
    printf("%lf,pnga_add_diagonal,(%ld;%ld)\n",MPI_Wtime(),g_a,g_v);
    pnga_add_diagonal(g_a, g_v);
    printf("%lf,/pnga_add_diagonal,(%ld;%ld)\n",MPI_Wtime(),g_a,g_v);

}


void wnga_add_patch(void *alpha, Integer g_a, Integer *alo, Integer *ahi, void *beta, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    printf("%lf,pnga_add_patch,(%p;%ld;%p;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),alpha,g_a,alo,ahi,beta,g_b,blo,bhi,g_c,clo,chi);
    pnga_add_patch(alpha, g_a, alo, ahi, beta, g_b, blo, bhi, g_c, clo, chi);
    printf("%lf,/pnga_add_patch,(%p;%ld;%p;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),alpha,g_a,alo,ahi,beta,g_b,blo,bhi,g_c,clo,chi);

}


logical wnga_allocate(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_allocate,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_allocate(g_a);
    printf("%lf,/pnga_allocate,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


void wnga_bin_index(Integer g_bin, Integer g_cnt, Integer g_off, Integer *values, Integer *subs, Integer n, Integer sortit)
{
    printf("%lf,pnga_bin_index,(%ld;%ld;%ld;%p;%p;%ld;%ld)\n",MPI_Wtime(),g_bin,g_cnt,g_off,values,subs,n,sortit);
    pnga_bin_index(g_bin, g_cnt, g_off, values, subs, n, sortit);
    printf("%lf,/pnga_bin_index,(%ld;%ld;%ld;%p;%p;%ld;%ld)\n",MPI_Wtime(),g_bin,g_cnt,g_off,values,subs,n,sortit);

}


void wnga_bin_sorter(Integer g_bin, Integer g_cnt, Integer g_off)
{
    printf("%lf,pnga_bin_sorter,(%ld;%ld;%ld)\n",MPI_Wtime(),g_bin,g_cnt,g_off);
    pnga_bin_sorter(g_bin, g_cnt, g_off);
    printf("%lf,/pnga_bin_sorter,(%ld;%ld;%ld)\n",MPI_Wtime(),g_bin,g_cnt,g_off);

}


void wnga_brdcst(Integer type, void *buf, Integer len, Integer originator)
{
    printf("%lf,pnga_brdcst,(%ld;%p;%ld;%ld)\n",MPI_Wtime(),type,buf,len,originator);
    pnga_brdcst(type, buf, len, originator);
    printf("%lf,/pnga_brdcst,(%ld;%p;%ld;%ld)\n",MPI_Wtime(),type,buf,len,originator);

}


void wnga_check_handle(Integer g_a, char *string)
{
    printf("%lf,pnga_check_handle,(%ld;%s)\n",MPI_Wtime(),g_a,string);
    pnga_check_handle(g_a, string);
    printf("%lf,/pnga_check_handle,(%ld;%s)\n",MPI_Wtime(),g_a,string);

}


Integer wnga_cluster_nnodes()
{
    Integer retval;
    printf("%lf,pnga_cluster_nnodes,\n",MPI_Wtime());
    retval = pnga_cluster_nnodes();
    printf("%lf,/pnga_cluster_nnodes,\n",MPI_Wtime());
    return retval;

}


Integer wnga_cluster_nodeid()
{
    Integer retval;
    printf("%lf,pnga_cluster_nodeid,\n",MPI_Wtime());
    retval = pnga_cluster_nodeid();
    printf("%lf,/pnga_cluster_nodeid,\n",MPI_Wtime());
    return retval;

}


Integer wnga_cluster_nprocs(Integer node)
{
    Integer retval;
    printf("%lf,pnga_cluster_nprocs,(%ld)\n",MPI_Wtime(),node);
    retval = pnga_cluster_nprocs(node);
    printf("%lf,/pnga_cluster_nprocs,(%ld)\n",MPI_Wtime(),node);
    return retval;

}


Integer wnga_cluster_proc_nodeid(Integer proc)
{
    Integer retval;
    printf("%lf,pnga_cluster_proc_nodeid,(%ld)\n",MPI_Wtime(),proc);
    retval = pnga_cluster_proc_nodeid(proc);
    printf("%lf,/pnga_cluster_proc_nodeid,(%ld)\n",MPI_Wtime(),proc);
    return retval;

}


Integer wnga_cluster_procid(Integer node, Integer loc_proc_id)
{
    Integer retval;
    printf("%lf,pnga_cluster_procid,(%ld;%ld)\n",MPI_Wtime(),node,loc_proc_id);
    retval = pnga_cluster_procid(node, loc_proc_id);
    printf("%lf,/pnga_cluster_procid,(%ld;%ld)\n",MPI_Wtime(),node,loc_proc_id);
    return retval;

}


logical wnga_comp_patch(Integer andim, Integer *alo, Integer *ahi, Integer bndim, Integer *blo, Integer *bhi)
{
    logical retval;
    printf("%lf,pnga_comp_patch,(%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),andim,alo,ahi,bndim,blo,bhi);
    retval = pnga_comp_patch(andim, alo, ahi, bndim, blo, bhi);
    printf("%lf,/pnga_comp_patch,(%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),andim,alo,ahi,bndim,blo,bhi);
    return retval;

}


logical wnga_compare_distr(Integer g_a, Integer g_b)
{
    logical retval;
    printf("%lf,pnga_compare_distr,(%ld;%ld)\n",MPI_Wtime(),g_a,g_b);
    retval = pnga_compare_distr(g_a, g_b);
    printf("%lf,/pnga_compare_distr,(%ld;%ld)\n",MPI_Wtime(),g_a,g_b);
    return retval;

}


void wnga_copy(Integer g_a, Integer g_b)
{
    printf("%lf,pnga_copy,(%ld;%ld)\n",MPI_Wtime(),g_a,g_b);
    pnga_copy(g_a, g_b);
    printf("%lf,/pnga_copy,(%ld;%ld)\n",MPI_Wtime(),g_a,g_b);

}


void wnga_copy_patch(char *trans, Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi)
{
    printf("%lf,pnga_copy_patch,(%s;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),trans,g_a,alo,ahi,g_b,blo,bhi);
    pnga_copy_patch(trans, g_a, alo, ahi, g_b, blo, bhi);
    printf("%lf,/pnga_copy_patch,(%s;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),trans,g_a,alo,ahi,g_b,blo,bhi);

}


void wnga_copy_patch_dp(char *t_a, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi)
{
    printf("%lf,pnga_copy_patch_dp,(%s;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),t_a,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi);
    pnga_copy_patch_dp(t_a, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi);
    printf("%lf,/pnga_copy_patch_dp,(%s;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),t_a,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi);

}


logical wnga_create(Integer type, Integer ndim, Integer *dims, char *name, Integer *chunk, Integer *g_a)
{
    logical retval;
    printf("%lf,pnga_create,(%ld;%ld;%p;%s;%p;%p)\n",MPI_Wtime(),type,ndim,dims,name,chunk,g_a);
    retval = pnga_create(type, ndim, dims, name, chunk, g_a);
    printf("%lf,/pnga_create,(%ld;%ld;%p;%s;%p;%p)\n",MPI_Wtime(),type,ndim,dims,name,chunk,g_a);
    return retval;

}


logical wnga_create_bin_range(Integer g_bin, Integer g_cnt, Integer g_off, Integer *g_range)
{
    logical retval;
    printf("%lf,pnga_create_bin_range,(%ld;%ld;%ld;%p)\n",MPI_Wtime(),g_bin,g_cnt,g_off,g_range);
    retval = pnga_create_bin_range(g_bin, g_cnt, g_off, g_range);
    printf("%lf,/pnga_create_bin_range,(%ld;%ld;%ld;%p)\n",MPI_Wtime(),g_bin,g_cnt,g_off,g_range);
    return retval;

}


logical wnga_create_config(Integer type, Integer ndim, Integer *dims, char *name, Integer *chunk, Integer p_handle, Integer *g_a)
{
    logical retval;
    printf("%lf,pnga_create_config,(%ld;%ld;%p;%s;%p;%ld;%p)\n",MPI_Wtime(),type,ndim,dims,name,chunk,p_handle,g_a);
    retval = pnga_create_config(type, ndim, dims, name, chunk, p_handle, g_a);
    printf("%lf,/pnga_create_config,(%ld;%ld;%p;%s;%p;%ld;%p)\n",MPI_Wtime(),type,ndim,dims,name,chunk,p_handle,g_a);
    return retval;

}


logical wnga_create_ghosts(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *chunk, Integer *g_a)
{
    logical retval;
    printf("%lf,pnga_create_ghosts,(%ld;%ld;%p;%p;%s;%p;%p)\n",MPI_Wtime(),type,ndim,dims,width,name,chunk,g_a);
    retval = pnga_create_ghosts(type, ndim, dims, width, name, chunk, g_a);
    printf("%lf,/pnga_create_ghosts,(%ld;%ld;%p;%p;%s;%p;%p)\n",MPI_Wtime(),type,ndim,dims,width,name,chunk,g_a);
    return retval;

}


logical wnga_create_ghosts_config(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *chunk, Integer p_handle, Integer *g_a)
{
    logical retval;
    printf("%lf,pnga_create_ghosts_config,(%ld;%ld;%p;%p;%s;%p;%ld;%p)\n",MPI_Wtime(),type,ndim,dims,width,name,chunk,p_handle,g_a);
    retval = pnga_create_ghosts_config(type, ndim, dims, width, name, chunk, p_handle, g_a);
    printf("%lf,/pnga_create_ghosts_config,(%ld;%ld;%p;%p;%s;%p;%ld;%p)\n",MPI_Wtime(),type,ndim,dims,width,name,chunk,p_handle,g_a);
    return retval;

}


logical wnga_create_ghosts_irreg(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *map, Integer *block, Integer *g_a)
{
    logical retval;
    printf("%lf,pnga_create_ghosts_irreg,(%ld;%ld;%p;%p;%s;%p;%p;%p)\n",MPI_Wtime(),type,ndim,dims,width,name,map,block,g_a);
    retval = pnga_create_ghosts_irreg(type, ndim, dims, width, name, map, block, g_a);
    printf("%lf,/pnga_create_ghosts_irreg,(%ld;%ld;%p;%p;%s;%p;%p;%p)\n",MPI_Wtime(),type,ndim,dims,width,name,map,block,g_a);
    return retval;

}


logical wnga_create_ghosts_irreg_config(Integer type, Integer ndim, Integer *dims, Integer *width, char *name, Integer *map, Integer *block, Integer p_handle, Integer *g_a)
{
    logical retval;
    printf("%lf,pnga_create_ghosts_irreg_config,(%ld;%ld;%p;%p;%s;%p;%p;%ld;%p)\n",MPI_Wtime(),type,ndim,dims,width,name,map,block,p_handle,g_a);
    retval = pnga_create_ghosts_irreg_config(type, ndim, dims, width, name, map, block, p_handle, g_a);
    printf("%lf,/pnga_create_ghosts_irreg_config,(%ld;%ld;%p;%p;%s;%p;%p;%ld;%p)\n",MPI_Wtime(),type,ndim,dims,width,name,map,block,p_handle,g_a);
    return retval;

}


Integer wnga_create_handle()
{
    Integer retval;
    printf("%lf,pnga_create_handle,\n",MPI_Wtime());
    retval = pnga_create_handle();
    printf("%lf,/pnga_create_handle,\n",MPI_Wtime());
    return retval;

}


logical wnga_create_irreg(Integer type, Integer ndim, Integer *dims, char *name, Integer *map, Integer *block, Integer *g_a)
{
    logical retval;
    printf("%lf,pnga_create_irreg,(%ld;%ld;%p;%s;%p;%p;%p)\n",MPI_Wtime(),type,ndim,dims,name,map,block,g_a);
    retval = pnga_create_irreg(type, ndim, dims, name, map, block, g_a);
    printf("%lf,/pnga_create_irreg,(%ld;%ld;%p;%s;%p;%p;%p)\n",MPI_Wtime(),type,ndim,dims,name,map,block,g_a);
    return retval;

}


logical wnga_create_irreg_config(Integer type, Integer ndim, Integer *dims, char *name, Integer *map, Integer *block, Integer p_handle, Integer *g_a)
{
    logical retval;
    printf("%lf,pnga_create_irreg_config,(%ld;%ld;%p;%s;%p;%p;%ld;%p)\n",MPI_Wtime(),type,ndim,dims,name,map,block,p_handle,g_a);
    retval = pnga_create_irreg_config(type, ndim, dims, name, map, block, p_handle, g_a);
    printf("%lf,/pnga_create_irreg_config,(%ld;%ld;%p;%s;%p;%p;%ld;%p)\n",MPI_Wtime(),type,ndim,dims,name,map,block,p_handle,g_a);
    return retval;

}


logical wnga_create_mutexes(Integer num)
{
    logical retval;
    printf("%lf,pnga_create_mutexes,(%ld)\n",MPI_Wtime(),num);
    retval = pnga_create_mutexes(num);
    printf("%lf,/pnga_create_mutexes,(%ld)\n",MPI_Wtime(),num);
    return retval;

}


DoublePrecision wnga_ddot_patch_dp(Integer g_a, char *t_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, char *t_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi)
{
    DoublePrecision retval;
    printf("%lf,pnga_ddot_patch_dp,(%ld;%s;%ld;%ld;%ld;%ld;%ld;%s;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,t_a,ailo,aihi,ajlo,ajhi,g_b,t_b,bilo,bihi,bjlo,bjhi);
    retval = pnga_ddot_patch_dp(g_a, t_a, ailo, aihi, ajlo, ajhi, g_b, t_b, bilo, bihi, bjlo, bjhi);
    printf("%lf,/pnga_ddot_patch_dp,(%ld;%s;%ld;%ld;%ld;%ld;%ld;%s;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,t_a,ailo,aihi,ajlo,ajhi,g_b,t_b,bilo,bihi,bjlo,bjhi);
    return retval;

}


logical wnga_destroy(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_destroy,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_destroy(g_a);
    printf("%lf,/pnga_destroy,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


logical wnga_destroy_mutexes()
{
    logical retval;
    printf("%lf,pnga_destroy_mutexes,\n",MPI_Wtime());
    retval = pnga_destroy_mutexes();
    printf("%lf,/pnga_destroy_mutexes,\n",MPI_Wtime());
    return retval;

}


void wnga_diag(Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    printf("%lf,pnga_diag,(%ld;%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_s,g_v,eval);
    pnga_diag(g_a, g_s, g_v, eval);
    printf("%lf,/pnga_diag,(%ld;%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_s,g_v,eval);

}


void wnga_diag_reuse(Integer reuse, Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    printf("%lf,pnga_diag_reuse,(%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime(),reuse,g_a,g_s,g_v,eval);
    pnga_diag_reuse(reuse, g_a, g_s, g_v, eval);
    printf("%lf,/pnga_diag_reuse,(%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime(),reuse,g_a,g_s,g_v,eval);

}


void wnga_diag_seq(Integer g_a, Integer g_s, Integer g_v, DoublePrecision *eval)
{
    printf("%lf,pnga_diag_seq,(%ld;%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_s,g_v,eval);
    pnga_diag_seq(g_a, g_s, g_v, eval);
    printf("%lf,/pnga_diag_seq,(%ld;%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_s,g_v,eval);

}


void wnga_diag_std(Integer g_a, Integer g_v, DoublePrecision *eval)
{
    printf("%lf,pnga_diag_std,(%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_v,eval);
    pnga_diag_std(g_a, g_v, eval);
    printf("%lf,/pnga_diag_std,(%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_v,eval);

}


void wnga_diag_std_seq(Integer g_a, Integer g_v, DoublePrecision *eval)
{
    printf("%lf,pnga_diag_std_seq,(%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_v,eval);
    pnga_diag_std_seq(g_a, g_v, eval);
    printf("%lf,/pnga_diag_std_seq,(%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_v,eval);

}


void wnga_distribution(Integer g_a, Integer proc, Integer *lo, Integer *hi)
{
    printf("%lf,pnga_distribution,(%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,proc,lo,hi);
    pnga_distribution(g_a, proc, lo, hi);
    printf("%lf,/pnga_distribution,(%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,proc,lo,hi);

}


void wnga_dot(int type, Integer g_a, Integer g_b, void *value)
{
    printf("%lf,pnga_dot,(%d;%ld;%ld;%p)\n",MPI_Wtime(),type,g_a,g_b,value);
    pnga_dot(type, g_a, g_b, value);
    printf("%lf,/pnga_dot,(%d;%ld;%ld;%p)\n",MPI_Wtime(),type,g_a,g_b,value);

}


void wnga_dot_patch(Integer g_a, char *t_a, Integer *alo, Integer *ahi, Integer g_b, char *t_b, Integer *blo, Integer *bhi, void *retval)
{
    printf("%lf,pnga_dot_patch,(%ld;%s;%p;%p;%ld;%s;%p;%p;%p)\n",MPI_Wtime(),g_a,t_a,alo,ahi,g_b,t_b,blo,bhi,retval);
    pnga_dot_patch(g_a, t_a, alo, ahi, g_b, t_b, blo, bhi, retval);
    printf("%lf,/pnga_dot_patch,(%ld;%s;%p;%p;%ld;%s;%p;%p;%p)\n",MPI_Wtime(),g_a,t_a,alo,ahi,g_b,t_b,blo,bhi,retval);

}


logical wnga_duplicate(Integer g_a, Integer *g_b, char *array_name)
{
    logical retval;
    printf("%lf,pnga_duplicate,(%ld;%p;%s)\n",MPI_Wtime(),g_a,g_b,array_name);
    retval = pnga_duplicate(g_a, g_b, array_name);
    printf("%lf,/pnga_duplicate,(%ld;%p;%s)\n",MPI_Wtime(),g_a,g_b,array_name);
    return retval;

}


void wnga_elem_divide(Integer g_a, Integer g_b, Integer g_c)
{
    printf("%lf,pnga_elem_divide,(%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_c);
    pnga_elem_divide(g_a, g_b, g_c);
    printf("%lf,/pnga_elem_divide,(%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_c);

}


void wnga_elem_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    printf("%lf,pnga_elem_divide_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);
    pnga_elem_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    printf("%lf,/pnga_elem_divide_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);

}


void wnga_elem_maximum(Integer g_a, Integer g_b, Integer g_c)
{
    printf("%lf,pnga_elem_maximum,(%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_c);
    pnga_elem_maximum(g_a, g_b, g_c);
    printf("%lf,/pnga_elem_maximum,(%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_c);

}


void wnga_elem_maximum_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    printf("%lf,pnga_elem_maximum_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);
    pnga_elem_maximum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    printf("%lf,/pnga_elem_maximum_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);

}


void wnga_elem_minimum(Integer g_a, Integer g_b, Integer g_c)
{
    printf("%lf,pnga_elem_minimum,(%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_c);
    pnga_elem_minimum(g_a, g_b, g_c);
    printf("%lf,/pnga_elem_minimum,(%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_c);

}


void wnga_elem_minimum_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    printf("%lf,pnga_elem_minimum_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);
    pnga_elem_minimum_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    printf("%lf,/pnga_elem_minimum_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);

}


void wnga_elem_multiply(Integer g_a, Integer g_b, Integer g_c)
{
    printf("%lf,pnga_elem_multiply,(%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_c);
    pnga_elem_multiply(g_a, g_b, g_c);
    printf("%lf,/pnga_elem_multiply,(%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_c);

}


void wnga_elem_multiply_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    printf("%lf,pnga_elem_multiply_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);
    pnga_elem_multiply_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    printf("%lf,/pnga_elem_multiply_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);

}


void wnga_elem_step_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    printf("%lf,pnga_elem_step_divide_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);
    pnga_elem_step_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    printf("%lf,/pnga_elem_step_divide_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);

}


void wnga_elem_stepb_divide_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    printf("%lf,pnga_elem_stepb_divide_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);
    pnga_elem_stepb_divide_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    printf("%lf,/pnga_elem_stepb_divide_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);

}


void wnga_error(char *string, Integer icode)
{
    printf("%lf,pnga_error,(%s;%ld)\n",MPI_Wtime(),string,icode);
    pnga_error(string, icode);
    printf("%lf,/pnga_error,(%s;%ld)\n",MPI_Wtime(),string,icode);

}


void wnga_fence()
{
    printf("%lf,pnga_fence,\n",MPI_Wtime());
    pnga_fence();
    printf("%lf,/pnga_fence,\n",MPI_Wtime());

}


void wnga_fill(Integer g_a, void *val)
{
    printf("%lf,pnga_fill,(%ld;%p)\n",MPI_Wtime(),g_a,val);
    pnga_fill(g_a, val);
    printf("%lf,/pnga_fill,(%ld;%p)\n",MPI_Wtime(),g_a,val);

}


void wnga_fill_patch(Integer g_a, Integer *lo, Integer *hi, void *val)
{
    printf("%lf,pnga_fill_patch,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,val);
    pnga_fill_patch(g_a, lo, hi, val);
    printf("%lf,/pnga_fill_patch,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,val);

}


void wnga_gather(Integer g_a, void *v, Integer subscript[], Integer nv)
{
    printf("%lf,pnga_gather,(%ld;%p;%p;%ld)\n",MPI_Wtime(),g_a,v,subscript,nv);
    pnga_gather(g_a, v, subscript, nv);
    printf("%lf,/pnga_gather,(%ld;%p;%p;%ld)\n",MPI_Wtime(),g_a,v,subscript,nv);

}


void wnga_gather2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv)
{
    printf("%lf,pnga_gather2d,(%ld;%p;%p;%p;%ld)\n",MPI_Wtime(),g_a,v,i,j,nv);
    pnga_gather2d(g_a, v, i, j, nv);
    printf("%lf,/pnga_gather2d,(%ld;%p;%p;%p;%ld)\n",MPI_Wtime(),g_a,v,i,j,nv);

}


void wnga_get(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    printf("%lf,pnga_get,(%ld;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld);
    pnga_get(g_a, lo, hi, buf, ld);
    printf("%lf,/pnga_get,(%ld;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld);

}


void wnga_get_block_info(Integer g_a, Integer *num_blocks, Integer *block_dims)
{
    printf("%lf,pnga_get_block_info,(%ld;%p;%p)\n",MPI_Wtime(),g_a,num_blocks,block_dims);
    pnga_get_block_info(g_a, num_blocks, block_dims);
    printf("%lf,/pnga_get_block_info,(%ld;%p;%p)\n",MPI_Wtime(),g_a,num_blocks,block_dims);

}


logical wnga_get_debug()
{
    logical retval;
    printf("%lf,pnga_get_debug,\n",MPI_Wtime());
    retval = pnga_get_debug();
    printf("%lf,/pnga_get_debug,\n",MPI_Wtime());
    return retval;

}


void wnga_get_diag(Integer g_a, Integer g_v)
{
    printf("%lf,pnga_get_diag,(%ld;%ld)\n",MPI_Wtime(),g_a,g_v);
    pnga_get_diag(g_a, g_v);
    printf("%lf,/pnga_get_diag,(%ld;%ld)\n",MPI_Wtime(),g_a,g_v);

}


Integer wnga_get_dimension(Integer g_a)
{
    Integer retval;
    printf("%lf,pnga_get_dimension,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_get_dimension(g_a);
    printf("%lf,/pnga_get_dimension,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


void wnga_get_ghost_block(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    printf("%lf,pnga_get_ghost_block,(%ld;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld);
    pnga_get_ghost_block(g_a, lo, hi, buf, ld);
    printf("%lf,/pnga_get_ghost_block,(%ld;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld);

}


Integer wnga_get_pgroup(Integer g_a)
{
    Integer retval;
    printf("%lf,pnga_get_pgroup,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_get_pgroup(g_a);
    printf("%lf,/pnga_get_pgroup,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


Integer wnga_get_pgroup_size(Integer grp_id)
{
    Integer retval;
    printf("%lf,pnga_get_pgroup_size,(%ld)\n",MPI_Wtime(),grp_id);
    retval = pnga_get_pgroup_size(grp_id);
    printf("%lf,/pnga_get_pgroup_size,(%ld)\n",MPI_Wtime(),grp_id);
    return retval;

}


void wnga_get_proc_grid(Integer g_a, Integer *dims)
{
    printf("%lf,pnga_get_proc_grid,(%ld;%p)\n",MPI_Wtime(),g_a,dims);
    pnga_get_proc_grid(g_a, dims);
    printf("%lf,/pnga_get_proc_grid,(%ld;%p)\n",MPI_Wtime(),g_a,dims);

}


void wnga_get_proc_index(Integer g_a, Integer iproc, Integer *index)
{
    printf("%lf,pnga_get_proc_index,(%ld;%ld;%p)\n",MPI_Wtime(),g_a,iproc,index);
    pnga_get_proc_index(g_a, iproc, index);
    printf("%lf,/pnga_get_proc_index,(%ld;%ld;%p)\n",MPI_Wtime(),g_a,iproc,index);

}


void wnga_ghost_barrier()
{
    printf("%lf,pnga_ghost_barrier,\n",MPI_Wtime());
    pnga_ghost_barrier();
    printf("%lf,/pnga_ghost_barrier,\n",MPI_Wtime());

}


void wnga_gop(Integer type, void *x, Integer n, char *op)
{
    printf("%lf,pnga_gop,(%ld;%p;%ld;%s)\n",MPI_Wtime(),type,x,n,op);
    pnga_gop(type, x, n, op);
    printf("%lf,/pnga_gop,(%ld;%p;%ld;%s)\n",MPI_Wtime(),type,x,n,op);

}


logical wnga_has_ghosts(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_has_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_has_ghosts(g_a);
    printf("%lf,/pnga_has_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


void wnga_init_fence()
{
    printf("%lf,pnga_init_fence,\n",MPI_Wtime());
    pnga_init_fence();
    printf("%lf,/pnga_init_fence,\n",MPI_Wtime());

}


void wnga_initialize_ltd(Integer limit)
{
    printf("%lf,pnga_initialize_ltd,(%ld)\n",MPI_Wtime(),limit);
    pnga_initialize_ltd(limit);
    printf("%lf,/pnga_initialize_ltd,(%ld)\n",MPI_Wtime(),limit);

}


void wnga_inquire(Integer g_a, Integer *type, Integer *ndim, Integer *dims)
{
    printf("%lf,pnga_inquire,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,type,ndim,dims);
    pnga_inquire(g_a, type, ndim, dims);
    printf("%lf,/pnga_inquire,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,type,ndim,dims);

}


Integer wnga_inquire_memory()
{
    Integer retval;
    printf("%lf,pnga_inquire_memory,\n",MPI_Wtime());
    retval = pnga_inquire_memory();
    printf("%lf,/pnga_inquire_memory,\n",MPI_Wtime());
    return retval;

}


void wnga_inquire_name(Integer g_a, char **array_name)
{
    printf("%lf,pnga_inquire_name,(%ld;%s)\n",MPI_Wtime(),g_a,array_name);
    pnga_inquire_name(g_a, array_name);
    printf("%lf,/pnga_inquire_name,(%ld;%s)\n",MPI_Wtime(),g_a,array_name);

}


void wnga_inquire_type(Integer g_a, Integer *type)
{
    printf("%lf,pnga_inquire_type,(%ld;%p)\n",MPI_Wtime(),g_a,type);
    pnga_inquire_type(g_a, type);
    printf("%lf,/pnga_inquire_type,(%ld;%p)\n",MPI_Wtime(),g_a,type);

}


logical wnga_is_mirrored(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_is_mirrored,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_is_mirrored(g_a);
    printf("%lf,/pnga_is_mirrored,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


void wnga_list_nodeid(Integer *list, Integer nprocs)
{
    printf("%lf,pnga_list_nodeid,(%p;%ld)\n",MPI_Wtime(),list,nprocs);
    pnga_list_nodeid(list, nprocs);
    printf("%lf,/pnga_list_nodeid,(%p;%ld)\n",MPI_Wtime(),list,nprocs);

}


Integer wnga_llt_solve(Integer g_a, Integer g_b)
{
    Integer retval;
    printf("%lf,pnga_llt_solve,(%ld;%ld)\n",MPI_Wtime(),g_a,g_b);
    retval = pnga_llt_solve(g_a, g_b);
    printf("%lf,/pnga_llt_solve,(%ld;%ld)\n",MPI_Wtime(),g_a,g_b);
    return retval;

}


logical wnga_locate(Integer g_a, Integer *subscript, Integer *owner)
{
    logical retval;
    printf("%lf,pnga_locate,(%ld;%p;%p)\n",MPI_Wtime(),g_a,subscript,owner);
    retval = pnga_locate(g_a, subscript, owner);
    printf("%lf,/pnga_locate,(%ld;%p;%p)\n",MPI_Wtime(),g_a,subscript,owner);
    return retval;

}


logical wnga_locate_nnodes(Integer g_a, Integer *lo, Integer *hi, Integer *np)
{
    logical retval;
    printf("%lf,pnga_locate_nnodes,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,np);
    retval = pnga_locate_nnodes(g_a, lo, hi, np);
    printf("%lf,/pnga_locate_nnodes,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,np);
    return retval;

}


Integer wnga_locate_num_blocks(Integer g_a, Integer *lo, Integer *hi)
{
    Integer retval;
    printf("%lf,pnga_locate_num_blocks,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);
    retval = pnga_locate_num_blocks(g_a, lo, hi);
    printf("%lf,/pnga_locate_num_blocks,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);
    return retval;

}


logical wnga_locate_region(Integer g_a, Integer *lo, Integer *hi, Integer *map, Integer *proclist, Integer *np)
{
    logical retval;
    printf("%lf,pnga_locate_region,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,map,proclist,np);
    retval = pnga_locate_region(g_a, lo, hi, map, proclist, np);
    printf("%lf,/pnga_locate_region,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,map,proclist,np);
    return retval;

}


void wnga_lock(Integer mutex)
{
    printf("%lf,pnga_lock,(%ld)\n",MPI_Wtime(),mutex);
    pnga_lock(mutex);
    printf("%lf,/pnga_lock,(%ld)\n",MPI_Wtime(),mutex);

}


void wnga_lu_solve(char *tran, Integer g_a, Integer g_b)
{
    printf("%lf,pnga_lu_solve,(%s;%ld;%ld)\n",MPI_Wtime(),tran,g_a,g_b);
    pnga_lu_solve(tran, g_a, g_b);
    printf("%lf,/pnga_lu_solve,(%s;%ld;%ld)\n",MPI_Wtime(),tran,g_a,g_b);

}


void wnga_lu_solve_alt(Integer tran, Integer g_a, Integer g_b)
{
    printf("%lf,pnga_lu_solve_alt,(%ld;%ld;%ld)\n",MPI_Wtime(),tran,g_a,g_b);
    pnga_lu_solve_alt(tran, g_a, g_b);
    printf("%lf,/pnga_lu_solve_alt,(%ld;%ld;%ld)\n",MPI_Wtime(),tran,g_a,g_b);

}


void wnga_lu_solve_seq(char *trans, Integer g_a, Integer g_b)
{
    printf("%lf,pnga_lu_solve_seq,(%s;%ld;%ld)\n",MPI_Wtime(),trans,g_a,g_b);
    pnga_lu_solve_seq(trans, g_a, g_b);
    printf("%lf,/pnga_lu_solve_seq,(%s;%ld;%ld)\n",MPI_Wtime(),trans,g_a,g_b);

}


void wnga_mask_sync(Integer begin, Integer end)
{
    printf("%lf,pnga_mask_sync,(%ld;%ld)\n",MPI_Wtime(),begin,end);
    pnga_mask_sync(begin, end);
    printf("%lf,/pnga_mask_sync,(%ld;%ld)\n",MPI_Wtime(),begin,end);

}


void wnga_matmul(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi, Integer g_c, Integer cilo, Integer cihi, Integer cjlo, Integer cjhi)
{
    printf("%lf,pnga_matmul,(%s;%s;%p;%p;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),transa,transb,alpha,beta,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi,g_c,cilo,cihi,cjlo,cjhi);
    pnga_matmul(transa, transb, alpha, beta, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi, g_c, cilo, cihi, cjlo, cjhi);
    printf("%lf,/pnga_matmul,(%s;%s;%p;%p;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),transa,transb,alpha,beta,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi,g_c,cilo,cihi,cjlo,cjhi);

}


void wnga_matmul_mirrored(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer ailo, Integer aihi, Integer ajlo, Integer ajhi, Integer g_b, Integer bilo, Integer bihi, Integer bjlo, Integer bjhi, Integer g_c, Integer cilo, Integer cihi, Integer cjlo, Integer cjhi)
{
    printf("%lf,pnga_matmul_mirrored,(%s;%s;%p;%p;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),transa,transb,alpha,beta,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi,g_c,cilo,cihi,cjlo,cjhi);
    pnga_matmul_mirrored(transa, transb, alpha, beta, g_a, ailo, aihi, ajlo, ajhi, g_b, bilo, bihi, bjlo, bjhi, g_c, cilo, cihi, cjlo, cjhi);
    printf("%lf,/pnga_matmul_mirrored,(%s;%s;%p;%p;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),transa,transb,alpha,beta,g_a,ailo,aihi,ajlo,ajhi,g_b,bilo,bihi,bjlo,bjhi,g_c,cilo,cihi,cjlo,cjhi);

}


void wnga_matmul_patch(char *transa, char *transb, void *alpha, void *beta, Integer g_a, Integer alo[], Integer ahi[], Integer g_b, Integer blo[], Integer bhi[], Integer g_c, Integer clo[], Integer chi[])
{
    printf("%lf,pnga_matmul_patch,(%s;%s;%p;%p;%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),transa,transb,alpha,beta,g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);
    pnga_matmul_patch(transa, transb, alpha, beta, g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    printf("%lf,/pnga_matmul_patch,(%s;%s;%p;%p;%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),transa,transb,alpha,beta,g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);

}


void wnga_median(Integer g_a, Integer g_b, Integer g_c, Integer g_m)
{
    printf("%lf,pnga_median,(%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_c,g_m);
    pnga_median(g_a, g_b, g_c, g_m);
    printf("%lf,/pnga_median,(%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_c,g_m);

}


void wnga_median_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi, Integer g_m, Integer *mlo, Integer *mhi)
{
    printf("%lf,pnga_median_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi,g_m,mlo,mhi);
    pnga_median_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi, g_m, mlo, mhi);
    printf("%lf,/pnga_median_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi,g_m,mlo,mhi);

}


Integer wnga_memory_avail()
{
    Integer retval;
    printf("%lf,pnga_memory_avail,\n",MPI_Wtime());
    retval = pnga_memory_avail();
    printf("%lf,/pnga_memory_avail,\n",MPI_Wtime());
    return retval;

}


Integer wnga_memory_avail_type(Integer datatype)
{
    Integer retval;
    printf("%lf,pnga_memory_avail_type,(%ld)\n",MPI_Wtime(),datatype);
    retval = pnga_memory_avail_type(datatype);
    printf("%lf,/pnga_memory_avail_type,(%ld)\n",MPI_Wtime(),datatype);
    return retval;

}


logical wnga_memory_limited()
{
    logical retval;
    printf("%lf,pnga_memory_limited,\n",MPI_Wtime());
    retval = pnga_memory_limited();
    printf("%lf,/pnga_memory_limited,\n",MPI_Wtime());
    return retval;

}


void wnga_merge_distr_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi)
{
    printf("%lf,pnga_merge_distr_patch,(%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi);
    pnga_merge_distr_patch(g_a, alo, ahi, g_b, blo, bhi);
    printf("%lf,/pnga_merge_distr_patch,(%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi);

}


void wnga_merge_mirrored(Integer g_a)
{
    printf("%lf,pnga_merge_mirrored,(%ld)\n",MPI_Wtime(),g_a);
    pnga_merge_mirrored(g_a);
    printf("%lf,/pnga_merge_mirrored,(%ld)\n",MPI_Wtime(),g_a);

}


void wnga_msg_brdcst(Integer type, void *buffer, Integer len, Integer root)
{
    printf("%lf,pnga_msg_brdcst,(%ld;%p;%ld;%ld)\n",MPI_Wtime(),type,buffer,len,root);
    pnga_msg_brdcst(type, buffer, len, root);
    printf("%lf,/pnga_msg_brdcst,(%ld;%p;%ld;%ld)\n",MPI_Wtime(),type,buffer,len,root);

}


void wnga_msg_pgroup_sync(Integer grp_id)
{
    printf("%lf,pnga_msg_pgroup_sync,(%ld)\n",MPI_Wtime(),grp_id);
    pnga_msg_pgroup_sync(grp_id);
    printf("%lf,/pnga_msg_pgroup_sync,(%ld)\n",MPI_Wtime(),grp_id);

}


void wnga_msg_sync()
{
    printf("%lf,pnga_msg_sync,\n",MPI_Wtime());
    pnga_msg_sync();
    printf("%lf,/pnga_msg_sync,\n",MPI_Wtime());

}


void wnga_nbacc(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer *nbhndl)
{
    printf("%lf,pnga_nbacc,(%ld;%p;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld,alpha,nbhndl);
    pnga_nbacc(g_a, lo, hi, buf, ld, alpha, nbhndl);
    printf("%lf,/pnga_nbacc,(%ld;%p;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld,alpha,nbhndl);

}


void wnga_nbget(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
    printf("%lf,pnga_nbget,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld,nbhandle);
    pnga_nbget(g_a, lo, hi, buf, ld, nbhandle);
    printf("%lf,/pnga_nbget,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld,nbhandle);

}


void wnga_nbget_ghost_dir(Integer g_a, Integer *mask, Integer *nbhandle)
{
    printf("%lf,pnga_nbget_ghost_dir,(%ld;%p;%p)\n",MPI_Wtime(),g_a,mask,nbhandle);
    pnga_nbget_ghost_dir(g_a, mask, nbhandle);
    printf("%lf,/pnga_nbget_ghost_dir,(%ld;%p;%p)\n",MPI_Wtime(),g_a,mask,nbhandle);

}


void wnga_nblock(Integer g_a, Integer *nblock)
{
    printf("%lf,pnga_nblock,(%ld;%p)\n",MPI_Wtime(),g_a,nblock);
    pnga_nblock(g_a, nblock);
    printf("%lf,/pnga_nblock,(%ld;%p)\n",MPI_Wtime(),g_a,nblock);

}


void wnga_nbput(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, Integer *nbhandle)
{
    printf("%lf,pnga_nbput,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld,nbhandle);
    pnga_nbput(g_a, lo, hi, buf, ld, nbhandle);
    printf("%lf,/pnga_nbput,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld,nbhandle);

}


Integer wnga_nbtest(Integer *nbhandle)
{
    Integer retval;
    printf("%lf,pnga_nbtest,(%p)\n",MPI_Wtime(),nbhandle);
    retval = pnga_nbtest(nbhandle);
    printf("%lf,/pnga_nbtest,(%p)\n",MPI_Wtime(),nbhandle);
    return retval;

}


void wnga_nbwait(Integer *nbhandle)
{
    printf("%lf,pnga_nbwait,(%p)\n",MPI_Wtime(),nbhandle);
    pnga_nbwait(nbhandle);
    printf("%lf,/pnga_nbwait,(%p)\n",MPI_Wtime(),nbhandle);

}


Integer wnga_ndim(Integer g_a)
{
    Integer retval;
    printf("%lf,pnga_ndim,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_ndim(g_a);
    printf("%lf,/pnga_ndim,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


Integer wnga_nnodes()
{
    Integer retval;
    printf("%lf,pnga_nnodes,\n",MPI_Wtime());
    retval = pnga_nnodes();
    printf("%lf,/pnga_nnodes,\n",MPI_Wtime());
    return retval;

}


Integer wnga_nodeid()
{
    Integer retval;
    printf("%lf,pnga_nodeid,\n",MPI_Wtime());
    retval = pnga_nodeid();
    printf("%lf,/pnga_nodeid,\n",MPI_Wtime());
    return retval;

}


void wnga_norm1(Integer g_a, double *nm)
{
    printf("%lf,pnga_norm1,(%ld;%p)\n",MPI_Wtime(),g_a,nm);
    pnga_norm1(g_a, nm);
    printf("%lf,/pnga_norm1,(%ld;%p)\n",MPI_Wtime(),g_a,nm);

}


void wnga_norm_infinity(Integer g_a, double *nm)
{
    printf("%lf,pnga_norm_infinity,(%ld;%p)\n",MPI_Wtime(),g_a,nm);
    pnga_norm_infinity(g_a, nm);
    printf("%lf,/pnga_norm_infinity,(%ld;%p)\n",MPI_Wtime(),g_a,nm);

}


void wnga_pack(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer *icount)
{
    printf("%lf,pnga_pack,(%ld;%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_b,g_sbit,lo,hi,icount);
    pnga_pack(g_a, g_b, g_sbit, lo, hi, icount);
    printf("%lf,/pnga_pack,(%ld;%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_b,g_sbit,lo,hi,icount);

}


void wnga_patch_enum(Integer g_a, Integer lo, Integer hi, void *start, void *stride)
{
    printf("%lf,pnga_patch_enum,(%ld;%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,start,stride);
    pnga_patch_enum(g_a, lo, hi, start, stride);
    printf("%lf,/pnga_patch_enum,(%ld;%ld;%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,start,stride);

}


logical wnga_patch_intersect(Integer *lo, Integer *hi, Integer *lop, Integer *hip, Integer ndim)
{
    logical retval;
    printf("%lf,pnga_patch_intersect,(%p;%p;%p;%p;%ld)\n",MPI_Wtime(),lo,hi,lop,hip,ndim);
    retval = pnga_patch_intersect(lo, hi, lop, hip, ndim);
    printf("%lf,/pnga_patch_intersect,(%p;%p;%p;%p;%ld)\n",MPI_Wtime(),lo,hi,lop,hip,ndim);
    return retval;

}


void wnga_periodic(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld, void *alpha, Integer op_code)
{
    printf("%lf,pnga_periodic,(%ld;%p;%p;%p;%p;%p;%ld)\n",MPI_Wtime(),g_a,lo,hi,buf,ld,alpha,op_code);
    pnga_periodic(g_a, lo, hi, buf, ld, alpha, op_code);
    printf("%lf,/pnga_periodic,(%ld;%p;%p;%p;%p;%p;%ld)\n",MPI_Wtime(),g_a,lo,hi,buf,ld,alpha,op_code);

}


Integer wnga_pgroup_absolute_id(Integer grp, Integer pid)
{
    Integer retval;
    printf("%lf,pnga_pgroup_absolute_id,(%ld;%ld)\n",MPI_Wtime(),grp,pid);
    retval = pnga_pgroup_absolute_id(grp, pid);
    printf("%lf,/pnga_pgroup_absolute_id,(%ld;%ld)\n",MPI_Wtime(),grp,pid);
    return retval;

}


void wnga_pgroup_brdcst(Integer grp_id, Integer type, void *buf, Integer len, Integer originator)
{
    printf("%lf,pnga_pgroup_brdcst,(%ld;%ld;%p;%ld;%ld)\n",MPI_Wtime(),grp_id,type,buf,len,originator);
    pnga_pgroup_brdcst(grp_id, type, buf, len, originator);
    printf("%lf,/pnga_pgroup_brdcst,(%ld;%ld;%p;%ld;%ld)\n",MPI_Wtime(),grp_id,type,buf,len,originator);

}


Integer wnga_pgroup_create(Integer *list, Integer count)
{
    Integer retval;
    printf("%lf,pnga_pgroup_create,(%p;%ld)\n",MPI_Wtime(),list,count);
    retval = pnga_pgroup_create(list, count);
    printf("%lf,/pnga_pgroup_create,(%p;%ld)\n",MPI_Wtime(),list,count);
    return retval;

}


logical wnga_pgroup_destroy(Integer grp)
{
    logical retval;
    printf("%lf,pnga_pgroup_destroy,(%ld)\n",MPI_Wtime(),grp);
    retval = pnga_pgroup_destroy(grp);
    printf("%lf,/pnga_pgroup_destroy,(%ld)\n",MPI_Wtime(),grp);
    return retval;

}


Integer wnga_pgroup_get_default()
{
    Integer retval;
    printf("%lf,pnga_pgroup_get_default,\n",MPI_Wtime());
    retval = pnga_pgroup_get_default();
    printf("%lf,/pnga_pgroup_get_default,\n",MPI_Wtime());
    return retval;

}


Integer wnga_pgroup_get_mirror()
{
    Integer retval;
    printf("%lf,pnga_pgroup_get_mirror,\n",MPI_Wtime());
    retval = pnga_pgroup_get_mirror();
    printf("%lf,/pnga_pgroup_get_mirror,\n",MPI_Wtime());
    return retval;

}


Integer wnga_pgroup_get_world()
{
    Integer retval;
    printf("%lf,pnga_pgroup_get_world,\n",MPI_Wtime());
    retval = pnga_pgroup_get_world();
    printf("%lf,/pnga_pgroup_get_world,\n",MPI_Wtime());
    return retval;

}


void wnga_pgroup_gop(Integer p_grp, Integer type, void *x, Integer n, char *op)
{
    printf("%lf,pnga_pgroup_gop,(%ld;%ld;%p;%ld;%s)\n",MPI_Wtime(),p_grp,type,x,n,op);
    pnga_pgroup_gop(p_grp, type, x, n, op);
    printf("%lf,/pnga_pgroup_gop,(%ld;%ld;%p;%ld;%s)\n",MPI_Wtime(),p_grp,type,x,n,op);

}


Integer wnga_pgroup_nnodes(Integer grp)
{
    Integer retval;
    printf("%lf,pnga_pgroup_nnodes,(%ld)\n",MPI_Wtime(),grp);
    retval = pnga_pgroup_nnodes(grp);
    printf("%lf,/pnga_pgroup_nnodes,(%ld)\n",MPI_Wtime(),grp);
    return retval;

}


Integer wnga_pgroup_nodeid(Integer grp)
{
    Integer retval;
    printf("%lf,pnga_pgroup_nodeid,(%ld)\n",MPI_Wtime(),grp);
    retval = pnga_pgroup_nodeid(grp);
    printf("%lf,/pnga_pgroup_nodeid,(%ld)\n",MPI_Wtime(),grp);
    return retval;

}


void wnga_pgroup_set_default(Integer grp)
{
    printf("%lf,pnga_pgroup_set_default,(%ld)\n",MPI_Wtime(),grp);
    pnga_pgroup_set_default(grp);
    printf("%lf,/pnga_pgroup_set_default,(%ld)\n",MPI_Wtime(),grp);

}


Integer wnga_pgroup_split(Integer grp, Integer grp_num)
{
    Integer retval;
    printf("%lf,pnga_pgroup_split,(%ld;%ld)\n",MPI_Wtime(),grp,grp_num);
    retval = pnga_pgroup_split(grp, grp_num);
    printf("%lf,/pnga_pgroup_split,(%ld;%ld)\n",MPI_Wtime(),grp,grp_num);
    return retval;

}


Integer wnga_pgroup_split_irreg(Integer grp, Integer mycolor)
{
    Integer retval;
    printf("%lf,pnga_pgroup_split_irreg,(%ld;%ld)\n",MPI_Wtime(),grp,mycolor);
    retval = pnga_pgroup_split_irreg(grp, mycolor);
    printf("%lf,/pnga_pgroup_split_irreg,(%ld;%ld)\n",MPI_Wtime(),grp,mycolor);
    return retval;

}


void wnga_pgroup_sync(Integer grp_id)
{
    printf("%lf,pnga_pgroup_sync,(%ld)\n",MPI_Wtime(),grp_id);
    pnga_pgroup_sync(grp_id);
    printf("%lf,/pnga_pgroup_sync,(%ld)\n",MPI_Wtime(),grp_id);

}


void wnga_print(Integer g_a)
{
    printf("%lf,pnga_print,(%ld)\n",MPI_Wtime(),g_a);
    pnga_print(g_a);
    printf("%lf,/pnga_print,(%ld)\n",MPI_Wtime(),g_a);

}


void wnga_print_distribution(int fstyle, Integer g_a)
{
    printf("%lf,pnga_print_distribution,(%d;%ld)\n",MPI_Wtime(),fstyle,g_a);
    pnga_print_distribution(fstyle, g_a);
    printf("%lf,/pnga_print_distribution,(%d;%ld)\n",MPI_Wtime(),fstyle,g_a);

}


void wnga_print_file(FILE *file, Integer g_a)
{
    printf("%lf,pnga_print_file,(%p;%ld)\n",MPI_Wtime(),file,g_a);
    pnga_print_file(file, g_a);
    printf("%lf,/pnga_print_file,(%p;%ld)\n",MPI_Wtime(),file,g_a);

}


void wnga_print_patch(Integer g_a, Integer *lo, Integer *hi, Integer pretty)
{
    printf("%lf,pnga_print_patch,(%ld;%p;%p;%ld)\n",MPI_Wtime(),g_a,lo,hi,pretty);
    pnga_print_patch(g_a, lo, hi, pretty);
    printf("%lf,/pnga_print_patch,(%ld;%p;%p;%ld)\n",MPI_Wtime(),g_a,lo,hi,pretty);

}


void wnga_print_patch2d(Integer g_a, Integer ilo, Integer ihi, Integer jlo, Integer jhi, Integer pretty)
{
    printf("%lf,pnga_print_patch2d,(%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,ilo,ihi,jlo,jhi,pretty);
    pnga_print_patch2d(g_a, ilo, ihi, jlo, jhi, pretty);
    printf("%lf,/pnga_print_patch2d,(%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,ilo,ihi,jlo,jhi,pretty);

}


void wnga_print_patch_file(FILE *file, Integer g_a, Integer *lo, Integer *hi, Integer pretty)
{
    printf("%lf,pnga_print_patch_file,(%p;%ld;%p;%p;%ld)\n",MPI_Wtime(),file,g_a,lo,hi,pretty);
    pnga_print_patch_file(file, g_a, lo, hi, pretty);
    printf("%lf,/pnga_print_patch_file,(%p;%ld;%p;%p;%ld)\n",MPI_Wtime(),file,g_a,lo,hi,pretty);

}


void wnga_print_patch_file2d(FILE *file, Integer g_a, Integer ilo, Integer ihi, Integer jlo, Integer jhi, Integer pretty)
{
    printf("%lf,pnga_print_patch_file2d,(%p;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),file,g_a,ilo,ihi,jlo,jhi,pretty);
    pnga_print_patch_file2d(file, g_a, ilo, ihi, jlo, jhi, pretty);
    printf("%lf,/pnga_print_patch_file2d,(%p;%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),file,g_a,ilo,ihi,jlo,jhi,pretty);

}


void wnga_print_stats()
{
    printf("%lf,pnga_print_stats,\n",MPI_Wtime());
    pnga_print_stats();
    printf("%lf,/pnga_print_stats,\n",MPI_Wtime());

}


void wnga_proc_topology(Integer g_a, Integer proc, Integer *subscript)
{
    printf("%lf,pnga_proc_topology,(%ld;%ld;%p)\n",MPI_Wtime(),g_a,proc,subscript);
    pnga_proc_topology(g_a, proc, subscript);
    printf("%lf,/pnga_proc_topology,(%ld;%ld;%p)\n",MPI_Wtime(),g_a,proc,subscript);

}


void wnga_put(Integer g_a, Integer *lo, Integer *hi, void *buf, Integer *ld)
{
    printf("%lf,pnga_put,(%ld;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld);
    pnga_put(g_a, lo, hi, buf, ld);
    printf("%lf,/pnga_put,(%ld;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,buf,ld);

}


void wnga_randomize(Integer g_a, void *val)
{
    printf("%lf,pnga_randomize,(%ld;%p)\n",MPI_Wtime(),g_a,val);
    pnga_randomize(g_a, val);
    printf("%lf,/pnga_randomize,(%ld;%p)\n",MPI_Wtime(),g_a,val);

}


Integer wnga_read_inc(Integer g_a, Integer *subscript, Integer inc)
{
    Integer retval;
    printf("%lf,pnga_read_inc,(%ld;%p;%ld)\n",MPI_Wtime(),g_a,subscript,inc);
    retval = pnga_read_inc(g_a, subscript, inc);
    printf("%lf,/pnga_read_inc,(%ld;%p;%ld)\n",MPI_Wtime(),g_a,subscript,inc);
    return retval;

}


void wnga_recip(Integer g_a)
{
    printf("%lf,pnga_recip,(%ld)\n",MPI_Wtime(),g_a);
    pnga_recip(g_a);
    printf("%lf,/pnga_recip,(%ld)\n",MPI_Wtime(),g_a);

}


void wnga_recip_patch(Integer g_a, Integer *lo, Integer *hi)
{
    printf("%lf,pnga_recip_patch,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);
    pnga_recip_patch(g_a, lo, hi);
    printf("%lf,/pnga_recip_patch,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);

}


void wnga_release(Integer g_a, Integer *lo, Integer *hi)
{
    printf("%lf,pnga_release,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);
    pnga_release(g_a, lo, hi);
    printf("%lf,/pnga_release,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);

}


void wnga_release_block(Integer g_a, Integer iblock)
{
    printf("%lf,pnga_release_block,(%ld;%ld)\n",MPI_Wtime(),g_a,iblock);
    pnga_release_block(g_a, iblock);
    printf("%lf,/pnga_release_block,(%ld;%ld)\n",MPI_Wtime(),g_a,iblock);

}


void wnga_release_block_grid(Integer g_a, Integer *index)
{
    printf("%lf,pnga_release_block_grid,(%ld;%p)\n",MPI_Wtime(),g_a,index);
    pnga_release_block_grid(g_a, index);
    printf("%lf,/pnga_release_block_grid,(%ld;%p)\n",MPI_Wtime(),g_a,index);

}


void wnga_release_block_segment(Integer g_a, Integer iproc)
{
    printf("%lf,pnga_release_block_segment,(%ld;%ld)\n",MPI_Wtime(),g_a,iproc);
    pnga_release_block_segment(g_a, iproc);
    printf("%lf,/pnga_release_block_segment,(%ld;%ld)\n",MPI_Wtime(),g_a,iproc);

}


void wnga_release_ghost_element(Integer g_a, Integer subscript[])
{
    printf("%lf,pnga_release_ghost_element,(%ld;%p)\n",MPI_Wtime(),g_a,subscript);
    pnga_release_ghost_element(g_a, subscript);
    printf("%lf,/pnga_release_ghost_element,(%ld;%p)\n",MPI_Wtime(),g_a,subscript);

}


void wnga_release_ghosts(Integer g_a)
{
    printf("%lf,pnga_release_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    pnga_release_ghosts(g_a);
    printf("%lf,/pnga_release_ghosts,(%ld)\n",MPI_Wtime(),g_a);

}


void wnga_release_update(Integer g_a, Integer *lo, Integer *hi)
{
    printf("%lf,pnga_release_update,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);
    pnga_release_update(g_a, lo, hi);
    printf("%lf,/pnga_release_update,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);

}


void wnga_release_update_block(Integer g_a, Integer iblock)
{
    printf("%lf,pnga_release_update_block,(%ld;%ld)\n",MPI_Wtime(),g_a,iblock);
    pnga_release_update_block(g_a, iblock);
    printf("%lf,/pnga_release_update_block,(%ld;%ld)\n",MPI_Wtime(),g_a,iblock);

}


void wnga_release_update_block_grid(Integer g_a, Integer *index)
{
    printf("%lf,pnga_release_update_block_grid,(%ld;%p)\n",MPI_Wtime(),g_a,index);
    pnga_release_update_block_grid(g_a, index);
    printf("%lf,/pnga_release_update_block_grid,(%ld;%p)\n",MPI_Wtime(),g_a,index);

}


void wnga_release_update_block_segment(Integer g_a, Integer iproc)
{
    printf("%lf,pnga_release_update_block_segment,(%ld;%ld)\n",MPI_Wtime(),g_a,iproc);
    pnga_release_update_block_segment(g_a, iproc);
    printf("%lf,/pnga_release_update_block_segment,(%ld;%ld)\n",MPI_Wtime(),g_a,iproc);

}


void wnga_release_update_ghost_element(Integer g_a, Integer subscript[])
{
    printf("%lf,pnga_release_update_ghost_element,(%ld;%p)\n",MPI_Wtime(),g_a,subscript);
    pnga_release_update_ghost_element(g_a, subscript);
    printf("%lf,/pnga_release_update_ghost_element,(%ld;%p)\n",MPI_Wtime(),g_a,subscript);

}


void wnga_release_update_ghosts(Integer g_a)
{
    printf("%lf,pnga_release_update_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    pnga_release_update_ghosts(g_a);
    printf("%lf,/pnga_release_update_ghosts,(%ld)\n",MPI_Wtime(),g_a);

}


void wnga_scale(Integer g_a, void *alpha)
{
    printf("%lf,pnga_scale,(%ld;%p)\n",MPI_Wtime(),g_a,alpha);
    pnga_scale(g_a, alpha);
    printf("%lf,/pnga_scale,(%ld;%p)\n",MPI_Wtime(),g_a,alpha);

}


void wnga_scale_cols(Integer g_a, Integer g_v)
{
    printf("%lf,pnga_scale_cols,(%ld;%ld)\n",MPI_Wtime(),g_a,g_v);
    pnga_scale_cols(g_a, g_v);
    printf("%lf,/pnga_scale_cols,(%ld;%ld)\n",MPI_Wtime(),g_a,g_v);

}


void wnga_scale_patch(Integer g_a, Integer *lo, Integer *hi, void *alpha)
{
    printf("%lf,pnga_scale_patch,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,alpha);
    pnga_scale_patch(g_a, lo, hi, alpha);
    printf("%lf,/pnga_scale_patch,(%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,alpha);

}


void wnga_scale_rows(Integer g_a, Integer g_v)
{
    printf("%lf,pnga_scale_rows,(%ld;%ld)\n",MPI_Wtime(),g_a,g_v);
    pnga_scale_rows(g_a, g_v);
    printf("%lf,/pnga_scale_rows,(%ld;%ld)\n",MPI_Wtime(),g_a,g_v);

}


void wnga_scan_add(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer excl)
{
    printf("%lf,pnga_scan_add,(%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_sbit,lo,hi,excl);
    pnga_scan_add(g_a, g_b, g_sbit, lo, hi, excl);
    printf("%lf,/pnga_scan_add,(%ld;%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_sbit,lo,hi,excl);

}


void wnga_scan_copy(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi)
{
    printf("%lf,pnga_scan_copy,(%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_sbit,lo,hi);
    pnga_scan_copy(g_a, g_b, g_sbit, lo, hi);
    printf("%lf,/pnga_scan_copy,(%ld;%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,g_b,g_sbit,lo,hi);

}


void wnga_scatter(Integer g_a, void *v, Integer *subscript, Integer nv)
{
    printf("%lf,pnga_scatter,(%ld;%p;%p;%ld)\n",MPI_Wtime(),g_a,v,subscript,nv);
    pnga_scatter(g_a, v, subscript, nv);
    printf("%lf,/pnga_scatter,(%ld;%p;%p;%ld)\n",MPI_Wtime(),g_a,v,subscript,nv);

}


void wnga_scatter2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv)
{
    printf("%lf,pnga_scatter2d,(%ld;%p;%p;%p;%ld)\n",MPI_Wtime(),g_a,v,i,j,nv);
    pnga_scatter2d(g_a, v, i, j, nv);
    printf("%lf,/pnga_scatter2d,(%ld;%p;%p;%p;%ld)\n",MPI_Wtime(),g_a,v,i,j,nv);

}


void wnga_scatter_acc(Integer g_a, void *v, Integer subscript[], Integer nv, void *alpha)
{
    printf("%lf,pnga_scatter_acc,(%ld;%p;%p;%ld;%p)\n",MPI_Wtime(),g_a,v,subscript,nv,alpha);
    pnga_scatter_acc(g_a, v, subscript, nv, alpha);
    printf("%lf,/pnga_scatter_acc,(%ld;%p;%p;%ld;%p)\n",MPI_Wtime(),g_a,v,subscript,nv,alpha);

}


void wnga_scatter_acc2d(Integer g_a, void *v, Integer *i, Integer *j, Integer nv, void *alpha)
{
    printf("%lf,pnga_scatter_acc2d,(%ld;%p;%p;%p;%ld;%p)\n",MPI_Wtime(),g_a,v,i,j,nv,alpha);
    pnga_scatter_acc2d(g_a, v, i, j, nv, alpha);
    printf("%lf,/pnga_scatter_acc2d,(%ld;%p;%p;%p;%ld;%p)\n",MPI_Wtime(),g_a,v,i,j,nv,alpha);

}


void wnga_select_elem(Integer g_a, char *op, void *val, Integer *subscript)
{
    printf("%lf,pnga_select_elem,(%ld;%s;%p;%p)\n",MPI_Wtime(),g_a,op,val,subscript);
    pnga_select_elem(g_a, op, val, subscript);
    printf("%lf,/pnga_select_elem,(%ld;%s;%p;%p)\n",MPI_Wtime(),g_a,op,val,subscript);

}


void wnga_set_array_name(Integer g_a, char *array_name)
{
    printf("%lf,pnga_set_array_name,(%ld;%s)\n",MPI_Wtime(),g_a,array_name);
    pnga_set_array_name(g_a, array_name);
    printf("%lf,/pnga_set_array_name,(%ld;%s)\n",MPI_Wtime(),g_a,array_name);

}


void wnga_set_block_cyclic(Integer g_a, Integer *dims)
{
    printf("%lf,pnga_set_block_cyclic,(%ld;%p)\n",MPI_Wtime(),g_a,dims);
    pnga_set_block_cyclic(g_a, dims);
    printf("%lf,/pnga_set_block_cyclic,(%ld;%p)\n",MPI_Wtime(),g_a,dims);

}


void wnga_set_block_cyclic_proc_grid(Integer g_a, Integer *dims, Integer *proc_grid)
{
    printf("%lf,pnga_set_block_cyclic_proc_grid,(%ld;%p;%p)\n",MPI_Wtime(),g_a,dims,proc_grid);
    pnga_set_block_cyclic_proc_grid(g_a, dims, proc_grid);
    printf("%lf,/pnga_set_block_cyclic_proc_grid,(%ld;%p;%p)\n",MPI_Wtime(),g_a,dims,proc_grid);

}


void wnga_set_chunk(Integer g_a, Integer *chunk)
{
    printf("%lf,pnga_set_chunk,(%ld;%p)\n",MPI_Wtime(),g_a,chunk);
    pnga_set_chunk(g_a, chunk);
    printf("%lf,/pnga_set_chunk,(%ld;%p)\n",MPI_Wtime(),g_a,chunk);

}


void wnga_set_data(Integer g_a, Integer ndim, Integer *dims, Integer type)
{
    printf("%lf,pnga_set_data,(%ld;%ld;%p;%ld)\n",MPI_Wtime(),g_a,ndim,dims,type);
    pnga_set_data(g_a, ndim, dims, type);
    printf("%lf,/pnga_set_data,(%ld;%ld;%p;%ld)\n",MPI_Wtime(),g_a,ndim,dims,type);

}


void wnga_set_debug(logical flag)
{
    printf("%lf,pnga_set_debug,(%ld)\n",MPI_Wtime(),flag);
    pnga_set_debug(flag);
    printf("%lf,/pnga_set_debug,(%ld)\n",MPI_Wtime(),flag);

}


void wnga_set_diagonal(Integer g_a, Integer g_v)
{
    printf("%lf,pnga_set_diagonal,(%ld;%ld)\n",MPI_Wtime(),g_a,g_v);
    pnga_set_diagonal(g_a, g_v);
    printf("%lf,/pnga_set_diagonal,(%ld;%ld)\n",MPI_Wtime(),g_a,g_v);

}


void wnga_set_ghost_corner_flag(Integer g_a, logical flag)
{
    printf("%lf,pnga_set_ghost_corner_flag,(%ld;%ld)\n",MPI_Wtime(),g_a,flag);
    pnga_set_ghost_corner_flag(g_a, flag);
    printf("%lf,/pnga_set_ghost_corner_flag,(%ld;%ld)\n",MPI_Wtime(),g_a,flag);

}


logical wnga_set_ghost_info(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_set_ghost_info,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_set_ghost_info(g_a);
    printf("%lf,/pnga_set_ghost_info,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


void wnga_set_ghosts(Integer g_a, Integer *width)
{
    printf("%lf,pnga_set_ghosts,(%ld;%p)\n",MPI_Wtime(),g_a,width);
    pnga_set_ghosts(g_a, width);
    printf("%lf,/pnga_set_ghosts,(%ld;%p)\n",MPI_Wtime(),g_a,width);

}


void wnga_set_irreg_distr(Integer g_a, Integer *mapc, Integer *nblock)
{
    printf("%lf,pnga_set_irreg_distr,(%ld;%p;%p)\n",MPI_Wtime(),g_a,mapc,nblock);
    pnga_set_irreg_distr(g_a, mapc, nblock);
    printf("%lf,/pnga_set_irreg_distr,(%ld;%p;%p)\n",MPI_Wtime(),g_a,mapc,nblock);

}


void wnga_set_irreg_flag(Integer g_a, logical flag)
{
    printf("%lf,pnga_set_irreg_flag,(%ld;%ld)\n",MPI_Wtime(),g_a,flag);
    pnga_set_irreg_flag(g_a, flag);
    printf("%lf,/pnga_set_irreg_flag,(%ld;%ld)\n",MPI_Wtime(),g_a,flag);

}


void wnga_set_memory_limit(Integer mem_limit)
{
    printf("%lf,pnga_set_memory_limit,(%ld)\n",MPI_Wtime(),mem_limit);
    pnga_set_memory_limit(mem_limit);
    printf("%lf,/pnga_set_memory_limit,(%ld)\n",MPI_Wtime(),mem_limit);

}


void wnga_set_pgroup(Integer g_a, Integer p_handle)
{
    printf("%lf,pnga_set_pgroup,(%ld;%ld)\n",MPI_Wtime(),g_a,p_handle);
    pnga_set_pgroup(g_a, p_handle);
    printf("%lf,/pnga_set_pgroup,(%ld;%ld)\n",MPI_Wtime(),g_a,p_handle);

}


void wnga_set_restricted(Integer g_a, Integer *list, Integer size)
{
    printf("%lf,pnga_set_restricted,(%ld;%p;%ld)\n",MPI_Wtime(),g_a,list,size);
    pnga_set_restricted(g_a, list, size);
    printf("%lf,/pnga_set_restricted,(%ld;%p;%ld)\n",MPI_Wtime(),g_a,list,size);

}


void wnga_set_restricted_range(Integer g_a, Integer lo_proc, Integer hi_proc)
{
    printf("%lf,pnga_set_restricted_range,(%ld;%ld;%ld)\n",MPI_Wtime(),g_a,lo_proc,hi_proc);
    pnga_set_restricted_range(g_a, lo_proc, hi_proc);
    printf("%lf,/pnga_set_restricted_range,(%ld;%ld;%ld)\n",MPI_Wtime(),g_a,lo_proc,hi_proc);

}


logical wnga_set_update4_info(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_set_update4_info,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_set_update4_info(g_a);
    printf("%lf,/pnga_set_update4_info,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


logical wnga_set_update5_info(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_set_update5_info,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_set_update5_info(g_a);
    printf("%lf,/pnga_set_update5_info,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


void wnga_shift_diagonal(Integer g_a, void *c)
{
    printf("%lf,pnga_shift_diagonal,(%ld;%p)\n",MPI_Wtime(),g_a,c);
    pnga_shift_diagonal(g_a, c);
    printf("%lf,/pnga_shift_diagonal,(%ld;%p)\n",MPI_Wtime(),g_a,c);

}


Integer wnga_solve(Integer g_a, Integer g_b)
{
    Integer retval;
    printf("%lf,pnga_solve,(%ld;%ld)\n",MPI_Wtime(),g_a,g_b);
    retval = pnga_solve(g_a, g_b);
    printf("%lf,/pnga_solve,(%ld;%ld)\n",MPI_Wtime(),g_a,g_b);
    return retval;

}


Integer wnga_spd_invert(Integer g_a)
{
    Integer retval;
    printf("%lf,pnga_spd_invert,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_spd_invert(g_a);
    printf("%lf,/pnga_spd_invert,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


void wnga_step_bound_info(Integer g_xx, Integer g_vv, Integer g_xxll, Integer g_xxuu, void *boundmin, void *wolfemin, void *boundmax)
{
    printf("%lf,pnga_step_bound_info,(%ld;%ld;%ld;%ld;%p;%p;%p)\n",MPI_Wtime(),g_xx,g_vv,g_xxll,g_xxuu,boundmin,wolfemin,boundmax);
    pnga_step_bound_info(g_xx, g_vv, g_xxll, g_xxuu, boundmin, wolfemin, boundmax);
    printf("%lf,/pnga_step_bound_info,(%ld;%ld;%ld;%ld;%p;%p;%p)\n",MPI_Wtime(),g_xx,g_vv,g_xxll,g_xxuu,boundmin,wolfemin,boundmax);

}


void wnga_step_bound_info_patch(Integer g_xx, Integer *xxlo, Integer *xxhi, Integer g_vv, Integer *vvlo, Integer *vvhi, Integer g_xxll, Integer *xxlllo, Integer *xxllhi, Integer g_xxuu, Integer *xxuulo, Integer *xxuuhi, void *boundmin, void *wolfemin, void *boundmax)
{
    printf("%lf,pnga_step_bound_info_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p;%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_xx,xxlo,xxhi,g_vv,vvlo,vvhi,g_xxll,xxlllo,xxllhi,g_xxuu,xxuulo,xxuuhi,boundmin,wolfemin,boundmax);
    pnga_step_bound_info_patch(g_xx, xxlo, xxhi, g_vv, vvlo, vvhi, g_xxll, xxlllo, xxllhi, g_xxuu, xxuulo, xxuuhi, boundmin, wolfemin, boundmax);
    printf("%lf,/pnga_step_bound_info_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p;%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_xx,xxlo,xxhi,g_vv,vvlo,vvhi,g_xxll,xxlllo,xxllhi,g_xxuu,xxuulo,xxuuhi,boundmin,wolfemin,boundmax);

}


void wnga_step_mask_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, Integer g_c, Integer *clo, Integer *chi)
{
    printf("%lf,pnga_step_mask_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);
    pnga_step_mask_patch(g_a, alo, ahi, g_b, blo, bhi, g_c, clo, chi);
    printf("%lf,/pnga_step_mask_patch,(%ld;%p;%p;%ld;%p;%p;%ld;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,g_c,clo,chi);

}


void wnga_step_max(Integer g_a, Integer g_b, void *retval)
{
    printf("%lf,pnga_step_max,(%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_b,retval);
    pnga_step_max(g_a, g_b, retval);
    printf("%lf,/pnga_step_max,(%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_b,retval);

}


void wnga_step_max_patch(Integer g_a, Integer *alo, Integer *ahi, Integer g_b, Integer *blo, Integer *bhi, void *result)
{
    printf("%lf,pnga_step_max_patch,(%ld;%p;%p;%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,result);
    pnga_step_max_patch(g_a, alo, ahi, g_b, blo, bhi, result);
    printf("%lf,/pnga_step_max_patch,(%ld;%p;%p;%ld;%p;%p;%p)\n",MPI_Wtime(),g_a,alo,ahi,g_b,blo,bhi,result);

}


void wnga_strided_acc(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld, void *alpha)
{
    printf("%lf,pnga_strided_acc,(%ld;%p;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,skip,buf,ld,alpha);
    pnga_strided_acc(g_a, lo, hi, skip, buf, ld, alpha);
    printf("%lf,/pnga_strided_acc,(%ld;%p;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,skip,buf,ld,alpha);

}


void wnga_strided_get(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld)
{
    printf("%lf,pnga_strided_get,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,skip,buf,ld);
    pnga_strided_get(g_a, lo, hi, skip, buf, ld);
    printf("%lf,/pnga_strided_get,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,skip,buf,ld);

}


void wnga_strided_put(Integer g_a, Integer *lo, Integer *hi, Integer *skip, void *buf, Integer *ld)
{
    printf("%lf,pnga_strided_put,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,skip,buf,ld);
    pnga_strided_put(g_a, lo, hi, skip, buf, ld);
    printf("%lf,/pnga_strided_put,(%ld;%p;%p;%p;%p;%p)\n",MPI_Wtime(),g_a,lo,hi,skip,buf,ld);

}


void wnga_summarize(Integer verbose)
{
    printf("%lf,pnga_summarize,(%ld)\n",MPI_Wtime(),verbose);
    pnga_summarize(verbose);
    printf("%lf,/pnga_summarize,(%ld)\n",MPI_Wtime(),verbose);

}


void wnga_symmetrize(Integer g_a)
{
    printf("%lf,pnga_symmetrize,(%ld)\n",MPI_Wtime(),g_a);
    pnga_symmetrize(g_a);
    printf("%lf,/pnga_symmetrize,(%ld)\n",MPI_Wtime(),g_a);

}


void wnga_sync()
{
    printf("%lf,pnga_sync,\n",MPI_Wtime());
    pnga_sync();
    printf("%lf,/pnga_sync,\n",MPI_Wtime());

}


double wnga_timer()
{
    double retval;
    printf("%lf,pnga_timer,\n",MPI_Wtime());
    retval = pnga_timer();
    printf("%lf,/pnga_timer,\n",MPI_Wtime());
    return retval;

}


Integer wnga_total_blocks(Integer g_a)
{
    Integer retval;
    printf("%lf,pnga_total_blocks,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_total_blocks(g_a);
    printf("%lf,/pnga_total_blocks,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


void wnga_transpose(Integer g_a, Integer g_b)
{
    printf("%lf,pnga_transpose,(%ld;%ld)\n",MPI_Wtime(),g_a,g_b);
    pnga_transpose(g_a, g_b);
    printf("%lf,/pnga_transpose,(%ld;%ld)\n",MPI_Wtime(),g_a,g_b);

}


Integer wnga_type_c2f(Integer type)
{
    Integer retval;
    printf("%lf,pnga_type_c2f,(%ld)\n",MPI_Wtime(),type);
    retval = pnga_type_c2f(type);
    printf("%lf,/pnga_type_c2f,(%ld)\n",MPI_Wtime(),type);
    return retval;

}


Integer wnga_type_f2c(Integer type)
{
    Integer retval;
    printf("%lf,pnga_type_f2c,(%ld)\n",MPI_Wtime(),type);
    retval = pnga_type_f2c(type);
    printf("%lf,/pnga_type_f2c,(%ld)\n",MPI_Wtime(),type);
    return retval;

}


void wnga_unlock(Integer mutex)
{
    printf("%lf,pnga_unlock,(%ld)\n",MPI_Wtime(),mutex);
    pnga_unlock(mutex);
    printf("%lf,/pnga_unlock,(%ld)\n",MPI_Wtime(),mutex);

}


void wnga_unpack(Integer g_a, Integer g_b, Integer g_sbit, Integer lo, Integer hi, Integer *icount)
{
    printf("%lf,pnga_unpack,(%ld;%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_b,g_sbit,lo,hi,icount);
    pnga_unpack(g_a, g_b, g_sbit, lo, hi, icount);
    printf("%lf,/pnga_unpack,(%ld;%ld;%ld;%ld;%ld;%p)\n",MPI_Wtime(),g_a,g_b,g_sbit,lo,hi,icount);

}


void wnga_update1_ghosts(Integer g_a)
{
    printf("%lf,pnga_update1_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    pnga_update1_ghosts(g_a);
    printf("%lf,/pnga_update1_ghosts,(%ld)\n",MPI_Wtime(),g_a);

}


logical wnga_update2_ghosts(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_update2_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_update2_ghosts(g_a);
    printf("%lf,/pnga_update2_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


logical wnga_update3_ghosts(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_update3_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_update3_ghosts(g_a);
    printf("%lf,/pnga_update3_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


logical wnga_update44_ghosts(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_update44_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_update44_ghosts(g_a);
    printf("%lf,/pnga_update44_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


logical wnga_update4_ghosts(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_update4_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_update4_ghosts(g_a);
    printf("%lf,/pnga_update4_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


logical wnga_update55_ghosts(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_update55_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_update55_ghosts(g_a);
    printf("%lf,/pnga_update55_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


logical wnga_update5_ghosts(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_update5_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_update5_ghosts(g_a);
    printf("%lf,/pnga_update5_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


logical wnga_update6_ghosts(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_update6_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_update6_ghosts(g_a);
    printf("%lf,/pnga_update6_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


logical wnga_update7_ghosts(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_update7_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_update7_ghosts(g_a);
    printf("%lf,/pnga_update7_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


logical wnga_update_ghost_dir(Integer g_a, Integer pdim, Integer pdir, logical pflag)
{
    logical retval;
    printf("%lf,pnga_update_ghost_dir,(%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,pdim,pdir,pflag);
    retval = pnga_update_ghost_dir(g_a, pdim, pdir, pflag);
    printf("%lf,/pnga_update_ghost_dir,(%ld;%ld;%ld;%ld)\n",MPI_Wtime(),g_a,pdim,pdir,pflag);
    return retval;

}


void wnga_update_ghosts(Integer g_a)
{
    printf("%lf,pnga_update_ghosts,(%ld)\n",MPI_Wtime(),g_a);
    pnga_update_ghosts(g_a);
    printf("%lf,/pnga_update_ghosts,(%ld)\n",MPI_Wtime(),g_a);

}


logical wnga_uses_ma()
{
    logical retval;
    printf("%lf,pnga_uses_ma,\n",MPI_Wtime());
    retval = pnga_uses_ma();
    printf("%lf,/pnga_uses_ma,\n",MPI_Wtime());
    return retval;

}


logical wnga_uses_proc_grid(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_uses_proc_grid,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_uses_proc_grid(g_a);
    printf("%lf,/pnga_uses_proc_grid,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


logical wnga_valid_handle(Integer g_a)
{
    logical retval;
    printf("%lf,pnga_valid_handle,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_valid_handle(g_a);
    printf("%lf,/pnga_valid_handle,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


Integer wnga_verify_handle(Integer g_a)
{
    Integer retval;
    printf("%lf,pnga_verify_handle,(%ld)\n",MPI_Wtime(),g_a);
    retval = pnga_verify_handle(g_a);
    printf("%lf,/pnga_verify_handle,(%ld)\n",MPI_Wtime(),g_a);
    return retval;

}


DoublePrecision wnga_wtime()
{
    DoublePrecision retval;
    printf("%lf,pnga_wtime,\n",MPI_Wtime());
    retval = pnga_wtime();
    printf("%lf,/pnga_wtime,\n",MPI_Wtime());
    return retval;

}


void wnga_zero(Integer g_a)
{
    printf("%lf,pnga_zero,(%ld)\n",MPI_Wtime(),g_a);
    pnga_zero(g_a);
    printf("%lf,/pnga_zero,(%ld)\n",MPI_Wtime(),g_a);

}


void wnga_zero_diagonal(Integer g_a)
{
    printf("%lf,pnga_zero_diagonal,(%ld)\n",MPI_Wtime(),g_a);
    pnga_zero_diagonal(g_a);
    printf("%lf,/pnga_zero_diagonal,(%ld)\n",MPI_Wtime(),g_a);

}


void wnga_zero_patch(Integer g_a, Integer *lo, Integer *hi)
{
    printf("%lf,pnga_zero_patch,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);
    pnga_zero_patch(g_a, lo, hi);
    printf("%lf,/pnga_zero_patch,(%ld;%p;%p)\n",MPI_Wtime(),g_a,lo,hi);

}

void wnga_initialize()
{
    pnga_initialize();
}

void wnga_terminate()
{
    static int count_pnga_terminate=0;

    ++count_pnga_terminate;
    pnga_terminate();
    /* don't dump info if terminate more than once */
    if (1 == count_pnga_terminate) {

    }
}

