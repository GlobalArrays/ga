/*$Id: base.h,v 1.2 2001-10-25 21:06:39 d3g293 Exp $ */
extern int _max_global_array;
extern int gai_getval(int *ptr);
extern Integer *_ga_map;
extern Integer GAme, GAnproc;
extern Integer *GA_proclist;
extern int* GA_Proc_list;
extern int* GA_inv_Proc_list;
extern global_array_t GA[MAX_ARRAYS]; 
extern int** GA_Update_Flags;
#define ERR_STR_LEN 256             /* length of string for error reporting */
char err_string[ ERR_STR_LEN];        /* string for extended error reporting */

/**************************** MACROS ************************************/


#define ga_check_handleM(g_a, string) \
{\
    if(GA_OFFSET+ (*g_a) < 0 || GA_OFFSET+(*g_a) >=_max_global_array){ \
      sprintf(err_string, "%s: INVALID ARRAY HANDLE", string);         \
      ga_error(err_string, (*g_a));                                    \
    }\
    if( ! (GA[GA_OFFSET+(*g_a)].actv) ){                               \
      sprintf(err_string, "%s: ARRAY NOT ACTIVE", string);             \
      ga_error(err_string, (*g_a));                                    \
    }                                                                  \
}

/* this macro finds cordinates of the chunk of array owned by processor proc */
#define ga_ownsM_no_handle(ndim, dims, nblock, mapc, proc, lo, hi)                                      \
{                                                                              \
   Integer _loc, _nb, _d, _index, _dim=ndim,_dimstart=0, _dimpos;\
   for(_nb=1, _d=0; _d<_dim; _d++)_nb *= nblock[_d];             \
   if(proc > _nb - 1 || proc<0)for(_d=0; _d<_dim; _d++){                       \
         lo[_d] = (Integer)0;                                                  \
         hi[_d] = (Integer)-1;                                                 \
   }else{                                                                      \
         _index = proc;                                                        \
         if(GA_inv_Proc_list) _index = GA_inv_Proc_list[proc];                 \
         for(_d=0; _d<_dim; _d++){                                             \
             _loc = _index% nblock[_d];                          \
             _index  /= nblock[_d];                              \
             _dimpos = _loc + _dimstart; /* correction to find place in mapc */\
             _dimstart += nblock[_d];                            \
             lo[_d] = mapc[_dimpos];                             \
             if(_loc==nblock[_d]-1)hi[_d]=dims[_d];\
             else hi[_d] = mapc[_dimpos+1]-1;                    \
         }                                                                     \
   }                                                                           \
}

/* this macro finds cordinates of the chunk of array owned by processor proc */
#define ga_ownsM(ga_handle, proc, lo, hi)				\
  ga_ownsM_no_handle(GA[ga_handle].ndim, GA[ga_handle].dims, GA[ga_handle].nblock, GA[ga_handle].mapc, proc, lo, hi )
