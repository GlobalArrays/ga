#ifdef CRAY
#define print_range_  PRINT_RANGE
#define init_array_   INIT_ARRAY
#define scale_patch_  SCALE_PATCH
#endif
extern void get_range( int ndim, int dims[], int lo[], int hi[]);
extern void new_range(int ndim, int dims[], int lo[], int hi[],
                             int new_lo[], int new_hi[]);
extern void print_range(char *pre,int ndim, int lo[], int hi[], char* post);
extern void print_subscript(char *pre,int ndim, int subscript[], char* post);
extern void print_distribution(int g_a);
