#ifndef  _GLOBALARRAY_H
#define  _GLOBALARRAY_H


#define DEF_NDIM 2
#define DEF_DIMS 10

/**
 * This is the GlobalArray class.
 */
class GlobalArray { 

 public:
  
  /**  
   * Creates an ndim-dimensional array using the regular distribution model 
   * and returns integer handle representing the array. 
   
   * The array can be distributed evenly or not. The control over the 
   * distribution is accomplished by specifying chunk (block) size for all or 
   * some of array dimensions.
   
   * For example, for a 2-dimensional array, setting chunk[0]=dim[0] gives 
   * distribution by vertical strips (chunk[0]*dims[0]); 
   * setting chunk[1]=dim[1] gives distribution by horizontal strips 
   * (chunk[1]*dims[1]). Actual chunks will be modified so that they are at 
   * least the size of the minimum and each process has either zero or one 
   * chunk. Specifying chunk[i] as <1 will cause that dimension to be 
   * distributed evenly. 
   
   * As a convenience, when chunk is specified as NULL, the entire array is 
   * distributed evenly.
   
   * This is a collective operation. 
   
   * @param arrayname  - a unique character string               [input]
   * @param type        - data type(MT_F_DBL,MT_F_INT,MT_F_DCPL)  [input]
   * @param ndim        - number of array dimensions              [input]
   * @param dims[ndim]  - array of dimensions                     [input]
   * @param chunk[ndim] - array of chunks, each element specifies 
   * minimum size that given dimensions should be chunked up into [input]
   * @param p_handle    - processor group handle                  [input]
   */
  GlobalArray(int type, int ndim, int dims[], char *arrayname, int chunk[]);
  GlobalArray(int type, int ndim, int dims[], char *arrayname, int chunk[],
              GA::PGroup* p_handle);

  /**
   * "long" interface of above methods for large array creations 
   */
  GlobalArray(int type, int ndim, int64_t dims[], char *arrayname,
              int64_t chunk[]);
  GlobalArray(int type, int ndim, int64_t dims[], char *arrayname,
              int64_t chunk[], GA::PGroup *p_handle);
      
  
  /**
   * Creates an array by following the user-specified distribution and 
   * returns integer handle representing the array. 
     
   * The distribution is specified as a Cartesian product of distributions 
   * for each dimension. The array indices start at 0. For example, the 
   * following figure demonstrates distribution of a 2-dimensional array 8x10 
   * on 6 (or more) processors. nblock[2]={3,2}, the size of map array is s=5 
   * and array map contains the following elements map={0,2,6, 0, 5}. The 
   * distribution is nonuniform because, P1 and P4 get 20 elements each and 
   * processors P0,P2,P3, and P5 only 10 elements each. 
   *
   * <TABLE>
   * <TR> <TD>5</TD>  <TD>5</TD>  </TR>
   * <TR> <TD>P0</TD> <TD>P3</TD> <TD>2</TD> </TR>
   * <TR> <TD>P1</TD> <TD>P4</TD> <TD>4</TD> </TR>
   * <TR> <TD>P2</TD> <TD>P5</TD> <TD>2</TD> </TR>
   *  </TABLE>
   *
   * This is a collective operation. 
   * @param arrayname    - a unique character string           [input]
   * @param type  - MA data type (MT_F_DBL,MT_F_INT,MT_F_DCPL) [input]
   * @param ndim  - number of array dimensions                 [input]
   * @param  dims - array of dimension values                  [input]
   * @param block[ndim] - no. of blocks each dimension is divided into [input]
   * @param maps[s]  - starting index for for each block; the size s is a sum 
   * all elements of nblock array                              [input]
   * @param p_handle - processor group handle                  [input]
   */
  GlobalArray(int type, int ndim, int dims[], char *arrayname, int block[],
	      int maps[]);
  GlobalArray(int type, int ndim, int dims[], char *arrayname, int block[],
	      int maps[], GA::PGroup* p_handle);
  
  /**
   * "long" interface of above methods for large array creations 
   */
  GlobalArray(int type, int ndim, int64_t dims[], char *arrayname,
              int64_t block[], int64_t maps[]);
  GlobalArray(int type, int ndim, int64_t dims[], char *arrayname,
              int64_t block[], int64_t maps[], GA::PGroup* p_handle);
      
  /**
   * Creates an ndim-dimensional array with a layer of ghost cells around 
   * the visible data on each processor using the regular distribution 
   * model and returns an integer handle representing the array. 
   * The array can be distributed evenly or not evenly. The control over 
   * the distribution is accomplished by specifying chunk (block) size for 
   * all or some of the array dimensions. For example, for a 2-dimensional 
   * array, setting chunk(1)=dim(1) gives distribution by vertical strips 
   * (chunk(1)*dims(1)); setting chunk(2)=dim(2) gives distribution by 
   * horizontal strips (chunk(2)*dims(2)). Actual chunks will be modified 
   * so that they are at least the size of the minimum and each process 
   * has either zero or one chunk. Specifying chunk(i) as <1 will cause
   * that dimension (i-th) to be distributed evenly. The  width of the 
   * ghost cell layer in each dimension is specified using the array 
   * width().  The local data of the global array residing on each 
   * processor will have a layer width[n] ghosts cells wide on either 
   * side of the visible data along the dimension n. 
   * 
   * @param array_name   - a unique character string                [input]
   * @param type         - data type (MT_DBL,MT_INT,MT_DCPL)        [input]
   * @param ndim         - number of array dimensions               [input]
   * @param dims[ndim]   - array of dimensions                      [input]
   * @param width[ndim]  - array of ghost cell widths               [input]
   * @param chunk[ndim]  - array of chunks, each element specifies
   *                       minimum size that given dimensions should be
   *                       chunked up into                          [input]
   * @param ghosts       - this is a dummy parameter: added to increase the 
   *                       number of arguments, inorder to avoid the conflicts
   *                       among constructors. (ghosts = 'g' or 'G')
   * @param p_handle - processor group handle                       [input]
   */
  GlobalArray(int type, int ndim, int dims[], int width[], char *arrayname, 
	      int chunk[], char ghosts);
  GlobalArray(int type, int ndim, int dims[], int width[], char *arrayname, 
	      int chunk[], GA::PGroup* p_handle, char ghosts);
  
  /**
   * "long" interface of above methods for large array creations 
   */
  GlobalArray(int type, int ndim, int64_t dims[], int64_t width[], char *arrayname, 
	      int64_t chunk[], char ghosts);
  GlobalArray(int type, int ndim, int64_t dims[], int64_t width[], char *arrayname, 
	      int64_t chunk[], GA::PGroup* p_handle, char ghosts);
  

  /**
   * Creates an array with ghost cells by following the user-specified 
   * distribution and returns integer handle representing the array. 
   * The distribution is specified as a Cartesian product of distributions 
   * for each dimension. For example, the following figure demonstrates 
   * distribution of a 2-dimensional array 8x10 on 6 (or more) processors. 
   * nblock(2)={3,2}, the size of map array is s=5 and array map contains 
   * the following elements map={1,3,7, 1, 6}. The distribution is 
   * nonuniform because, P1 and P4 get 20 elements each and processors 
   * P0,P2,P3, and P5 only 10 elements each. 
   *
   * <TABLE>
   * <TR> <TD>5</TD>  <TD>5</TD>  </TR>
   * <TR> <TD>P0</TD> <TD>P3</TD> <TD>2</TD> </TR>
   * <TR> <TD>P1</TD> <TD>P4</TD> <TD>4</TD> </TR>
   * <TR> <TD>P2</TD> <TD>P5</TD> <TD>2</TD> </TR>
   *  </TABLE>
   *
   * The array width[] is used to control the width of the ghost cell 
   * boundary around the visible data on each processor. The local data 
   * of the global array residing on each processor will have a layer 
   * width[n] ghosts cells wide on either side of the visible data along 
   * the dimension n. This is a collective operation. 
   *
   * @param array_name   - a unique character string                [input]
   * @param type         - data type (MT_DBL,MT_INT,MT_DCPL)        [input]
   * @param ndim         - number of array dimensions               [input]
   * @param dims[ndim]   - array of dimensions                      [input]
   * @param width[ndim]  - array of ghost cell widths               [input]
   * @param nblock[ndim] - no. of blocks each dimension is divided into[input]
   * @param  map[s]      - starting index for for each block; the size     
   *                       s is a sum of all elements of nblock array[input]
   * @param ghosts       - this is a dummy parameter: added to increase the 
   *                       number of arguments, inorder to avoid the conflicts
   *                       among constructors. (ghosts = 'g' or 'G')
   * @param p_handle - processor group handle                       [input]
   */
  GlobalArray(int type, int ndim, int dims[], int width[], char *arrayname, 
	      int block[], int maps[], char ghosts);
  GlobalArray(int type, int ndim, int dims[], int width[], char *arrayname, 
	      int block[], int maps[], GA::PGroup* p_handle, char ghosts);

  /**
   * "long" interface of above methods for large array creations 
   */
  GlobalArray(int type, int ndim, int64_t dims[], int64_t width[], char *arrayname, 
	      int64_t block[], int64_t maps[], char ghosts);
  GlobalArray(int type, int ndim, int64_t dims[], int64_t width[], char *arrayname, 
	      int64_t block[], int64_t maps[], GA::PGroup* p_handle, char ghosts);

  
  /**
   * Creates a new array by applying all the properties of another existing 
   * array.
   * This is a collective operation. 
   * @param arrayname    - a character string                 [input]
   * @param g_a           - integer handle for reference array [input]
   */
  GlobalArray(const GlobalArray &g_a, char *arrayname); 
  
  /**
   * Creates a new array by applying all the properties of another existing 
   * array.
   * This is a collective operation. 
   * @param g_a           - integer handle for reference array [input]
   */
  GlobalArray(const GlobalArray &g_a);/* copy constructor */

  /**
   * Creates a new array with no existing attributes. These must all
   * be set using the "set" methods.
   * This is a collective operation. 
   */
  GlobalArray();
  
  /** Destructor */
  ~GlobalArray();

  /* access the data */
  /** @return returns the array handler*/
  int handle() const { return mHandle; }
  
  /* Global Array operations */
  
  /** 
   * Combines data from local array buffer with data in the global array 
   * section. The local array is assumed to be have the same number of 
   * dimensions as the global array. 

   * global array section (lo[],hi[]) += *alpha * buffer

   * This is a one-sided and atomic operation.  
   * @param lo[ndim]   - array of starting indices for array section[input] 
   * @param hi[ndim]   - array of ending indices for array section  [input]
   * @param buf        - pointer to the local buffer array          [input]
   * @param ld[ndim-1] - array specifying leading dimensions/strides/extents 
   *                     for buffer array                           [input]
   * @param alpha      - scale factor (double/DoubleComplex/long *) [input]
   */
  void acc(int lo[], int hi[], void *buf, int ld[], void *alpha) const;

  /**
   * "long" interface for accumulate operation on large arrays
   */
  void acc(int64_t lo[], int64_t hi[], void *buf, int64_t ld[], void *alpha) const;
  
  /**
   * Provides access to the specified patch of a global array. Returns 
   * array of leading dimensions ld and a pointer to the first element 
   * in the patch. This routine allows to access directly, in place 
   * elements in the local section of a global array. It useful for 
   * writing new GA operations. A call to ga_access normally follows a 
   * previous call to ga_distribution that returns coordinates of the 
   * patch associated with a processor. You need to make sure that the 
   * coordinates of the patch are valid (test values returned from 
   * ga_distribution). 
   *
   * Each call to ga_access has to be followed by a call to either 
   * ga_release or ga_release_update. You can access in this fashion only 
   * local data. Since the data is shared with other processes, you need 
   * to consider issues of mutual exclusion. This operation is local. 
   * 
   * @param ndim      - number of dimensions of the global array
   * @param lo[ndim]  - array of starting indices for array section [input]
   * @param hi[ndim]  - array of ending indices for array section   [input]
   * @param ptr       - points to location of first element in patch[output]
   * @param ld[ndim-1]- leading dimensions for the pacth elements   [output]
   */
  void access(int lo[], int hi[], void *ptr, int ld[]) const;

  /**
   * "long" interface of access() method for large array creations 
   */
  void access(int64_t lo[], int64_t hi[], void *ptr, int64_t ld[]) const;

  /**
   * Provides access to the specified block of a global array that is using
   * simple block-cyclic data distribution. Returns array of leading
   * dimensions ld and a pointer to the first element in the patch. This
   * routine allows user to access directly, in-place * elements in the
   * local section of a global array. It useful for writing new GA
   * operations. A call to ga_access normally follows a previous call to
   * ga_distribution that returns coordinates of the patch associated with
   * a processor. You need to make sure that the coordinates of the patch
   * are valid (test values returned from * ga_distribution). 
   *
   * Each call to ga_access_block has to be followed by a call to either 
   * ga_release_block or ga_release_block_update. You can access in this
   * fashion only local data. Since the data is shared with other processes,
   * you need to consider issues of mutual exclusion. This operation is
   * local. 
   * 
   * @param ndim      - number of dimensions of the global array
   * @param idx       - index of block                              [input]
   * @param ptr       - points to location of first element in patch[output]
   * @param ld[ndim-1]- leading dimensions for the pacth elements   [output]
   */
  void accessBlock(int idx, void *ptr, int ld[]) const;

  /**
   * "long" interface for accessBlock
   */
  void accessBlock(int64_t idx, void *ptr, int64_t ld[]) const;

  /**
   * Provides access to the specified block of a global array that is using
   * SCALAPACK type block-cyclic data distribution. Returns array of leading
   * dimensions ld and a pointer to the first element in the patch. This
   * routine allows user to access directly, in-place * elements in the
   * local section of a global array. It useful for writing new GA
   * operations. A call to ga_access_block normally follows a previous call to
   * ga_distribution that returns coordinates of the patch associated with
   * a processor. You need to make sure that the coordinates of the patch
   * are valid (test values returned from * ga_distribution). 
   *
   * Each call to ga_access_block_grid has to be followed by a call to either 
   * ga_release_block_grid or ga_release_block_grid_update. You can access in
   * this fashion only local data. Since the data is shared with other
   * processes, you need to consider issues of mutual exclusion. This
   * operation is local. 
   * 
   * @param ndim       - number of dimensions of the global array
   * @param index[ndim]- indices of block in processor grid          [input]
   * @param ptr        - points to location of first element in patch[output]
   * @param ld[ndim-1] - leading dimensions for the pacth elements   [output]
   */
  void accessBlockGrid(int index[], void *ptr, int ld[]) const;

  /**
   * "long" interface for accessBlockGrid
   */
  void accessBlockGrid(int64_t index[], void *ptr, int64_t ld[]) const;

  /**
   * Provides access to the local data of a global array that is using
   * either the simple or SCALAPACK type block-cyclic data distribution.
   * Returns the length of the local data block and a pointer to the first
   * element. This routine allows user to access directly, in-place
   * elements in the local section of a global array. It useful for writing new GA
   * operations.
   *
   * Each call to ga_access_segment has to be followed by a call to either 
   * ga_release_segment or ga_release_segmentupdate. You can access in
   * this fashion only local data. Since the data is shared with other
   * processes, you need to consider issues of mutual exclusion. This
   * operation is local. 
   * 
   * @param proc       - processor ID                                [input]
   * @param ptr        - points to location of first element         [output]
   * @param len        - length of locally held data                 [output]
   */
  void accessBlockSegment(int index, void *ptr, int *len) const;

  /**
   * "long" interface for accessBlockSegment
   */
  void accessBlockSegment(int index, void *ptr, int64_t *len) const;

  /**
   * Provides access to the local patch of the  global array. Returns 
   * leading dimension ld and and pointer for the data.  This routine 
   * will provide access to the ghost cell data residing on each processor. 
   * Calls to accessGhosts should normally follow a call to  
   * distribution  that returns coordinates of the visible data patch 
   * associated with a processor. You need to make sure that the coordinates 
   * of the patch are valid (test values returned from distribution). 
   *    
   * You can only access local data. 
   * This is a local operation. 
   * 
   * @param g_a                                                   [input]
   * @param dims[ndim] - array of dimensions of local patch, 
   *                          including ghost cells               [output]
   * @param ptr        - returns an index corresponding to the origin
   *                     the global array patch held locally on the
   *                     processor                                [output]
   * @param ld[ndim-1] - physical dimenstions of the local array patch,
   *                     including ghost cells                    [output]
   */
  void accessGhosts(int dims[], void *ptr, int ld[]) const;

  /**
   * "long" interface for accessGhosts
   */
  void accessGhosts(int64_t dims[], void *ptr, int64_t ld[]) const;
  
  /**
   * @param g_a                                                [input]
   * @param index           - index pointing to location of element
   *                          indexed by subscript[]           [output]
   * @param subscript[ndim] - array of integers that index desired
   *                          element                          [input]
   * @param ld[ndim-1]      - array of strides for local data patch.
   *                          These include ghost cell widths. [output]
   * 
   * This function can be used to return a pointer to any data element 
   * in the locally held portion of the global array and can be used to 
   * directly access ghost cell data. The array subscript refers to the 
   * local index of the  element relative to the origin of the local 
   * patch (which is assumed to be indexed by (0,0,...)). 
   * This is a  local operation. 
   */
  void accessGhostElement(void *ptr, int subscript[], int ld[]) const;

  /**
   * "long" interface for accessGhostElement
   */
  void accessGhostElement(void *ptr, int64_t subscript[], int64_t ld[]) const;
  
 /**
   * The arrays are aded together elemet-wise:
   * [for example: g_c.add(...,g_a, .., g_b);]
   * c = alpha * a + beta * b
   * The result c may replace one of he input arrays(a/b).
   * This is a collective operation.
   */
  void add(void *alpha, const GlobalArray * g_a, 
	   void *beta,  const GlobalArray * g_b) const;
  

  /**
   * Patches of arrays (which must have the same number of elements) are 
   * added together element-wise. 
   * c[ ][ ] = alpha * a[ ][ ] + beta * b[ ][ ]. 
   * This is a collective operation. 
   * @param g_a, g_b, g_c             global array        [input] 
   * @param alo[], ahi[]              patch of g_a         [input]
   * @param blo[], bhi[]              patch of g_b         [input]
   * @param clo[], chi[]              patch of g_c         [input]
   * @param alpha, beta               scale factors        [input]  
   */
  void addPatch (void *alpha, const GlobalArray * g_a, int alo[], int ahi[],
		 void *beta,  const GlobalArray * g_b, int blo[], int bhi[],
		 int clo[], int chi[]) const;

  /**
   * "long" interface for addPatch
   */
  void addPatch (void *alpha, const GlobalArray * g_a, int64_t alo[], int64_t ahi[],
		 void *beta,  const GlobalArray * g_b, int64_t blo[], int64_t bhi[],
		 int64_t clo[], int64_t chi[]) const;
  
 
  /**
   * Allocate internal memory etc. to create a global array
   */
  int allocate() const;

  /**
   * @param string - message string             [input]
   *
   * Check that the global array handle g_a is valid ... if not call 
   * ga_error with the string provided and some more info. 
   * This operation is local. 
   */
  void checkHandle(char* string) const;
  
  /**   
   * Compares distributions of two global arrays. Returns 0 if 
   * distributions are identical and 1 when they are not. 
   * This is a collective operation. 
   * @param g_a   - global array               [input]
   */
  int compareDistr(const GlobalArray *g_a) const;

  /**   
   * Copies elements in array represented by g_a into the array 
   * represented by g_b [say for example: g_b.copy(g_a);]. The arrays must be the same type, shape, 
   * and identically aligned. 
   * This is a collective operation.
   * @param g_a - global array       [input]
   */
  void copy(const GlobalArray *g_a) const; 
  
  /**
   * Copies elements in a patch of one array (ga) into another one (say for 
   * example:gb.copyPatch(...,ga,....); ). The patches of arrays may be of 
   * different shapes but must have the same number of elements. Patches must 
   * be nonoverlapping (if gb=ga). 
   *
   * trans = 'N' or 'n' means that the transpose operator should not be 
   * applied. trans = 'T' or 't' means that transpose operator should be 
   * applied. This is a collective operation. 
   * @param ga               - global array         [input]
   * @param alo[]             - ga patch coordinates [input] 
   * @param ahi[]             - ga patch coordinates [input]
   * @param blo[]            - gb patch coordinates [input]  
   * @param bhi[]            - gb patch coordinates [input]  
   */
  void copyPatch(char trans, const GlobalArray* ga, int alo[], int ahi[], 
		 int blo[], int bhi[]) const;

  /**
   * "long" interface for copyPatch
   */
  void copyPatch(char trans, const GlobalArray* ga, int64_t alo[], int64_t ahi[], 
		 int64_t blo[], int64_t bhi[]) const;
  
  /**
   * Computes element-wise dot product of the two arrays which must be of
   * the same types and same number of elements.
   *          return value = SUM_ij a(i,j)*b(i,j)
   * This is a collective operation. 
   * @param g_a   - array handle                 [input]
   */
  double ddot(const GlobalArray * g_a) const; 
  
  /**
   * Computes the element-wise dot product of the two (possibly transposed) 
   * patches which must be of the same type and have the same number of 
   * elements. 
   * @param  g_a                  - global array             [input]
   * @param  alo[], ahi[]         - g_a patch coordinates     [input]
   * @param  blo[], bhi[]         - g_b patch coordinates     [input]
   * @param  ta, tb               - transpose flags           [input]
   */
  double ddotPatch(char ta, int alo[], int ahi[], const GlobalArray * g_a, 
		   char tb, int blo[], int bhi[]) const;

  /**
   * "long" interface for ddotPatch
   */
  double ddotPatch(char ta, int64_t alo[], int64_t ahi[], const GlobalArray * g_a, 
		   char tb, int64_t blo[], int64_t bhi[]) const;
  
  /** Deallocates the array and frees any associated resources. */
  void destroy();
  
  /**
   * Performs one of the matrix-matrix operations: 
   * [say: g_c.dgemm(..., g_a, g_b,..);]
   *
   *     C := alpha*op( A )*op( B ) + beta*C, \n 
   * where op( X ) is one of \n 
   *     op( X ) = X   or   op( X ) = X', \n 
   * alpha and beta are scalars, and A, B and C are matrices, with op( A ) 
   * an m by k matrix, op( B ) a k by n matrix and C an m by n matrix. 
   * On entry, transa specifies the form of op( A ) to be used in the 
   * matrix multiplication as follows:\n  
   *         ta = 'N' or 'n', op( A ) = A.  \n 
   *         ta = 'T' or 't', op( A ) = A'. \n 
   * This is a collective operation. 
   * @param g_a,g_b- input arrays  [input]
   * @param ta, tb - transpose operators      [input]
   * @param m  - number of rows of op(A) and of matrix  C           [input]
   * @param n  - number of columns of op(B) and of matrix  C        [input]
   * @param k  - number of columns of op(A) and rows of matrix op(B)[input]
   * @param alpha, beta  - scale factors                            [input]
   */
  void dgemm(char ta, char tb, int m, int n, int k, double alpha,  
	     const GlobalArray *g_a, const GlobalArray *g_b,double beta) const;
  /**
   * "long" interface for dgemm
   */
  void dgemm(char ta, char tb, int64_t m, int64_t n, int64_t k, double alpha,  
	     const GlobalArray *g_a, const GlobalArray *g_b,double beta) const;
  
  /**
   * @param g_s     - Metric                         [input] 
   * @param g_v     - Global matrix to return evecs  [output]
   * @param eval    - Local array to return evals    [output]
   * 
   * Solve the generalized eigen-value problem returning all eigen-vectors 
   * and values in ascending order. The input matrices are not overwritten 
   * or destroyed. 
   * This is a collective operation. 
   */
  void diag(const GlobalArray *g_s, GlobalArray *g_v, void *eval) const;

  /**
   * Solve the generalized eigen-value problem returning all eigen-vectors 
   * and values in ascending order. Recommended for REPEATED calls if g_s 
   * is unchanged. Values of the control flag: 
   * 
   *          value       action/purpose 
   * 
   *           0          indicates first call to the eigensolver
   * 
   *          >0          consecutive calls (reuses factored g_s) 
   *
   *          <0          only erases factorized g_s; g_v and eval unchanged 
   *                      (should be called after previous use if another 
   *                      eigenproblem, i.e., different g_a and g_s, is to 
   *                      be solved) 
   *
   * The input matrices are not destroyed. 
   * This is a collective operation. 
   * @param control - Control flag                   [input]
   * @param g_a     - Matrix to diagonalize          [input]
   * @param g_s     - Metric                         [input] 
   * @param g_v     - Global matrix to return evecs  [output]
   * @param eval    - Local array to return evals    [output]
   */
  void diagReuse(int control, const GlobalArray *g_s, GlobalArray *g_v, 
		 void *eval) const;
  
  /**
   * Solve the standard (non-generalized) eigenvalue problem returning 
   * all eigenvectors and values in the ascending order. The input matrix 
   * is neither overwritten nor destroyed. 
   * This is a collective operation. 
   * @param g_v     - Global matrix to return evecs  [output]
   * @param eval    - Local array to return evals    [output]
   */
  void diagStd(GlobalArray *g_v, void *eval) const;

  void diagSeq(const GlobalArray * g_s, const GlobalArray * g_v, 
	       void *eval) const;
  
  void diagStdSeq(const GlobalArray * g_v, void *eval) const;
  
  /** 
   * If no array elements are owned by process 'me', the range is returned
   * as lo[]=-1 and hi[]=-2 for all dimensions. The operation is local.
   * @param iproc      - process number                            [input]
   * @param ndim       - number of dimensions of the global array
   * @param lo[ndim]   - array of starting indices for array section[input]
   * @param hi[ndim]   - array of ending indices for array section  [input]
   */
  void distribution(int me, int* lo, int* hi) const;
      
  /**
   * "long" interface of distribution() method for large array creations 
   */
  void distribution(int me, int64_t* lo, int64_t* hi) const;

  float fdot(const GlobalArray * g_a) const;

  float fdotPatch(char t_a, int alo[], int ahi[], const GlobalArray * g_b, 
		  char t_b, int blo[], int bhi[]) const;
  /**
   * "long" interface for fdotPatch
   */
  float fdotPatch(char t_a, int64_t alo[], int64_t ahi[], const GlobalArray * g_b, 
		  char t_b, int64_t blo[], int64_t bhi[]) const;

  /**
   * @param value   - pointer to the value of appropriate type 
   * (double/DoubleComplex/long) that matches array type.
   *
   * Assign a single value to all elements in the array.
   * This is a collective operation. 
   */
  void fill(void *value) const;
  
  /**
   * @param lo[], hi[]                patch of g_a         [input]
   * @param val                       value to fill        [input]
   *
   * Fill the patch with  value of 'val' 
   * This is a collective operation. 
   */
  void fillPatch (int lo[], int hi[], void *val) const;

  /**
   * "long" interface for fillPatch
   */
  void fillPatch (int64_t lo[], int64_t hi[], void *val) const;
  
  /** 
   * Gathers array elements from a global array into a local array. 
   * The contents of the input arrays (v, subscrArray) are preserved, 
   * but their contents might be (consistently) shuffled on return. 
   
   * for(k=0; k<= n; k++){
   * 
   *     v[k] = a[subsArray[k][0]][subsArray[k][1]][subsArray[k][2]]...;    
   *
   * }
   *
   * This is a one-sided operation.  
   * @param n   - number of elements              [input] 
   * @param  v[n]                - array containing values [input]        
   * @param  subsarray[n][ndim]  - array of subscripts for each element [input]
   */
  void gather(void *v, int * subsarray[], int n) const;

  /**
   * "long" interface for gather
   */
  void gather(void *v, int64_t * subsarray[], int64_t n) const;

  /**
   * One-side operations. 
   * Copies data from global array section to the local array buffer. The 
   * local array is assumed to be have the same number of dimensions as the 
   * global array. Any detected inconsitencies/errors in the input arguments
   * are fatal. 
   * 
   * Example: For ga_get operation transfering data from the [10:14,0:4] 
   * section of 2-dimensional 15x10 global array into local buffer 5x10 
   * array we have: lo={10,0}, hi={14,4}, ld={10}  
   *
   * @param lo[ndim] -array of starting indices for global array section[input]
   * @param hi[ndim] - array of ending indices for global array section[input]
   * @param buf - pointer to the local buffer array where the data goes[output]
   * @param ld[ndim-1] - array specifying leading dimensions/strides/extents 
   * for buffer array [input]
   */
  void get(int lo[], int hi[], void *buf, int ld[]) const;

  /**
   * "long" interface of get() method for large array creations 
   */
  void get(int64_t lo[], int64_t hi[], void *buf, int64_t ld[]) const;

  /**
   * The function retrieves the number of blocks along each coordinate dimension
   * and the dimensions of the individual blocks for a global array with a
   * block-cyclic data distribution.
   *
   * This is a local operation.
   *
   * @param num_blocks[ndim] - array containing number of blocks along each
   * coordinate direction
   * @param block_dims[ndim] - array containing block dimensions
   */
  void getBlockInfo(int num_blocks[], int block_dims[]);

  /**
   * This function returns 1 if the global array has some dimensions for 
   * which the ghost cell width is greater than zero, it returns 0 otherwise. 
   * This is a local operation. 
   */
  int hasGhosts() const;
  
  /**
   * Computes element-wise dot product of the two arrays which must be of
   * the same types and same number of elements.
   *          return value = SUM_ij a(i,j)*b(i,j)
   * This is a collective operation. 
   * @param g_a   - array handle                 [input]
   */
  Integer idot(const GlobalArray * g_a) const; 

  /**
   * @param  g_a                  - global array             [input]
   * @param  alo[], ahi[]         - g_a patch coordinates     [input]
   * @param  blo[], bhi[]         - g_b patch coordinates     [input]
   * @param  ta, tb               - transpose flags           [input]
   * 
   * Computes the element-wise dot product of the two (possibly transposed) 
   * patches which must be of the same type and have the same number of 
   * elements. 
   */
  long idotPatch(char ta, int alo[], int ahi[], const GlobalArray * g_a, 
		 char tb, int blo[], int bhi[]) const;

  /**
   * "long" interface for idotPatch
   */
  long idotPatch(char ta, int64_t alo[], int64_t ahi[], const GlobalArray * g_a, 
		 char tb, int64_t blo[], int64_t bhi[]) const;

  
  /** 
   * Returns data type and dimensions of the array. 
   * This operation is local.   
   * @param type - data type                     [output]
   * @param ndim - number of dimensions          [output]
   * @param dims - array of dimensions           [output]
   */
  void inquire(int *type, int *ndim, int dims[]) const;

  /**
   * "long" interface for inquire
   */
  void inquire(int *type, int *ndim, int64_t dims[]) const;

  /** 
   * Returns the name of an array represented by the handle g_a. 
   * This operation is local. 
   */
  char* inquireName() const;

  /**
   * Computes element-wise dot product of the two arrays which must be of
   * the same types and same number of elements.
   *          return value = SUM_ij a(i,j)*b(i,j)
   * This is a collective operation. 
   * @param g_a   - array handle                 [input]
   */
   long ldot(const GlobalArray * g_a) const; 

  /**
   * @param g_a     - coefficient matrix        [input]
   * Solves a system of linear equations 
   * 
   *            A * X = B 
   * using the Cholesky factorization of an NxN double precision symmetric 
   * positive definite matrix A (epresented by handle g_a). On successful 
   * exit B will contain the solution X. 
   * It returns: 
   * 
   *         = 0 : successful exit 
   * 
   *         > 0 : the leading minor of this order is not positive 
   *               definite and the factorization could 
   *               not be completed 
   * 
   * This is a collective operation. 
   */
  int lltSolve(const GlobalArray * g_a) const;

  /**
   * Return in owner the GA compute process id that 'owns' the data. If any 
   * element of subscript[] is out of bounds "-1" is returned. This operation 
   * is local. 
   * @param subscript[ndim]  element subscript    [output]
   */
  int locate(int subscript[]) const;

  /**
   * "long" interface for locate
   */
  int locate(int64_t subscript[]) const;
  
  /**
   * Return the list of the GA processes id that 'own' the data. Parts of the 
   * specified patch might be actually 'owned' by several processes. If lo/hi 
   * are out of bounds "0" is returned, otherwise return value is equal to the 
   * number of processes that hold the data. This operation is local.
   * 
   *   map[i][0:ndim-1]       - lo[i]
   * 
   *   map[i][ndim:2*ndim-1]  - hi[i]
   * 
   *   procs[i]               - processor id that owns data in patch 
   *                            lo[i]:hi[i] 
   * 
   * @param ndim          - number of dimensions of the global array
   * @param lo[ndim]      - array of starting indices for array section[input]
   * @param hi[ndim]      - array of ending indices for array section  [input]
   * @param map[][2*ndim] - array with mapping information           [output]
   * @param procs[nproc]  - list of processes that own a part of array 
   *                        section[output]
   */
  int locateRegion(int lo[], int hi[], int map[], int procs[]) const;

  /**
   * "long" interface for locateRegion
   */
  int locateRegion(int64_t lo[], int64_t hi[], int64_t map[], int procs[]) const;

  /**
   * @param trans   - transpose or not transpose [input]
   * @param g_a     - coefficient matrix         [input]
   * 
   * Solve the system of linear equations op(A)X = B based on the LU 
   * factorization. 
   * 
   * op(A) = A or A' depending on the parameter trans: 
   * 
   * trans = 'N' or 'n' means that the transpose operator should not 
   *          be applied. 
   * 
   * trans = 'T' or 't' means that the transpose operator should be applied. 
   * 
   * Matrix A is a general real matrix. Matrix B contains possibly multiple 
   * rhs vectors. The array associated with the handle g_b is overwritten 
   * by the solution matrix X. 
   * This is a collective operation. 
   */
  void luSolve(char trans, const GlobalArray * g_a) const;
  
  /**
   * @param g_a, g_b                  global array        [input] 
   * @param ailo, aihi, ajlo, ajhi    patch of g_a         [input]
   * @param bilo, bihi, bjlo, bjhi    patch of g_b         [input]
   * @param cilo, cihi, cjlo, cjhi    patch of g_c         [input]
   * @param alpha, beta               scale factors        [input]
   * @param transa, transb            transpose operators  [input]
   * 
   * ga_matmul_patch is a patch version of ga_dgemm: 
   * 
   *      C[cilo:cihi,cjlo:cjhi] := alpha* AA[ailo:aihi,ajlo:ajhi] *
   *                                BB[bilo:bihi,bjlo:bjhi] ) + 
   *                                beta*C[cilo:cihi,cjlo:cjhi],
   * 
   * where AA = op(A), BB = op(B), and op( X ) is one of 
   *      op( X ) = X   or   op( X ) = X',
   * 
   * Valid values for transpose arguments: 'n', 'N', 't', 'T'. It works 
   * for both double and DoubleComplex data tape. 
   * This is a collective operation. 
   */
  void matmulPatch(char transa, char transb, void* alpha, void *beta,
		   const GlobalArray *g_a, 
		   int ailo, int aihi, int ajlo, int ajhi,
		   const GlobalArray *g_b, 
		   int bilo, int bihi, int bjlo, int bjhi,
		   int cilo, int cihi, int cjlo, int cjhi) const;

  /**
   * "long" interface for matmulPatch
   */
  void matmulPatch(char transa, char transb, void* alpha, void *beta,
		   const GlobalArray *g_a, 
		   int64_t ailo, int64_t aihi, int64_t ajlo, int64_t ajhi,
		   const GlobalArray *g_b, 
		   int64_t bilo, int64_t bihi, int64_t bjlo, int64_t bjhi,
		   int64_t cilo, int64_t cihi, int64_t cjlo, int64_t cjhi) const;

  /**
   * N-dimensional Arrays:
   * @param g_a, g_b          global array                [input] 
   * @param alo, ahi          array of patch of g_a        [input]
   * @param blo, bhi          array of patch of g_b        [input]
   * @param clo, chi          array of patch of g_c        [input]
   * @param alpha, beta               scale factors        [input]
   * @param transa, transb            transpose operators  [input]
   * 
   * nga_matmul_patch is a n-dimensional patch version of ga_dgemm: 
   * 
   *      C[clo[]:chi[]] := alpha* AA[alo[]:ahi[]] *
   *                               BB[blo[]:bhi[]]) + 
   *                               beta*C[clo[]:chi[]],
   * 
   * where AA = op(A), BB = op(B), and op( X ) is one of 
   *      op( X ) = X   or   op( X ) = X',
   * 
   * Valid values for transpose arguments: 'n', 'N', 't', 'T'. It works 
   * for both double and DoubleComplex data tape. 
   * This is a collective operation. 
   */
  void matmulPatch(char transa, char transb, void* alpha, void *beta,
		   const GlobalArray *g_a, int *alo, int *ahi,
		   const GlobalArray *g_b, int *blo, int *bhi,
		   int *clo, int *chi) const;
  /**
   * "long" interface for matmulPatch
   */
  void matmulPatch(char transa, char transb, void* alpha, void *beta,
		   const GlobalArray *g_a, int64_t *alo, int64_t *ahi,
		   const GlobalArray *g_b, int64_t *blo, int64_t *bhi,
		   int64_t *clo, int64_t *chi) const;

  /**
   * This function merges all values in a patch of a mirrored array into
   * a patch in another global array g_b.
   *
   * This is a collective operation.
   *
   * @param alo[ndim],ahi[ndim] - patch indices of mirrored array [input]
   * @param blo[ndim],bhi[ndim] - patch indices of result array   [input]
   * @param g_a                 - global array containing result  [output]
   */
  void mergeDistrPatch(int alo[], int ahi[], GlobalArray *g_a,
                       int blo[], int bhi[]);

  /**
   * "long" interface for mergeDistrPatch
   */
  void mergeDistrPatch(int64_t alo[], int64_t ahi[], GlobalArray *g_a,
                       int64_t blo[], int64_t bhi[]);

  /**
   * This function adds together all copies of a mirrored array so that all
   * copies are the same.
   *
   * This is a collective operation.
   */
  void mergeMirrored();

  /**
   * Non-blocking accumalate operation. This is function performs an
   * accumulate operation and returns a nblocking handle. Completion of the
   * operation can be forced by calling the nbwait method on the handle.
   *
   * This is a onesided operation.
   *
   * @param lo[ndim],hi[ndim] - patch coordinates of block             [input]
   * @param buf               - local buffer containing data           [input]
   * @param ld[ndim-1]        - array of strides for local data        [input]
   * @param alpha             - multiplier for data before adding to existing
   *                            results                                [input]
   * @param nbhandle          - nonblocking handle                     [output]
   */
  void nbAcc(int lo[], int hi[], void *buf, int ld[], void *alpha,
             GANbhdl *nbhandle);

  /**
   * "long" interface for non-blocking accumulate on large arrays
   */
  void nbAcc(int64_t lo[], int64_t hi[], void *buf, int64_t ld[], void *alpha,
             GANbhdl *nbhandle);
  
  /**
   * Non-blocking get operation. This is function gets a data block from a
   * global array, copies it into a local buffer, and returns a nonblocking
   * handle. Completion of the operation can be forced by calling the nbwait
   * method on the handle.
   *
   * This is a onesided operation.
   *
   * @param lo[ndim],hi[ndim] - patch coordinates of block             [input]
   * @param buf               - local buffer to receive data           [input]
   * @param ld[ndim-1]        - array of strides for local data        [input]
   * @param nbhandle          - nonblocking handle                     [output]
   */
  void nbGet(int lo[], int hi[], void *buf, int ld[], GANbhdl *nbhandle);

  /**
   * "long" interface for nbGet for large arrays
   */
  void nbGet(int64_t lo[], int64_t hi[], void *buf, int64_t ld[], GANbhdl *nbhandle);

  /**
   * Non-blocking update operation for arrays with ghost cells. Ghost cells
   * along the coordinates specified in the mask array are updated with
   * non-blocking get calls. The mask array must contain either 0's or 1's.
   *
   * This is a onesided operation.
   *
   * @param mask[ndim]        - array with flags for directions that
   *                            are to be updated                     [input]
   * @param nbhandle          - nonblocking handle                    [output]
   */
  void nbGetGhostDir(int mask[], GANbhdl *hbhandle);

  /**
   * "long" interface for nbGetGhostDir
   */
  void nbGetGhostDir(int64_t mask[], GANbhdl *hbhandle);
  
  /**
   * @param nblock[ndim]  - number of partitions for each dimension [output]
   * 
   * Given a distribution of an array represented by the handle g_a, 
   * returns the number of partitions of each array dimension. 
   * This operation is local. 
   */
  void nblock(int numblock[]) const;

  /**
   * Non-blocking put operation. This is function puts a data block from a
   * local array, copies it into a global array, and returns a nonblocking
   * handle. Completion of the operation can be forced by calling the nbwait
   * method on the handle.
   *
   * This is a onesided operation.
   *
   * @param lo[ndim],hi[ndim] - patch coordinates of block             [input]
   * @param buf               - local buffer that supplies data        [input]
   * @param ld[ndim-1]        - array of strides for local data        [input]
   * @param nbhandle          - nonblocking handle                     [output]
   */
  void nbPut(int lo[], int hi[], void *buf, int ld[], GANbhdl *nbhandle);

  /**
   *  "long" interface for nbPut operation on large arrays.
   */
  void nbPut(int64_t lo[], int64_t hi[], void *buf, int64_t ld[], GANbhdl *nbhandle);

  /**
   * Returns the number of dimensions in array represented by the handle 
   * g_a. This operation is local. 
   */
  int ndim() const;
  
  /**
   * The pack subroutine is designed to compress the values in the source vector
   * g_src into a smaller destination array g_dest based on the values in an
   * integer mask array g_mask. The values lo and hi denote the range of
   * elements that should be compressed and icount is a variable that on output
   * lists the number of values placed in the compressed array. This operation
   * is the complement of the ga_unpack operation. An example is shown below
   *
   *  g_src->pack(g_dest, g_mask, 1, n, icount)
   *  g_mask:   1  0  0  0  0  0  1  0  1  0  0  1  0  0  1  1  0
   *  g_src:    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
   *  g_dest:   1  7  9 12 15 16
   *  icount:   6
   *
   *  The calling array is the source array. This is a collective operation.
   *
   * @param g_dest     - destination array                 [output]
   * @param g_mask     - mask array                        [input]
   * @param lo, hi     - coordinate interval to pack       [input]
   * @param icount     - number of packed elements         [output]
   */
  void pack(const GlobalArray *g_dest, const GlobalArray *g_mask,
            int lo, int hi, int *icount) const;

  /**
   * "long" interface for pack
   */
  void pack(const GlobalArray *g_dest, const GlobalArray *g_mask,
            int64_t lo, int64_t hi, int64_t *icount) const;

  /**
   * This subroutine enumerates the values of an array between elements lo and
   * hi starting with the value istart and incrementing each subsequent value by
   * inc. This operation is only applicable to 1-dimensional arrays. An example
   * of its use is shown below:
   *
   * call g_a->patch_enum(g_a, 1, n, 7, 2)
   * g_a:  7  9 11 13 15 17 19 21 23 ...
   *
   * This is a collective operation.
   *
   * @param lo, hi     - coordinate interval to enumerate     [input]
   * @param istart     - starting value of enumeration        [input]
   * @param inc        - increment value                      [input]
   */
   void patchEnum(int lo, int hi, int istart, int inc);

  /**
   * "long" interface for patchEnum
   */
   void patchEnum(int64_t lo, int64_t hi, int64_t istart, int64_t inc);

  /**  
   * @param ndim       - number of dimensions of the global array
   * @param lo[ndim]   - array of starting indices for array section [input] 
   * @param hi[ndim]   - array of ending indices for array section   [input]
   * @param buf        - pointer to the local buffer array           [input]
   * @param ld[ndim-1] - array specifying leading 
   *                     dimensions/strides/extents for buffer array [input]
   * @param double/DoubleComplex/long *alpha     scale factor 
   * 
   * Same as nga_acc except the indices can extend beyond the array 
   * boundary/dimensions in which case the library wraps them around. 
   * This is a one-sided and atomic operation.     
   */
  void periodicAcc(int lo[], int hi[], void* buf, int ld[], void* alpha) const;

  /**
   * "long" interface for periodicAcc
   */
  void periodicAcc(int64_t lo[], int64_t hi[], void* buf, int64_t ld[], void* alpha) const;
  
  /**
   * @param ndim       - number of dimensions of the global array
   * @param lo[ndim]   - array of starting indices for global array 
   *                     section [input] 
   * @param hi[ndim]   - array of ending indices for global array 
   *                     section [input] 
   * @param buf        - pointer to the local buffer array where the data 
   *                     goes    [output]
   * @param ld[ndim-1] - array specifying leading dimensions/strides/extents 
   *                     for buffer array [input]
   * 
   * Same as nga_get except the indices can extend beyond the array 
   * boundary/dimensions in which case the library wraps them around. 
   * This is a one-sided operation.
   */
  void periodicGet(int lo[], int hi[], void* buf, int ld[]) const;

  /**
   * "long" interface for periodicGet
   */
  void periodicGet(int64_t lo[], int64_t hi[], void* buf, int64_t ld[]) const;
  
  /**
   * @param ndim       - number of dimensions of the global array
   * @param lo[ndim]   - array of starting indices for global array 
   *                     section [input] 
   * @param hi[ndim]   - array of ending indices for global array 
   *                     section [input] 
   * @param buf        - pointer to the local buffer array where the data 
   *                     goes    [output]
   * @param ld[ndim-1] - array specifying leading dimensions/strides/extents 
   *                     for buffer array [input]
   * 
   * Same as nga_put except the indices can extend beyond the array 
   * boundary/dimensions in which case the library wraps them around. 
   * This is a one-sided operation.
   */
  void periodicPut(int lo[], int hi[], void* buf, int ld[]) const;

  /**
   * "long" interface for periodicPut
   */
  void periodicPut(int64_t lo[], int64_t hi[], void* buf, int64_t ld[]) const;
  
  /** 
   * Prints an entire array to the standard output. 
   * This is a collective operation. 
   */
  void print() const ; 
  
  /** 
   * Prints the array distribution. 
   * This is a collective operation. 
   */
  void printDistribution() const ;

  /** 
   * Prints the array distribution to a file. 
   * This is a collective operation. 
   */
   void printFile(FILE *file) const;

  /**
   * Prints a patch of g_a array to the standard output. If pretty has the 
   * value 0 then output is printed in a dense fashion. If pretty has the 
   * value 1 then output is formatted and rows/columns labeled. 
   
   * This is a collective operation.  
   * @param lo[]         - coordinates of the patch    [input]
   * @param hi[]         - coordinates of the patch    [input]
   * @param int pretty        - formatting flag        [input]
   */
  void printPatch(int* lo, int* hi, int pretty)  const;

  /**
   * "long" interface for printPatch
   */
  void printPatch(int64_t* lo, int64_t* hi, int pretty)  const;
  
  /**
   * @param ndim              number of array dimensions 
   * @param proc              process id                    [input] 
   * @param coord[ndim]       coordinates in processor grid [output] 
   * 
   * Based on the distribution of an array associated with handle g_a, 
   * determines coordinates of the specified processor in the virtual 
   * processor grid corresponding to the distribution of array g_a. The 
   * numbering starts from 0. The values of -1 means that the processor 
   * doesn't 'own' any section of array represented by g_a. 
   * This operation is local. 
   */
  void procTopology(int proc, int coord[]) const;
  
  /*void procTopology(int proc, int *prow, int *pcol);*/
  
  /**
   * Copies data from local array buffer to the global array section . The 
   * local array is assumed to be have the same number of dimensions as the 
   * global array. Any detected inconsitencies/errors in input arguments are 
   * fatal. This is a one-sided operation. 
   *
   * @param lo[ndim]-array of starting indices for global array section[input] 
   * @param hi[ndim]- array of ending indices for global array section [input] 
   * @param buf - pointer to the local buffer array where the data is [input]
   * @param ld[ndim-1]-array specifying leading dimensions/strides/extents for 
   * @param buffer array [input]
   */
  void put(int lo[], int hi[], void *buf, int ld[]) const;
  
  /**
   * "long" interface of put() method for large array creations 
   */
  void put(int64_t lo[], int64_t hi[], void *buf, int64_t ld[]) const;

  /**
   * @param ndim            - number of dimensions of the global array
   * @param subscript[ndim] - subscript array for the referenced element[input]
   * 
   * Atomically read and increment an element in an integer array. 
   * 
   *      *BEGIN CRITICAL SECTION*
   * 
   *       old_value = a(subscript)
   * 
   *       a(subscript) += inc
   * 
   *      *END CRITICAL SECTION*
   * 
   *       return old_value
   *
   * This is a one-sided and atomic operation. 
   */
  long readInc(int subscript[], long inc) const;

  /**
   * "long" interface for readInc
   */
  long readInc(int64_t subscript[], long inc) const;

  /**
   * @param ndim       - number of dimensions of the global array
   * @param lo[ndim]   - array of starting indices for array section [input]
   * @param hi[ndim]   - array of ending indices for array section   [input]
   * 
   * Releases access to a global array when the data was read only. 
   * Your code should look like: 
   * 
   *        g_a->distribution(myproc, lo,hi);
   *
   *        g_a->access(lo, hi, &ptr, ld);
   *            
   *          <operate on the data referenced by ptr> 
   *
   *        g_a->release(lo, hi);
   * 
   * NOTE: see restrictions specified for ga_access. 
   * This operation is local. 
   */
  void release(int lo[], int hi[]) const;

  /**
   * "long" interface for release
   */
  void release(int64_t lo[], int64_t hi[]) const;

  /**
   * @param index      - block index                       [input]
   *
   * Releases access to the block of data specified by the integer
   * index when data was accessed as read only. This is only applicable to
   * block-cyclic data distributions created using the simple block-cyclic
   * distribution.
   *
   * This is a local operation.
   */
  void releaseBlock(int idx) const;

  /**
   * @param index[ndim] - indices of block in array                [input]
   * @param ndim       - number of dimensions of the global array
   *
   * Releases access to the block of data specified by the subscript
   * array when data was accessed as read only. This is only applicable to
   * block-cyclic data distributions created using the SCALAPACK data
   * distribution.
   *
   * This is a local operation.
   */      
  void releaseBlockGrid(int index[]) const;
      
  /**
   * @param proc         - process ID/rank                   [input]
   *
   * Releases access to the block of locally held data for a block-cyclic
   * array, when data was accessed as read-only. This is a local operation.
   */
  void releaseBlockSegment(int proc) const;
      
  /**
   * @param ndim       - number of dimensions of the global array
   * @param lo[ndim]   - array of starting indices for array section [input]
   * @param hi[ndim]   - array of ending indices for array section   [input]
   * 
   * Releases access to the data. It must be used if the data was accessed 
   * for writing. NOTE: see restrictions specified for ga_access. 
   * This operation is local. 
   */
  void releaseUpdate(int lo[], int hi[]) const;

  /**
   * "long" interface for releaseUpdate
   */
  void releaseUpdate(int64_t lo[], int64_t hi[]) const;

  /**
   * @param index      - block index                       [input]

   * Releases access to the block of data specified by the integer index when
   * data was accessed in read-write mode. This is only applicable to
   * block-cyclic data distributions created using the simple block-cyclic
   * distribution.
  *
  * This is a local operation.
  */
  void releaseUpdateBlock(int idx) const;
      
  /**
   * @param index[ndim] - indices of block in array                [input
   * @param ndim            - number of dimensions of the global array
   *
   * Releases access to the block of data specified by the subscript
   * array when data was accessed in read-write mode. This is only applicable
   * to block-cyclic data distributions created using the SCALAPACK data
   * distribution.
   *
   * This is a local operation.
   */   
  void releaseUpdateBlockGrid(int index[]) const;

  /**
   * @param proc         - process ID/rank                   [input]
   *
   * Releases access to the block of locally held data for a block-cyclic
   * array, when data was accessed as read-only.
   *
   * This is a local operation.
   */
  void releaseUpdateBlockSegment(int proc) const;    
      
  /** 
   * Scales an array by the constant s. Note that the library is unable 
   * to detect errors when the pointed value is of different type than 
   * the array. 
   * This is a collective operation. 
   * @param value   - pointer to the value of appropriate type 
   *                  (double/DoubleComplex/long) that matches array type    
   */
  void scale(void *value) const;

  /** 
   * @param lo[], hi[]                patch of g_a         [input]
   * @param val                       scale factor         [input]
   * 
   * Scale an array by the factor 'val'. 
   * This is a collective operation. 
   */
  void scalePatch (int lo[], int hi[], void *val) const;

  /**
   * "long" interface for scalePatch
   */
  void scalePatch (int64_t lo[], int64_t hi[], void *val) const;
      
  /**
   * @param g_src     - handle for source array    [input]
   * @param g_dest    - handle for destination array [output]
   * @param g_mask    - handle for integer array representing mask [input]
   * @param lo, hi,   - low and high values of range on which operation
   *                    is performed       [input]
   * @param excl      - value to signify if masked values are included in
   *                    in add [input]
   * 
   * This operation will add successive elements in a source vector g_src
   * and put the results in a destination vector g_dest. The addition will
   * restart based on the values of the integer mask vector g_mask. The scan
   * is performed within the range specified by the integer values lo and
   * hi. Note that this operation can only be applied to 1-dimensional
   * arrays. The excl flag determines whether the sum starts with the value
   * in the source vector corresponding to the location of a 1 in the mask
   * vector (excl=0) or whether the first value is set equal to 0
   * (excl=1). Some examples of this operation are given below.
   *
   * g_src->scanAdd(g_dest, g_mask, 1, n, 0);
   * g_mask:   1  0  0  0  0  0  1  0  1  0  0  1  0  0  1  1  0
   * g_src:    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
   * g_dest:   1  3  6 10 16 21  7 15  9 19 30 12 25 39 15 16 33
   * 
   * g_src->scanAdd(g_dest, g_mask, 1, n, 1);
   * g_mask:   1  0  0  0  0  0  1  0  1  0  0  1  0  0  1  1  0
   * g_src:    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
   * g_dest:   0  1  3  6 10 15  0  7  0  9 19  0 12 25  0  0 16
   * 
   * This is a collective operation.
   */
  void scanAdd(const GlobalArray *g_dest, const GlobalArray *g_mask,
               int lo, int hi, int excl) const;
  /**
   * "long" interface for scanAdd
   */
  void scanAdd(const GlobalArray *g_dest, const GlobalArray *g_mask,
               int64_t lo, int64_t hi, int excl) const;

  /**
   * @param g_src    - handle for source array [input]
   * @param g_dest   - handle for destination array [output]
   * @param g_mask   - handle for integer array representing mask      [input]
   * @param lo, hi   - low and high values of range on which operation
   *                   is performed [input]
   *
   * This subroutine does a segmented scan-copy of values in the
   * source array g_src into a destination array g_dest with segments
   * defined by values in the integer mask array g_mask. The scan-copy
   * operation is only applied to the range between the lo and hi
   * indices. This operation is restriced to 1-dimensional arrays. The
   * resulting destination array will consist of segments of consecutive
   * elements with the same value. An example is shown below
   *
   * g_src->scanCopy(g_src, g_dest, g_mask, 1, n);
   * g_mask:   1  0  0  0  0  0  1  0  1  0  0  1  0  0  1  1  0
   * g_src:    1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17
   * g_dest:   1  1  1  1  1  1  7  7  9  9  9 12 12 12 15 16 16
   *
   * This is  a collective operation.
   */
  void scanCopy(const GlobalArray *g_dest, const GlobalArray *g_mask,
                int lo, int hi) const;

  /**
   * "long" interface for scanCopy
   */
  void scanCopy(const GlobalArray *g_dest, const GlobalArray *g_mask,
                int64_t lo, int64_t hi) const;
                 
  /** 
   * Scatters array elements into a global array. The contents of the input 
   * arrays (v,subscrArray) are preserved, but their contents might be 
   * (consistently) shuffled on return.    
   *
   *   for(k=0; k<= n; k++){
   *
   *     a[subsArray[k][0]][subsArray[k][1]][subsArray[k][2]]... = v[k];    
   *
   *   }
   * This is a one-sided operation.  
   * @param n   - number of elements              [input] 
   * @param  v[n]                - array containing values [input]        
   * @param  subsarray[n][ndim]  - array of subscripts for each element [input]
   */
  void scatter(void *v, int *subsarray[], int n) const;

  /**
   * "long" interface for scatter
   */
  void scatter(void *v, int64_t *subsarray[], int64_t n) const;

  /**
   * @param  op              - operator {"min","max"}               [input]
   * @param  val             - address where value should be stored [output] 
   * @param  subscript[ndim] - array index for the selected element [output]
   *
   * Returns the value and index for an element that is selected by the 
   * specified operator  in a global array corresponding to g_a handle. 
   * This is a collective operation. 
   */
  void selectElem(char *op, void* val, int index[]) const;

  /**
   * "long" interface for selectElem
   */
  void selectElem(char *op, void* val, int64_t index[]) const;

  /**
   * @param name       - array name [input]
   *
   * This function can be used to assign a unique character
   * string name to a global array handle that was obtained
   * using the createHandle function.
   *
   * This is a collective operation.
   */
  void setArrayName(char *name) const;

  /**
   *
   * @param dims[]       - array of block dimensions        [input]
   *
   * This subroutine is used to create a global array with a simple
   * block-cyclic data distribution. The array is broken up into blocks of
   * size dims and each block is numbered sequentially using a column major
   * indexing scheme. The blocks are then assigned in a simple round-robin
   * fashion to processors. This is illustrated in the figure below for an
   * array containing 25 blocks distributed on 4 processors. Blocks at the
   * edge of the array may be smaller than the block size specified in
   * dims. In the example below, blocks 4,9,14,19,20,21,22,23, and 24 might
   * be smaller thatn the remaining blocks. Most global array operations
   * are insensitive to whether or not a block-cyclic data distribution is
   * used, although performance may be slower in some cases if the global
   * array is using a block-cyclic data distribution. Individual data
   * blocks can be accessesed using the block-cyclic access functions.
   *
   * This is a collective operation.
   */
  void setBlockCyclic(int dims[]) const;

  /**
   * @param dims[]         - array of block dimensions   [input]
   * @param proc_grid[]    - processor grid dimensions   [input]
   *
   * This subroutine is used to create a global array with a
   * SCALAPACK-type block cyclic data distribution. The user  specifies
   * the dimensions of the processor grid in the array proc_grid. The
   * product of the processor grid dimensions must equal the number of
   * total number of processors  and the number of dimensions in the
   * processor grid must be the same as the number of dimensions in the
   * global array. The data blocks are mapped onto the processor grid
   * in a cyclic manner along each of the processor grid axes. This is
   * illustrated below for an array consisting of 25 data blocks
   * disributed on 6 processors. The 6 processors are configured in a 3
   * by 2 processor grid. Blocks at the edge of the array may be
   * smaller than the block size specified in dims. Most global array
   * operations are insensitive to whether or not a block-cyclic data
   * distribution is used, although performance may be slower in some
   * cases if the global array is using a block-cyclic data
   * distribution. Individual data blocks can be accessesed using the
   * block-cyclic access functions.
   *
   * This is a collective operation.
   */
  void setBlockCyclicProcGrid(int dims[], int proc_grid[]) const;

  /**
   * @param chunk[]  - array of chunk widths              [input]
   *
   * This function is used to set the chunk array for a global array handle
   * that was obtained using the createHandle function. The chunk array
   * is used to determine the minimum number of array elements assigned to
   * each processor along each coordinate direction.
   *
   * This is a collective operation.
   */
  void setChunk(int chunk[]) const;

  /**
   * "long" interface for setChunk
   */
  void setChunk(int64_t chunk[]) const;

  /**
   * @param ndim     - dimension of global array                 [input]
   * @param dims[]   - dimensions of global array                [input]
   * @param type     - data type of global array                 [input]
   *
   * This function can be used to set the array dimension, the coordinate
   * dimensions, and the data type assigned to a global array handle obtained
   * using the GA_Create_handle function.
   *
   * This is a collective operation.
   */
  void setData(int ndim, int dims[], int type) const;

  /**
   * "long" interface for setData
   */
  void setData(int ndim, int64_t dims[], int type) const;

      
  /**
   * @param width[ndim] - array of ghost cell widths     [input]
   *
   * This function can be used to set the ghost cell widths for a global
   * array handle that was obtained using the createHandle function. The
   * ghosts cells widths indicate how many ghost cells are used to pad the
   * locally held array data along each dimension. The padding can be set
   * independently for each coordinate dimension.
   *
   * This is a collective operation.
   */
  void setGhosts(int width[]) const;

  /**
   * "long" interface for setGhosts
   */
  void setGhosts(int64_t width[]) const;

  /**
   * @param mapc[s]       - starting index for each block; the size
   *                        s is the sum of all elements of the array
   *                        nblock                                     [input]
   * @param nblock[ndim]  - number of blocks that each dimension is
   *                        divided into                               [input]
   *
   * This function can be used to partition the array data among the
   * individual processors for a global array handle obtained using the
   * GA_Create_handle function.
   *
   * The distribution is specified as a Cartesian product of distributions
   * for each dimension. For example, the following figure demonstrates
   * distribution of a 2-dimensional array 8x10 on 6 (or more)
   * processors. nblock(2)={3, 2}, the size of mapc array is s=5 and array
   * mapc contains the following elements mapc={1, 3, 7, 1, 6}. The
   * distribution is nonuniform because, P1 and P4 get 20 elements each and
   * processors P0,P2,P3, and P5 only 10 elements each.
   *
   * The array width() is used to control the width of the ghost cell
   * boundary around the visible data on each processor. The local data of
   * the global array residing on each processor will have a layer width(n)
   * ghosts cells wide on either side of the visible data along the dimension
   * n.
   *
   * This is a collective operation.
   */
  void setIrregDistr(int mapc[], int nblock[]) const;

  /**
   * @param pHandle - processor group handle     [input]
   *
   * This function can be used to set the processor configuration assigned to
   * a global array handle that was obtained using the
   * createHandle function. It can be used to create mirrored arrays by
   * using the mirrored array processor configuration in this function
   * call. It can also be used to create an array on a processor group by
   * using a processor group handle in this call.
   *
   * This is a collective operation.
   */
  void setPGroup(PGroup *pHandle) const;

  /**
   * @param list     - list of processors that should contain data [input]
   * @param nprocs   - number of processors in list                [input]
   *
   * This function is used to restrict the number of processors in a global
   * array that actually contain data. It can also be used to rearrange the
   * layout of data on a processor from the default distribution. Only the
   * processes listed in list[] will actually contain data, the remaining
   * processes will be able to see the data in the global array but they will
   * not contain any of the global array data locally.
   */
  void setRestricted(int list[], int nprocs) const;

  /**
   * @param lo_proc  - low end of processor range   [input]
   * @param hi_proc  - high end of processor range  [input]
   *
   * This function is used to restrict the number of processors in a global
   * array that actually contain data. Only the processors in the range
   * [lo_proc:hi_proc] (inclusive) will actually contain data, the remaining
   * processes will be able to see the data in the global array but they will
   * not contain any of the global array data locally.
   */
  void setRestrictedRange(int lo_proc, int hi_proc) const;

      
  /**
   * @param g_a,g_b- handles to input arrays  [input]
   * @param g_c    - handles to output array  [input]
   * @param ta, tb - transpose operators      [input]
   * @param m  - number of rows of op(A) and of matrix  C           [input]
   * @param n  - number of columns of op(B) and of matrix  C        [input]
   * @param k  - number of columns of op(A) and rows of matrix op(B)[input]
   * @param alpha, beta  - scale factors                            [input]
   *
   * Performs one of the matrix-matrix operations: 
   *
   *     C := alpha*op( A )*op( B ) + beta*C,
   * where op( X ) is one of 
   *     op( X ) = X   or   op( X ) = X',
   * alpha and beta are scalars, and A, B and C are matrices, with op( A ) 
   * an m by k matrix, op( B ) a k by n matrix and C an m by n matrix. 
   * On entry, transa specifies the form of op( A ) to be used in the 
   * matrix multiplication as follows: 
   *
   *         ta = 'N' or 'n', op( A ) = A. 
   *
   *         ta = 'T' or 't', op( A ) = A'. 
   *
   * This is a collective operation. 
   */
  void sgemm(char ta, char tb, int m, int n, int k, float alpha,  
	     const GlobalArray *g_a, const GlobalArray *g_b, float beta) const;

  /**
   * "long" interface for sgemm
   */
  void sgemm(char ta, char tb, int64_t m, int64_t n, int64_t k, float alpha,  
	     const GlobalArray *g_a, const GlobalArray *g_b, float beta) const;
  
  /**
   * @param  g_a     - coefficient matrix         [input]
   *
   * Solves a system of linear equations 
   *            A * X = B 
   * It first will call the Cholesky factorization routine and, if 
   * sucessfully, will solve the system with the Cholesky solver. If 
   * Cholesky will be not be able to factorize A, then it will call the 
   * LU factorization routine and will solve the system with forward/backward 
   * substitution. On exit B will contain the solution X. 
   * It returns 
   *
   *            = 0 : Cholesky factoriztion was succesful 
   *
   *            > 0 : the leading minor of this order 
   *                  is not positive definite, Cholesky factorization 
   *                  could not be completed and LU factoriztion was used 
   *
   * This is a collective operation. 
   */
  int solve(const GlobalArray * g_a) const;

  /**
   * It computes the inverse of a double precision using the Cholesky 
   * factorization of a NxN double precision symmetric positive definite 
   * matrix A stored in the global array represented by g_a. On successful 
   * exit, A will contain the inverse. 
   * It returns 
   *
   *            = 0 : successful exit *
   
   *            > 0 : the leading minor of this order is not positive 
   *                  definite and the factorization could not be completed 
   *
   *            < 0 : it returns the index i of the (i,i) 
   *                  element of the factor L/U that is zero and 
   *                  the inverse could not be computed 
   *
   * This is a collective operation. 
   */
  int spdInvert() const;

  /**
   * @param ndim    - number of dimensions of the global array
   * @param lo[ndim]- array of starting indices for glob array section [input]
   * @param hi[ndim]- array of ending indices for global array section [input]
   * @param skip[ndim] - array of strides for each dimension [input]
   * @param buf        - pointer to local buffer array where data goes [output]
   * @param ld[ndim-1] - array specifying leading dimensions/strides/extents
   *                     for buffer array [input]
   * @param double/DoublComplex/long *alpha     - scale factor [input]
   *
   * This operation is the same as "acc", except that the values
   * corresponding to dimension n in buf are accumulated to every skip[n]
   * values of the global array.
   *
   * This is a one-sided operation.
   */
  void stridedAcc(int lo[], int hi[], int skip[], void*buf, int ld[]) const;
      
  /**
   * @param ndim    - number of dimensions of the global array
   * @param lo[ndim]- array of starting indices for glob array section [input]
   * @param hi[ndim]- array of ending indices for global array section [input]
   * @param skip[ndim] - array of strides for each dimension [input]
   * @param buf        - pointer to local buffer array where data goes [output]
   * @param ld[ndim-1] - array specifying leading dimensions/strides/extents
   *                     for buffer array [input]
   * @param double/DoublComplex/long *alpha     - scale factor [input]
   *
   * This operation is the same as "get", except that the values
   * corresponding to dimension n in buf are accumulated to every skip[n]
   * values of the global array.
   *
   * This is a one-sided operation.
   */
  void stridedGet(int lo[], int hi[], int skip[], void*buf, int ld[]) const;
      
  /**
   * @param ndim    - number of dimensions of the global array
   * @param lo[ndim]- array of starting indices for glob array section [input]
   * @param hi[ndim]- array of ending indices for global array section [input]
   * @param skip[ndim] - array of strides for each dimension [input]
   * @param buf        - pointer to local buffer array where data goes [output]
   * @param ld[ndim-1] - array specifying leading dimensions/strides/extents
   *                     for buffer array [input]
   * @param double/DoublComplex/long *alpha     - scale factor [input]
   *
   * This operation is the same as "put", except that the values
   * corresponding to dimension n in buf are accumulated to every skip[n]
   * values of the global array.
   *
   * This is a one-sided operation.
   */
  void stridedPut(int lo[], int hi[], int skip[], void*buf, int ld[]) const;
  
  /**
   * "long" interface for stridedPut
   */
  void stridedPut(int64_t lo[], int64_t hi[], int64_t skip[], void*buf, int64_t ld[]) const;

  /**
   * @param verbose     - If true print distribution info [input]
   *
   * Prints info about allocated arrays.
   */
  void summarize(int verbose) const;
      
  /** 
   * Symmmetrizes matrix A with handle A:=.5 * (A+A').
   * This is a collective operation 
   */
  void symmetrize() const;

  /**
   * This function returns the total number of blocks contained in a global
   * array with a block-cyclic data distribution. This is a local operation.
   */
  int totalBlocks() const;
      
  /**
   * Transposes a matrix: B = A', where A and B are represented by 
   * handles g_a and g_b [say, g_b.transpose(g_a);]. This is a collective 
   * operation.
   */
  void transpose(const GlobalArray * g_a) const;
  
  /**
   * @param g_src   - handle for source array      [input]
   * @param g_dest  - handle for destination array [output]
   * @param g_mask  - handle for integer array representing mask [input]
   * @param lo, hi  - low and high values of range on which operation
   *                  is performed
   * @param icount - number of values in uncompressed array [output]
   *
   * The unpack subroutine is designed to expand the values in the source
   * vector g_src into a larger destination array g_dest based on the values
   * in an integer mask array g_mask. The values lo and hi denote the range
   * of elements that should be compressed and icount is a variable that on
   * output lists the number of values placed in the uncompressed array. This
   * operation is the complement of the pack operation. An example is
   * shown below
   *
   * g_src->unpack(g_dest, g_mask, 1, n, &icount);
   * g_src:    1  7  9 12 15 16
   * g_mask:   1  0  0  0  0  0  1  0  1  0  0  1  0  0  1  1  0
   * g_dest:   1  0  0  0  0  0  7  0  9  0  0 12  0  0 15 16  0
   * icount:   6
   *
   *  This is a collective operation.
   */
  void unpack(GlobalArray *g_dest, GlobalArray *g_mask, int lo, int hi,
              int *icount) const;
  /**
   * "long" interface for unpack
   */
  void unpack(GlobalArray *g_dest, GlobalArray *g_mask,
            int64_t lo, int64_t hi, int64_t *icount) const;
      
  /**
   * This call updates the ghost cell regions on each processor with the 
   * corresponding neighbor data from other processors. The operation assumes 
   * that all data is wrapped around using periodic boundary data so that 
   * ghost cell data that goes beyound an array boundary is wrapped around to
   * the other end of the array. The updateGhosts call contains two   
   * sync   calls before and after the actual update operation. For some 
   * applications these calls may be unecessary, if so they can be removed 
   * using the maskSync subroutine. 
   * This is a collective operation. 
   */
  void updateGhosts() const;
 
  /**
   * This function can be used to update the ghost cells along individual 
   * directions. It is designed for algorithms that can overlap updates 
   * with computation. The variable dimension indicates which coordinate 
   * direction is to be updated (e.g. dimension = 1 would correspond to the 
   * y axis in a two or three dimensional system), the variable idir can take
   * the values +/-1 and indicates whether the side that is to be updated lies 
   * in the positive or negative direction, and cflag indicates whether or not
   * the corners on the side being updated are to be included in the update. 
   * The following calls would be equivalent to a call to   updateGhosts 
   * for a 2-dimensional system: 
   *
   * status = g_a->updateGhostDir(0,-1,1);\n 
   * status = g_a->updateGhostDir(0,1,1);\n 
   * status = g_a->updateGhostDir(1,-1,0);\n 
   * status = g_a->updateGhostDir(1,1,0);\n 
   *
   * The variable cflag is set equal to 1 (or non-zero) in the first two
   * calls so that the corner ghost cells are update, it is set equal to 0 in
   * the second two calls to avoid redundant updates of the corners. Note
   * that updating the ghosts cells using several independent calls to the
   * nga_update_ghost_dir functions is generally not as efficient as using
   * updateGhosts  unless the individual calls can be effectively overlapped
   * with computation.
   *
   * This is a  collective operation. 
   * @param g_a                                                    [input]
   * @param dimension    - array dimension that is to be updated   [input]
   * @param idir         - direction of update (+/- 1)             [input]
   * @param cflag        - flag (0/1) to include corners in update [input]
   */
  int updateGhostDir(int dimension, int idir, int cflag) const;

  /**
   * This operation is designed to extract ghost cell data from a global array
   * and copy it to a local array. If the request can be satisfied using
   * completely local data, then a local copy will be used. Otherwise, the
   * method calls periodicGet. The request can be satisfied locally if
   * lo is greater than or equal to the lower bound of data held on the
   * processor minus the ghost cell width and hi is less than or equal to the
   * upper bound of data held on the processor plus the ghost cell width. Cell
   * indices using the global address space should be used for lo and hi. These
   * may exceed the global array dimensions.
   *
   * @param lo[ndim] -array of starting indices for global array section[input]
   * @param hi[ndim] - array of ending indices for global array section[input]
   * @param buf - pointer to the local buffer array where the data goes[output]
   * @param ld[ndim-1] - array specifying leading dimensions/strides/extents 
   * for buffer array [input]
   */
  void getGhostBlock(int lo[], int hi[], void *buf, int ld[]) const;

  /**
   * "long" interface for getGhostBlock
   */
  void getGhostBlock(int64_t lo[], int64_t hi[], void *buf, int64_t ld[]) const;

  /**
   * Computes element-wise dot product of the two arrays which must be of
   * the same types and same number of elements.
   *          return value = SUM_ij a(i,j)*b(i,j)
   * This is a collective operation. 
   * @param g_a   - array handle                 [input]
   */
  DoubleComplex zdot(const GlobalArray * g_a) const; 

  /**
   * @param  g_a                  - global array             [input]
   * @param  alo[], ahi[]         - g_a patch coordinates     [input]
   * @param  blo[], bhi[]         - g_b patch coordinates     [input]
   * @param  ta, tb               - transpose flags           [input]
   *
   * Computes the element-wise dot product of the two (possibly transposed) 
   * patches which must be of the same type and have the same number of 
   * elements. 
   */
  DoubleComplex zdotPatch(char ta, int alo[], int ahi[], 
			  const GlobalArray * g_a, char tb, int blo[], 
			  int bhi[]) const;

  /**
   * "long" interface for zdotPatch
   */
  DoubleComplex zdotPatch(char ta, int64_t alo[], int64_t ahi[], 
			  const GlobalArray * g_a, char tb, int64_t blo[], 
			  int64_t bhi[]) const;
  
  /** 
   * Sets value of all elements in the array to zero. 
   * This is a collective operation. 
   */
  void zero() const;
  
  /**
   * @param  lo[], hi[]               [input]
   *
   * Set all the elements in the patch to zero. 
   * This is a collective operation. 
   */
  void zeroPatch (int lo[], int hi[]) const;

  /**
   * "long" interface for zeroPatch
   */
  void zeroPatch (int64_t lo[], int64_t hi[]) const;
  
  /**
   * @param g_a,g_b- handles to input arrays  [input]
   * @param g_c    - handles to output array  [input]
   * @param ta, tb - transpose operators      [input]
   * @param m  - number of rows of op(A) and of matrix  C           [input]
   * @param n  - number of columns of op(B) and of matrix  C        [input]
   * @param k  - number of columns of op(A) and rows of matrix op(B)[input]
   * @param alpha, beta  - scale factors                            [input]
   *
   * Performs one of the matrix-matrix operations: 
   *     C := alpha*op( A )*op( B ) + beta*C,
   * where op( X ) is one of 
   *     op( X ) = X   or   op( X ) = X',
   * alpha and beta are scalars, and A, B and C are matrices, with op( A ) 
   * an m by k matrix, op( B ) a k by n matrix and C an m by n matrix. 
   * On entry, transa specifies the form of op( A ) to be used in the 
   * matrix multiplication as follows: 
   *
   *         ta = 'N' or 'n', op( A ) = A. 
   *
   *         ta = 'T' or 't', op( A ) = A'. *
   *
   * This is a collective operation. 
   */
  void zgemm(char ta, char tb, int m, int n, int k, DoubleComplex alpha,  
	     const GlobalArray *g_a, const GlobalArray *g_b, 
	     DoubleComplex beta) const;

  /**
   * "long" interface for zgemm
   */
  void zgemm(char ta, char tb, int64_t m, int64_t n, int64_t k, DoubleComplex alpha,  
	     const GlobalArray *g_a, const GlobalArray *g_b, 
	     DoubleComplex beta) const;
  
  /**
   * New additional functionalities from Limin. 
   */
  /**
   * Take element-wise absolute value of the array. 
   * This is a collective operation. 
   */
   void absValue() const; 

  /**
   * Take element-wise absolute value of the patch. 
   * This is a collective operation.
   * @param lo[], hi[]           - g_a patch coordinates   [input] 
   */
   void absValuePatch(int *lo, int *hi) const;

  /**
   * "long" interface for absValuePatch
   */
   void absValuePatch(int64_t *lo, int64_t *hi) const;

  /**
   * Add the constant pointed by alpha to each element of the array. 
   * This is a collective operation. 
   * @param double/complex/int/long/float      *alpha      [input] 
   */
   void addConstant(void* alpha) const;
 
  /**
   * Add the constant pointed by alpha to each element of the patch. 
   * This is a collective operation. 
   * @param lo[], hi[]           - g_a patch coordinates  [input] 
   * @param double/complex/int/long/float      *alpha     [input] 
   */
   void addConstantPatch(int *lo, int *hi, void *alpha) const;

  /**
   * "long" interface for addConstantPatch
   */
   void addConstantPatch(int64_t *lo, int64_t *hi, void *alpha) const;
  
  /**
   * Take element-wise reciprocal of the array. 
   * This is a collective operation. 
   */
   void recip() const;
  
  /**
   * Take element-wise reciprocal of the patch. 
   * This is a collective operation. 
   * @param   lo[], hi[]           - g_a patch coordinates     [input] 
   */
   void recipPatch(int *lo, int *hi) const;

  /**
   * "long" interface for recipPatch
   */
   void recipPatch(int64_t *lo, int64_t *hi) const;
  
  
  /**
   * Computes the element-wise product of the two arrays 
   * which must be of the same types and same number of 
   * elements. For two-dimensional arrays, 
   *
   *            c(i, j)  = a(i,j)*b(i,j) 
   *
   * The result (c) may replace one of the input arrays (a/b). 
   * This is a collective operation. 
   * @param g_a, g_b - global array         [input] 
   */
   void elemMultiply(const GlobalArray * g_a, const GlobalArray * g_b) const;
  
  
  /**
   * Computes the element-wise product of the two patches 
   * which must be of the same types and same number of 
   * elements. For two-dimensional arrays, 
   * 
   *             c(i, j)  = a(i,j)*b(i,j) 
   * 
   * The result (c) may replace one of the input arrays (a/b). 
   * This is a collective operation.
   * @param g_a, g_b - global array                           [input] 
   * @param alo[], ahi[]           - g_a patch coordinates     [input] 
   * @param blo[], bhi[]           - g_b patch coordinates     [input] 
   * @param clo[], chi[]           - g_c patch coordinates     [output]  
   */ 
   void elemMultiplyPatch(const GlobalArray * g_a,int *alo,int *ahi,
				 const GlobalArray * g_b,int *blo,int *bhi,
				 int *clo,int *chi) const;
  /**
   * "long" interface for elemMultiplyPatch
   */
   void elemMultiplyPatch(const GlobalArray * g_a,int64_t *alo,int64_t *ahi,
				 const GlobalArray * g_b,int64_t *blo,int64_t *bhi,
				 int64_t *clo,int64_t *chi) const;

  /**
   * @param g_a, g_b - global array         [input] 
   *
   * Computes the element-wise quotient of the two arrays 
   * which must be of the same types and same number of 
   * elements. For two-dimensional arrays, 
   * 
   *             c(i, j)  = a(i,j)/b(i,j) 
   * 
   * The result (c) may replace one of the input arrays (a/b). If one of 
   * the elements of array g_b is zero, the quotient for the element of g_c 
   * will be set to GA_NEGATIVE_INFINITY. 
   * This is a collective operation. 
   */
   void elemDivide(const GlobalArray * g_a, const GlobalArray * g_b) const;
  
  /**
   * Computes the element-wise quotient of the two patches 
   * which must be of the same types and same number of 
   * elements. For two-dimensional arrays, 
   *
   *            c(i, j)  = a(i,j)/b(i,j) 
   *
   * The result (c) may replace one of the input arrays (a/b). 
   * This is a collective operation. 
   * @param g_a, g_b - global array                           [input] 
   * @param alo[], ahi[]           - g_a patch coordinates     [input] 
   * @param blo[], bhi[]           - g_b patch coordinates     [input] 
   * @param clo[], chi[]           - g_c patch coordinates     [output] 
   *
   */ 
   void elemDividePatch(const GlobalArray * g_a,int *alo,int *ahi,
			       const GlobalArray * g_b,int *blo,int *bhi,
			       int *clo,int *chi) const;
  /**
   * "long" interface for elemDividePatch
   */
   void elemDividePatch(const GlobalArray * g_a,int64_t *alo,int64_t *ahi,
			       const GlobalArray * g_b,int64_t *blo,int64_t *bhi,
			       int64_t *clo,int64_t *chi) const;

  /**
   * Computes the element-wise maximum of the two arrays 
   * which must be of the same types and same number of 
   * elements. For two dimensional arrays, 
   *
   *                c(i, j)  = max{a(i,j), b(i,j)} 
   *
   * The result (c) may replace one of the input arrays (a/b). 
   * This is a collective operation. 
   * @param g_a, g_b - global array         [input] 
   */
   void elemMaximum(const GlobalArray * g_a, const GlobalArray * g_b) const;
  
  
  /**
   * Computes the element-wise maximum of the two patches 
   * which must be of the same types and same number of 
   * elements. For two-dimensional of noncomplex arrays, 
   *
   *             c(i, j)  = max{a(i,j), b(i,j)} 
   *
   * If the data type is complex, then 
   *     c(i, j).real = max{ |a(i,j)|, |b(i,j)|} while c(i,j).image = 0. 
   *
   * The result (c) may replace one of the input arrays (a/b). 
   * This is a collective operation. 
   * @param g_a, g_b - global array         [input]   
   * @param alo[], ahi[]           - g_a patch coordinates     [input] 
   * @param blo[], bhi[]           - g_b patch coordinates     [input] 
   * @param clo[], chi[]           - g_c patch coordinates     [output] 
   */
    void elemMaximumPatch(const GlobalArray * g_a,int *alo,int *ahi,
				 const GlobalArray * g_b,int *blo,int *bhi,
				 int *clo,int *chi) const;
   /**
    * "long" interface for elemMaximumPatch
    */
    void elemMaximumPatch(const GlobalArray * g_a,int64_t *alo,int64_t *ahi,
				 const GlobalArray * g_b,int64_t *blo,int64_t *bhi,
				 int64_t *clo,int64_t *chi) const;

   /**
    * Computes the element-wise minimum of the two arrays 
    * which must be of the same types and same number of 
    * elements. For two dimensional arrays, 
    * 
    *             c(i, j)  = min{a(i,j), b(i,j)} 
    * 
    * The result (c) may replace one of the input arrays (a/b). 
    * This is a collective operation. 
    * @param g_a, g_b - global array         [input] 
    */
    void elemMinimum(const GlobalArray * g_a, const GlobalArray * g_b) const;
   
   /**
    * Computes the element-wise minimum of the two patches 
    * which must be of the same types and same number of 
    * elements. For two-dimensional of noncomplex arrays, 
    * 
    *             c(i, j)  = min{a(i,j), b(i,j)} 
    * 
    * If the data type is complex, then 
    *             c(i, j).real = min{ |a(i,j)|, |b(i,j)|} while c(i,j).image = 0. 
    * 
    * The result (c) may replace one of the input arrays (a/b). 
    * This is a collective operation. 
    * @param g_a, g_b - global array                           [input] 
    * @param alo[], ahi[]           - g_a patch coordinates     [input] 
    * @param blo[], bhi[]           - g_b patch coordinates     [input] 
    * @param clo[], chi[]           - g_c patch coordinates     [output] 
    */
    void elemMinimumPatch(const GlobalArray * g_a,int *alo,int *ahi,
				 const GlobalArray * g_b,int *blo,int *bhi,
				 int *clo,int *chi) const;

    /**
     * "long" interface for elemMinimumPatch
     */
    void elemMinimumPatch(const GlobalArray * g_a, int64_t *alo, int64_t *ahi,
				 const GlobalArray * g_b, int64_t *blo, int64_t *bhi,
				 int64_t *clo, int64_t *chi) const;
  
   /** 
    * Calculates the largest multiple of a vector g_b that can be added 
    * to this vector g_a while keeping each element of this vector 
    * nonnegative. 
    * This is a collective operation. 
    * @param g_a, g_b - global array where g_b is the step direction.[input] 
    * @param step        - the maximum step                           [output] 
    */  
    void stepMax(const GlobalArray * g_a, double *step) const;
   
   void stepMaxPatch(int *alo, int *ahi, 
			    const GlobalArray * g_b, int *blo, int *bhi, 
			    double *step) const;
   /**
    * "long" interface for stepMaxPatch
    */
   void stepMaxPatch(int64_t *alo, int64_t *ahi, 
			    const GlobalArray * g_b, int64_t *blo, int64_t *bhi, 
			    double *step) const;
  
  /** Matrix Operations */
  
  /** 
   * Adds this constant to the diagonal elements of the matrix. 
   * This is a collective operation. 
   * @param double/complex/int/long/float      *c               [input] 
   */
   void shiftDiagonal(void *c) const;
  
  /**
   * Sets the diagonal elements of this matrix g_a with the elements of the 
   * vector g_v.  This is a collective operation. 
   * @param g_v - global array       [input]  
   */
  void setDiagonal(const GlobalArray * g_v) const;
  
  /**
   * Sets the diagonal elements of this matrix g_a with zeros. 
   * This is a collective operation. 
   */
   void zeroDiagonal() const;
  
  /**
   * Adds the elements of the vector g_v to the diagonal of this matrix g_a. 
   * This is a collective operation. 
   * @param g_v - global array       [input] 
   */
   void addDiagonal(const GlobalArray * g_v) const;
  
  /**
   * Inserts the diagonal elements of this matrix g_a into the vector g_v. 
   * This is a collective operation. 
   * @param g_v - global array       [input] 
   */
   void getDiagonal(const GlobalArray * g_a) const;
  
  /**
   * Scales the rows of this matrix g_a using the vector g_v. 
   * This is a collective operation.  
   * @param g_v - global array       [input] 
   */
   void scaleRows(const GlobalArray * g_v) const;
  
  /** 
   * Scales the columns of this matrix g_a using the vector g_v. 
   * This is a collective operation. 
   * @param g_v - global array       [input] 
   */
   void scaleCols(const GlobalArray * g_v) const;
  
  /**
   * Computes the 1-norm of the matrix or vector g_a. 
   * This is a collective operation. 
   * @param nm - matrix/vector 1-norm value 
   */
   void norm1(double *nm) const;
  
  /**
   * Computes the 1-norm of the matrix or vector g_a. 
   * This is a collective operation. 
   * @param nm - matrix/vector 1-norm value 
   */
   void normInfinity(double *nm) const;
  
  /**
   * Computes the componentwise Median of three arrays g_a, g_b, and g_c, and 
   * stores the result in this array g_m.  The result (m) may replace one of the 
   * input arrays (a/b/c). This is a collective operation. 
   * @param g_a, g_b, g_c- global array       [input] 
   */
   void median(const GlobalArray * g_a, const GlobalArray * g_b, 
		      const GlobalArray * g_c) const;
  
  /**
   * Computes the componentwise Median of three patches g_a, g_b, and g_c, and 
   * stores the result in this patch g_m.  The result (m) may replace one of the 
   * input patches (a/b/c). This is a collective operation. 
   * @param g_a, g_b, g_c - global array         [input] 
   * @param alo[], ahi[]           - g_a patch coordinates     [input] 
   * @param blo[], bhi[]           - g_b patch coordinates     [input] 
   * @param clo[], chi[]           - g_c patch coordinates     [intput] 
   * @param mlo[], mhi[]         - g_m patch coordinates     [output] 
   */
   void medianPatch(const GlobalArray * g_a, int *alo, int *ahi, 
			       const GlobalArray * g_b, int *blo, int *bhi, 
			       const GlobalArray * g_c, int *clo, int *chi, 
			       int *mlo, int *mhi) const;
  /**
   * "long" interface for medianPatch
   */
   void medianPatch(const GlobalArray * g_a, int64_t *alo, int64_t *ahi, 
			       const GlobalArray * g_b, int64_t *blo, int64_t *bhi, 
			       const GlobalArray * g_c, int64_t *clo, int64_t *chi, 
			       int64_t *mlo, int64_t *mhi) const;

   GlobalArray& operator=(const GlobalArray &g_a);
   int operator==(const GlobalArray &g_a) const;
   int operator!=(const GlobalArray &g_a) const;
  
 private:
  int mHandle;      /* g_a handle */
};

#endif // _GLOBALARRAY_H 
