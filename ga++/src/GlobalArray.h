#define DEF_NDIM 2
#define DEF_DIMS 10

/**
 * This is the GlobalArray class.
 */
class GlobalArray { 

 public:
  explicit GlobalArray(int dim1);
  GlobalArray(int type, int ndim, int dims[], char *arrayname, int chunk[]);
  GlobalArray(int type, int ndim, int dims[], char *arrayname, int block[],
	      int maps[]);
  GlobalArray(int type, int ndim, int dims[], int width[], char *arrayname, 
	      int chunk[], char dummy);
  GlobalArray(int type, int ndim, int dims[], int width[], char *arrayname, 
	      int block[], int maps[], char dummy);
  GlobalArray(const GlobalArray &g_a, char *arrayname); // check: for temp object created here
  GlobalArray(const GlobalArray &g_a);/* copy constructor */
  GlobalArray();
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
  void acc(int lo[], int hi[], void *buf, int ld[], void *alpha);
  
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
  void access(int lo[], int hi[], void *ptr, int ld[]);

  /**
   * Provides access to the local patch of the  global array. Returns 
   * leading dimension ld and and pointer for the data.  This routine 
   * will provide access to the ghost cell data residing on each processor. 
   * Calls to NGA_Access_ghosts should normally follow a call to  
   * NGA_Distribution  that returns coordinates of the visible data patch 
   * associated with a processor. You need to make sure that the coordinates 
   * of the patch are valid (test values returned from NGA_Distribution). 
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
  void accessGhosts(int dims[], void *ptr, int ld[]);
  
  /**
   * @param g_a                                                [input]
   * @param index           - index pointing to location of element
   *                          indexed by subscript[]           [output]
   * @param subscript[ndim] - array of integers that index desired
   *                          element                          [input]
   * @param ld[ndim-1]      - array of strides for local data patch.
   *                          These include ghost cell widths. [output]
   * This function can be used to return a pointer to any data element 
   * in the locally held portion of the global array and can be used to 
   * directly access ghost cell data. The array subscript refers to the 
   * local index of the  element relative to the origin of the local 
   * patch (which is assumed to be indexed by (0,0,...)). 
   * This is a  local operation. 
   */
  void accessGhostElement(void *ptr, int subscript[], int ld[]);
  
 /**
   * The arrays are aded together elemet-wise:
   * c = alpha * a + beta * b
   * The result c may replace one of he input arrays(a/b).
   * This is a collective operation.
   */
  void add(void *alpha, const GlobalArray * g_a, 
	   void *beta,  const GlobalArray * g_b);
  

  /**
   * Patches of arrays (which must have the same number of elements) are 
   * added together element-wise. 
   * c[ ][ ] = alpha * a[ ][ ] + beta * b[ ][ ]. 
   * This is a collective operation. 
   * @param g_a, g_b, g_c             array handles        [input] 
   * @param alo[], ahi[]              patch of g_a         [input]
   * @param blo[], bhi[]              patch of g_b         [input]
   * @param clo[], chi[]              patch of g_c         [input]
   * @param alpha, beta               scale factors        [input]  
   */
  void addPatch (void *alpha, const GlobalArray * g_a, int alo[], int ahi[],
		 void *beta,  const GlobalArray * g_b, int blo[], int bhi[],
		 int clo[], int chi[]);
  
 
  /**
   * @param g_a    - array handle               [input]
   * @param string - message string             [input]
   *
   * Check that the global array handle g_a is valid ... if not call 
   * ga_error with the string provided and some more info. 
   * This operation is local. 
   */
  void checkHandle(char* string);
  
  /**   
   * @param g_a, g_b    - array handles               [input]
   * Compares distributions of two global arrays. Returns 0 if 
   * distributions are identical and 1 when they are not. 
   * This is a collective operation. 
   */
  int compareDistr(const GlobalArray *g_a);

  /**   
   * @param g_a - array handles       [input]
   * Copies elements in array represented by g_a into the array 
   * represented by g_b. The arrays must be the same type, shape, 
   * and identically aligned. 
   * This is a collective operation.
   */
  void copy(const GlobalArray *g_a); 
  
  /**
   * Copies elements in a patch of one array into another one. The patches of 
   * arrays may be of different shapes but must have the same number of 
   * elements. Patches must be nonoverlapping (if gb=ga). 

   * trans = 'N' or 'n' means that the transpose operator should not be 
   * applied. trans = 'T' or 't' means that transpose operator should be 
   * applied. This is a collective operation. 
   * @param ga               - array handles         [input]
   * @param alo[]             - ga patch coordinates [input] 
   * @param ahi[]             - ga patch coordinates [input]
   * @param blo[]            - gb patch coordinates [input]  
   * @param bhi[]            - gb patch coordinates [input]  
   */
  void copyPatch(char trans, GlobalArray* ga, int alo[], int ahi[], int blo[], int bhi[]);
  
  /**
   * Computes element-wise dot product of the two arrays which must be of
   * the same types and same number of elements.
   *          return value = SUM_ij a(i,j)*b(i,j)
   * This is a collective operation. 
   * @param g_a   - array handle                 [input]
   */
  double ddot(const GlobalArray * g_a); 
  
  /**
   * @param  g_a                  - array handles             [input]
   * @param  alo[], ahi[]         - g_a patch coordinates     [input]
   * @param  blo[], bhi[]         - g_b patch coordinates     [input]
   * @param  ta, tb               - transpose flags           [input]
   * Computes the element-wise dot product of the two (possibly transposed) 
   * patches which must be of the same type and have the same number of 
   * elements. 
   */
  double ddotPatch(char ta, int alo[], int ahi[],
		   const GlobalArray * g_a, char tb, int blo[], int bhi[]);
  
  /** Deallocates the array and frees any associated resources. */
  void destroy();
  
  /**
   * @param g_a,g_b- handles to input arrays  [input]
   * @param g_c    - handles to output array  [input]
   * @param ta, tb - transpose operators      [input]
   * @param m  - number of rows of op(A) and of matrix  C           [input]
   * @param n  - number of columns of op(B) and of matrix  C        [input]
   * @param k  - number of columns of op(A) and rows of matrix op(B)[input]
   * @param alpha, beta  - scale factors                            [input]
   * Performs one of the matrix-matrix operations: 
   *     C := alpha*op( A )*op( B ) + beta*C,
   * where op( X ) is one of 
   *     op( X ) = X   or   op( X ) = X',
   * alpha and beta are scalars, and A, B and C are matrices, with op( A ) 
   * an m by k matrix, op( B ) a k by n matrix and C an m by n matrix. 
   * On entry, transa specifies the form of op( A ) to be used in the 
   * matrix multiplication as follows: 
   *         ta = 'N' or 'n', op( A ) = A. 
   *         ta = 'T' or 't', op( A ) = A'. 
   * This is a collective operation. 
   */
  void dgemm(char ta, char tb, int m, int n, int k, double alpha,  
		const GlobalArray *g_a, const GlobalArray *g_b, double beta);
  
  /**
   * @param g_s     - Metric                         [input] 
   * @param g_v     - Global matrix to return evecs  [output]
   * @param eval    - Local array to return evals    [output]
   * Solve the generalized eigen-value problem returning all eigen-vectors 
   * and values in ascending order. The input matrices are not overwritten 
   * or destroyed. 
   * This is a collective operation. 
   */
  void diag(const GlobalArray *g_s, GlobalArray *g_v, void *eval);

  /**
   * @param control - Control flag                   [input]
   * @param g_a     - Matrix to diagonalize          [input]
   * @param g_s     - Metric                         [input] 
   * @param g_v     - Global matrix to return evecs  [output]
   * @param eval    - Local array to return evals    [output]
   * Solve the generalized eigen-value problem returning all eigen-vectors 
   * and values in ascending order. Recommended for REPEATED calls if g_s 
   * is unchanged. Values of the control flag: 
   *          value       action/purpose 
   *           0          indicates first call to the eigensolver 
   *          >0          consecutive calls (reuses factored g_s) 
   *          <0          only erases factorized g_s; g_v and eval unchanged 
   *                      (should be called after previous use if another 
   *                      eigenproblem, i.e., different g_a and g_s, is to 
   *                      be solved) 
   * The input matrices are not destroyed. 
   * This is a collective operation. 
   */
  void diagReuse(int control, const GlobalArray *g_s, GlobalArray *g_v, 
		 void *eval);
  
  /**
   * @param g_v     - Global matrix to return evecs  [output]
   * @param eval    - Local array to return evals    [output]
   * Solve the standard (non-generalized) eigenvalue problem returning 
   * all eigenvectors and values in the ascending order. The input matrix 
   * is neither overwritten nor destroyed. 
   * This is a collective operation. 
   */
  void diagStd(GlobalArray *g_v, void *eval);

  void diagSeq(const GlobalArray * g_s, const GlobalArray * g_v, void *eval);
  
  void diagStdSeq(const GlobalArray * g_v, void *eval);
  
  /** 
   * If no array elements are owned by process 'me', the range is returned
   * as lo[]=0 and hi[]=-1 for all dimensions. The operation is local.
   * @param iproc      - process number                            [input]
   * @param ndim       - number of dimensions of the global array
   * @param lo[ndim]   - array of starting indices for array section[input]
   * @param hi[ndim]   - array of ending indices for array section  [input]
   */
  void distribution(int me, int* lo, int* hi);

  float fdot(const GlobalArray * g_a);

  float fdotPatch(char t_a, int alo[], int ahi[], 
		  const GlobalArray * g_b, char t_b, int blo[], int bhi[]);

  /**
   * @param value   - pointer to the value of appropriate type 
   * (double/DoubleComplex/long) that matches array type.
   * Assign a single value to all elements in the array.
   * This is a collective operation. 
   */
  void fill(void *value);
  
  /**
   * @param   g_a                       array handles        [input] 
   * @param lo[], hi[]                patch of g_a         [input]
   * @param val                       value to fill        [input]
   * Fill the patch of g_a with  value of 'val' 
   * This is a collective operation. 
   */
  void fillPatch (int lo[], int hi[], void *val);
  
  /** 
   * Gathers array elements from a global array into a local array. 
   * The contents of the input arrays (v, subscrArray) are preserved, 
   * but their contents might be (consistently) shuffled on return. 
   
   * for(k=0; k<= n; k++){
   *     v[k] = a[subsArray[k][0]][subsArray[k][1]][subsArray[k][2]]...;    
   * }

   * This is a one-sided operation.  
   * @param n   - number of elements              [input] 
   * @param  v[n]                - array containing values [input]        
   * @param  subsarray[n][ndim]  - array of subscripts for each element [input]
   */
  void gather(void *v, int * subsarray[], int n);

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
  void get(int lo[], int hi[], void *buf, int ld[]);
  
  /**
   * This function returns 1 if the global array has some dimensions for 
   * which the ghost cell width is greater than zero, it returns 0 otherwise. 
   * This is a collective operation. 
   */
  int hasGhosts();
  
  /**
   * Computes element-wise dot product of the two arrays which must be of
   * the same types and same number of elements.
   *          return value = SUM_ij a(i,j)*b(i,j)
   * This is a collective operation. 
   * @param g_a   - array handle                 [input]
   */
  Integer idot(const GlobalArray * g_a); 

  /**
   * @param  g_a                  - array handles             [input]
   * @param  alo[], ahi[]         - g_a patch coordinates     [input]
   * @param  blo[], bhi[]         - g_b patch coordinates     [input]
   * @param  ta, tb               - transpose flags           [input]
   * Computes the element-wise dot product of the two (possibly transposed) 
   * patches which must be of the same type and have the same number of 
   * elements. 
   */
  long idotPatch(char ta, int alo[], int ahi[], const GlobalArray * g_a, 
		 char tb, int blo[], int bhi[]);
  
  /** 
   * Returns data type and dimensions of the array. 
   * This operation is local.   
   * @param type - data type                     [output]
   * @param ndim - number of dimensions          [output]
   * @param dims - array of dimensions           [output]
   */
  void inquire(int *type, int *ndim, int dims[]);

  /** 
   * Returns the name of an array represented by the handle g_a. 
   * This operation is local. 
   */
  char* inquireName();

  /**
   * Computes element-wise dot product of the two arrays which must be of
   * the same types and same number of elements.
   *          return value = SUM_ij a(i,j)*b(i,j)
   * This is a collective operation. 
   * @param g_a   - array handle                 [input]
   */
   long ldot(const GlobalArray * g_a); 

  /**
   * @param g_a     - coefficient matrix        [input]
   * Solves a system of linear equations 
   *            A * X = B 
   * using the Cholesky factorization of an NxN double precision symmetric 
   * positive definite matrix A (epresented by handle g_a). On successful 
   * exit B will contain the solution X. 
   * It returns: 
   *         = 0 : successful exit 
   *         > 0 : the leading minor of this order is not positive 
   *               definite and the factorization could 
   *               not be completed 
   * This is a collective operation. 
   */
  int lltSolve(const GlobalArray * g_a);

  /**
   * Return in owner the GA compute process id that 'owns' the data. If any 
   * element of subscript[] is out of bounds "-1" is returned. This operation 
   * is local. 
   * @param subscript[ndim]  element subscript    [output]
   */
  int locate(int subscript[]);
  
  /**
   * Return the list of the GA processes id that 'own' the data. Parts of the 
   * specified patch might be actually 'owned' by several processes. If lo/hi 
   * are out of bounds "0" is returned, otherwise return value is equal to the 
   * number of processes that hold the data. This operation is local.
   * 
   *   map[i][0:ndim-1]       - lo[i]
   *   map[i][ndim:2*ndim-1]  - hi[i]
   *   procs[i]               - processor id that owns data in patch 
   *                            lo[i]:hi[i] 
   * @param ndim          - number of dimensions of the global array
   * @param lo[ndim]      - array of starting indices for array section[input]
   * @param hi[ndim]      - array of ending indices for array section  [input]
   * @param map[][2*ndim] - array with mapping information           [output]
   * @param procs[nproc]  - list of processes that own a part of array 
   *                        section[output]
   */
  int locateRegion(int lo[], int hi[], int map[], int procs[]);

  /**
   * @param trans   - transpose or not transpose [input]
   * @param g_a     - coefficient matrix         [input]
   * Solve the system of linear equations op(A)X = B based on the LU 
   * factorization. 
   * op(A) = A or A' depending on the parameter trans: 
   * trans = 'N' or 'n' means that the transpose operator should not 
   *          be applied. 
   * trans = 'T' or 't' means that the transpose operator should be applied. 
   * Matrix A is a general real matrix. Matrix B contains possibly multiple 
   * rhs vectors. The array associated with the handle g_b is overwritten 
   * by the solution matrix X. 
   * This is a collective operation. 
   */
  void luSolve(char trans, const GlobalArray * g_a);
  
  /**
   * @param g_a, g_b                  array handles        [input] 
   * @param ailo, aihi, ajlo, ajhi    patch of g_a         [input]
   * @param bilo, bihi, bjlo, bjhi    patch of g_b         [input]
   * @param cilo, cihi, cjlo, cjhi    patch of g_c         [input]
   * @param alpha, beta               scale factors        [input]
   * @param transa, transb            transpose operators  [input]
   * ga_matmul_patch is a patch version of ga_dgemm: 
   *      C[cilo:cihi,cjlo:cjhi] := alpha* AA[ailo:aihi,ajlo:ajhi] *
   *                                BB[bilo:bihi,bjlo:bjhi] ) + 
   *                                beta*C[cilo:cihi,cjlo:cjhi],
   * where AA = op(A), BB = op(B), and op( X ) is one of 
   *      op( X ) = X   or   op( X ) = X',
   * Valid values for transpose arguments: 'n', 'N', 't', 'T'. It works 
   * for both double and DoubleComplex data tape. 
   * This is a collective operation. 
   */
  void matmulPatch(char transa, char transb, void* alpha, void *beta,
		   const GlobalArray *g_a, 
		   int ailo, int aihi, int ajlo, int ajhi,
		   const GlobalArray *g_b, 
		   int bilo, int bihi, int bjlo, int bjhi,
		   int cilo, int cihi, int cjlo, int cjhi);

  /**
   * N-dimensional Arrays:
   * @param g_a, g_b          array handles                [input] 
   * @param alo, ahi          array of patch of g_a        [input]
   * @param blo, bhi          array of patch of g_b        [input]
   * @param clo, chi          array of patch of g_c        [input]
   * @param alpha, beta               scale factors        [input]
   * @param transa, transb            transpose operators  [input]
   * nga_matmul_patch is a n-dimensional patch version of ga_dgemm: 
   *      C[clo[]:chi[]] := alpha* AA[alo[]:ahi[]] *
   *                               BB[blo[]:bhi[]]) + 
   *                               beta*C[clo[]:chi[]],
   * where AA = op(A), BB = op(B), and op( X ) is one of 
   *      op( X ) = X   or   op( X ) = X',
   * Valid values for transpose arguments: 'n', 'N', 't', 'T'. It works 
   * for both double and DoubleComplex data tape. 
   * This is a collective operation. 
   */
  void matmulPatch(char transa, char transb, void* alpha, void *beta,
		   const GlobalArray *g_a, int *alo, int *ahi,
		   const GlobalArray *g_b, int *blo, int *bhi,
		   int *clo, int *chi);
  
  /**
   * @param nblock[ndim]  - number of partitions for each dimension [output]
   * Given a distribution of an array represented by the handle g_a, 
   * returns the number of partitions of each array dimension. 
   * This operation is local. 
   */
  void nblock(int numblock[]);

  /**
   * Returns the number of dimensions in array represented by the handle 
   * g_a. This operation is local. 
   */
  int ndim();
  
  /**  
   * @param ndim       - number of dimensions of the global array
   * @param lo[ndim]   - array of starting indices for array section [input] 
   * @param hi[ndim]   - array of ending indices for array section   [input]
   * @param buf        - pointer to the local buffer array           [input]
   * @param ld[ndim-1] - array specifying leading 
   *                     dimensions/strides/extents for buffer array [input]
   * @param double/DoubleComplex/long *alpha     scale factor 
   * Same as nga_acc except the indices can extend beyond the array 
   * boundary/dimensions in which case the library wraps them around. 
   * This is a one-sided and atomic operation.     
   */
  void periodicAcc(int lo[], int hi[], void* buf, int ld[], void* alpha);
  
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
   * Same as nga_get except the indices can extend beyond the array 
   * boundary/dimensions in which case the library wraps them around. 
   * This is a one-sided operation.
   */
  void periodicGet(int lo[], int hi[], void* buf, int ld[]);
  
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
   * Same as nga_put except the indices can extend beyond the array 
   * boundary/dimensions in which case the library wraps them around. 
   * This is a one-sided operation.
   */
  void periodicPut(int lo[], int hi[], void* buf, int ld[]);
  
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
   void printFile(FILE *file);

  /**
   * Prints a patch of g_a array to the standard output. If pretty has the 
   * value 0 then output is printed in a dense fashion. If pretty has the 
   * value 1 then output is formatted and rows/columns labeled. 
   
   * This is a collective operation.  
   * @param lo[]         - coordinates of the patch    [input]
   * @param hi[]         - coordinates of the patch    [input]
   * @param int pretty        - formatting flag        [input]
   */
  void printPatch(int* lo, int* hi, int pretty) ;
  
  /**
   * @param ndim              number of array dimensions 
   * @param proc              process id                    [input] 
   * @param coord[ndim]       coordinates in processor grid [output] 
   * Based on the distribution of an array associated with handle g_a, 
   * determines coordinates of the specified processor in the virtual 
   * processor grid corresponding to the distribution of array g_a. The 
   * numbering starts from 0. The values of -1 means that the processor 
   * doesn't 'own' any section of array represented by g_a. 
   * This operation is local. 
   */
  void procTopology(int proc, int coord[]);
  
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
  void put(int lo[], int hi[], void *buf, int ld[]);
  

  /**
   * @param ndim            - number of dimensions of the global array
   * @param subscript[ndim] - subscript array for the referenced element[input]
   * Atomically read and increment an element in an integer array. 
   *      *BEGIN CRITICAL SECTION*
   *       old_value = a(subscript)
   *       a(subscript) += inc
   *      *END CRITICAL SECTION*
   *       return old_value
   * This is a one-sided and atomic operation. 
   */
  long readInc(int subscript[], long inc);

  /**
   * @param ndim       - number of dimensions of the global array
   * @param lo[ndim]   - array of starting indices for array section [input]
   * @param hi[ndim]   - array of ending indices for array section   [input]
   * Releases access to a global array when the data was read only. 
   * Your code should look like: 
   *        NGA_Distribution(g_a, myproc, lo,hi);
   *        NGA_Access(g_a, lo, hi, &ptr, ld);
   *            
   *          <operate on the data referenced by ptr> 
   *        GA_Release(g_a, lo, hi);
   * NOTE: see restrictions specified for ga_access 
   * This operation is local. 
   */
  void release(int lo[], int hi[]);

  /**
   * @param ndim       - number of dimensions of the global array
   * @param lo[ndim]   - array of starting indices for array section [input]
   * @param hi[ndim]   - array of ending indices for array section   [input]
   * Releases access to the data. It must be used if the data was accessed 
   * for writing. NOTE: see restrictions specified for ga_access. 
   * This operation is local. 
   */
  void releaseUpdate(int lo[], int hi[]);

  /** 
   * Scales an array by the constant s. Note that the library is unable 
   * to detect errors when the pointed value is of different type than 
   * the array. 
   * This is a collective operation. 
   * @param value   - pointer to the value of appropriate type 
   *                  (double/DoubleComplex/long) that matches array type    
   */
  void scale(void *value);

  /** 
   * @param lo[], hi[]                patch of g_a         [input]
   * @param val                       scale factor         [input]
   * Scale an array by the factor 'val' 
   * This is a collective operation. 
   */
  void scalePatch (int lo[], int hi[], void *val);
  
  /** 
   * Scatters array elements into a global array. The contents of the input 
   * arrays (v,subscrArray) are preserved, but their contents might be 
   * (consistently) shuffled on return.    
   *   for(k=0; k<= n; k++){
   *     a[subsArray[k][0]][subsArray[k][1]][subsArray[k][2]]... = v[k];    
   *   }
   * This is a one-sided operation.  
   * @param n   - number of elements              [input] 
   * @param  v[n]                - array containing values [input]        
   * @param  subsarray[n][ndim]  - array of subscripts for each element [input]
   */
  void scatter(void *v, int *subsarray[], int n);

  /**
   * @param g_a,g_b- handles to input arrays  [input]
   * @param g_c    - handles to output array  [input]
   * @param ta, tb - transpose operators      [input]
   * @param m  - number of rows of op(A) and of matrix  C           [input]
   * @param n  - number of columns of op(B) and of matrix  C        [input]
   * @param k  - number of columns of op(A) and rows of matrix op(B)[input]
   * @param alpha, beta  - scale factors                            [input]
   * Performs one of the matrix-matrix operations: 
   *     C := alpha*op( A )*op( B ) + beta*C,
   * where op( X ) is one of 
   *     op( X ) = X   or   op( X ) = X',
   * alpha and beta are scalars, and A, B and C are matrices, with op( A ) 
   * an m by k matrix, op( B ) a k by n matrix and C an m by n matrix. 
   * On entry, transa specifies the form of op( A ) to be used in the 
   * matrix multiplication as follows: 
   *         ta = 'N' or 'n', op( A ) = A. 
   *         ta = 'T' or 't', op( A ) = A'. 
   * This is a collective operation. 
   */
  void sgemm(char ta, char tb, int m, int n, int k, float alpha,  
	     const GlobalArray *g_a, const GlobalArray *g_b, float beta);
  
  /**
   * @param  g_a     - coefficient matrix         [input]
   * Solves a system of linear equations 
   *            A * X = B 
   * It first will call the Cholesky factorization routine and, if 
   * sucessfully, will solve the system with the Cholesky solver. If 
   * Cholesky will be not be able to factorize A, then it will call the 
   * LU factorization routine and will solve the system with forward/backward 
   * substitution. On exit B will contain the solution X. 
   * It returns 
   *            = 0 : Cholesky factoriztion was succesful 
   *            > 0 : the leading minor of this order 
   *                  is not positive definite, Cholesky factorization 
   *                  could not be completed and LU factoriztion was used 
   * This is a collective operation. 
   */
  int solve(const GlobalArray * g_a);

  /**
   * It computes the inverse of a double precision using the Cholesky 
   * factorization of a NxN double precision symmetric positive definite 
   * matrix A stored in the global array represented by g_a. On successful 
   * exit, A will contain the inverse. 
   * It returns 
   *            = 0 : successful exit 
   *            > 0 : the leading minor of this order is not positive 
   *                  definite and the factorization could not be completed 
   *            < 0 : it returns the index i of the (i,i) 
   *                  element of the factor L/U that is zero and 
   *                  the inverse could not be computed 
   * This is a collective operation. 
   */
  int spdInvert();

  /**
   * @param  op              - operator {"min","max"}               [input]
   * @param  val             - address where value should be stored [output] 
   * @param  subscript[ndim] - array index for the selected element [output]
   * Returns the value and index for an element that is selected by the 
   * specified operator  in a global array corresponding to g_a handle. 
   * This is a collective operation. 
   */
  void selectElem(char *op, void* val, int index[]);

  /** 
   * Symmmetrizes matrix A with handle A:=.5 * (A+A').
   * This is a collective operation 
   */
  void symmetrize();
  
  /**
   * Transposes a matrix: B = A', where A and B are represented by 
   * handles g_a and g_b. This is a collective operation.
   */
  void transpose(const GlobalArray * g_a);
  
  /**
   * This call updates the ghost cell regions on each processor with the 
   * corresponding neighbor data from other processors. The operation assumes 
   * that all data is wrapped around using periodic boundary data so that 
   * ghost cell data that goes beyound an array boundary is wrapped around to
   * the other end of the array. The GA_Update_ghosts call contains two   
   * GA_Sync   calls before and after the actual update operation. For some 
   * applications these calls may be unecessary, if so they can be removed 
   * using the GA_Mask_sync subroutine. 
   * This is a collective operation. 
   */
  void updateGhosts();
 
  /**
   * This function can be used to update the ghost cells along individual 
   * directions. It is designed for algorithms that can overlap updates 
   * with computation. The variable dimension indicates which coordinate 
   * direction is to be updated (e.g. dimension = 1 would correspond to the 
   * y axis in a two or three dimensional system), the variable idir can take
   * the values +/-1 and indicates whether the side that is to be updated lies 
   * in the positive or negative direction, and cflag indicates whether or not
   * the corners on the side being updated are to be included in the update. 
   * The following calls would be equivalent to a call to   GA_Update_ghosts 
   * for a 2-dimensional system: 
   *
   * status = NGA_Update_ghost_dir(g_a,0,-1,1);
   * status = NGA_Update_ghost_dir(g_a,0,1,1);
   * status = NGA_Update_ghost_dir(g_a,1,-1,0);
   * status = NGA_Update_ghost_dir(g_a,1,1,0);
   *
   * The variable cflag is set equal to 1 (or non-zero) in the first two calls so that the corner ghost cells are update, it is set equal to 0 in the second two calls to avoid redundant updates of the corners. Note that updating the ghosts cells using several independent calls to the nga_update_ghost_dir functions is generally not as efficient as using   GA_Update_ghosts  unless the individual calls can be effectively overlapped with computation. 
   * This is a  collective operation. 
   * @param g_a                                                    [input]
   * @param dimension    - array dimension that is to be updated   [input]
   * @param idir         - direction of update (+/- 1)             [input]
   * @param cflag        - flag (0/1) to include corners in update [input]
   */
  int updateGhostDir(int dimension, int idir, int cflag);

  
  /**
   * Computes element-wise dot product of the two arrays which must be of
   * the same types and same number of elements.
   *          return value = SUM_ij a(i,j)*b(i,j)
   * This is a collective operation. 
   * @param g_a   - array handle                 [input]
   */
  DoubleComplex zdot(const GlobalArray * g_a); 

  /**
   * @param  g_a                  - array handles             [input]
   * @param  alo[], ahi[]         - g_a patch coordinates     [input]
   * @param  blo[], bhi[]         - g_b patch coordinates     [input]
   * @param  ta, tb               - transpose flags           [input]
   * Computes the element-wise dot product of the two (possibly transposed) 
   * patches which must be of the same type and have the same number of 
   * elements. 
   */
  DoubleComplex zdotPatch(char ta, int alo[], int ahi[], 
			  const GlobalArray * g_a, char tb, int blo[], 
			  int bhi[]) ;
  
  /** 
   * Sets value of all elements in the array to zero. 
   * This is a collective operation. 
   */
  void zero();
  
  /**
   * @param  lo[], hi[]                patch of g_a         [input]
   * Set all the elements in the patch to zero. 
   * This is a collective operation. 
   */
  void zeroPatch (int lo[], int hi[]);
  
  /**
   * @param g_a,g_b- handles to input arrays  [input]
   * @param g_c    - handles to output array  [input]
   * @param ta, tb - transpose operators      [input]
   * @param m  - number of rows of op(A) and of matrix  C           [input]
   * @param n  - number of columns of op(B) and of matrix  C        [input]
   * @param k  - number of columns of op(A) and rows of matrix op(B)[input]
   * @param alpha, beta  - scale factors                            [input]
   * Performs one of the matrix-matrix operations: 
   *     C := alpha*op( A )*op( B ) + beta*C,
   * where op( X ) is one of 
   *     op( X ) = X   or   op( X ) = X',
   * alpha and beta are scalars, and A, B and C are matrices, with op( A ) 
   * an m by k matrix, op( B ) a k by n matrix and C an m by n matrix. 
   * On entry, transa specifies the form of op( A ) to be used in the 
   * matrix multiplication as follows: 
   *         ta = 'N' or 'n', op( A ) = A. 
   *         ta = 'T' or 't', op( A ) = A'. 
   * This is a collective operation. 
   */
  void zgemm(char ta, char tb, int m, int n, int k, DoubleComplex alpha,  
	     const GlobalArray *g_a, const GlobalArray *g_b, 
	     DoubleComplex beta);
  
  
  /**
   * New additional functionalities from Limin. 
   */
  /**
   * Take element-wise absolute value of the array. 
   * This is a collective operation. 
   */
   void absValue(); 

  /**
   * Take element-wise absolute value of the patch. 
   * This is a collective operation.
   * @param lo[], hi[]           - g_a patch coordinates   [input] 
   */
   void absValuePatch(int *lo, int *hi);

  /**
   * Add the constant pointed by alpha to each element of the array. 
   * This is a collective operation. 
   * @param double/complex/int/long/float      *alpha      [input] 
   */
   void addConstant(void* alpha);
 
  /**
   * Add the constant pointed by alpha to each element of the patch. 
   * This is a collective operation. 
   * @param lo[], hi[]           - g_a patch coordinates  [input] 
   * @param double/complex/int/long/float      *alpha     [input] 
   */
   void addConstantPatch(int *lo, int *hi, void *alpha);
  
  /**
   * Take element-wise reciprocal of the array. 
   * This is a collective operation. 
   */
   void recip();
  
  /**
   * Take element-wise reciprocal of the patch. 
   * This is a collective operation. 
   * @param   lo[], hi[]           - g_a patch coordinates     [input] 
   */
   void recipPatch(int *lo, int *hi);
  
  
  /**
   * Computes the element-wise product of the two arrays 
   * which must be of the same types and same number of 
   * elements. For two-dimensional arrays, 
   *
   *            c(i, j)  = a(i,j)*b(i,j) 
   *
   * The result (c) may replace one of the input arrays (a/b). 
   * This is a collective operation. 
   * @param g_a, g_b - array handles         [input] 
   */
   void elemMultiply(const GlobalArray * g_a, const GlobalArray * g_b);
  
  
  /**
   * Computes the element-wise product of the two patches 
   * which must be of the same types and same number of 
   * elements. For two-dimensional arrays, 
   * 
   *             c(i, j)  = a(i,j)*b(i,j) 
   * 
   * The result (c) may replace one of the input arrays (a/b). 
   * This is a collective operation.
   * @param g_a, g_b - array handles                           [input] 
   * @param alo[], ahi[]           - g_a patch coordinates     [input] 
   * @param blo[], bhi[]           - g_b patch coordinates     [input] 
   * @param clo[], chi[]           - g_c patch coordinates     [output]  
   */ 
   void elemMultiplyPatch(const GlobalArray * g_a,int *alo,int *ahi,
				 const GlobalArray * g_b,int *blo,int *bhi,
				 int *clo,int *chi);
  /**
   * @param g_a, g_b - array handles         [input] 
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
   void elemDivide(const GlobalArray * g_a, const GlobalArray * g_b);
  
  /**
   * Computes the element-wise quotient of the two patches 
   * which must be of the same types and same number of 
   * elements. For two-dimensional arrays, 
   *
   *            c(i, j)  = a(i,j)/b(i,j) 
   *
   * The result (c) may replace one of the input arrays (a/b). 
   * This is a collective operation. 
   * @param g_a, g_b - array handles                           [input] 
   * @param alo[], ahi[]           - g_a patch coordinates     [input] 
   * @param blo[], bhi[]           - g_b patch coordinates     [input] 
   * @param clo[], chi[]           - g_c patch coordinates     [output] 
   *
   */ 
   void elemDividePatch(const GlobalArray * g_a,int *alo,int *ahi,
			       const GlobalArray * g_b,int *blo,int *bhi,
			       int *clo,int *chi);
  /**
   * Computes the element-wise maximum of the two arrays 
   * which must be of the same types and same number of 
   * elements. For two dimensional arrays, 
   *
   *                c(i, j)  = max{a(i,j), b(i,j)} 
   *
   * The result (c) may replace one of the input arrays (a/b). 
   * This is a collective operation. 
   * @param g_a, g_b - array handles         [input] 
   */
   void elemMaximum(const GlobalArray * g_a, const GlobalArray * g_b);
  
  
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
   * @param g_a, g_b - array handles         [input]   
   * @param alo[], ahi[]           - g_a patch coordinates     [input] 
   * @param blo[], bhi[]           - g_b patch coordinates     [input] 
   * @param clo[], chi[]           - g_c patch coordinates     [output] 
   */
    void elemMaximumPatch(const GlobalArray * g_a,int *alo,int *ahi,
				 const GlobalArray * g_b,int *blo,int *bhi,
				 int *clo,int *chi);
   /**
    * Computes the element-wise minimum of the two arrays 
    * which must be of the same types and same number of 
    * elements. For two dimensional arrays, 
    * 
    *             c(i, j)  = min{a(i,j), b(i,j)} 
    * 
    * The result (c) may replace one of the input arrays (a/b). 
    * This is a collective operation. 
    * @param g_a, g_b - array handles         [input] 
    */
    void elemMinimum(const GlobalArray * g_a, const GlobalArray * g_b);
   
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
    * @param g_a, g_b - array handles                           [input] 
    * @param alo[], ahi[]           - g_a patch coordinates     [input] 
    * @param blo[], bhi[]           - g_b patch coordinates     [input] 
    * @param clo[], chi[]           - g_c patch coordinates     [output] 
    */
    void elemMinimumPatch(const GlobalArray * g_a,int *alo,int *ahi,
				 const GlobalArray * g_b,int *blo,int *bhi,
				 int *clo,int *chi);
  
   /** 
    * Calculates the largest multiple of a vector g_b that can be added 
    * to this vector g_a while keeping each element of this vector 
    * nonnegative. 
    * This is a collective operation. 
    * @param g_a, g_b - array handles where g_b is the step direction.[input] 
    * @param step        - the maximum step                           [output] 
    */  
    void stepMax(const GlobalArray * g_a, double *step);
   
   /** 
    * Calculates the largest step size that should be used in a projected 
    * bound line search. 
    * This is a collective operation. 
    * @param g_vv, g_xxll, g_xxuu - array handles              [input] 
    * @param step2 - the maximum step size                     [output] 
    * where 
    * g_vv - the step direction 
    * g_xxll - lower bounds 
    * g_xxuu - upper bounds 
    */
    void stepMax2(const GlobalArray * g_vv, 
			 const GlobalArray * g_xxll, 
			 const GlobalArray * g_xxuu, double *step2);
   void stepMaxPatch(int *alo, int *ahi, 
			    const GlobalArray * g_b, int *blo, int *bhi, 
			    double *step);
   void stepMax2Patch(int *xxlo, int *xxhi, 
			     const GlobalArray * g_vv, int *vvlo, 
			     int *vvhi, 
			     const GlobalArray * g_xxll, int *xxlllo, 
			     int *xxllhi,
			     const GlobalArray * g_xxuu, int *xxuulo, 
			     int *xxuuhi, double *step2);
 

 
 
  
  /** Matrix Operations */
  
  /** 
   * Adds this constant to the diagonal elements of the matrix. 
   * This is a collective operation. 
   * @param double/complex/int/long/float      *c               [input] 
   */
   void shiftDiagonal(void *c);
  
  /**
   * Sets the diagonal elements of this matrix g_a with the elements of the 
   * vector g_v.  This is a collective operation. 
   * @param g_v - array handles       [input]  
   */
   void setDiagonal(const GlobalArray * g_v);
  
  /**
   * Sets the diagonal elements of this matrix g_a with zeros. 
   * This is a collective operation. 
   */
   void zeroDiagonal();
  
  /**
   * Adds the elements of the vector g_v to the diagonal of this matrix g_a. 
   * This is a collective operation. 
   * @param g_v - array handles       [input] 
   */
   void addDiagonal(const GlobalArray * g_v);
  
  /**
   * Inserts the diagonal elements of this matrix g_a into the vector g_v. 
   * This is a collective operation. 
   * @param g_v - array handles       [input] 
   */
   void getDiagonal(const GlobalArray * g_a);
  
  /**
   * Scales the rows of this matrix g_a using the vector g_v. 
   * This is a collective operation.  
   * @param g_v - array handles       [input] 
   */
   void scaleRows(const GlobalArray * g_v);
  
  /** 
   * Scales the columns of this matrix g_a using the vector g_v. 
   * This is a collective operation. 
   * @param g_v - array handles       [input] 
   */
   void scaleCols(const GlobalArray * g_v);
  
  /**
   * Computes the 1-norm of the matrix or vector g_a. 
   * This is a collective operation. 
   * @param nm - matrix/vector 1-norm value 
   */
   void norm1(double *nm);
  
  /**
   * Computes the 1-norm of the matrix or vector g_a. 
   * This is a collective operation. 
   * @param nm - matrix/vector 1-norm value 
   */
   void normInfinity(double *nm);
  
  /**
   * Computes the componentwise Median of three arrays g_a, g_b, and g_c, and 
   * stores the result in this array g_m.  The result (m) may replace one of the 
   * input arrays (a/b/c). This is a collective operation. 
   * @param g_a, g_b, g_c- array handles       [input] 
   */
   void median(const GlobalArray * g_a, const GlobalArray * g_b, 
		      const GlobalArray * g_c);
  
  /**
   * Computes the componentwise Median of three patches g_a, g_b, and g_c, and 
   * stores the result in this patch g_m.  The result (m) may replace one of the 
   * input patches (a/b/c). This is a collective operation. 
   * @param g_a, g_b, g_c - array handles         [input] 
   * @param alo[], ahi[]           - g_a patch coordinates     [input] 
   * @param blo[], bhi[]           - g_b patch coordinates     [input] 
   * @param clo[], chi[]           - g_c patch coordinates     [intput] 
   * @param mlo[], mhi[]         - g_m patch coordinates     [output] 
   */
   void medianPatch(const GlobalArray * g_a, int *alo, int *ahi, 
			       const GlobalArray * g_b, int *blo, int *bhi, 
			       const GlobalArray * g_c, int *clo, int *chi, 
			       int *mlo, int *mhi);
  
 private:
  int mHandle;      /* g_a handle */
};

