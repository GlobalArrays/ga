
/**
 * Ga Initialize and Terminate calls.
 */

/**
 * Initialize Global Arrays.
 * Allocate and initialize internal data structures in Global Arrays.
 * This is a collective operation. 
 */

_GA_STATIC_ void
Initialize(int argc, char *argv[], size_t limit = 0);


_GA_STATIC_ void
Initialize(int argc, char *argv[], unsigned long heapSize, 
	   unsigned long stackSize, int type, size_t limit = 0);


/**
 * Delete all active arrays and destroy internal data structures. 
 * This is a collective operation. 
 */
_GA_STATIC_ void 
Terminate();
