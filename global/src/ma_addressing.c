/*\
 *   The routine used to provide addresses of MA base arrays for
 *   Fortran double precision (dbl_mb) and integer types to C programs.
 *
 *   Called in ga_initialize in the distributed memory version of GAs. 
 *   
 * 
\*/


#include "types.f2c.h"

DoublePrecision *DBL_MB;
Integer         *INT_MB;


void f2c_ma_address_(dpointer, ipointer)
DoublePrecision *dpointer;
Integer         *ipointer;
{
extern void ga_error();

	if(dpointer == (DoublePrecision*)0)
                    ga_error("f2c_ma_address: null DBL pointer",0L);
	if(ipointer == (Integer*)0)  
	            ga_error("f2c_ma_address: null INT pointer",0L);
	DBL_MB = dpointer;
	INT_MB = ipointer;
        /* printf("DBL_MB=%d INT_MB=%d\n",(int)DBL_MB, (int)INT_MB);*/
}
    
