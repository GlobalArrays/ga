/* $Id: datatypes.c,v 1.9.10.1 2006-12-14 13:24:47 manoj Exp $
 * conversion of MA identifiers between C to Fortran data types 
 * Note that ga_type_c2f(MT_F_INT) == MT_F_INT 
 */

#include "global.h"
#include "globalp.h"
int ga_type_f2c(int type)
{
int ctype;
   switch(type){
   case MT_F_INT: 
#       ifdef EXT_INT 
                ctype = C_LONG;
#          ifdef EXT_INT64
		ctype = C_LONGLONG;
#          endif
#       else
                ctype = C_INT;
#       endif
                break;
   case MT_F_REAL: 
#       if defined(NO_REAL_32)
                ctype = C_DBL;
#       else
                ctype = C_FLOAT;
#       endif
                break;
   case MT_F_DBL: 
		ctype = C_DBL;
                break;
   case MT_F_DCPL: 
		ctype = C_DCPL;
                break;
   case MT_F_SCPL: 
#       if defined(NO_REAL_32)
		ctype = C_DCPL;
#       else
		ctype = C_SCPL;
#       endif
                break;
   default:     ctype = type;
                break;
   }
   
   return(ctype);
}


int ga_type_c2f(int type)
{
int ftype;
   switch(type){
   case C_INT: 
                ftype = (sizeof(int) != sizeof(Integer))? -1: MT_F_INT;
                break;
   case C_LONG: 
                ftype = (sizeof(long) != sizeof(Integer))? -1: MT_F_INT;
                break;
   case C_LONGLONG: 
                ftype = (sizeof(long long) != sizeof(Integer))? -1: MT_F_INT;
                break;
   case C_FLOAT:
#       if defined(NO_REAL_32)
                ftype = -1;
#       else
                ftype = MT_F_REAL; 
#       endif
                break;
   case C_DBL: 
                ftype = MT_F_DBL;
                break;
   case C_DCPL:
                ftype = MT_F_DCPL;
                break;
   case C_SCPL:
#       if defined(NO_REAL_32)
                ftype = -1;
#       else
                ftype = MT_F_SCPL;
#       endif
                break;
   default:     ftype = type;
                break;
   }
   
   return(ftype);
}
