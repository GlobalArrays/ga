/* $Id: datatypes.c,v 1.3 2001-06-29 18:06:57 d3h325 Exp $
 * conversion of MA identifiers between C to Fortran data types 
 * Note that ga_type_c2f(MT_F_INT) == MT_F_INT 
 */

#include <macdecls.h>

int ga_type_f2c(int type)
{
int ctype;
   switch(type){
   case MT_F_INT: 
#       ifdef EXT_INT
		ctype = MT_INT;
#       else
		ctype = MT_LONGINT;
#       endif
                break;
   case MT_F_REAL: 
#       if defined(CRAY) || defined(NEC)
		ctype = MT_DBL;
#       else
		ctype = MT_FLOAT;
#       endif
                break;
   case MT_F_DBL: 
		ctype = MT_DBL;
                break;
   case MT_F_DCPL: 
		ctype = MT_DCPL;
                break;
   case MT_F_SCPL: 
#       if defined(CRAY) || defined(NEC)
		ctype = MT_DCPL;
#       else
		ctype = MT_SCPL;
#       endif
   default:     ctype = type;
                break;
   }
   
   return(ctype);
}


int ga_type_c2f(int type)
{
int ftype;
   switch(type){
   case MT_INT: 
                ftype = (sizeof(int) != sizeof(Integer))? -1: MT_F_INT;
                break;
   case MT_LONGINT: 
                ftype = (sizeof(long) != sizeof(Integer))? -1: MT_F_INT;
                break;
   case MT_FLOAT:
#       if defined(CRAY) || defined(NEC)
                ftype = -1;
#       else
                ftype = MT_F_REAL; 
#       endif
                break;
   case MT_DBL: 
                ftype = MT_F_DBL;
                break;
   case MT_DCPL:
                ftype = MT_F_DCPL;
                break;
   case MT_SCPL:
#       if defined(CRAY) || defined(NEC)
                ftype = -1;
#       else
                ftype = MT_F_SCPL;
#       endif
   default:     ftype = type;
                break;
   }
   
   return(ftype);
}
