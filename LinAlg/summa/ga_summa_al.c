#include "global.h"
#include "macommon.h"

#include <stdlib.h>

#ifndef NULL
#  define NULL 0
#endif

DoublePrecision *ga_summa_alloc( Integer n_elements, char string[], Integer *ma_handle )

/*
   Allocate DoublePrecision array with n_elements using ma_push_stack.  The function
   returns the address of the start of the allocated array.  Returns NULL
   on failure.

   n_elements: size of "DoublePrecision" array to allocate
   string:     string to use with MA_push_stack
   ma_handle:  handle for memory allocated by ma_push_stack,
               needed for deallocting memory latter.
*/

{
  Integer    nele;
  Void       *pd_name;

  DoublePrecision     *p_name;

  nele = (Integer) n_elements;

  if( nele < 1 )
     nele = 1;

  if( MA_inquire_avail( MT_F_DBL ) < nele )
     pd_name = NULL;
  else if( MA_push_stack(MT_F_DBL, nele, string, ma_handle ) )
     MA_get_pointer( *ma_handle, &pd_name);
  else
     pd_name = NULL;

  p_name = (DoublePrecision *) pd_name;

  return p_name;
}
