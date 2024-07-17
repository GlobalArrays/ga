#include <openblas_config.h>

int main() {
  int blis_int_size = sizeof(blasint)*8;
  if( blis_int_size == 32 ) return 0;
  else                      return 1;
}
