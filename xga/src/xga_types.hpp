/* Define some enumerations used throughout XGA */
#ifndef _XGA_TYPES_H
#define _XGA_TYPES_H
namespace XGA {

  enum xga_types{XGA_UNKNOWN = 0,
                 XGA_INT,
                 XGA_LONG,
                 XGA_LONGLONG,
                 XGA_FLOAT,
                 XGA_DOUBLE,
                 XGA_COMPLEX,
                 XGA_DCOMPLEX};

  enum data_distribution{REGULAR, SCALAPACK, TILED, TILED_IRREG};

  /**
   * Struct that tracks non-blocking operations in XGA
   * @param index index into a list of non-blocking operations
   * @param tag unique identifier for each non-blocking operation
   */
  typedef struct {
    unsigned int index;
    unsigned int tag;
  } NBHandle;

}
#endif
