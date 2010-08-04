/** @file */
#ifndef XDRSTUFF_H_
#define XDRSTUFF_H_

/**
 * Called automatically at start to allocate the XDR buffers.
 */
extern void CreateXdrBuf();

/**
 * Call to free the xdr buffers.
 */
extern void DestroyXdrBuf();

/**
 * Write DoublePrecision x[n_DoublePrecision] to the socket translating to XDR representation.
 * 
 * Returned is the number of bytes written to the socket.
 * 
 * All errors are treated as fatal.
 */
extern int WriteXdrDouble(int sock, DoublePrecision *x, Integer n_DoublePrecision);

/**
 * Read DoublePrecision x[n_DoublePrecision] from the socket translating from XDR representation.
 *
 * Returned is the number of bytes read from the socket.
 *
 * All errors are treated as fatal.
 */
extern int ReadXdrDouble(int sock, DoublePrecision *x, Integer n_DoublePrecision);

/**
 * Write Integer x[n_Integer] to the socket translating to XDR representation.
 *
 * Returned is the number of bytes written to the socket.
 *
 * All errors are treated as fatal.
 */
extern int WriteXdrLong(int sock, Integer *x, Integer n_Integer);

/**
 * Read Integer x[n_Integer] from the socket translating from XDR representation.
 *
 * Returned is the number of bytes read from the socket.
 *
 * All errors are treated as fatal.
 */
extern int ReadXdrLong(int sock, Integer *x, Integer n_Integer);

#endif /* XDRSTUFF_H_ */
