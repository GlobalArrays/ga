#if !defined(_WINF2C_H_)
#define _WINF2C_H_

/*
 * $Id: winf2c.h,v 1.2 2000-07-04 05:54:56 d3g001 Exp $
 */

typedef struct{
        char *string;
        int  len;
}_fcd;

#define _fcdlen(x)   (x).len
#define _fcdtocp(x)  (x).string
#define _cptofcd(str,len) str,len

#define FATR __stdcall

#endif /* _WINF2C_H_ */
