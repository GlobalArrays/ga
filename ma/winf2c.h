#if !defined(_WINF2C_H_)
#define _WINF2C_H_

typedef struct{
        char *string;
        int  len;
}_fcd;

#define _fcdlen(x)   (x).len
#define _fcdtocp(x)  (x).string
#define _cptofcd(str,len) str,len

#define FATR __stdcall

#endif
