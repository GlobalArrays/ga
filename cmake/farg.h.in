#ifndef FARG_H_
#define FARG_H_

#include "typesf2c.h"

#if ENABLE_F77
#define F2C_GETARG F77_FUNC_(f2c_getarg,F2C_GETARG)
#define F2C_IARGC F77_FUNC_(f2c_iargc,F2C_IARGC)
#endif

#define F2C_GETARG_ARGV_MAX 255
#define F2C_GETARG_ARGLEN_MAX 255

extern void F2C_GETARG(Integer*, char*, int);
extern Integer F2C_IARGC();

extern void ga_c2fstring(char *cstring, char *fstring, int flength);
extern void ga_f2cstring(char *fstring, int flength, char *cstring, int clength);
extern void ga_f2c_get_cmd_args(int *argc, char ***argv);

#endif /* FARG_H_ */
