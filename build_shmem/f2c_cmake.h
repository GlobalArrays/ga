#ifndef F77_FUNC_HEADER_INCLUDED
#define F77_FUNC_HEADER_INCLUDED
#define F77_FUNC_GLOBAL(name,NAME) name##_
#define F77_FUNC_GLOBAL_(name,NAME) name##_
#define F77_FUNC_MODULE(mod_name,name, mod_NAME,NAME) __##mod_name##_MOD_##name
#define F77_FUNC_MODULE_(mod_name,name, mod_NAME,NAME) __##mod_name##_MOD_##name
#endif
