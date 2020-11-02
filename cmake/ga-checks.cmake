INCLUDE( CheckCSourceCompiles )
# Check for restrict keyword
FOREACH( ac_kw __restrict __restrict__ _Restrict restrict )
    CHECK_C_SOURCE_COMPILES(
"
typedef int * int_ptr;
int foo (int_ptr ${ac_kw} ip) {
    return ip[0];
}
int main() {
    int s[1];
    int * ${ac_kw} t = s;
    t[0] = 0;
    return foo(t); 
}   
"
    HAVE_RESTRICT )
    IF( HAVE_RESTRICT )
        SET( ac_cv_c_restrict ${ac_kw} )
        BREAK( )
    ENDIF( )
ENDFOREACH( )
IF( HAVE_RESTRICT )
    SET( restrict ${ac_cv_c_restrict} )
ELSE( )
    SET( restrict " " )
ENDIF( )

# Check for inline keyword
CHECK_C_SOURCE_COMPILES(
"
typedef int foo_t;
static inline foo_t static_foo(){return 0;}
foo_t foo(){return 0;}
int main(int argc, char *argv[]){return 0;}
"
    HAVE_INLINE_NATIVE )
IF( HAVE_INLINE_NATIVE )
ELSE ( )
    FOREACH( ac_kw __inline__ __inline )
        CHECK_C_SOURCE_COMPILES(
"
typedef int foo_t;
static ${ac_kw} foo_t static_foo(){return 0;}
foo_t foo(){return 0;}
int main(int argc, char *argv[]){return 0;}
"
        HAVE_INLINE )
        IF( HAVE_INLINE )
            SET( ac_cv_c_inline ${ac_kw} )
            BREAK( )
        ENDIF( )
    ENDFOREACH( )
    IF( HAVE_INLINE )
        SET( inline ${ac_cv_c_inline} )
    ELSE( )
        SET( inline " " )
    ENDIF( )
ENDIF( )

# check for availability of functions by seeing if small programs compile
CHECK_C_SOURCE_COMPILES(
"
int main(int argc, char *argv[])
{
pause();
return 0;
}
"
    HAVE_PAUSE )

# check for availability of long double C-type
CHECK_C_SOURCE_COMPILES(
"
int main(int argc, char *argv[])
{
long double x;
return 0;
}
"
    HAVE_LONG_DOUBLE )

CHECK_C_SOURCE_COMPILES(
"
extern void weakf(int c);
#pragma weak weakf = __weakf
void __weakf(int c) {}
int main(int argc, char **argv) {
  weakf(0);
  return(0);
}
"
   HAVE_SYS_WEAK_ALIAS_PRAGMA )

if (HAVE_LONG_DOUBLE)
  set(MA_LONG_DOUBLE "long double")
else()
  set(MA_LONG_DOUBLE "struct {double dummy[2];}")
endif()

# check size of different variables
include(CheckTypeSize)
check_type_size("int" CM_SIZEOF_INT)
check_type_size("double" CM_SIZEOF_DOUBLE)
check_type_size("float" CM_SIZEOF_FLOAT)
check_type_size("long" CM_SIZEOF_LONG)
check_type_size("long double" CM_SIZEOF_LONG_DOUBLE)
check_type_size("long long" CM_SIZEOF_LONG_LONG)
check_type_size("short" CM_SIZEOF_SHORT)

# check for standard C/C++ include files
include(CheckIncludeFiles)
check_include_files("assert.h" HAVE_ASSERT_H)
check_include_files("limits.h" HAVE_LIMITS_H)
check_include_files("linux/limits.h" HAVE_LINUX_LIMITS_H)
check_include_files("malloc.h" HAVE_MALLOC_H)
check_include_files("math.h" HAVE_MATH_H)
check_include_files("stddef.h" HAVE_STDDEF_H)
check_include_files("stdint.h" HAVE_STDINT_H)
check_include_files("stdio.h" HAVE_STDIO_H)
check_include_files("stdlib.h" HAVE_STDLIB_H)
check_include_files("strings.h" HAVE_STRINGS_H)
check_include_files("string.h" HAVE_STRING_H)
check_include_files("sys/types.h" HAVE_SYS_TYPES_H)
check_include_files("unistd.h" HAVE_UNISTD_H)
check_include_files("windows.h" HAVE_WINDOWS_H)

# check for certain functions
include(CheckFunctionExists)
check_function_exists("bzero" HAVE_BZERO)
