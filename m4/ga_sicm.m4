# GA_MSG_COMMS()
# --------------
# Establishes all things related to SICM.
AC_DEFUN([GA_SICM], [
SICM_LIBS=
SICM_LDFLAGS=
SICM_CPPFLAGS=
AC_ARG_WITH([sicm],
    [AS_HELP_STRING([--with-sicm[[=ARG]]],
        [specify SICM library (optional)])],
    [],
    [with_sicm=no])

AC_ARG_WITH([sicm_dev],
    [AS_HELP_STRING([--with-sicm_dev[[=ARG]]],
        [specify SICM hardware (default=dram)])],
    [],
    [with_sicm_dev=no])

#TODO: AC_CHECK_HEADERS, AC_SEARCH_LIBS
AS_IF([test "x$with_sicm" != "xno"],
      [
       AM_CONDITIONAL(WITH_SICM, true)
       have_sicm=1
       GA_ARG_PARSE([with_sicm], [SICM_LIBS], [SICM_LDFLAGS], [SICM_CPPFLAGS])
       AC_SUBST([SICM_LIBS])                                
       AC_SUBST([SICM_LDFLAGS])                             
       AC_SUBST([SICM_CPPFLAGS])                            
       AC_DEFINE([USE_SICM], [1], [enable sicm define])        
       AC_DEFINE([TEST_SICM], [1], [enable test_sicm define])  
       AC_DEFINE([TEST_SICM_DEV], [dram], [specify sicm test hw])   
      ],
      [
         have_sicm=0
         AM_CONDITIONAL(WITH_SICM, false)
      ]    
   )
AC_SUBST([have_sicm])  

#TODO:Check if sicm_dev provided is valid
AS_IF([test "x$with_sicm_dev" != "xno"],
      [
       have_sicm_dev=1
       AC_DEFINE_UNQUOTED([TEST_SICM_DEV], [$with_sicm_dev], [specify sicm test hw])   
      ],
      [
         have_sicm_dev=0
      ]    
   )
AC_SUBST([with_sicm_dev])  

])dnl
