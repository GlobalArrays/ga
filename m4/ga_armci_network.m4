# _GA_ARMCI_NETWORK_WITH(KEY, DESCRIPTION)
# --------------------------------------------------
# A helper macro for generating all of the AC_ARG_WITHs.
# Also may establish value of ga_armci_network.
# Counts how many armci networks were specified by user.
AC_DEFUN([_GA_ARMCI_NETWORK_WITH], [
AC_ARG_WITH([$1],
    [AS_HELP_STRING([--with-$1[[=ARG]]], [select armci network as $2])])
AS_VAR_PUSHDEF([KEY],      m4_toupper(m4_translit([$1],      [-.], [__])))
AS_VAR_PUSHDEF([with_key],            m4_translit([with_$1], [-.], [__]))
dnl Can't have AM_CONDITIONAL here in case configure must find armci network
dnl without user intervention.
dnl AM_CONDITIONAL([ARMCI_NETWORK_]KEY, [test "x$with_key" != x])
AS_IF([test "x$with_key" != x],
    [GA_ARG_PARSE([with_key], [ARMCI_NETWORK_LIBS], [ARMCI_NETWORK_LDFLAGS],
                  [ARMCI_NETWORK_CPPFLAGS])])
AS_IF([test "x$with_key" != xno && test "x$with_key" != x],
    [ga_armci_network=KEY
     AS_VAR_ARITH([armci_network_count], [$armci_network_count + 1])])
AS_VAR_POPDEF([KEY])
AS_VAR_POPDEF([with_key])
])dnl

# _GA_ARMCI_NETWORK_WARN(KEY)
# ---------------------------
# Helper macro for idicating value of armci network arguments.
AC_DEFUN([_GA_ARMCI_NETWORK_WARN], [
AS_VAR_PUSHDEF([with_key],            m4_translit([with_$1], [-.], [__]))
AS_IF([test "x$with_key" != x && test "x$with_key" != xno],
    [AC_MSG_WARN([--with-$1=$with_key])])
AS_VAR_POPDEF([with_key])
])dnl

# _GA_ARMCI_NETWORK_AM_CONDITIONAL(KEY)
#--------------------------------------
# Helper macro for generating all AM_CONDITIONALs.
AC_DEFUN([_GA_ARMCI_NETWORK_AM_CONDITIONAL], [
AS_VAR_PUSHDEF([KEY],      m4_toupper(m4_translit([$1],      [-.], [__])))
AS_VAR_PUSHDEF([with_key],            m4_translit([with_$1], [-.], [__]))
AM_CONDITIONAL([ARMCI_NETWORK_]KEY,
    [test "x$with_key" != x && test "x$with_key" != xno])
AS_VAR_POPDEF([KEY])
AS_VAR_POPDEF([with_key])
])dnl

# _GA_ARMCI_NETWORK_ARMCI([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# ----------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_ARMCI], [
AC_MSG_NOTICE([searching for external ARMCI...])
happy=yes
CPPFLAGS="$CPPFLAGS $GA_MP_CPPFLAGS"
LDFLAGS="$LDFLAGS $GA_MP_LDFLAGS"
LIBS="$LIBS $GA_MP_LIBS"
AS_IF([test "x$happy" = xyes],
    [AC_CHECK_HEADER([armci.h], [], [happy=no])])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([ARMCI_Init], [armci], [], [happy=no])
     AS_CASE([$ac_cv_search_ARMCI_Init],
            ["none required"], [],
            [no], [],
            [# add missing lib to ARMCI_NETWORK_LIBS if not there
             AS_CASE([$ARMCI_NETWORK_LIBS],
                     [*$ac_cv_search_ARMCI_Init*], [],
                     [ARMCI_NETWORK_LIBS="$ARMCI_NETWORK_LIBS $ac_cv_search_ARMCI_Init"])])])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([armci_group_comm], [armci])
     AS_IF([test "x$ac_cv_search_armci_group_comm" != xno],
        [ac_cv_search_armci_group_comm=1],
        [ac_cv_search_armci_group_comm=0])
     AC_DEFINE_UNQUOTED([HAVE_ARMCI_GROUP_COMM],
        [$ac_cv_search_armci_group_comm],
        [set to 1 if ARMCI has armci_group_comm function])
    ])
AS_IF([test "x$happy" = xyes],
    [AC_CHECK_MEMBER([ARMCI_Group.comm], [], [], [[#include <armci.h>]])
     AS_IF([test "x$ac_cv_member_ARMCI_Group_comm" != xno],
        [ac_cv_member_ARMCI_Group_comm=1],
        [ac_cv_member_ARMCI_Group_comm=0])
     AC_DEFINE_UNQUOTED([HAVE_ARMCI_GROUP_COMM_MEMBER],
        [$ac_cv_member_ARMCI_Group_comm],
        [set to 1 if ARMCI has ARMCI_Group.comm member])
    ])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([ARMCI_Initialized], [armci])
     AS_IF([test "x$ac_cv_search_ARMCI_Initialized" != xno],
        [ac_cv_search_ARMCI_Initialized=1],
        [ac_cv_search_ARMCI_Initialized=0])
     AC_DEFINE_UNQUOTED([HAVE_ARMCI_INITIALIZED],
        [$ac_cv_search_ARMCI_Initialized],
        [set to 1 if ARMCI has ARMCI_Initialized function])
    ])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([armci_stride_info_init], [armci])
     AS_IF([test "x$ac_cv_search_armci_stride_info_init" != xno],
        [ac_cv_search_armci_stride_info_init=1],
        [ac_cv_search_armci_stride_info_init=0])
     AC_DEFINE_UNQUOTED([HAVE_ARMCI_STRIDE_INFO_INIT],
        [$ac_cv_search_armci_stride_info_init],
        [set to 1 if ARMCI has armci_stride_info_init function])
    ])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([armci_notify], [armci])
     AS_IF([test "x$ac_cv_search_armci_notify" != xno],
        [ac_cv_search_armci_notify=1],
        [ac_cv_search_armci_notify=0])
     AC_DEFINE_UNQUOTED([HAVE_ARMCI_NOTIFY],
        [$ac_cv_search_armci_notify],
        [set to 1 if ARMCI has armci_notify function])
    ])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([armci_msg_init], [armci])
     AS_IF([test "x$ac_cv_search_armci_msg_init" != xno],
        [ac_cv_search_armci_msg_init=1],
        [ac_cv_search_armci_msg_init=0])
     AC_DEFINE_UNQUOTED([HAVE_ARMCI_MSG_INIT],
        [$ac_cv_search_armci_msg_init],
        [set to 1 if ARMCI has armci_msg_init function])
    ])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([armci_msg_finalize], [armci])
     AS_IF([test "x$ac_cv_search_armci_msg_finalize" != xno],
        [ac_cv_search_armci_msg_finalize=1],
        [ac_cv_search_armci_msg_finalize=0])
     AC_DEFINE_UNQUOTED([HAVE_ARMCI_MSG_FINALIZE],
        [$ac_cv_search_armci_msg_finalize],
        [set to 1 if ARMCI has armci_msg_finalize function])
    ])
AM_CONDITIONAL([HAVE_ARMCI_GROUP_COMM],
   [test "x$ac_cv_search_armci_group_comm" = x1])
AM_CONDITIONAL([HAVE_ARMCI_GROUP_COMM_MEMBER],
   [test "x$ac_cv_member_ARMCI_Group_comm" = x1])
AM_CONDITIONAL([HAVE_ARMCI_INITIALIZED],
   [test "x$ac_cv_search_ARMCI_Initialized" = x1])
AM_CONDITIONAL([HAVE_ARMCI_STRIDE_INFO_INIT],
   [test "x$ac_cv_search_armci_stride_info_init" = x1])
AM_CONDITIONAL([HAVE_ARMCI_NOTIFY],
   [test "x$ac_cv_search_armci_notify" = x1])
AM_CONDITIONAL([HAVE_ARMCI_MSG_INIT],
   [test "x$ac_cv_search_armci_msg_init" = x1])
AM_CONDITIONAL([HAVE_ARMCI_MSG_FINALIZE],
   [test "x$ac_cv_search_armci_msg_finalize" = x1])
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=ARMCI; with_armci=yes; armci_network_external=1; $1],
    [armci_network_external=0; $2])
])dnl

# _GA_ARMCI_NETWORK_CRAY_SHMEM([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# ----------------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_CRAY_SHMEM], [
AC_MSG_NOTICE([searching for CRAY_SHMEM...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [AC_CHECK_HEADER([mpp/shmem.h], [],
        [AC_CHECK_HEADER([shmem.h], [], [happy=no])])])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([shmem_init], [sma], [], [happy=no])
     AS_CASE([$ac_cv_search_shmem_init],
        ["none required"], [],
        [no], [],
        [# add sma to ARMCI_NETWORK_LIBS if not there
         AS_CASE([$ARMCI_NETWORK_LIBS],
                 [*sma*], [],
                 [ARMCI_NETWORK_LIBS="$ARMCI_NETWORK_LIBS -lsma"])])])
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=CRAY_SHMEM; with_cray_shmem=yes; $1],
    [$2])
])dnl

# _GA_ARMCI_NETWORK_LAPI([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# ----------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_LAPI], [
AC_MSG_NOTICE([searching for LAPI...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [AC_CHECK_HEADER([lapi.h], [], [happy=no])])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([LAPI_Init], [lapi_r lapi], [], [happy=no])
     AS_CASE([$ac_cv_search_LAPI_Init],
            ["none required"], [],
            [no], [],
            [# add missing lib to ARMCI_NETWORK_LIBS if not there
             AS_CASE([$ARMCI_NETWORK_LIBS],
                     [*$ac_cv_search_LAPI_Init*], [],
                     [ARMCI_NETWORK_LIBS="$ARMCI_NETWORK_LIBS $ac_cv_search_LAPI_Init"])])])
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=LAPI; with_lapi=yes; $1],
    [$2])
])dnl

# _GA_ARMCI_NETWORK_MPI_TS([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# ---------------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_MPI_TS], [
AC_MSG_NOTICE([searching for MPI_TS...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=MPI_TS; with_mpi_ts=yes; $1],
    [$2])
])dnl

# _GA_ARMCI_NETWORK_MPI_MT([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# ---------------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_MPI_MT], [
AC_MSG_NOTICE([searching for MPI_MT...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=MPI_MT; with_mpi_mt=yes; $1],
    [$2])
])dnl

# _GA_ARMCI_NETWORK_MPI_PT([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# ---------------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_MPI_PT], [
AC_MSG_NOTICE([searching for MPI_PT...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=MPI_PT; with_mpi_pt=yes; $1],
    [$2])
])dnl

# _GA_ARMCI_NETWORK_MPI_PR([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# ---------------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_MPI_PR], [
AC_MSG_NOTICE([searching for MPI_PR...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=MPI_PR; with_mpi_pr=yes; $1],
    [$2])
])dnl

# _GA_ARMCI_NETWORK_MPI3([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# ---------------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_MPI3], [
AC_MSG_NOTICE([searching for MPI3...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=MPI3; with_mpi3=yes; $1],
    [$2])
])dnl

# _GA_ARMCI_NETWORK_MPI_SPAWN([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# ---------------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_MPI_SPAWN], [
AC_MSG_NOTICE([searching for MPI_SPAWN...])
happy=yes
CPPFLAGS="$CPPFLAGS $GA_MP_CPPFLAGS"
LDFLAGS="$LDFLAGS $GA_MP_LDFLAGS"
LIBS="$LIBS $GA_MP_LIBS"
AS_IF([test "x$happy" = xyes],
    [AC_CHECK_HEADER([mpi.h], [], [happy=no])])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([MPI_Comm_spawn_multiple], [mpi mpich.cnk mpich.rts],
        [], [happy=no])])
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=MPI_SPAWN; with_mpi_spawn=yes; $1],
    [$2])
])dnl

# _GA_ARMCI_NETWORK_OFA([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# ---------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_OFA], [
AC_MSG_NOTICE([searching for OFA...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [AC_CHECK_HEADER([infiniband/verbs.h], [], [happy=no])])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([ibv_open_device], [ibverbs], [], [happy=no])
     AS_CASE([$ac_cv_search_ibv_open_device],
        ["none required"], [],
        [no], [],
        [ARMCI_NETWORK_LIBS="$ARMCI_NETWORK_LIBS $ac_cv_search_ibv_open_device"])])
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=OFA; with_ofa=yes; $1],
    [$2])
])dnl

# _GA_ARMCI_NETWORK_OPENIB([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# ------------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_OPENIB], [
AC_MSG_NOTICE([searching for OPENIB...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [AC_CHECK_HEADER([infiniband/verbs.h], [], [happy=no])])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([ibv_open_device], [ibverbs], [], [happy=no])
     AS_CASE([$ac_cv_search_ibv_open_device],
        ["none required"], [],
        [no], [],
        [ARMCI_NETWORK_LIBS="$ARMCI_NETWORK_LIBS $ac_cv_search_ibv_open_device"])])
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=OPENIB; with_openib=yes; $1],
    [$2])
])dnl

# _GA_ARMCI_NETWORK_PORTALS4([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# -------------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_PORTALS4], [
AC_MSG_NOTICE([searching for PORTALS4...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=PORTALS4; with_portals4=yes; $1],
    [$2])
])dnl

# _GA_ARMCI_NETWORK_DMAPP([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# -----------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_DMAPP], [
AC_MSG_NOTICE([searching for DMAPP...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=DMAPP; with_dmapp=yes; $1],
    [$2])
AS_IF([test "x$happy" = xyes],
    [AC_SEARCH_LIBS([gethugepagesize], [hugetlbfs])
     AS_CASE([$ac_cv_search_gethugepagesize],
            ["none required"], [],
            [no], [],
            [# add missing lib to ARMCI_NETWORK_LIBS if not there
             AS_CASE([$ARMCI_NETWORK_LIBS],
                     [*$ac_cv_search_gethugepagesize*], [],
                     [ARMCI_NETWORK_LIBS="$ARMCI_NETWORK_LIBS $ac_cv_search_gethugepagesize"])])])
])dnl

# _GA_ARMCI_NETWORK_OFI([ACTION-IF-FOUND], [ACTION-IF-NOT-FOUND])
# -------------------------------------------------------------------
AC_DEFUN([_GA_ARMCI_NETWORK_OFI], [
AC_MSG_NOTICE([searching for OFI...])
happy=yes
AS_IF([test "x$happy" = xyes],
    [ga_armci_network=OFI; with_ofi=yes; $1],
    [$2])
])dnl

# GA_ARMCI_NETWORK
# ----------------
# This macro allows user to choose the armci network but also allows the
# network to be tested for automatically.
AC_DEFUN([GA_ARMCI_NETWORK], [
# Clear the variables we will be using, just in case.
ga_armci_network=
ARMCI_NETWORK_LIBS=
ARMCI_NETWORK_LDFLAGS=
ARMCI_NETWORK_CPPFLAGS=
AC_ARG_ENABLE([autodetect],
    [AS_HELP_STRING([--enable-autodetect],
        [attempt to locate ARMCI_NETWORK besides SOCKETS])])
# First, all of the "--with" stuff is taken care of.
armci_network_external=0
armci_network_count=0
_GA_ARMCI_NETWORK_WITH([armci],     [external; path to external ARMCI library])
_GA_ARMCI_NETWORK_WITH([cray-shmem],[Cray XT shmem])
_GA_ARMCI_NETWORK_WITH([dmapp],     [(Comex) Cray DMAPP])
_GA_ARMCI_NETWORK_WITH([lapi],      [IBM LAPI])
_GA_ARMCI_NETWORK_WITH([mpi-mt],    [(Comex) MPI-2 multi-threading])
_GA_ARMCI_NETWORK_WITH([mpi-pt],    [(Comex) MPI-2 multi-threading with progress thread])
_GA_ARMCI_NETWORK_WITH([mpi-pr],    [(Comex) MPI-1 two-sided with progress rank])
_GA_ARMCI_NETWORK_WITH([mpi-spawn], [MPI-2 dynamic process mgmt])
_GA_ARMCI_NETWORK_WITH([mpi-ts],    [(Comex) MPI-1 two-sided])
_GA_ARMCI_NETWORK_WITH([mpi3],      [(Comex) MPI-3 one-sided])
_GA_ARMCI_NETWORK_WITH([ofa],       [(Comex) Infiniband OpenIB])
_GA_ARMCI_NETWORK_WITH([ofi],       [(Comex) OFI])
_GA_ARMCI_NETWORK_WITH([openib],    [Infiniband OpenIB])
_GA_ARMCI_NETWORK_WITH([portals4],  [(Comex) Portals4])
_GA_ARMCI_NETWORK_WITH([sockets],   [Ethernet TCP/IP])
# Temporarily add ARMCI_NETWORK_CPPFLAGS to CPPFLAGS.
ga_save_CPPFLAGS="$CPPFLAGS"; CPPFLAGS="$CPPFLAGS $ARMCI_NETWORK_CPPFLAGS"
# Temporarily add ARMCI_NETWORK_LDFLAGS to LDFLAGS.
ga_save_LDFLAGS="$LDFLAGS"; LDFLAGS="$LDFLAGS $ARMCI_NETWORK_LDFLAGS"
# Temporarily add ARMCI_NETWORK_LIBS to LIBS.
ga_save_LIBS="$LIBS"; LIBS="$ARMCI_NETWORK_LIBS $LIBS"
AS_IF([test "x$enable_autodetect" = xyes],
    [AC_MSG_NOTICE([searching for ARMCI_NETWORK...])
     AS_IF([test "x$ga_armci_network" = x && test "x$with_cray_shmem" != xno],
        [_GA_ARMCI_NETWORK_CRAY_SHMEM()])
     AS_IF([test "x$ga_armci_network" = x && test "x$with_lapi" != xno],
        [_GA_ARMCI_NETWORK_LAPI()])
dnl     AS_IF([test "x$ga_armci_network" = x && test "x$with_mpi_ts" != xno],
dnl         [_GA_ARMCI_NETWORK_MPI_TS()])
dnl     AS_IF([test "x$ga_armci_network" = x && test "x$with_mpi_mt" != xno],
dnl         [_GA_ARMCI_NETWORK_MPI_MT()])
dnl     AS_IF([test "x$ga_armci_network" = x && test "x$with_mpi_pt" != xno],
dnl         [_GA_ARMCI_NETWORK_MPI_PT()])
dnl     AS_IF([test "x$ga_armci_network" = x && test "x$with_mpi_pr" != xno],
dnl         [_GA_ARMCI_NETWORK_MPI_PR()])
dnl     AS_IF([test "x$ga_armci_network" = x && test "x$with_mpi3" != xno],
dnl         [_GA_ARMCI_NETWORK_MPI3()])
dnl     AS_IF([test "x$ga_armci_network" = x && test "x$with_mpi_spawn" != xno],
dnl         [_GA_ARMCI_NETWORK_MPI_SPAWN()])
     AS_IF([test "x$ga_armci_network" = x && test "x$with_ofa" != xno],
        [_GA_ARMCI_NETWORK_OFA()])
     AS_IF([test "x$ga_armci_network" = x && test "x$with_openib" != xno],
        [_GA_ARMCI_NETWORK_OPENIB()])
     AS_IF([test "x$ga_armci_network" = x && test "x$with_portals4" != xno],
        [_GA_ARMCI_NETWORK_PORTALS4()])
     AS_IF([test "x$ga_armci_network" = x && test "x$with_dmapp" != xno],
        [_GA_ARMCI_NETWORK_DMAPP()])
     AS_IF([test "x$ga_armci_network" = x && test "x$with_armci" != xno],
        [_GA_ARMCI_NETWORK_ARMCI()])
     AS_IF([test "x$ga_armci_network" = x && test "x$with_ofi" != xno],
        [_GA_ARMCI_NETWORK_OFI()])
     AS_IF([test "x$ga_armci_network" = x],
        [AC_MSG_WARN([!!!])
         AC_MSG_WARN([No ARMCI_NETWORK detected, defaulting to MPI_TS])
         AC_MSG_WARN([!!!])
         ga_armci_network=MPI_TS; with_mpi_ts=yes])],
    [# Not autodetecting
     # Check whether multiple armci networks were selected by user.
     AS_CASE([$armci_network_count],
        [0], [AC_MSG_WARN([No ARMCI_NETWORK specified, defaulting to MPI_TS])
              ga_armci_network=MPI_TS; with_mpi_ts=yes],
        [1], [AS_IF([test "x$ga_armci_network" = xARMCI],
                 [_GA_ARMCI_NETWORK_ARMCI([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=ARMCI failed])])])
              AS_IF([test "x$ga_armci_network" = xCRAY_SHMEM],
                 [_GA_ARMCI_NETWORK_CRAY_SHMEM([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=CRAY_SHMEM failed])])])
              AS_IF([test "x$ga_armci_network" = xDMAPP],
                 [_GA_ARMCI_NETWORK_DMAPP([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=DMAPP failed])])])
              AS_IF([test "x$ga_armci_network" = xLAPI],
                 [_GA_ARMCI_NETWORK_LAPI([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=LAPI failed])])])
              AS_IF([test "x$ga_armci_network" = xMPI_TS],
                 [_GA_ARMCI_NETWORK_MPI_TS([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=MPI_TS failed])])])
              AS_IF([test "x$ga_armci_network" = xMPI_MT],
                 [_GA_ARMCI_NETWORK_MPI_MT([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=MPI_MT failed])])])
              AS_IF([test "x$ga_armci_network" = xMPI_PT],
                 [_GA_ARMCI_NETWORK_MPI_PT([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=MPI_PT failed])])])
              AS_IF([test "x$ga_armci_network" = xMPI_PR],
                 [_GA_ARMCI_NETWORK_MPI_PR([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=MPI_PR failed])])])
              AS_IF([test "x$ga_armci_network" = xMPI3],
                 [_GA_ARMCI_NETWORK_MPI3([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=MPI3 failed])])])
              AS_IF([test "x$ga_armci_network" = xMPI_SPAWN],
                 [_GA_ARMCI_NETWORK_MPI_SPAWN([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=MPI_SPAWN failed])])])
              AS_IF([test "x$ga_armci_network" = xOFA],
                 [_GA_ARMCI_NETWORK_OFA([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=OFA failed])])])
              AS_IF([test "x$ga_armci_network" = xOPENIB],
                 [_GA_ARMCI_NETWORK_OPENIB([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=OPENIB failed])])])
              AS_IF([test "x$ga_armci_network" = xPORTALS4],
                 [_GA_ARMCI_NETWORK_PORTALS4([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=PORTALS4 failed])])])
              AS_IF([test "x$ga_armci_network" = xOFI],
                 [_GA_ARMCI_NETWORK_OFI([],
                    [AC_MSG_ERROR([test for ARMCI_NETWORK=OFI failed])])])
             ],
        [AC_MSG_WARN([too many armci networks specified: $armci_network_count])
         AC_MSG_WARN([the following were specified:])
         _GA_ARMCI_NETWORK_WARN([armci])
         _GA_ARMCI_NETWORK_WARN([cray-shmem])
         _GA_ARMCI_NETWORK_WARN([dmapp])
         _GA_ARMCI_NETWORK_WARN([lapi])
         _GA_ARMCI_NETWORK_WARN([mpi-ts])
         _GA_ARMCI_NETWORK_WARN([mpi-mt])
         _GA_ARMCI_NETWORK_WARN([mpi-pt])
         _GA_ARMCI_NETWORK_WARN([mpi-pr])
         _GA_ARMCI_NETWORK_WARN([mpi-spawn])
         _GA_ARMCI_NETWORK_WARN([mpi3])
         _GA_ARMCI_NETWORK_WARN([ofa])
         _GA_ARMCI_NETWORK_WARN([openib])
         _GA_ARMCI_NETWORK_WARN([portals4])
         _GA_ARMCI_NETWORK_WARN([ofi])
         _GA_ARMCI_NETWORK_WARN([sockets])
         AC_MSG_ERROR([please select only one armci network])])])
# Remove ARMCI_NETWORK_CPPFLAGS from CPPFLAGS.
CPPFLAGS="$ga_save_CPPFLAGS"
# Remove ARMCI_NETWORK_LDFLAGS from LDFLAGS.
LDFLAGS="$ga_save_LDFLAGS"
# Remove ARMCI_NETWORK_LIBS from LIBS.
LIBS="$ga_save_LIBS"
_GA_ARMCI_NETWORK_AM_CONDITIONAL([armci])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([cray-shmem])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([dmapp])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([lapi])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([mpi-ts])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([mpi-mt])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([mpi-pt])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([mpi-pr])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([mpi-spawn])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([mpi3])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([ofa])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([openib])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([portals4])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([ofi])
_GA_ARMCI_NETWORK_AM_CONDITIONAL([sockets])
AC_SUBST([ARMCI_NETWORK_LDFLAGS])
AC_SUBST([ARMCI_NETWORK_LIBS])
AC_SUBST([ARMCI_NETWORK_CPPFLAGS])

# permanent hack
AS_CASE([$ga_armci_network],
[DMAPP],    [ARMCI_SRC_DIR=comex],
[MPI_MT],   [ARMCI_SRC_DIR=comex],
[MPI_PT],   [ARMCI_SRC_DIR=comex],
[MPI_PR],   [ARMCI_SRC_DIR=comex],
[MPI_TS],   [ARMCI_SRC_DIR=comex],
[MPI3],     [ARMCI_SRC_DIR=comex],
[OFA],      [ARMCI_SRC_DIR=comex],
[OFI],      [ARMCI_SRC_DIR=comex],
[OPENIB],   [ARMCI_SRC_DIR=src],
[PORTALS4], [ARMCI_SRC_DIR=comex],
            [ARMCI_SRC_DIR=src])
AC_SUBST([ARMCI_SRC_DIR])
AM_CONDITIONAL([ARMCI_SRC_DIR_COMEX],   [test "x$ARMCI_SRC_DIR" = "xcomex"])
AM_CONDITIONAL([ARMCI_SRC_DIR_SRC],     [test "x$ARMCI_SRC_DIR" = "xsrc"])
AS_IF([test "x$ARMCI_SRC_DIR" = "xcomex"], [armci_network_external=1])
AM_CONDITIONAL([ARMCI_NETWORK_EXTERNAL], [test "x$armci_network_external" = x1])
AM_CONDITIONAL([ARMCI_NETWORK_COMEX], [test "x$ARMCI_SRC_DIR" = "xcomex"])

# tcgmsg5 requires this
AS_IF([test x$ga_armci_network = xLAPI],
[AC_DEFINE([NOTIFY_SENDER], [1],
    [this was defined unconditionally when using LAPI for tcgmsg 5])
AC_DEFINE([LAPI], [1], [tcgmsg 5 requires this when using LAPI])
])

ga_cray_xt_networks=no
AS_IF([test x$ga_armci_network = xCRAY_SHMEM], [ga_cray_xt_networks=yes])
AM_CONDITIONAL([CRAY_XT_NETWORKS], [test x$ga_cray_xt_networks = xyes])

ga_cv_sysv_hack=no
# Only perform this hack for ARMCI build.
AS_IF([test "x$ARMCI_TOP_BUILDDIR" != x], [
    AS_IF([test x$ga_cv_sysv = xno],
        [AS_CASE([$ga_armci_network],
            [PORTALS|GEMINI], [ga_cv_sysv_hack=no],
                [ga_cv_sysv_hack=yes])],
        [ga_cv_sysv_hack=yes])
AS_IF([test x$ga_cv_sysv_hack = xyes],
    [AC_DEFINE([SYSV], [1],
        [Defined if we want this system to use SYSV shared memory])])
])
AM_CONDITIONAL([SYSV], [test x$ga_cv_sysv_hack = xyes])

# if not using external armci library, the following functions are always available
AS_IF([test "x$ga_armci_network" != xARMCI],
    [AC_DEFINE([HAVE_ARMCI_GROUP_COMM], [1], [])
     AC_DEFINE([HAVE_ARMCI_INITIALIZED], [1], [])
     AC_DEFINE([HAVE_ARMCI_NOTIFY], [1], [])
     AC_DEFINE([HAVE_ARMCI_MSG_INIT], [1], [])
     AC_DEFINE([HAVE_ARMCI_MSG_FINALIZE], [1], [])])
AM_CONDITIONAL([HAVE_ARMCI_GROUP_COMM_MEMBER],
   [test "x$ac_cv_member_ARMCI_Group_comm" = x1])
AM_CONDITIONAL([HAVE_ARMCI_GROUP_COMM],  [test "x$ga_armci_network" != xARMCI])
AM_CONDITIONAL([HAVE_ARMCI_INITIALIZED], [test "x$ga_armci_network" != xARMCI])
AM_CONDITIONAL([HAVE_ARMCI_NOTIFY],      [test "x$ga_armci_network" != xARMCI])
AM_CONDITIONAL([HAVE_ARMCI_MSG_INIT],    [test "x$ga_armci_network" != xARMCI])
AM_CONDITIONAL([HAVE_ARMCI_MSG_FINALIZE],[test "x$ga_armci_network" != xARMCI])
# the armci iterators only available in the conglomerate sources
AS_CASE([$ga_armci_network],
    [ARMCI], [],
    [AC_DEFINE([HAVE_ARMCI_STRIDE_INFO_INIT], [1], [])])
AM_CONDITIONAL([HAVE_ARMCI_STRIDE_INFO_INIT],
    [test "x$ga_armci_network" != xARMCI])

# ugly hack for working around NWChem memory requirements
# and MPI_PR startup verus the 'classic' ARMCI startup
delay_tcgmsg_mpi_startup=1
AS_CASE([$ga_armci_network],
[ARMCI],        [delay_tcgmsg_mpi_startup=0],
[CRAY_SHMEM],   [delay_tcgmsg_mpi_startup=1],
[DMAPP],        [delay_tcgmsg_mpi_startup=0],
[LAPI],         [delay_tcgmsg_mpi_startup=1],
[MPI_TS],       [delay_tcgmsg_mpi_startup=0],
[MPI_MT],       [delay_tcgmsg_mpi_startup=0],
[MPI_PT],       [delay_tcgmsg_mpi_startup=0],
[MPI_PR],       [delay_tcgmsg_mpi_startup=0],
[MPI_SPAWN],    [delay_tcgmsg_mpi_startup=1],
[MPI3],         [delay_tcgmsg_mpi_startup=0],
[OFA],          [delay_tcgmsg_mpi_startup=0],
[OFI],          [delay_tcgmsg_mpi_startup=0],
[OPENIB],       [delay_tcgmsg_mpi_startup=1],
[PORTALS4],     [delay_tcgmsg_mpi_startup=0],
[SOCKETS],      [delay_tcgmsg_mpi_startup=1])
AC_DEFINE_UNQUOTED([NEED_DELAY_TCGMSG_MPI_STARTUP],
    [$delay_tcgmsg_mpi_startup],
    [whether to wait until the last moment to call ARMCI_Init() in TCGMSG-MPI])
])dnl
