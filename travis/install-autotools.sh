#!/bin/sh

set -e
set -x

os=`uname`
TOP="$1"

if [ ! -d ${TOP} ] ; then
    mkdir ${TOP}
fi

case "$os" in
    Darwin|Linux)
        MAKE_JNUM=4
        M4_VERSION=1.4.17
        M4_VERSION_FOUND=
        M4_VERSION_MIN=1.4.12
        LIBTOOL_VERSION=2.4.6
        AUTOCONF_VERSION=2.69
        AUTOMAKE_VERSION=1.11

        # we need m4 at least version 1.4.13
        TOOL=m4
        M4_OKAY=no
        if m4 --version >/dev/null ; then
            M4_VERSION_FOUND=`m4 --version | head -n1 | cut -d' ' -f 4`
            rm -f m4.conftest
            cat > m4.conftest <<END
$M4_VERSION_FOUND
$M4_VERSION_MIN
END
            M4_VERSION_TOP=`sort -V m4.conftest | head -n 1`
            rm -f m4.conftest
            if [ "x$M4_VERSION_TOP" = "x$M4_VERSION_MIN" ] ; then
                M4_OKAY=yes
            fi
        fi

        if [ "x$M4_OKAY" = "xno" ] ; then
            cd ${TOP}
            TDIR=${TOOL}-${M4_VERSION}
            FILE=${TDIR}.tar.gz
            BIN=${TOP}/bin/${TOOL}
            if [ -f ${FILE} ] ; then
                echo ${FILE} already exists! Using existing copy.
            else
                wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
            fi
            if [ -d ${TDIR} ] ; then
                echo ${TDIR} already exists! Using existing copy.
            else
                echo Unpacking ${FILE}
                tar -xzf ${FILE}
            fi
            if [ -f ${BIN} ] ; then
                echo ${BIN} already exists! Skipping build.
            else
                cd ${TOP}/${TDIR}
                ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
                if [ "x$?" != "x0" ] ; then
                    echo FAILURE 1
                    exit
                fi
            fi
            # refresh the path
            export PATH=${TOP}/bin:$PATH
        else
            echo "${TOOL} found and is sufficiently new ($M4_VERSION_FOUND)"
        fi

        cd ${TOP}
        TOOL=autoconf
        TDIR=${TOOL}-${AUTOCONF_VERSION}
        FILE=${TDIR}.tar.gz
        BIN=${TOP}/bin/${TOOL}
        if [ ! -f ${FILE} ] ; then
          wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
        else
          echo ${FILE} already exists! Using existing copy.
        fi
        if [ ! -d ${TDIR} ] ; then
          echo Unpacking ${FILE}
          tar -xzf ${FILE}
        else
          echo ${TDIR} already exists! Using existing copy.
        fi
        if [ -f ${BIN} ] ; then
          echo ${BIN} already exists! Skipping build.
        else
          cd ${TOP}/${TDIR}
          ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
          if [ "x$?" != "x0" ] ; then
            echo FAILURE 3
            exit
          fi
        fi

        # refresh the path
        export PATH=${TOP}/bin:$PATH

        cd ${TOP}
        TOOL=automake
        TDIR=${TOOL}-${AUTOMAKE_VERSION}
        FILE=${TDIR}.tar.gz
        BIN=${TOP}/bin/${TOOL}
        if [ ! -f ${FILE} ] ; then
          wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
        else
          echo ${FILE} already exists! Using existing copy.
        fi
        if [ ! -d ${TDIR} ] ; then
          echo Unpacking ${FILE}
          tar -xzf ${FILE}
        else
          echo ${TDIR} already exists! Using existing copy.
        fi
        if [ -f ${BIN} ] ; then
          echo ${BIN} already exists! Skipping build.
        else
          cd ${TOP}/${TDIR}
          ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
          if [ "x$?" != "x0" ] ; then
            echo FAILURE 4
            exit
          fi
        fi

        # refresh the path
        export PATH=${TOP}/bin:$PATH

        cd ${TOP}
        TOOL=libtool
        TDIR=${TOOL}-${LIBTOOL_VERSION}
        FILE=${TDIR}.tar.gz
        BIN=${TOP}/bin/${TOOL}
        if [ ! -f ${FILE} ] ; then
          wget http://ftp.gnu.org/gnu/${TOOL}/${FILE}
        else
          echo ${FILE} already exists! Using existing copy.
        fi
        if [ ! -d ${TDIR} ] ; then
          echo Unpacking ${FILE}
          tar -xzf ${FILE}
        else
          echo ${TDIR} already exists! Using existing copy.
        fi
        if [ -f ${BIN} ] ; then
          echo ${BIN} already exists! Skipping build.
        else
          cd ${TOP}/${TDIR}
          ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
          if [ "x$?" != "x0" ] ; then
            echo FAILURE 2
            exit
          fi
        fi

        # refresh the path
        export PATH=${TOP}/bin:$PATH

        ;;
esac

