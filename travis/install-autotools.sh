#!/bin/sh

set -e
set -x

os=`uname`
TOP="$1"

if [ ! -d ${TOP} ] ; then
    mkdir ${TOP}
fi

if [ ! -d ${TOP}/bin ] ; then
    mkdir ${TOP}/bin
fi

download=""
if wget --version > /dev/null ; then
    download="wget -O"
else
    if curl --version > /dev/null ; then
        download="curl -o"
    fi
fi
if [ "x${download}" = x ] ; then
    echo "failed to determine download agent"
    exit 1
fi

case "$os" in
    Darwin|Linux)
        MAKE_JNUM=4
        # we need m4 at least version 1.4.16
        M4_VERSION=1.4.17
        LIBTOOL_VERSION=2.4.6
        AUTOCONF_VERSION=2.69
        AUTOMAKE_VERSION=1.11.6

        # check whether we can reach ftp.gnu.org
        TIMEOUT=timeout
        if [ "x$os" = "xDarwin" ] ; then
            TIMEOUT=gtimeout
        fi
        # do we have a working timeout command?
        HAVE_TIMEOUT=yes
        if ! $TIMEOUT --version > /dev/null ; then
            HAVE_TIMEOUT=no
        fi
        FTP_OK=yes
        if [ "x$HAVE_TIMEOUT" = xyes ] ; then
            if ! $TIMEOUT 2 bash -c "</dev/tcp/ftp.gnu.org/21" ; then
                FTP_OK=no
                # can we reach our backup URL?
                if ! $TIMEOUT 2 bash -c "</dev/tcp/github.com/443" ; then
                    echo FAILURE 0
                    exit 1
                fi
            fi
        fi

        ##########################################
        ### config.guess
        ##########################################
        cd ${TOP}/bin
        if [ -f config.guess ] ; then
            echo "config.guess already exists! Using existing copy."
        else
            ${download} config.guess 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.guess;hb=HEAD'
        fi

        ##########################################
        ### config.guess
        ##########################################
        cd ${TOP}/bin
        if [ -f config.sub ] ; then
            echo "config.sub already exists! Using existing copy."
        else
            ${download} config.sub 'http://git.savannah.gnu.org/gitweb/?p=config.git;a=blob_plain;f=config.sub;hb=HEAD'
        fi

        ##########################################
        ### m4
        ##########################################
        cd ${TOP}
        TOOL=m4
        TOOL_OKAY=no
        TOOL_VERSION=$M4_VERSION
        if ${TOOL} --version >/dev/null ; then
            TOOL_VERSION_FOUND=`${TOOL} --version | head -n1 | cut -d' ' -f 4`
            if [ "x$TOOL_VERSION" = "x$TOOL_VERSION_FOUND" ] ; then
                TOOL_OKAY=yes
            fi
        fi

        if [ "x$TOOL_OKAY" = "xno" ] ; then
            TDIR=${TOOL}-${TOOL_VERSION}
            FILE=${TDIR}.tar.gz
            BIN=${TOP}/bin/${TOOL}
            URL=http://ftp.gnu.org/gnu/${TOOL}/${FILE}
            if [ "x$FTP_OK" = "xno" ] ; then
                URL=https://github.com/GlobalArrays/autotools/blob/master/${FILE}?raw=true
            fi
            if [ -f ${FILE} ] ; then
                echo ${FILE} already exists! Using existing copy.
            else
                ${download} ${FILE} ${URL}
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
                cp ${TOP}/bin/config.guess ./build-aux/config.guess
                cp ${TOP}/bin/config.sub ./build-aux/config.sub
                ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
                if [ "x$?" != "x0" ] ; then
                    echo FAILURE 1
                    exit
                fi
            fi
            # refresh the path
            export PATH=${TOP}/bin:$PATH
        else
            echo "${TOOL} found and is sufficiently new ($TOOL_VERSION_FOUND)"
        fi

        ##########################################
        ### autoconf
        ##########################################

        cd ${TOP}
        TOOL=autoconf
        TOOL_OKAY=no
        TOOL_VERSION=$AUTOCONF_VERSION
        if ${TOOL} --version >/dev/null ; then
            TOOL_VERSION_FOUND=`${TOOL} --version | head -n1 | cut -d' ' -f 4`
            if [ "x$TOOL_VERSION" = "x$TOOL_VERSION_FOUND" ] ; then
                TOOL_OKAY=yes
            fi
        fi

        if [ "x$TOOL_OKAY" = "xno" ] ; then
            TDIR=${TOOL}-${TOOL_VERSION}
            FILE=${TDIR}.tar.gz
            BIN=${TOP}/bin/${TOOL}
            URL=http://ftp.gnu.org/gnu/${TOOL}/${FILE}
            if [ "x$FTP_OK" = "xno" ] ; then
                URL=https://github.com/GlobalArrays/autotools/blob/master/${FILE}?raw=true
            fi
            if [ ! -f ${FILE} ] ; then
                ${download} ${FILE} ${URL}
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
                cp ${TOP}/bin/config.guess ./build-aux/config.guess
                cp ${TOP}/bin/config.sub ./build-aux/config.sub
                ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
                if [ "x$?" != "x0" ] ; then
                    echo FAILURE 3
                    exit
                fi
            fi
            # refresh the path
            export PATH=${TOP}/bin:$PATH
        else
            echo "${TOOL} found and is exactly needed version ($TOOL_VERSION_FOUND)"
        fi

        ##########################################
        ### automake
        ##########################################

        cd ${TOP}
        TOOL=automake
        TOOL_OKAY=no
        TOOL_VERSION=$AUTOMAKE_VERSION
        if ${TOOL} --version >/dev/null ; then
            TOOL_VERSION_FOUND=`${TOOL} --version | head -n1 | cut -d' ' -f 4`
            if [ "x$TOOL_VERSION" = "x$TOOL_VERSION_FOUND" ] ; then
                TOOL_OKAY=yes
            fi
        fi

        if [ "x$TOOL_OKAY" = "xno" ] ; then
            TDIR=${TOOL}-${TOOL_VERSION}
            FILE=${TDIR}.tar.gz
            BIN=${TOP}/bin/${TOOL}
            URL=http://ftp.gnu.org/gnu/${TOOL}/${FILE}
            if [ "x$FTP_OK" = "xno" ] ; then
                URL=https://github.com/GlobalArrays/autotools/blob/master/${FILE}?raw=true
            fi
            if [ ! -f ${FILE} ] ; then
                ${download} ${FILE} ${URL}
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
                cp ${TOP}/bin/config.guess ./lib/config.guess
                cp ${TOP}/bin/config.sub ./lib/config.sub
                ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
                if [ "x$?" != "x0" ] ; then
                    echo FAILURE 4
                    exit
                fi
            fi
            # refresh the path
            export PATH=${TOP}/bin:$PATH
        else
            echo "${TOOL} found and is exactly needed version ($TOOL_VERSION_FOUND)"
        fi

        ##########################################
        ### libtool
        ##########################################

        cd ${TOP}
        TOOL=libtool
        TOOL_OKAY=no
        TOOL_VERSION=$LIBTOOL_VERSION
        if ${TOOL} --version >/dev/null ; then
            TOOL_VERSION_FOUND=`${TOOL} --version | head -n1 | cut -d' ' -f 4`
            if [ "x$TOOL_VERSION" = "x$TOOL_VERSION_FOUND" ] ; then
                TOOL_OKAY=yes
            fi
        fi

        if [ "x$TOOL_OKAY" = "xno" ] ; then
            TDIR=${TOOL}-${TOOL_VERSION}
            FILE=${TDIR}.tar.gz
            BIN=${TOP}/bin/${TOOL}
            URL=http://ftp.gnu.org/gnu/${TOOL}/${FILE}
            if [ "x$FTP_OK" = "xno" ] ; then
                URL=https://github.com/GlobalArrays/autotools/blob/master/${FILE}?raw=true
            fi
            if [ ! -f ${FILE} ] ; then
                ${download} ${FILE} ${URL}
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
                cp ${TOP}/bin/config.guess ./build-aux/config.guess
                cp ${TOP}/bin/config.sub ./build-aux/config.sub
                ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
                if [ "x$?" != "x0" ] ; then
                    echo FAILURE 2
                    exit
                fi
            fi
            # refresh the path
            export PATH=${TOP}/bin:$PATH
        else
            echo "${TOOL} found and is exactly needed version ($TOOL_VERSION_FOUND)"
        fi

        ;;
esac

