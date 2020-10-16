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
TOOL_VERSION=$M4_VERSION
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
    if [ -f secure_snprintf.patch ] ; then
        echo secure_snprintf.patch already exists! Using existing copy.
    else
        ${download} secure_snprintf.patch https://raw.githubusercontent.com/macports/macports-ports/master/devel/m4/files/secure_snprintf.patch
    fi
    if patch -p0 -N < secure_snprintf.patch ; then
        echo patch applied
    fi
    if [ -f 0001-fflush-adjust-to-glibc-2.28-libio.h-removal.patch ] ; then
        echo glibc v2.28 patch already exists! Using existing copy.
    else
        ${download} 0001-fflush-adjust-to-glibc-2.28-libio.h-removal.patch https://gitlab.com/atilla/buildroot/raw/c48f8a64626c60bd1b46804b7cf1a699ff53cdf3/package/m4/0001-fflush-adjust-to-glibc-2.28-libio.h-removal.patch
    fi
    if patch -p1 -N < 0001-fflush-adjust-to-glibc-2.28-libio.h-removal.patch ; then
        echo patch applied
    fi
    ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
    if [ "x$?" != "x0" ] ; then
        echo FAILURE 1
        exit 1
    fi
fi
# refresh the path
export PATH=${TOP}/bin:$PATH
export M4=${TOP}/bin/m4

##########################################
### autoconf
##########################################

cd ${TOP}
TOOL=autoconf
TOOL_VERSION=$AUTOCONF_VERSION
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
# patch for ifx -loopopt=0 issue
    wget https://github.com/autotools-mirror/autoconf/commit/ae26b9b44c183c41b84fc36ff70082713295d621.patch
    patch -p1 < ae26b9b44c183c41b84fc36ff70082713295d621.patch
    ./configure --prefix=${TOP} && make -j ${MAKE_JNUM} && make install
    if [ "x$?" != "x0" ] ; then
        echo FAILURE 3
        exit 3
    fi
fi
# refresh the path
export PATH=${TOP}/bin:$PATH
export AUTOCONF=${TOP}/bin/autoconf
export AUTOHEADER=${TOP}/bin/autoheader
export AUTOM4TE=${TOP}/bin/autom4te
export AUTORECONF=${TOP}/bin/autoreconf
export AUTOSCAN=${TOP}/bin/autoscan
export AUTOUPDATE=${TOP}/bin/autoupdate
export IFNAMES=${TOP}/bin/ifnames

##########################################
### automake
##########################################

cd ${TOP}
TOOL=automake
TOOL_VERSION=$AUTOMAKE_VERSION
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
        exit 4
    fi
fi
# refresh the path
export PATH=${TOP}/bin:$PATH
export AUTOMAKE=${TOP}/bin/automake
export ACLOCAL=${TOP}/bin/aclocal

# patch this old automake for new perl
if [ -f ${TOP}/bin/automake.patched ] ; then
    echo ${TOP}/bin/automake.patched already exists!
else
    awk 'NR==4159 {$0="  $text =~ s/\\$\\{([^ \\t=:+{}]+)}/&substitute_ac_subst_variables_worker ($1)/ge;"} 1' ${TOP}/bin/automake > ${TOP}/bin/automake.patched
fi

if [ diff ${TOP}/bin/automake.patched ${TOP}/bin/automake >/dev/null ] ; then
    echo  ${TOP}/bin/automake is already patched!
else
    cp ${TOP}/bin/automake.patched ${TOP}/bin/automake
fi

##########################################
### libtool
##########################################

cd ${TOP}
TOOL=libtool
TOOL_VERSION=$LIBTOOL_VERSION
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
        exit 2
    fi
fi
# refresh the path
export PATH=${TOP}/bin:$PATH

