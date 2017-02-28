#!/bin/sh

set -e
set -x

os=`uname`
TRAVIS_ROOT="$1"

case "$os" in
    Linux)
        echo "Linux"
        sudo apt-get install -y -q gfortran
        ;;
    Darwin)
        echo "Mac"
        brew update
        brew install gfortran
        ;;
esac

echo "using gfortran at `which gfortran`"
