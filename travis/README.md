Folder Contents
===============
This folder contains bash scripts used by the Travis CI build ([.travis.yml](../.travis.yml) in the root of the repo).

Build Overview
==============
The GA Travis build is a matrix build, meaning it combines the selected OS, compilers, and environment variables to create a set of builds run on each commit.

As of October 30, 2018 this means that there are 42 builds run for each commit (2 OS x 2 compilers x 10 env settings + 2 xcode installs on MacOS).

For each combination of the matrix build, the `before_install` followed by the `install` scripts are run. These get the required dependencies for the build installed and ready to use. Then the `script` section is run, followed by `after_failure` in the event the build fails.

Within the `script` section, the [build-run.sh](build-run.sh) file is executed that includes running the appropriate test suite depending on the language. E.g., [test.x](../global/testing/test.x) for Fortran or [testc.x](../global/testing/testc.x) for C.

Tutorials
=========
See https://docs.travis-ci.com/user/tutorial/ for more information on creating Travis CI builds.
