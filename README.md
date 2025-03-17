# GLOBAL ARRAYS

[![License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](LICENSE)
[![Documentation Status](https://readthedocs.org/projects/globalarrays/badge/?version=latest)](https://globalarrays.readthedocs.io/en/latest/?badge=latest)
[![GitHub Downloads](https://img.shields.io/github/downloads/GlobalArrays/ga/total)](https://github.com/GlobalArrays/ga/releases)
[![CI](https://github.com/GlobalArrays/ga/actions/workflows/github_actions.yml/badge.svg)](https://github.com/GlobalArrays/ga/actions?query=workflow:GlobalArrays_CI)

## Table of Contents

* [ACKNOWLEDGMENTS](#acknowledgment)
* [ABOUT THIS SOFTWARE](#about-this-software)
* [WHERE IS THE DOCUMENTATION?](#where-is-the-documentation)
* [QUESTIONS/HELP/SUPPORT/BUG-REPORT](#questionshelpsupportbug-report)

## ACKNOWLEDGMENTS

This software and its documentation were produced with United States Government support under Contract Number DE-AC06-76RLO-1830 awarded by the United States Department of Energy. The United States Government retains a paid-up non-exclusive, irrevocable worldwide license to reproduce, prepare derivative works, perform publicly and display publicly by or for the US Government, including the right to distribute to other US Government contractors.

The most recent source of funding for development of GA is the [Exascale Computing Project](https://exascaleproject.org).

## ABOUT THIS SOFTWARE

More information about Global Arrays can be found at the webpage
[https://globalarrays.github.io](httpss://globalarrays.github.io).

Global Arrays is a portable Non-Uniform Memory Access (NUMA) shared-memory programming environment for distributed and shared memory computers. It augments the message-passing model by providing a shared-memory like access to distributed dense arrays. This is also known as the Partitioned Global Address Space (PGAS) model.

This library contains the Global Arrays (GA), Communications Runtime for Exascale (ComEx) run-time library, Aggregate Remote Memory Copy Interface (ARMCI) run-time library, Memory Allocator (MA), parallel I/O libraries (DRA,EAF,SF), TCGMSG, and TCGMSG-MPI packages bundled together. 

ARMCI provides one-sided remote memory operations used by GA.

ComEx is a successor to ARMCI and provides an ARMCI-compatible interface. New parallel runtime development takes place within ComEx including the MPI-only runtimes.

DRA (Disk Resident Arrays) is a parallel I/O library that maintains dense two-dimensional arrays on disk. 

SF (Shared Files) is a parallel I/O library that allows noncollective I/O to a parallel file.

EAF (Exclusive Access Files) is parallel I/O library that supports I/O to private files.

TCGMSG is a simple, efficient, but obsolete message-passing library.

TCGMSG-MPI is a TCGMSG interface implementation on top of MPI and ARMCI. 

MA is a dynamic memory allocator/manager for Fortran and C programs.

GA++ is a C++ binding for global arrays.

## WHERE IS THE DOCUMENTATION?

The [GA manual](https://globalarrays.readthedocs.io) contains all the documentation.  
The API reference can be found [here](https://globalarrays.github.io/userinterface.html)

## QUESTIONS/HELP/SUPPORT/BUG-REPORT

Please submit issues to our [GitHub issue tracker](https://github.com/GlobalArrays/ga/issues).  We use Google Groups to host [our discussion forum](https://groups.google.com/forum/#!forum/hpctools).
