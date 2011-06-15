:author: Jeff Daily
:email: jeff.daily@pnnl.gov
:institution: Pacific Northwest National Laboratory

:author: Robert R. Lewis
:email: bobl@tricity.wsu.edu
:institution: Washington State University

--------------------------------------------------------------------------------
Using the Global Arrays Toolkit to Reimplement NumPy for Distributed Computation
--------------------------------------------------------------------------------

.. class:: abstract

   Global Arrays (GA) is a software system from Pacific Northwest National
   Laboratory that enables an efficient, portable, and parallel shared-memory
   programming interface to manipulate distributed dense arrays. Using a
   combination of GA and NumPy, we have reimplemented NumPy as a distributed
   drop-in replacement called Global Arrays in Numpy (GAiN). Scalability
   studies will be presented showing the utility of developing serial NumPy
   codes which can later run on more capable clusters or supercomputers.

.. class:: keywords

   Global Arrays, Python, NumPy, MPI

Introduction
------------

Scientific computing with Python typically involves using the NumPy package.
NumPy provides an efficient multi-dimensional array and array processing
routines. Unfortunately, like many Python programs, NumPy is serial in nature.
This limits both the size of the arrays as well as the speed with which the
arrays can be processed to the available resources on a single compute node.

NumPy programs are written, debugged, and run on single machines. This may be
sufficient for certain problem domains. However, NumPy may also be used to
develop prototype software. Such software is usually ported to a different,
compiled language and/or explicitly parallelized to take advantage of
additional hardware.

Global Arrays in NumPy (GAiN) is an extension to Python and provides parallel,
distributed processing of arrays. It implements a subset of the NumPy API so
that for some programs, by simply importing GAiN in place of NumPy they may be
able to take advantage of parallel processing automatically. Other programs
may require slight modification. This allows those programs to take advantage
of the additional cores available on single compute nodes and to increase
problem sizes by distributing across clustered environments.

Background
----------

Like any complex piece of software, GAiN builds on many other foundational
ideas and implementations. This background is not intended to be a complete
reference of the subjects herein, rather only what is necessary to
understand the design and implementation of GAiN. Further details may be found
by examining the references or as otherwise noted.

NumPy
=====

NumPy [Oli06]_ is a Python extension module which adds a powerful
multidimensional array class ``ndarray`` to the Python language. NumPy also
provides scientific computing capabilities such as basic linear algebra and
Fourier transform support. NumPy is the de facto standard for scientific
computing in Python and the successor of the other numerical Python packages
Numarray [Dub96]_ and numeric [Asc99]_.

NumPy's ndarray
===============

The primary class defined by NumPy is the ``ndarray``. The ``ndarray`` is
implemented as a contiguous memory segment that is either FORTRAN- or
C-ordered. Recall that in FORTRAN, the first dimension changes the fastest
while it is the opposite (last) dimension in C. All ``ndarray`` instances have
a pointer to the location of the first element as well as the attributes
``shape``, ``ndim``, and ``strides``. ``ndim`` describes the number of
dimensions in the array, ``shape`` describes the number of elements in each
dimension, and ``strides`` describes the number of bytes between consecutive
elements per dimension. ``shape`` can be modified while ``ndim`` and
``strides`` are read-only and used internally, although their exposure to the
programmer may help in developing certain algorithms.

The creation of ``ndarray`` instances is complicated by the various ways in
which they can be created such as explicit constructor calls, view casting, or
creating new instances from template instances. To this end, the ``ndarray``
does not implement Python’s ``__init__()`` object constructor.  Instead,
``ndarrays`` use the ``__new__()`` classmethod. Recall that ``__new__()`` is
Python’s hook for subclassing its built-in objects. If ``__new__()`` returns
an instance of the class on which it is defined, then the class's
``__init__()`` method is also called. Otherwise, the ``__init__()`` method is
not called. But since views of ``ndarray`` instances can be created based on
subtypes, the ``__new__()`` classmethod might not always get called to
properly initialize the instance. ``__array_finalize__()`` is called instead
of ``__init__()`` for ndarray subclasses to avoid this limitation.

NumPy's Universal Functions
===========================

The element-wise operators in NumPy are known as *Universal Functions*, or
*ufuncs*. Many of the methods of the ``ndarray`` simply invoke the
corresponding ufunc. For example, the operator + calls ``ndarray.__add__()``
which invokes the ufunc ``add``. Ufuncs are either unary or binary, taking
either one or two arrays as input, respectively. Ufuncs always return the
result of the operation as an array. Optionally, an additional array may be
specified to receive the results of the operation. Specifying this output
array to the ufunc avoids the sometimes unnecessary creation of a new array.

Ufuncs are more than just callable functions. They also have some special
methods such as ``reduce`` and ``accumulate``. ``reduce`` is similar to
Python’s built-in function of the same name that repeatedly applies a callable
object to its last result and the next item of the sequence. This effectively
reduces a sequence to a single value. When applied to arrays the reduction
occurs along the first axis by default, but other axes may be specified. Each
ufunc defines the function that is used for the reduction. For example,
``add`` will sum the values along an axis while ``multiply`` will generate the
running product.  ``accumulate`` is similar to reduce, but it returns the
intermediate results of the reduction.

Ufuncs can operate on objects that are not ``ndarrays``. In order for
subclasses of the ``ndarray`` or array-like objects to utilize the ufuncs,
they may define three methods or one attribute which are
``__array_prepare__()``, ``__array_wrap__()``, ``__array__()``, and
``__array_priority__``, respectively.  The ``__array_prepare__()`` and
``__array_wrap__()`` methods will be called on either the output, if
specified, or the input with the highest ``__array_priority__``.
``__array_prepare__()`` is called on the way into the ufunc after the output
array is created but before any computation has been performed and
``__array_wrap__()`` is called on the way out of the ufunc. If an output is
specified and defines ``__array__()`` method, results will be written to the
object returned by calling ``__array__()``.

Parallel Programming Paradigms
==============================

Parallel applications can be classified into a few well defined programming
paradigms. Each paradigm is a class of algorithms that have the same control
structure. The literature differs in how these paradigms are classified and
the boundaries between paradigms can sometimes be fuzzy or intentionally
blended into hybrid models [Buy99]_. The Master/Slave and Single Program
Multiple Data (SPMD) paradigms are discussed further.

The master/slave paradigm, also known as task-farming, is where a single
master process farms out tasks to multiple slave processes. The control is
always maintained by the master, dispatching commands to the slaves. Usually,
the communication takes place only between the master and slaves. This model
may either use static or dynamic load-balancing. The former involves the
allocation of tasks to happen when the computation begins whereas the latter
allows the application to adjust to changing conditions within the
computation. Dynamic load-balancing may involve recovering after the failure
of a subset of slave processes or handling the case where the number of tasks
is not known at the start of the application.

With SPMD, each process executes essentially the same code but on a different
part of the data. The communication pattern is highly structured and
predictable. Occasionally, a global synchronization may be needed. The
efficiency of these types of programs depends on the decomposition of the data
and the degree to which the data is independent of its neighbors. These
programs are also highly susceptible to process failure. If any single process
fails, generally it causes deadlock since global synchronizations thereafter
would fail.

Message Passing Interface (MPI)
===============================

Message passing is one form of inter-process communication. Each process is
considered to have access only to its local memory. Data is transferred
between processes by the sending and receiving of messages which usually
requires the cooperation of participating processes. Communication can take
the form of one-to-one, one-to-many, many-to-one, or many-to-many.

Message passing libraries allow efficient parallel programs to be written for
distributed memory systems. MPI [Gro99a]_, also known as MPI-1, is a library
specification for message- passing that was standardized in May 1994 by the
MPI Forum. It is designed for high performance on both massively parallel
machines and on workstation clusters. An MPI implementation exists on nearly
all modern parallel systems and there are a number of freely available,
portable implementations for those systems that do not [Buy99]_.  As such, MPI
is the de facto standard for writing massively parallel application codes in
either FORTRAN, C, or C++.

MPI programs are typically started with either mpirun or mpiexec, specifying
the number of processes to invoke. If the MPI program is run without the use
of those, then it is run as if only one process was specified. Not all MPI
implementations support running without the use of the mpirun or mpiexec
programs. MPI programs can query their environment to determine how many
processes were specified. Further, each process can query to determine which
process they are out of the total number specified.

MPI programs are typically conform to the SPMD paradigm [Buy99]_. The mpiexec
pro- grams by default launch programs for this type of parallelism. A single
program is specified on the command line which gets replicated to all
participating processes. This same pro- gram is then executed within its own
address space on each process, such that any process knows only its own data
until it communicates with other processes, passing messages (data) around. A
“hello world” program executed in this fashion would print ”hello world” once
per process.

MPI-2
=====

The MPI-2 standard [Gro99b]_ was first completed in 1997 and added a number of
important additions to MPI including, but not limited to, process creation and
management, one-sided communication, parallel file I/O, and the C++ language
binding. With MPI-2, any single MPI process or group of processes can invoke
additional MPI processes. This is useful when the total number of processes
required for the problem at hand cannot be known a priori.

Before MPI-2, all communication required explicit handshaking between the
sender and receiver via MPI_Send() and MPI_Recv() in addition to non-blocking
variants.  MPI-2’s one-sided communication model allows reads, writes, and
accumulates of remote memory without the explicit cooperation of the process
owning the memory. If synchro- nization is required at a later time, it can be
requested via MPI_Barrier(). Otherwise, there is no strict guarantee that a
one-sided operation will complete before the data segment it accessed is used
by another process.

Parallel I/O in MPI-2, sometimes referred to as MPI-IO, allows for single,
collective files to be output by an MPI process. Before MPI-IO, one such I/O
model for SPMD pro- grams was to have each process write to its own file.
Having each process write to its own file may be fast, however in most cases
it requires substantial post-processing in order to stitch those files back
together into a coherent, single-file representation thus diminish- ing the
benefit of parallel computation. Other forms of parallel I/O before MPI-IO was
introduced included having all other processes send their data to a single
process for out- put. However, any computational speed-ups from the
parallelism are reduced by having to communicate all data back to a single
node. MPI-IO hides the I/O model behind calls to the API, allowing efficient
I/O routines to be developed independently of the calling MPI programs. One
such popular implementation of MPI-IO is ROMIO [Tha04]_.

mpi4py
======

mpi4py is a Python wrapper around MPI written to mimic the C++ language
bindings. It supports point-to-point communication as well as the collective
communication models. Typical communication of arbitrary objects in the
FORTRAN or C bindings of MPI require the programmer to define new MPI
datatypes. These datatypes describe the number and order of the bytes to be
communicated. On the other hand, strings could be sent without defining a new
datatype so long as the length of the string was understood by the recipient.
mpi4py is able to communicate any pickleable Python object since pickled
objects are just byte streams. mpi4py also has special enhancements to
efficiently communicate any object implementing Python’s buffer protocol, such
as NumPy arrays. It also supports dynamic process management and parallel I/O
[Dal05]_ [Dal08]_.

Global Arrays
=============

The GA toolkit [Nie06]_ [Nie10]_ [Pnl11]_ is a software system from Battelle
Pacific Northwest National Laboratory that enables an efficient, portable, and
parallel shared-memory programming interface to manipulate physically
distributed dense multidimensional arrays, without the need for explicit
cooperation by other processes. GA compliments the message-passing programming
model and is compatible with MPI so that the programmer can use both in the
same program. The GA library handles the distribution of arrays across
processes and recognizes that accessing local memory is faster than accessing
remote memory. However, the library allows access mechanisms for any part of
the entire distributed array regardless of where its data is located. Local
memory is acquired via ``NGA_Access()`` returning a pointer while remote
memory is retrieved via ``NGA_Get()`` filling an already allocated array
buffer. GA has been leveraged in several large computational chemistry codes
and has been shown to scale well [Apr09]_.

Aggregate Remote Memory Copy Interface (ARMCI)
==============================================

ARMCI provides general-purpose, efficient, and widely portable remote memory
access (RMA) operations (one-sided communication). ARMCI operations are
optimized for contiguous and noncontiguous (strided, scatter/gather, I/O
vector) data transfers. It also exploits native network communication in-
terfaces and system resources such as shared memory [Nie00]_.  ARMCI provides
simpler progress rules and a less synchronous model of RMA than MPI-2. ARMCI
has been used to implement the Global Arrays library, GPSHMEM - a portable
version of Cray SHMEM library, and the portable Co-Array FORTAN compiler from
Rice University [Dot04]_.

Cython
======

TODO

Previous Work
-------------

GAiN is similar in many ways to other parallel computation software packages.
It attempts to leverage the best ideas for transparent, parallel processing
found in current systems. The following packages provided insight into how
GAiN was to be developed.

Star-P
======

MITMatlab [Hus98]_, which was later rebranded as Star-P [Ede07]_, provids a
client-server model for interactive, large-scale scientific computation. It
provids a transparently parallel front-en through the popular MATLAB [Pal07]_
numerical package and sends the parallel computations to its Parallel Problem
Server workhorse. Separating the interactive, serial nature of MATLAB from the
parallel computation server allows the user to leverage both of their
strengths. This also allows much larger arrays to be operated over than is
allowed by a single compute node.

Global Arrays Meets MATLAB
==========================

Global Arrays Meets MATLAB (GAMMA) [Pan06]_ provides a MATLAB binding to the
GA toolkit, thus allowing for larger problem sizes and parallel computation.
MATLAB provides an interactive interpreter, however to fully utilize GAMMA one
must run within a parallel environment such as provided by MPI and a cluster
of compute nodes.  GAMMA was shown to scale well even within an interpreted
environment like MATLAB.

IPython
=======

IPython [Per07]_ provides an enhanced interactive Python shell as well as an
architecture for interactive parallel computing. IPython supports practically
all models of parallelism but more importantly in an interactive way. For
instance, a single interactive Python shell could be controlling a parallel
program running on a super computer. This is done by having a Python engine
running on a remote machine which is able to receive Python commands.

IPython's distarray
===================

distarray [Gra09]_ is an experimental package for the IPython project.
distarray uses IPython’s architecture as well as MPI extensively in order to
look and feel like NumPy’s ndarray. Only the SPMD model of parallel
computation is supported, unlike other parallel models supported directly by
IPython.  Further, the status of distarray is that of a proof of concept and
not production ready.

GpuPy
=====

A Graphics Process Unit (GPU) is a powerful parallel processor that is capable
of more floating point calculations per second than a traditional CPU.
However, GPUs are more difficult to program and require other special
considerations such as copying data from main memory to the GPU’s on-board
memory in order for it to be processed, then copying the results back. The
GpuPy [Eit07]_ Python extension package was developed to lessen these burdens
by providing a NumPy-like interface for the GPU. Preliminary results
demonstrate considerable speedups for certain single-precision floating point
operations.

pyGA
====

The Global Arrays toolkit was wrapped in Python for the 3.x series of GA by
Robert Harrison [Har99]. It was written as a C extension to Python and only
wrapped a subset of the complete GA functionality. It illustrated some
important concepts such as the benefits of integration with NumPy and the
difficulty of compiling GA on certain systems. In pyGA, the local or remote
portions of the global arrays were retrieved as NumPy arrays at which point
they could be used as inputs to NumPy functions like the ufuncs. However, the
burden was still on the programmer to understand the SPMD nature of the
program. For example, when accessing the global array as an ndarray, the array
shape and dimensions would match that of the local array maintained by the
process calling the access function. Such an implementation is entirely
correct, however there was no attempt to handle slicing at the global level as
it is implemented in NumPy. In short, pyGA recognized the benefit of
returning portions of the global array wrapped in a NumPy array, but it did
not treat the global arrays as if they were themselves a subclass of the
ndarray.

Co-Array Python
===============

Co-Array Python [Ras04]_ is modeled after the Co-Array FORTRAN extensions to
FORTRAN 95. It allows the programmer to access data elements on non-local
processors via an extra array dimension, called the co-dimension. The
``CoArray`` module provided a local data structure existing on all processors
executing in a SPMD fashion. The CoArray was designed as an extension to
Numeric Python [Asc99]_.

Design
------

There comes a point at which a single compute node does not have the resources
necessary for executing a given problem. The need for parallel programming and
running these programs on parallel architectures is obvious, however,
efficiently programming for a parallel environment can be a daunting task. One
area of research is to automatically parallelize otherwise serial programs and
to do so with the least amount of user intervention [Buy99]_. GAiN attempts to
do this for certain Python programs utilizing the NumPy module. It will be
shown that some NumPy program can be parallelized in a nearly transparent way
with GAiN.

Both NumPy and Global Arrays are well established in their respective
communities. However, NumPy is inherently serial.  Also, the size of its
arrays are limited by the resources of a single compute node. NumPy’s
computational capabilities may be efficient, however parallelizing them using
the SPMD paradigm will allow for larger problem sizes and may also see
performance gains. This design attempts to leverage the substantial work that
is Global Arrays in support of large parallel array computation within the
NumPy framework.

Python is known for among other things its ease of use, elegant syntax, and
its interactive interpreter. Python users would expect these capabilities to
remain intact for any extension written for it. The IPython project is a good
example of supporting the interactive interpreter and parallel computation
simultaneously. Users familiar with NumPy would expect its syntax and
semantics to remain intact if large parallel array computation were added to
its feature set.

High performance computing users are familiar with writing codes that optimize
every last bit of performance out of the system where they are are being run.
Although message-passing is a useful and widely adopted computation model,
Global Arrays users have come to appreciate the abstraction of a shared-memory
interface to a distributed memory array. In either case, users are familiar
with the challenges involved in maintaining scalability as problem sizes
increase or as additional hardware is added. Maintaining these codes may be
difficult if they are muddled with FORTRAN and/or C and various
message-passing API calls. If one of these users were to switch to NumPy in
order to leverage its strengths, they would hope to not sacrifice the
performance and scalability they once may have enjoyed. Given the two user
communities discussed above, our GAiN module attempts to bridge the gap and
support both.

There are a few assumptions which govern the design of GAiN. First, all public
GAiN functions are collective. Since Python and NumPy were designed to run
serially on workstations, it naturally follows that GAiN -- running in an SPMD
fashion -- will execute every public function collectively. Second, not all
arrays should be distributed. If we assume that the cost of communication is
high such that communication should be avoided, cetain design goals become
clear. Small arrays and scalar values should be replicated on each process
rather than distributed, and data locality should be emphasized over
communiation. It follows, then, that GAiN operations should allow
mixed inputs of both distributed and local array-like objects. Further, NumPy
represents an extensive, useful, and hardened API. Every effort to reuse NumPy
should be made. Lastly, GA has its own strengths to offer such as processor
groups and custom data distributions. In order to maximize scalability of this
implementation, we should enable the use of processor groups [Nie05]_.

Both NumPy and GA provide multidimensional arrays and implement, among other
things, element-wise operations and linear algebra routines. Although they
have a number of differences, the primary one is that NumPy programs run
within a single address space while Global Arrys are distributed. When
translating from NumPy to Global Arrays, each process must translate NumPy
calls into calls with respect to their local array portions.

A distributed array representation must acknowledge the duality of a global
array and the many local arrays. Figure :ref:`fig1` will help illustrate.
Each local piece of the ``gain.ndarray`` has its own shape (in parenthesis)
and knows its portion of the distribution (in square brackets). Each local
piece also knows the global shape.

.. figure:: image1_crop.png

    :label:`fig1`
    Each local piece of the ``gain.ndarray`` has its own shape (in
    parenthesis) and knows its portion of the distribution (in square
    brackets). Each local piece also knows the global shape.

In order to handle the bookkeeping required for the distributed nature of
these arrays, a fundamental design decision was whether to subclass the
``ndarray`` or to provide a work-alike replacement module. The NumPy
documentation states that the ``ndarray`` implements ``__new__()`` in order to
control array creation via constructor calls, view casting, and slicing.
Subclasses implement ``__new__()`` for when the constructor is called directly, and ``__array_finalize__()`` in order to set additional
attributes or further modify the object from which the view has been taken.
The problem is that properly subclassing the ``ndarray`` will always result in
the allocation of memory. 

Implementation
--------------

We present a new Python module, ``gain``, developed as part of the main Global
Arrays software distribution. The release of GA v5.0 contained Python bindings
based on the complete GA C API, available in the extension module ``ga``. The
bindings were developed using Cython. With the upcoming release of GA v5.1,
the module ``ga.gain`` is available as a drop-in replacement for NumPy.  The
goal of the implementation is to allow users to write:

.. code-block:: python

    from ga import gain as numpy

Cython was also used to develop ``gain``. The implementation details for how
GA was used for distributed computation of NumPy arrays follows.

``gain.ndarray``
================

. Slice arithemetic
. Access
. Get (strided_get)

``gain.flatiter``
=================

. Gather/scatter

``gain.ufunc``
==============

 . Access the output, get the rest
 . Optimizations
 . . When inputs are the same object
 . . When inputs have the same distribution and slicing

Evaluation
----------

TODO

Conclusion
----------

TODO

Future Work
-----------

TODO

.. [Apr09]  E. Apra, A. P. Rendell, R. J. Harrison, V. Tipparaju, W. A.
            deJong, and S. S. Xantheas. *Liquid water: obtaining the right
            answer for the right reasons*, Proceedings of the Conference on
            High Performance Computing Networking, Storage, and Analysis,
            66:1-7, 2009.
.. [Asc99]  D. Ascher, P. F. Dubois, K. Hinsen, J. Hugunin, and T. Oliphant.
            *Numerical Python*, UCRL-MA-128569, 1999.
.. [Beh11]  S. Behnel, R. Bradshaw, C. Citro, L. Dalcin, D. S. Seljebotn, and
            K. Smith. *Cython: The Best of Both Worlds*, Computing in Science
            Engineering, 13(2):31-39, March/April 2011.
.. [Buy99]  R. Buyya. *High Performance Cluster Computing: Architectures and
            Systems*, Vol. 1, Prentice Hall PTR, 1 edition, May 1999.
.. [Dal05]  L. Dalcin, R. Paz, and M. Storti. *MPI for python*,
            Journal of Parallel and Distributed Computing, 65(9):1108-1115,
            September 2005.
.. [Dal08]  L. Dalcin, R. Paz, M. Storti, and J. D'Elia. *MPI for python:
            Performance improvements and MPI-2 extensions*,
            Journal of Parallel and Distributed Computing, 68(5):655-662,
            September 2005.
.. [Dot04]  Y. Dotsenko, C. Coarfa,. and J. Mellor-Crummmey. *A Multi-Platform
            Co-Array Fortran Compiler*, Proceedings of the 13th International
            Conference on Parallel Architectures and Compilation Techniques,
            29-40, 2004.
.. [Dub96]  P. F. Dubois, K. Hinsen, and J. Hugunin. *Numerical Python*,
            Computers in Physics, 10(3), May/June 1996.
.. [Ede07]  A. Edelman. *The Star-P High Performance Computing Platform*, IEEE
            International Conference on Acoustics, Speech, and Signal
            Processing, April 2007.
.. [Eit07]  B. Eitzen. *Gpupy: Efficiently using a gpu with python*, Master's
            thesis, Washington State University, Richland, WA, August 2007.
.. [Gra09]  B. Granger and F. Perez. *Distributed Data Structures, Parallel
            Computing and IPython*, SIAM CSE 2009.
.. [Gro99a] W. Gropp, E. Lusk, and A. Skjellum. *Using MPI: Portable Parallel
            Programming with the Message-Passing Interface*, second edition,
            MIT Press, November 1999.
.. [Gro99b] W. Gropp, E. Lusk, and R. Thakur. *Using MPI-2: Advanced Features of
            the Message-Passing Interface*, MIT Press, 1999.
.. [Har99]  R. J. Harrison. *Global Arrays Python Interface*,
            http://www.emsl.pnl.gov/docs/global/old/pyGA/, December 1999.
.. [Hus98]  P. Husbands and C. Isbell. *The Parallel Problems Server: A
            Client-Server Model for Interactive Large Scale Scientific
            Computation*, 3rd International Meeting on Vector and Parallel
            Processing, 1998.
.. [Nie00]  J. Nieplocha, J. Ju, and T. P. Straatsma. *A multiprotocol
            communication support for the global address space programming
            model on the IBM SP*, Proceedings of EuroPar, 2000.
.. [Nie05]  J. Nieplocha, M. Krishnan, B. Palmer, V. Tipparaju, and Y. Zhang.
            *Exploiting processor groups to extend scalability of the GA
            shared memory programming model*, Proceedings of the 2nd
            conference on Computing Frontiers, 262-272, 2005.
.. [Nie06]  J. Nieplocha, B. Palmer, V. Tipparaju, M. Krishnan, H. Trease, and
            E. Apra. *Advances, Applications and Performance of the Global
            Arrays Shared Memory Programming Toolkit*, International Journal of
            High Performance Computing Applications, 20(2):203-231, 2006.
.. [Nie10]  J. Nieplocha, M. Krishnan, B. Palmer, V. Tipparaju, and J. Ju. *The
            Global Arrays User's Manual*.
.. [Oli06]  T. E. Oliphant. *Guide to NumPy*, http://www.tramy.us/, March 2006.
.. [Pal07]  W. Palm III. *A Concise Introduction to Matlab*, McGraw-Hill, 1st
            edition, October 2007.
.. [Pan06]  R. Panuganti, M. M. Baskaran, D. E. Hudak, A. Krishnamurthy, J.
            Nieplocha, A. Rountev, and P. Sadayappan. *GAMMA: Global Arrays
            Meets Matlab*, Technical Report.
            ftp://ftp.cse.ohio-state.edu/pub/tech-report/ 2006/TR15.pdf
.. [Per07]  F. Perez and B. E. Granger. *IPython: a System for Interactive
            Scientific Computing*, Computing in Science Engineering,
            9(3):21-29, May 2007.
.. [Pnl11]  Global Arrays Webpage. http://www.emsl.pnl.gov/docs/global/
.. [Ras04]  C. E. Rasmussen, M. J. Sottile, J. Nieplocha, R. W. Numrich, and E.
            Jones. *Co-array Python: A Parallel Extension to the Python
            Language*, Euro-Par, 632-637, 2004.
.. [Tha04]  R. Thakur, E. Lusk, and W. Gropp. *Users guide for romio: A
            high-performance, portable, mpi-io implementation*, May 2004.
