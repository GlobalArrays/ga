#if HAVE_CONFIG_H
#   include "config.h"
#endif

#if HAVE_STDIO_H
#   include <stdio.h>
#endif
#if HAVE_STRING_H
#   include <string.h>
#endif

#include "farg.h"
#include "typesf2c.h"
#include "sndrcv.h"
#include "srftoc.h"
#define LEN 255

/**
 * Hewlett Packard Risc box, SparcWorks F77 2.* and Paragon compilers.
 * Have to construct the argument list by calling FORTRAN.
 */
void PBEGINF_()
{
    Integer argc;
    char **argv;

    ga_f2c_get_cmd_args(&argc, &argv);
    tcgi_pbegin(argc, argv);
}


/**
 * Alternative entry for those senstive to FORTRAN making reference
 * to 7 character external names
 */
void PBGINF_()
{
    PBEGINF_();
}
