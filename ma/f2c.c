/*
 * $Id: f2c.c,v 1.11 2002-09-14 05:40:30 d3g001 Exp $
 */

/*
 * C wrapper routines.  In general, f2c_foo_ is called by FORTRAN routine
 * MA_foo, performs any necessary argument munging, then calls C routine
 * MA_foo.
 */

#include "ma.h"
#include "scope.h"

#ifdef _CRAY
#include <fortran.h>
#endif /* _CRAY */

/**
 ** function types
 **/

private void fstring2cstring();

/**
 ** private routines
 **/

/* ------------------------------------------------------------------------- */
/*
 * Strip trailing blanks from fstring and copy it to cstring,
 * truncating if necessary to fit in cstring, and ensuring that
 * cstring is NUL-terminated.
 */
/* ------------------------------------------------------------------------- */

#if defined(_CRAY) || defined(WIN32)
private void fstring2cstring(fcd, cstring, clength)
    _fcd	fcd;		/* FORTRAN character descriptor */
    char	*cstring;	/* C buffer */
    Integer	clength;	/* max length (including NUL) of cstring */
#else /* _CRAY */
private void fstring2cstring(fstring, flength, cstring, clength)
    char	*fstring;	/* FORTRAN string */
    Integer	flength;	/* length of fstring */
    char	*cstring;	/* C buffer */
    Integer	clength;	/* max length (including NUL) of cstring */
#endif /* _CRAY */
{
#if defined(_CRAY) || defined(WIN32)
    char	*fstring;	/* FORTRAN string */
    Integer	flength;	/* length of fstring */

    fstring = _fcdtocp(fcd);
    flength = _fcdlen(fcd);
#endif /* _CRAY */

    /* remove trailing blanks from fstring */
    while (flength-- && fstring[flength] == ' ')
        ;

    /* the postdecrement above went one too far */
    flength++;

    /* truncate fstring to cstring size */
    if (flength >= clength)
        flength = clength - 1;

    /* ensure that cstring is NUL-terminated */
    cstring[flength] = '\0';

    /* copy fstring to cstring */
    while (flength--)
        cstring[flength] = fstring[flength];
}

/**
 ** public routines
 **/

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

#if defined(_CRAY) || defined(WIN32)
public Boolean FATR f2c_alloc_get_(datatype, nelem, name, memhandle, index)
    Integer	*datatype;
    Integer	*nelem;
    _fcd	name;
    Integer	*memhandle;
    Integer	*index;
#else /* _CRAY */
public Boolean FATR f2c_alloc_get_(datatype, nelem, name, memhandle, index, namesize)
    Integer	*datatype;
    Integer	*nelem;
    char	*name;
    Integer	*memhandle;
    Integer	*index;
    Integer	namesize;	/* implicitly passed by FORTRAN */
#endif /* _CRAY */
{
    Boolean	value;
    char	buf[MA_NAMESIZE];

    /* ensure that name is NUL-terminated */
#if defined(_CRAY) || defined(WIN32)
    fstring2cstring(name, buf, (Integer)sizeof(buf));
#else /* _CRAY */
    fstring2cstring(name, namesize, buf, (Integer)sizeof(buf));
#endif /* _CRAY */

    value = MA_alloc_get(*datatype, *nelem, buf, memhandle, index);

    /* FORTRAN array indexing is 1-based, so increment index */
    (*index)++;

    return value;
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

#if defined(_CRAY) || defined(WIN32)
public Boolean FATR f2c_allocate_heap_(datatype, nelem, name, memhandle)
    Integer	*datatype;
    Integer	*nelem;
    _fcd	name;
    Integer	*memhandle;
#else /* _CRAY */
public Boolean FATR f2c_allocate_heap_(datatype, nelem, name, memhandle, namesize)
    Integer	*datatype;
    Integer	*nelem;
    char	*name;
    Integer	*memhandle;
    Integer	namesize;	/* implicitly passed by FORTRAN */
#endif /* _CRAY */
{
    char	buf[MA_NAMESIZE];

    /* ensure that name is NUL-terminated */
#if defined(_CRAY) || defined(WIN32)
    fstring2cstring(name, buf, (Integer)sizeof(buf));
#else /* _CRAY */
    fstring2cstring(name, namesize, buf, (Integer)sizeof(buf));
#endif /* _CRAY */

    return MA_allocate_heap(*datatype, *nelem, buf, memhandle);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_chop_stack_(memhandle)
    Integer	*memhandle;
{
    return MA_chop_stack(*memhandle);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_free_heap_(memhandle)
    Integer	*memhandle;
{
    return MA_free_heap(*memhandle);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_free_heap_piece_(memhandle, nelem)
    Integer	*memhandle;
    Integer	*nelem;
{
    return MA_free_heap_piece(*memhandle, *nelem);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_get_index_(memhandle, index)
    Integer	*memhandle;
    Integer	*index;
{
    Boolean	value = MA_get_index(*memhandle, index);

    /* FORTRAN array indexing is 1-based, so increment index */
    (*index)++;

    return value;
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_get_next_memhandle_(ithandle, memhandle)
    Integer	*ithandle;
    Integer	*memhandle;
{
    return MA_get_next_memhandle(ithandle, memhandle);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_get_numalign_(value)
    Integer	*value;
{
    return MA_get_numalign(value);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_inform_base_(datatype, address1, address2)
    Integer	*datatype;
    Pointer	address1;
    Pointer	address2;
{
    return MAi_inform_base(*datatype, address1, address2);
}

#if defined(_CRAY) || defined(WIN32)
/* ------------------------------------------------------------------------- */
/*
 * Crays pass a FORTRAN character descriptor (_fcd) for byte_mb(i)
 * from ma_set_sizes, from which the character address must be extracted.
 * The C90 _fcd is 8 bytes, which fits into an 8-byte pointer, but the
 * T3D _fcd is 16 bytes, which doesn't fit.  To solve this problem, this
 * Cray-specific routine is provided.
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_inform_base_fcd_(datatype, fcd1, fcd2)
    Integer	*datatype;
    _fcd	fcd1;
    _fcd	fcd2;
{
    Pointer	address1 = _fcdtocp(fcd1);
    Pointer	address2 = _fcdtocp(fcd2);

    return MAi_inform_base(*datatype, address1, address2);
}
#endif /* _CRAY */

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_init_(datatype, nominal_stack, nominal_heap)
    Integer	*datatype;
    Integer	*nominal_stack;
    Integer	*nominal_heap;
{
    return MA_init(*datatype, *nominal_stack, *nominal_heap);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_initialized_()
{
    return MA_initialized();
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_init_memhandle_iterator_(ithandle)
    Integer	*ithandle;
{
    return MA_init_memhandle_iterator(ithandle);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Integer FATR f2c_inquire_avail_(datatype)
    Integer	*datatype;
{
    return MA_inquire_avail(*datatype);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Integer FATR f2c_inquire_heap_(datatype)
    Integer	*datatype;
{
    return MA_inquire_heap(*datatype);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Integer FATR f2c_inquire_heap_check_stack_(datatype)
    Integer	*datatype;
{
    return MA_inquire_heap_check_stack(*datatype);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Integer FATR f2c_inquire_heap_no_partition_(datatype)
    Integer	*datatype;
{
    return MA_inquire_heap_no_partition(*datatype);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Integer FATR f2c_inquire_stack_(datatype)
    Integer	*datatype;
{
    return MA_inquire_stack(*datatype);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Integer FATR f2c_inquire_stack_check_heap_(datatype)
    Integer	*datatype;
{
    return MA_inquire_stack_check_heap(*datatype);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Integer FATR f2c_inquire_stack_no_partition_(datatype)
    Integer	*datatype;
{
    return MA_inquire_stack_no_partition(*datatype);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_pop_stack_(memhandle)
    Integer	*memhandle;
{
    return MA_pop_stack(*memhandle);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public void FATR f2c_print_stats_(printroutines)
    Boolean	*printroutines;
{
    MA_print_stats(*printroutines);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

#if defined(_CRAY) || defined(WIN32)
public Boolean FATR f2c_push_get_(datatype, nelem, name, memhandle, index)
    Integer	*datatype;
    Integer	*nelem;
    _fcd	name;
    Integer	*memhandle;
    Integer	*index;
#else /* _CRAY */
public Boolean FATR f2c_push_get_(datatype, nelem, name, memhandle, index, namesize)
    Integer	*datatype;
    Integer	*nelem;
    char	*name;
    Integer	*memhandle;
    Integer	*index;
    Integer	namesize;	/* implicitly passed by FORTRAN */
#endif /* _CRAY */
{
    Boolean	value;
    char	buf[MA_NAMESIZE];

    /* ensure that name is NUL-terminated */
#if defined(_CRAY) || defined(WIN32)
    fstring2cstring(name, buf, (Integer)sizeof(buf));
#else /* _CRAY */
    fstring2cstring(name, namesize, buf, (Integer)sizeof(buf));
#endif /* _CRAY */

    value = MA_push_get(*datatype, *nelem, buf, memhandle, index);

    /* FORTRAN array indexing is 1-based, so increment index */
    (*index)++;

    return value;
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

#if defined(_CRAY) || defined(WIN32)
public Boolean FATR f2c_push_stack_(datatype, nelem, name, memhandle)
    Integer	*datatype;
    Integer	*nelem;
    _fcd	name;
    Integer	*memhandle;
#else /* _CRAY */
public Boolean FATR f2c_push_stack_(datatype, nelem, name, memhandle, namesize)
    Integer	*datatype;
    Integer	*nelem;
    char	*name;
    Integer	*memhandle;
    Integer	namesize;	/* implicitly passed by FORTRAN */
#endif /* _CRAY */
{
    char	buf[MA_NAMESIZE];

    /* ensure that name is NUL-terminated */
#if defined(_CRAY) || defined(WIN32)
    fstring2cstring(name, buf, (Integer)sizeof(buf));
#else /* _CRAY */
    fstring2cstring(name, namesize, buf, (Integer)sizeof(buf));
#endif /* _CRAY */

    return MA_push_stack(*datatype, *nelem, buf, memhandle);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_set_auto_verify_(value)
    Integer	*value;
{
    return MA_set_auto_verify((Boolean)*value);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_set_error_print_(value)
    Integer	*value;
{
    return MA_set_error_print((Boolean)*value);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_set_hard_fail_(value)
    Integer	*value;
{
    return MA_set_hard_fail((Boolean)*value);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_set_numalign_(value)
    Integer	*value;
{
    return MA_set_numalign(*value);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Integer FATR f2c_sizeof_(datatype1, nelem1, datatype2)
    Integer	*datatype1;
    Integer	*nelem1;
    Integer	*datatype2;
{
    return MA_sizeof(*datatype1, *nelem1, *datatype2);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Integer FATR f2c_sizeof_overhead_(datatype)
    Integer	*datatype;
{
    return MA_sizeof_overhead(*datatype);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public void FATR f2c_summarize_allocated_blocks_()
{
    /* FORTRAN indices are 1-based */
    MAi_summarize_allocated_blocks(1);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public void FATR f2c_trace_(value)
    Integer	*value;
{
    MA_trace((Boolean)*value);
}

/* ------------------------------------------------------------------------- */
/*
 * 
 */
/* ------------------------------------------------------------------------- */

public Boolean FATR f2c_verify_allocator_stuff_()
{
    return MA_verify_allocator_stuff();
}
