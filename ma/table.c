/*
 * $Id: table.c,v 1.5 2000-05-09 21:29:17 d3h325 Exp $
 */

/*
 * Dynamic table module.
 *
 * Each table entry consists of an external data item and internal state.
 * The size of the table is automatically increased if there isn't room
 * to add another entry.  The client can allocate, deallocate, verify
 * entries, and perform both handle-->data and data-->handle lookup.
 */

#include <stdio.h>
#include <stdlib.h>
#include "error.h"
#include "memcpy.h"
#include "scope.h"
#include "table.h"

/**
 ** constants
 **/

/* default # of initial table entries */
#define DEFAULT_TABLE_ENTRIES	32

/**
 ** types
 **/

/* state of a TableEntry */
typedef enum
{
    TES_Unused = 0,			/* never used */
    TES_Allocated,			/* currently in use */
    TES_Deallocated			/* formerly in use */
} TableEntryState;

/* table entry consists of data and state */
typedef struct _TableEntry
{
    TableData		data;		/* in this entry */
    TableEntryState	state;		/* of this entry */
} TableEntry;

/* table is an array of table entries */
typedef TableEntry * Table;

/**
 ** variables
 **/

/* currently only one table is managed */
private Table table = NULL;

private Integer table_capacity = 0;
private Integer table_entries = 0;
private Integer table_next_slot = 0;

/**
 ** public routines for internal use only
 **/

/* ------------------------------------------------------------------------- */
/*
 * Allocate a table slot, store the given data into it, and return a handle
 * by which the slot may be referenced.  If a slot cannot be allocated,
 * return TABLE_HANDLE_NONE.
 */
/* ------------------------------------------------------------------------- */

public Integer table_allocate(data)
    TableData	data;		/* to store */
{
    Table	new_table;
    Integer	new_table_capacity;
    unsigned	new_table_size;
    Integer	i;
    Integer	slots_examined;

    /* expand the table if necessary */
    if (table_entries >= table_capacity)
    {
        /* increase table capacity */
        if (table_capacity == 0)
            /* set the initial capacity */
            new_table_capacity = DEFAULT_TABLE_ENTRIES;
        else
            /* double the current capacity */
            new_table_capacity = 2 * table_capacity;

        /* allocate space for new table */
        new_table_size = (unsigned)(new_table_capacity * sizeof(TableEntry));
        if ((new_table = (Table)bytealloc(new_table_size)) == (Table)NULL)
        {
            (void)sprintf(ma_ebuf,
                "could not allocate %u bytes for table",
                new_table_size);
            ma_error(EL_Nonfatal, ET_Internal, "table_allocate", ma_ebuf);
            return TABLE_HANDLE_NONE;
        }

        /* copy and free old table */
        if (table_capacity > 0)
        {
            bytecopy(table, new_table, (table_capacity * sizeof(TableEntry)));
            bytefree(table);
        }

        /* initialize new part of new table */
        for (i = new_table_capacity-1; i >= table_capacity; i--)
            new_table[i].state = TES_Unused;

        /* remember the new table */
        table = new_table;
        table_next_slot = table_capacity;
        table_capacity = new_table_capacity;
    }

    /* perform a linear circular search to find the next available slot */
    for (slots_examined = 0, i = table_next_slot;
         slots_examined < table_capacity;
         slots_examined++, i = (i+1) % table_capacity)
    {
        if (table[i].state != TES_Allocated)
        {
            /* store the data */
            table[i].data = data;
            table[i].state = TES_Allocated;

            /* increment table_entries */
            table_entries++;

            /* advance table_next_slot */
            table_next_slot = (i+1) % table_capacity;

            /* return the handle */
            return i;
        }
    }

    /* if we get here, something is wrong */
    (void)sprintf(ma_ebuf,
        "no table slot available, %d/%d slots used",
        table_entries, table_capacity);
    ma_error(EL_Nonfatal, ET_Internal, "table_allocate", ma_ebuf);
    return TABLE_HANDLE_NONE;
}

/* ------------------------------------------------------------------------- */
/*
 * Deallocate the table slot corresponding to the given handle,
 * which should have been verified previously (e.g., via table_verify()).
 * If handle is invalid, print an error message.
 */
/* ------------------------------------------------------------------------- */

public void table_deallocate(handle)
    Integer	handle;		/* to deallocate */
{
    if (table_verify(handle, "table_deallocate"))
    {
        /* deallocate the slot */
        table[handle].state = TES_Deallocated;

        /* decrement table_entries */
        table_entries--;
    }
}

/* ------------------------------------------------------------------------- */
/*
 * Return the data in the table slot corresponding to the given handle,
 * which should have been verified previously (e.g., via table_verify()).
 * If handle is invalid, print an error message and return NULL.
 */
/* ------------------------------------------------------------------------- */

public TableData table_lookup(handle)
    Integer	handle;		/* to lookup */
{
    if (table_verify(handle, "table_lookup"))
        /* success */
        return table[handle].data;
    else
        /* failure */
        return (TableData)NULL;
}

/* ------------------------------------------------------------------------- */
/*
 * Return the handle for the table slot containing the given data (i.e.,
 * perform an associative or inverted lookup), or TABLE_HANDLE_NONE if
 * no currently allocated table slot contains the given data.
 *
 * If more than one table slot contains the given data, the one whose
 * handle is returned is undefined (i.e., implementation dependent).
 */
/* ------------------------------------------------------------------------- */

public Integer table_lookup_assoc(data)
    TableData	data;		/* to lookup */
{
    Integer	i;

    /* perform a linear search from the first table slot */
    for (i = 0; i < table_capacity; i++)
        if ((table[i].state == TES_Allocated) && (table[i].data == data))
            /* success */
            return i;

    /* failure */
    return TABLE_HANDLE_NONE;
}

/* ------------------------------------------------------------------------- */
/*
 * Return MA_TRUE if the given handle corresponds to a valid table slot
 * (one that is currently allocated), else return MA_FALSE and print an
 * error message.
 */
/* ------------------------------------------------------------------------- */

public Boolean table_verify(handle, caller)
    Integer	handle;		/* to verify */
    char	*caller;	/* name of calling routine */
{
    Boolean	badhandle;	/* is handle invalid? */

    badhandle = MA_FALSE;

    /* if handle is invalid, construct an error message */
    if ((handle < 0) ||
        (handle >= table_capacity) ||
        (table[handle].state == TES_Unused))
    {
        (void)sprintf(ma_ebuf,
            "handle %ld is not valid",
            (long)handle);
        badhandle = MA_TRUE;
    }
    else if (table[handle].state == TES_Deallocated)
    {
        (void)sprintf(ma_ebuf,
            "handle %ld already deallocated",
            (long)handle);
        badhandle = MA_TRUE;
    }

    if (badhandle)
    {
        /* invalid handle */
        ma_error(EL_Nonfatal, ET_External, caller, ma_ebuf);
        return MA_FALSE;
    }
    else
        /* valid handle */
        return MA_TRUE;
}
