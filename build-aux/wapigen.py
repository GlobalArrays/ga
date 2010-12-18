#!/usr/bin/env python

"""Generate the wapi.c source from the papi.h header."""

import re
import sys

if len(sys.argv) != 2:
    print "incorrect number of arguments"
    print "usage: wapi.py <papi.h> > <wapi.c>"
    sys.exit(len(sys.argv))

# first, gather all function signatures from papi.h aka argv[1]
accumulating = False
signatures = []
current_signature = ""
EXTERN = 'extern'
SEMICOLON = ';'
for line in open(sys.argv[1]):
    line = line.strip() # remove whitespace before and after line
    if not line:
        continue # skip blank lines
    if EXTERN in line and SEMICOLON in line:
        signatures.append(line)
    elif EXTERN in line:
        current_signature = line
        accumulating = True
    elif SEMICOLON in line and accumulating:
        current_signature += line
        signatures.append(current_signature)
        accumulating = False
    elif accumulating:
        current_signature += line

# print headers
print '#if HAVE_CONFIG_H'
print '#   include "config.h"'
print '#endif'
print ''
print '#include "papi.h"'
print '#include "typesf2c.h"'
print ''

# now process the signatures
call_remove = r'void|char|short|int|long|float|double|Integer|logical|Logical|SingleComplex|DoubleComplex|Real|DoublePrecision|AccessIndex|\*|\[|\]|FILE'
for sig in signatures:
    sig_wnga = re.sub('pnga', 'wnga', sig[:-1].split(None, 1)[-1])
    sig_wnga = re.sub('extern', '', sig_wnga)
    pnga_func = sig.split('(',1)[0].strip().split()[-1]
    args      = sig.split('(',1)[1].strip()
    wnga_func = re.sub('pnga', 'wnga', pnga_func)
    call_pnga = pnga_func + "(" + re.sub(call_remove, '', args).strip()
    print sig_wnga,'{'
    if sig.split()[1].startswith('void'):
        print '    %s;' % call_pnga
    else:
        print '    return %s;' % call_pnga
    print '}'
