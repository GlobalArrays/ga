#!/usr/bin/env python

"""Generate the wapi.c source from the papi.h header."""

import sys

if len(sys.argv) != 2:
    print "incorrect number of arguments"
    print "usage: wapi.py <papi.h> > <wapi.c>"
    sys.exit(len(sys.argv))

print '#if HAVE_CONFIG_H'
print '#   include "config.h"'
print '#endif'
print ''
print '#include "papi.h"'
print '#include "typesf2c.h"'
print ''

class Param:
    def __init__(self, arg):
        if "*" in arg:
            self.pointer = True
            arg = arg.replace("*","")
        else:
            self.pointer = False
        self.type,self.name = arg.strip().split()
        self.type = self.type.strip()
        self.name = self.name.strip()
    def sig(self):
        if self.pointer:
            return "%s *%s" % (self.type,self.name)
        else:
            return "%s %s" % (self.type,self.name)
    def arg(self):
        return self.name
    def __repr__(self):
        return self.sig()

accumulating = False
acc = "" # accumulate signature while printing each line as-is
skip = True # skip lines until the first "pnga" is found
for line in open(sys.argv[1]): # papi.h
    line = line.rstrip() # remove newline and any extra trailing whitespace
    if "#endif" in line: # avoid preprocessor guard
        skip = True
    if "pnga" in line:
        skip = False
        accumulating = True
    if skip:
        continue
    if accumulating:
        acc += line
    if "pnga" in line:
        line = line.replace("pnga","wnga")
    if "extern" in line:
        line = line.replace("extern","").lstrip()
    if ";" in line:
        # hit the end of the function signature, stop accumulating
        accumulating = False
        print line.replace(";","")
        line = line.replace(";","")
        extern,type,parts = acc.split(None,2)
        acc = "" # reset accumulation buffer
        parts = parts.strip()
        pfunc,parts = parts.split("(",1)
        pfunc = pfunc.strip()
        parts = parts.strip().replace(";","").replace(")","").split(",")
        arg = ""
        if parts:
            param = Param(parts[0])
            arg += param.arg()
        for part in parts[1:]:
            param = Param(part)
            arg += ", %s" % param.arg()
        print """{
    %s(%s);
}
""" % (pfunc,arg)
    else:
        print line
