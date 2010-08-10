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

for line in open(sys.argv[1]): # papi.h
    line = line.strip()
    if line.startswith("extern"):
        extern,type,parts = line.split(None,2)
        extern = extern.strip()
        type = type.strip()
        parts = parts.strip()
        pfunc,parts = parts.split("(",1)
        pfunc = pfunc.strip()
        wfunc = "w%s" % pfunc[1:]
        parts = parts.strip().replace(";","").replace(")","").split(",")
        sig = ""
        arg = ""
        if parts:
            param = Param(parts[0])
            sig += param.sig()
            arg += param.arg()
        for part in parts[1:]:
            param = Param(part)
            sig += ", %s" % param.sig()
            arg += ", %s" % param.arg()
        print """%s %s(%s)
{
    %s(%s);
}
""" % (type,wfunc,sig,pfunc,arg)
