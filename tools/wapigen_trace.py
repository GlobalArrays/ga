#!/usr/bin/env python

'''Generate the wapi_trace.c source from the papi.h header.'''

import sys

def get_signatures(header):
    # first, gather all function signatures from papi.h aka argv[1]
    accumulating = False
    signatures = []
    current_signature = ''
    EXTERN = 'extern'
    SEMICOLON = ';'
    for line in open(header):
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
    return signatures

format_from_type = {
        "char":         "%c", 
        "int":          "%d",
        "long":         "%ld",
        "long long":    "%lld",
        "float":        "%f",
        "double":       "%lf",
        "Integer":      "%ld",
        "logical":      "%ld",
        }

class FunctionArgument(object):

    def __init__(self, signature):
        self.pointer = signature.count('*')
        self.array = '[' in signature
        signature = signature.replace('*','').strip()
        signature = signature.replace('[','').strip()
        signature = signature.replace(']','').strip()
        self.type,self.name = signature.split()

    def __str__(self):
        ret = self.type[:]
        ret += ' '
        for p in range(self.pointer):
            ret += '*'
        ret += self.name
        if self.array:
            ret += '[]'
        return ret

class Function(object):
    def __init__(self, signature):
        signature = signature.replace('extern','').strip()
        self.return_type,signature = signature.split(None,1)
        self.return_type = self.return_type.strip()
        signature = signature.strip()
        self.name,signature = signature.split('(',1)
        self.name = self.name.strip()
        signature = signature.replace(')','').strip()
        signature = signature.replace(';','').strip()
        self.args = []
        if signature:
            for arg in signature.split(','):
                self.args.append(FunctionArgument(arg.strip()))

    def get_call(self, name=None):
        sig = ''
        if not name:
            sig += self.name
        else:
            sig += name
        sig += '('
        if self.args:
            for arg in self.args:
                sig += arg.name
                sig += ', '
            sig = sig[:-2] # remove last ', '
        sig += ')'
        return sig

    def get_signature(self, name=None):
        sig = self.return_type[:]
        sig += ' '
        if not name:
            sig += self.name
        else:
            sig += name
        sig += '('
        if self.args:
            for arg in self.args:
                sig += str(arg)
                sig += ', '
            sig = sig[:-2] # remove last ', '
        sig += ')'
        return sig

    def get_tracer_body(self):
        tracer = ''
        if 'void' not in func.return_type:
            tracer += '    %s retval;\n' % self.return_type
        tracer += '    %s;\n' % self.get_tracer_printf(False)
        tracer += '    '
        if 'void' not in func.return_type:
            tracer += 'retval = '
        tracer += '%s;\n' % self.get_call()
        tracer += '    %s;\n' % self.get_tracer_printf(True)
        if 'void' not in func.return_type:
            tracer += '    return retval;\n'
        return tracer

    def get_tracer_printf(self, end=False):
        tracer = 'printf("%lf,'
        if end: tracer += '/'
        tracer += self.name + ','
        if self.args:
            tracer += '('
            for arg in self.args:
                if arg.pointer == 1 and 'char' in arg.type:
                    tracer += '%s;'
                elif arg.pointer or arg.array:
                    tracer += '%p;'
                else:
                    tracer += '%s;' % format_from_type[arg.type]
            tracer = tracer[:-1]
            tracer += ')'
        tracer += '\\n",MPI_Wtime()'
        if self.args:
            for arg in self.args:
                tracer += ',%s' % arg.name
        tracer += ')'
        return tracer

    def __str__(self):
        return self.get_signature()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print 'incorrect number of arguments'
        print 'usage: wapigen_trace.py <papi.h> > <wapi_trace.c>'
        sys.exit(len(sys.argv))

    # print headers and other static stuff
    print '''
#if HAVE_CONFIG_H
#   include "config.h"
#endif

#include <assert.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <mpi.h>

#include "papi.h"
#include "typesf2c.h"

static FILE *fptrace=NULL;
int me, nproc;

#if HAVE_PROGNAME
extern const char * PROGNAME;
#endif

static void trace_finalize() {
    fclose(fptrace);
}

static void trace_initialize() {
    PMPI_Barrier(MPI_COMM_WORLD);
    PMPI_Comm_rank(MPI_COMM_WORLD, &me);
    PMPI_Comm_size(MPI_COMM_WORLD, &nproc);
    /* create files to write trace data */
    char *profile_dir=NULL;
    const char *program_name=NULL;
    char *file_name=NULL;
    struct stat f_stat;

    profile_dir = getenv("PNGA_PROFILE_DIR");
#if HAVE_PROGNAME
    program_name = PROGNAME;
#else
    program_name = "unknown";
#endif
    if (0 == me) {
        if (!profile_dir) {
            pnga_error("You need to set PNGA_PROFILE_DIR env var", 1);
        }
        fprintf(stderr, "PNGA_PROFILE_DIR=%s\\n", profile_dir);
        if (-1 == stat(profile_dir, &f_stat)) {
            perror("stat");
            fprintf(stderr, "Cannot successfully stat to PNGA_PROFILE_DIR.\\n");
            fprintf(stderr, "Check %s profile dir\\n", profile_dir);
            pnga_error("aborting", 1);
        }
    }
    PMPI_Barrier(MPI_COMM_WORLD);
    file_name = (char *)malloc(strlen(profile_dir)
            +1  /* / */
            + strlen(program_name)
            + 1 /* / */
            + 7 /* mpi id */
            + 6 /* .trace */
            + 2 /* NULL termination */);
    assert(file_name);
    sprintf(file_name,"%s/%s/%07d.trace%c",profile_dir,program_name,me,'\\0');
    fptrace = fopen(file_name,"w");
    if(!fptrace) {
        perror("fopen");
        printf("%d: Context summary file creation failed. file_name=%s Exiting\\n", me, file_name);
        exit(0);
    }
    free(file_name);
}

'''

    functions = {}
    # parse signatures into the Function class
    for sig in get_signatures(sys.argv[1]):
        function = Function(sig)
        functions[function.name] = function

    # now process the functions
    for name in sorted(functions):
        func = functions[name]
        if name in ['pnga_initialize','pnga_terminate']:
            continue
        func = functions[name]
        wnga_name = name.replace('pnga_','wnga_')
        print '''
%s
{
%s
}
''' % (func.get_signature(wnga_name), func.get_tracer_body())

    # output the initialize function
    name = 'pnga_initialize'
    wnga_name = name.replace('pnga_','wnga_')
    func = functions[name]
    print '''%s
{
    static int count_pnga_initialize=0;

    ++count_pnga_initialize;
    %s;
    if (1 == count_pnga_initialize) {
        trace_initialize();
    }
}
''' % (func.get_signature(wnga_name), func.get_call())

    # prepare to output the terminate function
    name = 'pnga_terminate'
    wnga_name = name.replace('pnga_','wnga_')
    func = functions[name]
    # output the terminate function
    print '''%s
{
    static int count_pnga_terminate=0;

    ++count_pnga_terminate;
    %s;
    /* don't dump info if terminate more than once */
    if (1 == count_pnga_terminate) {
        trace_finalize();
    }
}
''' % (func.get_signature(wnga_name), func.get_call()) 
