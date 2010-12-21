#
# f77funcgen.pl
#
# find all pnga_ functions and output as many F77_FUNC_ macros as possible
# 
if ($#ARGV+1 != 1) {
    die "usage: f77funcgen.pl filename";
}
open FILE, "<$ARGV[0]" or die $!;
while (<FILE>) {
    if (/pnga_/) {
        chomp;
        s/^.*pnga_(.*)\(.*$/\1/;
        $big = uc $_;
        print "#define ga_${_}_  F77_FUNC_(ga_$_, GA_$big)\n";
        print "#define ga_c${_}_ F77_FUNC_(ga_c$_,GA_C$big)\n";
        print "#define ga_d${_}_ F77_FUNC_(ga_d$_,GA_D$big)\n";
        print "#define ga_i${_}_ F77_FUNC_(ga_i$_,GA_I$big)\n";
        print "#define ga_s${_}_ F77_FUNC_(ga_s$_,GA_S$big)\n";
        print "#define ga_z${_}_ F77_FUNC_(ga_z$_,GA_Z$big)\n";
        print "#define nga_${_}_  F77_FUNC_(nga_$_, NGA_$big)\n";
        print "#define nga_c${_}_ F77_FUNC_(nga_c$_,NGA_C$big)\n";
        print "#define nga_d${_}_ F77_FUNC_(nga_d$_,NGA_D$big)\n";
        print "#define nga_i${_}_ F77_FUNC_(nga_i$_,NGA_I$big)\n";
        print "#define nga_s${_}_ F77_FUNC_(nga_s$_,NGA_S$big)\n";
        print "#define nga_z${_}_ F77_FUNC_(nga_z$_,NGA_Z$big)\n";
    }
}
