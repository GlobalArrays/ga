c $Id: peigstubs.f,v 1.1 1999-07-28 00:24:12 d3h325 Exp $
c stubs for subroutines that call the eigensolver library
c prevent unresolved externals when linking but abort program when called
c
	subroutine ga_diag(g_a, g_s, g_v, eval)
	implicit none
        integer g_a, g_s, g_v
        double precision eval(1)
        call ga_error('ga_diag:peigs not interfaced',0)
        end

	subroutine ga_diag_reuse(control,g_a, g_s, g_v, eval)
	implicit none
        integer g_a, g_s, g_v,control
        double precision eval(1)
        call ga_error('ga_diag_reuse:peigs not interfaced',0)
        end

	subroutine ga_diag_std(g_a, g_s, eval)
	implicit none
        integer g_a, g_s
        double precision eval(1)
        call ga_error('ga_diag:peigs not interfaced',0)
        end

