c $Id: sclstubs.f,v 1.1 1999-07-28 00:24:14 d3h325 Exp $
c  $Id: sclstubs.f,v 1.1 1999-07-28 00:24:14 d3h325 Exp $
c  stubs for operations that require scalapack 
c  avoid unresolved externals but abort user code when called  
	subroutine ga_lu_solve_alt(trans,g_a, g_b)
        implicit none
        integer g_a,g_b,trans
        call ga_error('ga_lu_solve:scalapack not interfaced',0)
        end

	subroutine ga_lu_solve(trans,g_a, g_b)
        integer g_a,g_b
        character*1 trans
        call ga_error('ga_lu_solve:scalapack not interfaced',0)
        end

        integer function ga_spd_invert(g_a)
        implicit none
        integer g_a
        call ga_error('ga_spd_invert:scalapack not interfaced',0)
        ga_spd_invert=0
        end
 
        integer function ga_solve(g_a, g_b)
        implicit none
        integer g_a,g_b
        call ga_error('ga_solve:scalapack not interfaced',0)
        ga_solve=0
        end

	integer function ga_llt_solve(g_a, g_b)
        implicit none
        integer g_a,g_b
        call ga_error('ga_llt_solve:scalapack not interfaced',0)
        ga_llt_solve=0
        end
