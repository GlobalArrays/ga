c $Id: nga_stubs.f,v 1.2 1999-07-28 00:18:58 d3h325 Exp $
c This file contains stubs for n-dimensional GA operations
c It should be linked with GA <= 2.4 to preserve backward compatibility
c of the GA Fortran header files
c
	logical function nga_create()
        nga_create =.false.
        call ga_error('n-dim GA not supported in this build',0)
        end

        logical function nga_locate() 
        nga_locate=.false.
        call ga_error('n-dim GA not supported in this build',0)
        end

        logical function nga_create_irreg()
        nga_create_irreg =.false.
        call ga_error('n-dim GA not supported in this build',0)
        end

        integer function nga_read_inc() 
        nga_read_inc = -12345
        call ga_error('n-dim GA not supported in this build',0)
        end


        subroutine nga_put()
        call ga_error('n-dim GA not supported in this build',0)
        end

        subroutine nga_get()
        call ga_error('n-dim GA not supported in this build',0)
        end

        subroutine nga_acc()
        call ga_error('n-dim GA not supported in this build',0)
        end
