module ARMCI_Mov_F90

use ARMCI_Types
interface ARMCI_Put_fa
  module procedure ARMCI_Put_1DI4
  module procedure ARMCI_Put_2DI4
  module procedure ARMCI_Put_3DI4
  module procedure ARMCI_Put_4DI4
  module procedure ARMCI_Put_5DI4
  module procedure ARMCI_Put_6DI4
  module procedure ARMCI_Put_7DI4
  module procedure ARMCI_Put_1DI8
  module procedure ARMCI_Put_2DI8
  module procedure ARMCI_Put_3DI8
  module procedure ARMCI_Put_4DI8
  module procedure ARMCI_Put_5DI8
  module procedure ARMCI_Put_6DI8
  module procedure ARMCI_Put_7DI8
  module procedure ARMCI_Put_1DR4
  module procedure ARMCI_Put_2DR4
  module procedure ARMCI_Put_3DR4
  module procedure ARMCI_Put_4DR4
  module procedure ARMCI_Put_5DR4
  module procedure ARMCI_Put_6DR4
  module procedure ARMCI_Put_7DR4
  module procedure ARMCI_Put_1DR8
  module procedure ARMCI_Put_2DR8
  module procedure ARMCI_Put_3DR8
  module procedure ARMCI_Put_4DR8
  module procedure ARMCI_Put_5DR8
  module procedure ARMCI_Put_6DR8
  module procedure ARMCI_Put_7DR8
  module procedure ARMCI_Put_1DC4
  module procedure ARMCI_Put_2DC4
  module procedure ARMCI_Put_3DC4
  module procedure ARMCI_Put_4DC4
  module procedure ARMCI_Put_5DC4
  module procedure ARMCI_Put_6DC4
  module procedure ARMCI_Put_7DC4
  module procedure ARMCI_Put_1DC8
  module procedure ARMCI_Put_2DC8
  module procedure ARMCI_Put_3DC8
  module procedure ARMCI_Put_4DC8
  module procedure ARMCI_Put_5DC8
  module procedure ARMCI_Put_6DC8
  module procedure ARMCI_Put_7DC8
end interface

interface ARMCI_Get_fa
  module procedure ARMCI_Get_1DI4
  module procedure ARMCI_Get_2DI4
  module procedure ARMCI_Get_3DI4
  module procedure ARMCI_Get_4DI4
  module procedure ARMCI_Get_5DI4
  module procedure ARMCI_Get_6DI4
  module procedure ARMCI_Get_7DI4
  module procedure ARMCI_Get_1DI8
  module procedure ARMCI_Get_2DI8
  module procedure ARMCI_Get_3DI8
  module procedure ARMCI_Get_4DI8
  module procedure ARMCI_Get_5DI8
  module procedure ARMCI_Get_6DI8
  module procedure ARMCI_Get_7DI8
  module procedure ARMCI_Get_1DR4
  module procedure ARMCI_Get_2DR4
  module procedure ARMCI_Get_3DR4
  module procedure ARMCI_Get_4DR4
  module procedure ARMCI_Get_5DR4
  module procedure ARMCI_Get_6DR4
  module procedure ARMCI_Get_7DR4
  module procedure ARMCI_Get_1DR8
  module procedure ARMCI_Get_2DR8
  module procedure ARMCI_Get_3DR8
  module procedure ARMCI_Get_4DR8
  module procedure ARMCI_Get_5DR8
  module procedure ARMCI_Get_6DR8
  module procedure ARMCI_Get_7DR8
  module procedure ARMCI_Get_1DC4
  module procedure ARMCI_Get_2DC4
  module procedure ARMCI_Get_3DC4
  module procedure ARMCI_Get_4DC4
  module procedure ARMCI_Get_5DC4
  module procedure ARMCI_Get_6DC4
  module procedure ARMCI_Get_7DC4
  module procedure ARMCI_Get_1DC8
  module procedure ARMCI_Get_2DC8
  module procedure ARMCI_Get_3DC8
  module procedure ARMCI_Get_4DC8
  module procedure ARMCI_Get_5DC8
  module procedure ARMCI_Get_6DC8
  module procedure ARMCI_Get_7DC8
end interface

contains 

subroutine ARMCI_Put_1DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I4), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_1DI4

subroutine ARMCI_Put_2DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I4), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 2
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_2DI4

subroutine ARMCI_Put_3DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I4), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 3
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_3DI4

subroutine ARMCI_Put_4DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I4), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 4
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_4DI4

subroutine ARMCI_Put_5DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I4), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 5
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_5DI4

subroutine ARMCI_Put_6DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I4), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 6
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_6DI4

subroutine ARMCI_Put_7DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_7DI4

subroutine ARMCI_Put_1DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I8), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_1DI8

subroutine ARMCI_Put_2DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I8), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 2
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_2DI8

subroutine ARMCI_Put_3DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I8), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 3
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_3DI8

subroutine ARMCI_Put_4DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I8), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 4
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_4DI8

subroutine ARMCI_Put_5DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I8), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 5
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_5DI8

subroutine ARMCI_Put_6DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I8), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 6
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_6DI8

subroutine ARMCI_Put_7DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  integer(kind=I8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_7DI8

subroutine ARMCI_Put_1DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R4), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_1DR4

subroutine ARMCI_Put_2DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R4), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 2
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_2DR4

subroutine ARMCI_Put_3DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R4), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 3
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_3DR4

subroutine ARMCI_Put_4DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R4), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 4
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_4DR4

subroutine ARMCI_Put_5DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R4), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 5
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_5DR4

subroutine ARMCI_Put_6DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R4), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 6
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_6DR4

subroutine ARMCI_Put_7DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_7DR4

subroutine ARMCI_Put_1DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R8), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_1DR8

subroutine ARMCI_Put_2DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R8), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 2
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_2DR8

subroutine ARMCI_Put_3DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R8), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 3
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_3DR8

subroutine ARMCI_Put_4DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R8), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 4
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_4DR8

subroutine ARMCI_Put_5DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R8), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 5
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_5DR8

subroutine ARMCI_Put_6DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R8), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 6
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_6DR8

subroutine ARMCI_Put_7DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  real(kind=R8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_7DR8

subroutine ARMCI_Put_1DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C4), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_1DC4

subroutine ARMCI_Put_2DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C4), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 2
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_2DC4

subroutine ARMCI_Put_3DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C4), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 3
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_3DC4

subroutine ARMCI_Put_4DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C4), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 4
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_4DC4

subroutine ARMCI_Put_5DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C4), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 5
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_5DC4

subroutine ARMCI_Put_6DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C4), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 6
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_6DC4

subroutine ARMCI_Put_7DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_7DC4

subroutine ARMCI_Put_1DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C8), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_1DC8

subroutine ARMCI_Put_2DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C8), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 2
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_2DC8

subroutine ARMCI_Put_3DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C8), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 3
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_3DC8

subroutine ARMCI_Put_4DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C8), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 4
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_4DC8

subroutine ARMCI_Put_5DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C8), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 5
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_5DC8

subroutine ARMCI_Put_6DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C8), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 6
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_6DC8

subroutine ARMCI_Put_7DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_put_farrays
  end interface
  complex(kind=C8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Put_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Put_7DC8

!Get

subroutine ARMCI_Get_1DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I4), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_1DI4

subroutine ARMCI_Get_2DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I4), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 2
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_2DI4

subroutine ARMCI_Get_3DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I4), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 3
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_3DI4

subroutine ARMCI_Get_4DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I4), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 4
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_4DI4

subroutine ARMCI_Get_5DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I4), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer  :: rank

  rank = 5
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_5DI4

subroutine ARMCI_Get_6DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I4), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer  :: rank

  rank = 6
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_6DI4

subroutine ARMCI_Get_7DI4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_7DI4

subroutine ARMCI_Get_1DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I8), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_1DI8

subroutine ARMCI_Get_2DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I8), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 2
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_2DI8

subroutine ARMCI_Get_3DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I8), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 3
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_3DI8

subroutine ARMCI_Get_4DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I8), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 4
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_4DI8

subroutine ARMCI_Get_5DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I8), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer  :: rank

  rank = 5
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_5DI8

subroutine ARMCI_Get_6DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I8), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer  :: rank

  rank = 6
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_6DI8

subroutine ARMCI_Get_7DI8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      integer(kind=I8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  integer(kind=I8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_7DI8

subroutine ARMCI_Get_1DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R4), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_1DR4

subroutine ARMCI_Get_2DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R4), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 2
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_2DR4

subroutine ARMCI_Get_3DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R4), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 3
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_3DR4

subroutine ARMCI_Get_4DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R4), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 4
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_4DR4

subroutine ARMCI_Get_5DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R4), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer  :: rank

  rank = 5
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_5DR4

subroutine ARMCI_Get_6DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R4), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer  :: rank

  rank = 6
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_6DR4

subroutine ARMCI_Get_7DR4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_7DR4

subroutine ARMCI_Get_1DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R8), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_1DR8

subroutine ARMCI_Get_2DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R8), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 2
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_2DR8

subroutine ARMCI_Get_3DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R8), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 3
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_3DR8

subroutine ARMCI_Get_4DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R8), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 4
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_4DR8

subroutine ARMCI_Get_5DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R8), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer  :: rank

  rank = 5
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_5DR8

subroutine ARMCI_Get_6DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R8), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer  :: rank

  rank = 6
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_6DR8

subroutine ARMCI_Get_7DR8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      real(kind=R8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  real(kind=R8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_7DR8

subroutine ARMCI_Get_1DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C4), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_1DC4

subroutine ARMCI_Get_2DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C4), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 2
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_2DC4

subroutine ARMCI_Get_3DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C4), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 3
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_3DC4

subroutine ARMCI_Get_4DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C4), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 4
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_4DC4

subroutine ARMCI_Get_5DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C4), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer  :: rank

  rank = 5
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_5DC4

subroutine ARMCI_Get_6DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C4), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer  :: rank

  rank = 6
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_6DC4

subroutine ARMCI_Get_7DC4(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C4), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_7DC4

subroutine ARMCI_Get_1DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C8), dimension(:), pointer :: src, dst
  type(ARMCI_slice), intent(in)           :: src_slc, dst_slc
  integer, intent(in)                     :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 1
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_1DC8

subroutine ARMCI_Get_2DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C8), dimension(:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 2
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_2DC8

subroutine ARMCI_Get_3DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C8), dimension(:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 3
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_3DC8

subroutine ARMCI_Get_4DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C8), dimension(:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer :: rank

  rank = 4
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_4DC8

subroutine ARMCI_Get_5DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C8), dimension(:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                    :: rc
  integer  :: rank

  rank = 5
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_5DC8

subroutine ARMCI_Get_6DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C8), dimension(:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer  :: rank

  rank = 6
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_6DC8

subroutine ARMCI_Get_7DC8(src, src_slc, dst, dst_slc, proc, rc)
  use DefineKind
!  implicit none
  interface
    subroutine ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)
      use DefineKind
      use ARMCI_Types
      complex(kind=C8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
      type(ARMCI_slice), intent(in) :: src_slc, dst_slc
      integer, intent(in) :: proc, rank
      integer, intent(out) :: rc
    end subroutine ARMCI_Get_farrays
  end interface
  complex(kind=C8), dimension(:,:,:,:,:,:,:), pointer :: src, dst
  type(ARMCI_slice), intent(in)             :: src_slc, dst_slc
  integer, intent(in)                       :: proc
  integer, intent(out)                      :: rc
  integer :: rank

  rank = 7
  call ARMCI_Get_farrays(src, src_slc, dst, dst_slc, proc, rank, rc)

end subroutine ARMCI_Get_7DC8

end module ARMCI_Mov_F90
