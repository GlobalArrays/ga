****************************************************************************
* GA to/from ScaLAPACK (square block scattered decomposition) interface    * 
*                                                                          *
* common variables:                                                        *
*     nnodes        - number of processors                                 *
*     iam           - my processor number                                  *
*     nprow/ npcol  - number of processor rows/cols in virtual proc. grid  *
*     myrow/mycol   - cordinates of my processor in virtual proc. grid     *
*                                                                          *
c***************************************************************************
c* 04/12/96  GVT  Changed the code to adapt to a new version of ScaLAPACK
c*           Giuseppe Vitillaro peppe@unipg.it
c***************************************************************************
      block data SLblock
      implicit none
c
#include "slface.fh"
      data init /.false./
      end



      subroutine SLinit
      implicit none
      integer  icontxt
c
#include "slface.fh"
      external blacs_pinfo, blacs_get, blacs_gridinit, blacs_gridinfo
c
      if(init)return
c
c**** call ga_sync before to enter in BLACS and after
      call ga_sync()
c     
      call blacs_pinfo(iam, nnodes)
c
      call FindGrid(nnodes, nprow, npcol)
c
      call blacs_get (0, 0, icontxt)
      igctxt = icontxt
c
      icontxt = igctxt
      call blacs_gridinit(icontxt, 'R', nprow, npcol)
      iSLctxt = icontxt
c
      call blacs_gridinfo(iSLctxt, nprow, npcol, myrow, mycol)
c
      init=.true.
c
      call ga_sync()
c    
      end



      subroutine FindGrid(nnodes, nprow, npcol)
c
c***  determine nprow, npcol from nnodes
c***  solution is searched in the neighborhood of the square grid 
c
      implicit none
      integer nnodes, nprow, npcol,i 
c
      npcol = int(sqrt(dble(nnodes)))
      do i = npcol, 1, -1 
         if(mod(nnodes,i).eq.0) goto 1
      enddo
1     continue
      npcol = i
      nprow = nnodes/npcol
      end



      subroutine blockpos(m, nb, idxb, pos)
c
c***  for m row/col find SL block num and its position within that block 
c
      implicit none
#include "slface.fh"
c
      integer m, nb, idxb, pos
      integer T
c
      T = nb * nnodes
      idxb = m/T
      pos = Mod(Mod(m,T),nb)
      end 



      subroutine blockplace(mypcord, npnum, dim, nb, lo,hi) 
c
c***  calculates range of indexes in the SBS array for data
c***  that the given processor holds (called for rows & cols)
c
      implicit none
c
      integer  dim, nb,lo, hi
      integer  np_or_q, npnum, mypcord 
      integer totblocks, minblocksproc, extrablocks, corr,BlocksBeforeMe 
      integer numroc
c
      np_or_q = numroc(dim, nb, mypcord, 0, npnum)
      if(np_or_q .lt. 1)then    ! this processor des not have any data 
         lo =  0
         hi = -1
         return
      endif
c
      totblocks    = (dim +nb-1)/nb
      minblocksproc = totblocks/npnum
      extrablocks  = Mod(totblocks,npnum) 
c
      if(extrablocks .le.  mypcord)then
         BlocksBeforeMe = minblocksproc * mypcord + extrablocks 
         if(extrablocks.gt.0)then       ! check if there are extra blocks
            corr = Mod(dim,nb)
            if(corr.ne.0)then           ! partial block might be among them
               BlocksBeforeMe = BlocksBeforeMe -1
            endif
         endif
      else
         BlocksBeforeMe = minblocksproc * mypcord + mypcord 
      endif
      lo = 1  + BlocksBeforeMe * nb + corr
      hi = lo + np_or_q -1
      end




      subroutine ga_to_SL(g_a, dim1, dim2, nbr, nbc, s_a, lda, np, nq)
c
c***  transforms a GA to SL format
c***  reference: Dongarra et al, 'A look at scalable dense lin. alg. libs'
c
      implicit none
#include "slface.fh"
      integer g_a, nbr, nbc, lda, np, nq, dim1, dim2
      double  precision  s_a(lda,nq)
      integer row, col, tcol, trow, rbase, cbase
      integer pcol, prow

c***  Synchronize at the beginning
      call ga_sync()
c    
      rbase = 1
      cbase = 1
c
      Trow = nbr * nprow
      Tcol = nbc * npcol
c     
      do col  = 1, dim2, nbc
         pcol = mod(col,Tcol)/nbc        ! processor column that holds "col"
         if(mycol.eq.pcol) then
            do row  = 1, dim1, nbr
               prow = mod(row,Trow)/nbr  ! processor row that holds "row"
               if(myrow.eq.prow) then
                  if(rbase.gt.np .or. cbase .gt. nq) then
                     call ga_error(' ga_to_SL: rows/cols error ',0)
                  endif
                  call ga_get(g_a,row,min(row+nbr -1,dim1),col,
     &                    min(col+nbc -1,dim2), s_a(rbase,cbase), lda) 
                  rbase = rbase + nbr
               endif
            enddo
            rbase = 1
            cbase = cbase + nbc
         endif
      enddo

c**** ... and at the end
      call ga_sync()
      end



      subroutine ga_from_SL(g_a, dim1, dim2, nbr, nbc, s_a, lda, np, nq)
c
c***  transforms a matrix from SL to GA format
c***  reference: Dongarra et al, 'A look at scalable dense lin. alg. libs'
c
      implicit none
#include "slface.fh"
      integer g_a, nbr, nbc, lda, np, nq, dim1, dim2
      double  precision s_a(lda,nq)
      integer row, col, tcol, trow, rbase, cbase
      integer pcol, prow
c
c**** Syncronize at the beginning
      call ga_sync()
c   
      rbase = 1
      cbase = 1
c
      Trow = nbr * nprow
      Tcol = nbc * npcol
c
      do col  = 1, dim2, nbc
         pcol = mod(col,Tcol)/nbc        ! processor column that holds "col"
         if(mycol.eq.pcol) then
            do row  = 1, dim1, nbr
               prow = mod(row,Trow)/nbr  ! processor row that holds "row"
               if(myrow.eq.prow) then
                  if(rbase.gt.np .or. cbase .gt. nq)
     &               call ga_error(' ga_from_SL: rows/cols error ',0)
                  call ga_put(g_a,row,min(row+nbr-1,dim1),col,
     &                    min(col+nbc-1,dim2), s_a(rbase,cbase), lda) 
                  rbase = rbase + nbr
               endif
            enddo
            rbase = 1
            cbase = cbase + nbc
         endif
      enddo

c**** ... and at the end
      call ga_sync()
      end
