!
!       endf6py.f90
!
      subroutine mf4_get_leg(awr,awi,awp,q,lct,e1,a1,nl1,e2,a2,nl2,ilaw,e,ne,xmu,nmu,f4)
!
!     Descrption:
!     Get the angular distribution f(E,u) given by Legendre expansion for a set
!     of incident energies e(ne) at different cosines xmu(nmu) supplied by the
!     user. The results are returned in the f4(ie,ju) array.
!
!     Input:
!     awr: relative atomic mass of the target
!     awi: relative nuclear mass of the incident particle
!     awp: relative nuclear mas of the outgoing particle in MF4
!     lct: reference system for angular distributions.(1=LAB, 2=CM)
!     e1: incident energy for the Legendre coefficients a1[l]
!     a1(l): Legendre coefficients at e1 (a0=1, not supplied)
!     nl1: order of the Legendre expansion at e1
!     e2: incident energy for the Legendre coefficients a2[l]
!     a2(l): Legendre coeffients at e2 (a0=1, not supplied)
!     nl1: order of the Legendre expansion at e2
!     ilaw: interpolation law between e1 and e2
!     e(ie): user's incident energy array
!     ne: number of user's incident energies
!     xmu: user's cosine array (in the LAB system)
!     nmu: number os user's cosines
!
!     Output:
!     f4(ie,ju): f(E,u) angular distribution in the lab system at ne incident
!                energies and for nmu cosine values
!
      implicit real*8 (a-h,o-z)
      parameter (nlmax=65)
!     externals
      dimension a1(*),a2(*),e(*),xmu(*),f4(ne,*)
!     internals
      dimension a(nlmax),zmu(nmu)
!
!     Check law to avoid numerical problems for log interpolation in y
      law=mod(ilaw,10)
      if (law.lt.4) then
        zero=0.0d0
      else
        zero=1.0d-38
      endif
!     Cycle for incident energies
      do ie=1,ne
        ei=e(ie)
        a(1)=1.0d0
        if (ei.eq.e1) then
!         case ei equal to e1
          nla=nl1
          do l=1,nl1
            a(l+1)=a1(l)
          enddo
        elseif (ei.eq.e2) then
!         case ei equal to e2
          nla=nl2
          do l=1,nl2
            a(l+1)=a2(l)
          enddo
        else
!         case e1<ei<e2
          nla1=min(nl1,nl2)
          nla=max(nl1,nl2)
          do l=1,nla1
            a(l+1)=yintp(e1,a1(l),e2,a2(l),law,ei)
          enddo
          if (nla.gt.nla1) then
            do l=nla1+1,nla
              if (l.gt.nl1) then
                a(l+1)=yintp(e1,zero,e2,a2(l),law,ei)
              else
                a(l+1)=yintp(e1,a1(l),e2,zero,law,ei)
              endif
            enddo
          endif
        endif
        if (lct.ne.1) then
!         evaluated distribution in the CM system
!         convert input cosines from LAB to CM using two-body kinematic
          r2=awr*(awr+awi-awp)/(awi*awp)*(1.0d0+(awr+awi)/awr*q/ei)
          r=sqrt(r2)
          do ju=1,nmu
            u=xmu(ju)
            u2=u*u
            zmu(ju)=(1.0d0-u2-r2*u2)/(r*(u2-1.0d0-u*sqrt(u2+r2-1.0d0)))
          enddo
        else
!         evaluated distribution in the LAB system
!         just copy the input cosines
          do ju=1,nmu
            zmu(ju)=xmu(ju)
          enddo
        endif
!       calculate the f(E,u) in the reference system of the evaluation
        do ju=1,nmu
          f4(ie,ju)=yleg(zmu(ju),a,nla)
        enddo
!       convert to LAB system if original data given in the CM system
!       f(E,ulab)=f(E,ucm)*J=f(E,ucm)*(ducm/dulab)
        if (lct.ne.1) then
          do ju=1,nmu
            w=zmu(ju)
            xw=1.0d0+2.0d0*r*w+r2
            f4(ie,ju)=f4(ie,ju)*xw*sqrt(xw)/(r2*(r+w))
          enddo
        endif
      enddo
      return
      end
! ------------------------------------------------------------------------------
      real*8 function yintp(x1,y1,x2,y2,i,x)
!
!      Description:
!      interpolate one point using ENDF-6 interpolation laws (1-5)
!
!      Input:
!      (x1,y1) and (x2,y2) are the end points
!      i is the endf-6 interpolation law (1-5)
!
!      Output:
!      (x,yintp) is the interpolated point
!

      implicit real*8 (a-h,o-z)
      parameter (zero=0.0d0)
!
!     *** x1=x2
      if (x2.eq.x1) then
        yintp=y1
!
!     ***y is constant
      elseif (i.eq.1.or.y2.eq.y1.or.x.eq.x1) then
         yintp=y1
!
!     ***y is linear in x
      else if (i.eq.2) then
         yintp=y1+(x-x1)*(y2-y1)/(x2-x1)
!
!     ***y is linear in ln(x)
      else if (i.eq.3) then
         yintp=y1+log(x/x1)*(y2-y1)/log(x2/x1)
!
!     ***ln(y) is linear in x
      else if (i.eq.4) then
         yintp=y1*exp((x-x1)*log(y2/y1)/(x2-x1))
!
!     ***ln(y) is linear in ln(x)
      else if (i.eq.5) then
         if (y1.eq.zero) then
            yintp=y1
         else
            yintp=y1*exp(log(x/x1)*log(y2/y1)/log(x2/x1))
         endif
!
!     ***coulomb penetrability law or other law
      else
        write(*,*) ' Interpolation law: ',i,' not allowed. y set to -1.0E38'
        yint=-1.0d38
      endif
      return
      end
! ------------------------------------------------------------------------------
      real*8 function yleg(x,a,na)
!
!     Description:
!     calculate y(x) given by a legendre expansion of order na
!
!     Input:
!      x: independent variable value
!      a: Legendre coefficients (na+1 coefficients)
!     na: Legendre expansion order
!
!     Output:
!      yleg: function value at x
!
      implicit real*8 (a-h,o-z)
      parameter (nlmax=65)
      dimension a(*),p(nlmax)
      call legndr(x,p,na)
      yleg=0.0d0
      n=na+1
      do l=1,n
        yleg=yleg+(dble(l)-0.5d0)*a(l)*p(l)
      enddo
      return
      end
! ------------------------------------------------------------------------------
      subroutine legndr(x,p,nl)
!
!     Description
!       generate legendre polynomials at x by recursion.
!
!     Input:
!      x: independent variable value
!     nl: Legendre expansion order
!
!     Output:
!      p(l): Legendre polynomials at x
!            p(1)=P0(x), p(2)=P1(x), ... p(nl+1)=Pnl(x)
!
      implicit real*8 (a-h,o-z)
      dimension p(*)
      p(1)=1.0d0
      p(2)=x
      if (nl.gt.1) then
        m1=nl-1
        do i=1,m1
          g=x*p(i+1)
          h=g-p(i)
          p(i+2)=h+g-h/(i+1)
        enddo
      endif
      return
      end
! ------------------------------------------------------------------------------
