!
!       endf6py.f90
!
! ----------------------------------------------------------------------------------
  subroutine mf4_get_leg(awr,awi,awp,q,lct,e1,a1,nl1,e2,a2,nl2,ilaw,e,ne,xmu,nmu,f4)
!
! Descrption:
! Get the angular distribution f(E,u) given by Legendre expansion for a set
! of incident energies e(ne) at different cosines xmu(nmu) supplied by the
! user. The results are returned in the f4(ie,ju) array.
!
! Input:
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the outgoing particle in MF4
! lct: reference system for angular distributions.(1=LAB, 2=CM)
! q: reaction q value
! e1: incident energy for the Legendre coefficients a1[l]
! a1(l): Legendre coefficients at e1 (a0=1, not supplied)
! nl1: order of the Legendre expansion at e1
! e2: incident energy for the Legendre coefficients a2[l]
! a2(l): Legendre coeffients at e2 (a0=1, not supplied)
! nl1: order of the Legendre expansion at e2
! ilaw: interpolation law between e1 and e2
! e(ie): user's incident energy array
! ne: number of user's incident energies
! xmu: user's cosine array (in the LAB system)
! nmu: number of user's cosines
!
! Output:
! f4(ie,ju): f(E,u) angular distribution in the lab system at ne incident
!            energies and for nmu cosine values
!
  implicit real*8 (a-h,o-z)
  parameter (nlmax=65)
! externals
  dimension a1(*),a2(*),e(*),xmu(*),f4(ne,*)
! internals
  dimension a(nlmax)
! Cycle for incident energies
  do ie=1,ne
    ei=e(ie)
!   Interpolate Legendre coefficients a(l) of order nla at ei, if required
    a(1)=1.0d0
    if (ei.eq.e1) then
!     case ei equal to e1
      nla=nl1
      do l=1,nl1
        a(l+1)=a1(l)
      enddo
    elseif (ei.eq.e2) then
!     case ei equal to e2
      nla=nl2
      do l=1,nl2
        a(l+1)=a2(l)
      enddo
    else
!     case e1<ei<e2
      law=mod(ilaw,10)
      nla1=min(nl1,nl2)
      nla=max(nl1,nl2)
      do l=1,nla1
        a(l+1)=yintp(e1,a1(l),e2,a2(l),law,ei)
      enddo
      if (nla.gt.nla1) then
!       Check law to avoid numerical problems for log interpolation in y
        zero=0.0d0
        do l=nla1+1,nla
          if (l.gt.nl1) then
            a(l+1)=yintp(e1,zero,e2,a2(l),law,ei)
          else
            a(l+1)=yintp(e1,a1(l),e2,zero,law,ei)
          endif
        enddo
      endif
    endif
    do ju=1,nmu
      u=xmu(ju)
!     cosine conversion if required
      w=ulab2cm(lct,awr,awi,awp,q,ei,u)
!     calculate the f(E,w) in the reference system of the evaluation
      f4(ie,ju)=yleg(w,a,nla)
!     multiply by Jacobian if required
      f4(ie,ju)=f4(ie,ju)*fcm2lab(lct,awr,awi,awp,q,ei,w)
    enddo
  enddo
  return
  end
! ------------------------------------------------------------------------------
  subroutine mf4_get_tab(awr,awi,awp,q,lct,e1,u1,f1,np1,nbt1,ibt1,nr1, &
                         e2,u2,f2,np2,nbt2,ibt2,nr2,ilaw,e,ne,xmu,nmu,f4)
!
! Description:
!
! Descrption:
! Get the angular distribution f(E,u) given by tabulated probabilities for
! a set of incident energies e(ne) at different cosines xmu(nmu) supplied
! by the user. The results are returned in the f4(ie,ju) array.
!
! Input:
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the outgoing particle in MF4
! q: reaction q value
! lct: reference system for angular distributions.(1=LAB, 2=CM)
! e1: incident energy for tabulated data set 1
! u1: cosine values at e1
! f1: tabulated probability values at e1
! np1: number of tabulated pair (u1,f1)
! nbt1: interpolation nodes for f1(u1)
! ibt1: interpolation laws for f1(u1)
! nr1: number of interpolation nodes for f1(u1)
! e2: incident energy for tabulated data set 2
! u2: cosine values at e2
! f2: tabulated probability values at e2
! np2: number of tabulated pair (u2,f2)
! nbt2: interpolation nodes for f2(u2)
! ibt2: interpolation laws for f2(u2)
! nr2: number of interpolation nodes for f2(u2)
! ilaw: interpolation law between e1 and e2
! e(ie): user's incident energy array
! ne: number of user's incident energies
! xmu: user's cosine array (in the LAB system)
! nmu: number of user's cosines
!
! Output:
! f4(ie,ju): f(E,u) angular distribution in the lab system at ne incident
!            energies and for nmu cosine values
!
  implicit real*8 (a-h,o-z)
  dimension u1(*),f1(*),nbt1(*),ibt1(*)
  dimension u2(*),f2(*),nbt2(*),ibt2(*)
  dimension e(*),xmu(*),f4(ne,*)
! interpolate in the original distribution
  do ie=1,ne
    ei=e(ie)
    do ju=1,nmu
      u=xmu(ju)
!     cosine conversion if required
      w=ulab2cm(lct,awr,awi,awp,q,ei,u)
      fe1=tab1intp(u1,f1,np1,nbt1,ibt1,nr1,w)
      fe2=tab1intp(u2,f2,np2,nbt2,ibt2,nr2,w)
      if (ei.eq.e1) then
        f4(ie,ju)=fe1
      elseif (ei.eq.e2) then
        f4(ie,ju)=fe2
      else
        law=mod(ilaw,10)
        f4(ie,ju)=yintp(e1,fe1,e2,fe2,law,ei)
      endif
!     multiply by Jacobian if required
      f4(ie,ju)=f4(ie,ju)*fcm2lab(lct,awr,awi,awp,q,ei,w)
    enddo
  enddo
  return
  end
! ------------------------------------------------------------------------------
  real*8 function yintp(x1,y1,x2,y2,i,x)
!
!  Description:
!  interpolate one point using ENDF-6 interpolation laws (1-5)
!
!  Input:
!  (x1,y1) and (x2,y2) are the end points
!  i is the endf-6 interpolation law (1-5)
!
!  Output:
!  (x,yintp) is the interpolated point
!

  implicit real*8 (a-h,o-z)
  parameter (zero=0.0d0, small=1.0d-38, big=1.0d+38)
!
! *** x1=x2 or x=x1
  if (x2.eq.x1.or.x.eq.x1) then
    yintp=y1
!
! *** x=x2
  elseif (x.eq.x2) then
    yintp=y2
!
! ***y is constant
  elseif (i.eq.1.or.y2.eq.y1) then
     yintp=y1
!
! ***y is linear in x
  else if (i.eq.2) then
     yintp=y1+(x-x1)*(y2-y1)/(x2-x1)
!
! ***y is linear in ln(x)
  else if (i.eq.3) then
     if (x1.eq.zero) x1=small
     yintp=y1+log(x/x1)*(y2-y1)/log(x2/x1)
!
! ***ln(y) is linear in x
  else if (i.eq.4) then
     if (y1.eq.zero) y1=small
     yintp=y1*exp((x-x1)*log(y2/y1)/(x2-x1))
!
! ***ln(y) is linear in ln(x)
  else if (i.eq.5) then
     if (x1.eq.zero) x1=small
     if (y1.eq.zero) y1=small
     yintp=y1*exp(log(x/x1)*log(y2/y1)/log(x2/x1))
!
! ***coulomb penetrability law or other law
  else
    write(*,*) ' Interpolation law: ',i,' not coded.'
    yint=-big
  endif
  return
  end
! ------------------------------------------------------------------------------
  real*8 function tab1intp(x,y,np,nbt,ibt,nr,x0)
!
! Description:
! Calculate the function value at x0
! The function is given by an ENDF-6/TAB1 record:
!   [x(i), y(i)]     (i=1 ... np) tabulated points
!   [nbt(j), ibt(j)] (j=1 ... nr) interpolation law table
!
! Input:
! x: array of abscissa points
! y: array of function values y(i)=f(x(i))
! np: number of points
! nbt: array of interpolation nodes
! ibt: array of ENDF-6 interpolation laws
! nr: interpolation ranges
! x0: input value of the abscissa to calculate the function
!
! Output:
! tab1intp=f(x0): function value at x0
!
  implicit real*8 (a-h, o-z)
  dimension nbt(*),ibt(*),x(*),y(*)
  if (x0.lt.x(1).or.x0.gt.x(np)) then
    tab1intp=0.0d0
  else
    i=2
    do while (i.le.np.and.x(i).lt.x0)
      i=i+1
    enddo
    i1=i-1
    x1=x(i1)
    y1=y(i1)
    x2=x(i)
    y2=y(i)
    if (x0.eq.x1) then
      tab1intp=y1
    elseif (x0.eq.x2) then
      tab1intp=y2
    else
      j=1
      do while (nbt(j).lt.i)
        j=j+1
      enddo
      law=ibt(j)
      tab1intp=yintp(x1,y1,x2,y2,law,x0)
    endif
  endif
  return
  end
! ------------------------------------------------------------------------------
  real*8 function ulab2cm(lct,awr,awi,awp,q,e,u)
!
! Description:
! Convert the input cosine values given in the LAB system to the CM system
! if the original distribution is given in the CM system (lct=2).
! If lct is not equal 2, no transformation is applied.
!
! Input:
! lct: original reference system for angular distributions.(1=LAB, 2=CM)
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the outgoing particle
! q: reaction q value
! e: incident energy
! u: input cosine value (u should be given in the LAB system if lct=1 or 2)
!
! Output:
! ulab2cm: cosine value in the reference system of the original data
!
  implicit real*8 (a-h,o-z)
  parameter (r2min=1.0d-76)
  if (lct.eq.2) then
!   distribution is in the CM system
!   convert input cosine from LAB to CM using two-body kinematic
    r2=awr*(awr+awi-awp)/(awi*awp)*(1.0d0+(awr+awi)/awr*q/e)
    if (r2.lt.r2min) r2=r2min
    r=sqrt(r2)
    u2=u*u
    ulab2cm=(1.0d0-u2-r2*u2)/(r*(u2-1.0d0-u*sqrt(u2+r2-1.0d0)))
  else
!   distribution is in the LAB system or no conversion is required
!   no transformation is applied
    ulab2cm=u
  endif
  return
  end
! ------------------------------------------------------------------------------
  real*8 function fcm2lab(lct,awr,awi,awp,q,e,w)
!
! Description:
! Calculate the Jacobian from CM to LAB for lct=2
! If lct is not equal 2, return 1.
!
! Input:
! lct: original reference system for angular distributions.(1=LAB, 2=CM)
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the outgoing particle
! q: reaction q value
! e: incident energy
! w: input cosine, if lct=2 should be given in the CM system
!
! Output:
! fcm2lab: Jacobian of transformation CM to LAB if lct=2, 1 otherwise
!
  implicit real*8 (a-h,o-z)
  parameter (r2min=1.0d-76)
  if (lct.eq.2) then
!   Calculate Jacobian = (ducm/dulab)
    r2=awr*(awr+awi-awp)/(awi*awp)*(1.0d0+(awr+awi)/awr*q/e)
    if (r2.lt.r2min) r2=r2min
    r=sqrt(r2)
    xw=1.0d0+2.0d0*r*w+r2
    fcm2lab=xw*sqrt(xw)/(r2*(r+w))
  else
!   return 1 ==> no transformation Jacobian=1
    fcm2lab=1.0d0
  endif
  return
  end
! ------------------------------------------------------------------------------
  real*8 function yleg(x,a,na)
!
! Description:
! calculate y(x) given by a legendre expansion of order na
!
! Input:
!  x: independent variable value
!  a: Legendre coefficients (na+1 coefficients)
! na: Legendre expansion order
!
! Output:
!  yleg: function value at x
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
! Description
!   generate legendre polynomials at x by recursion.
!
! Input:
!  x: independent variable value
! nl: Legendre expansion order
!
! Output:
!  p(l): Legendre polynomials at x
!        p(1)=P0(x), p(2)=P1(x), ... p(nl+1)=Pnl(x)
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
