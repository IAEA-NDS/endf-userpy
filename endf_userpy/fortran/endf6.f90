! ==============================================================================================================================
!
!  endf6py.f90 library for ENDF-6 processing
!
! ==============================================================================================================================
  subroutine mf4_get_leg(awr,awi,awp,q,lct,e1,a1,nl1,e2,a2,nl2,ilaw,e,ne,xmu,nmu,f4)
!
! Descrption:
! Get the angular distribution f(E,u) given by Legendre expansion for a set
! of incident energies e(ne) at different cosines xmu(nmu) supplied by the
! user. The results are given in the f4(ie,ju) array.
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
! externals
  dimension a1(*),a2(*),e(*),xmu(*),f4(ne,*)
! Cycle for incident energies
  do ie=1,ne
    ei=e(ie)
    do ju=1,nmu
      u=xmu(ju)
!     Reference system conversion, if required
      call mf4lab2cm(lct,awr,awi,awp,q,ei,u,w,dinv)
!     calculate the f(E,w) in the reference system of the evaluation
      f=f4leg(e1,a1,nl1,e2,a2,nl2,ilaw,ei,w)
!     multiply by Jacobian
      f4(ie,ju)=f*dinv
    enddo
  enddo
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  subroutine mf4_get_tab(awr,awi,awp,q,lct,e1,u1,f1,np1,nbt1,ibt1,nr1, &
                         e2,u2,f2,np2,nbt2,ibt2,nr2,ilaw,e,ne,xmu,nmu,f4)
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
  do ie=1,ne
    ei=e(ie)
    do ju=1,nmu
      u=xmu(ju)
!     Reference system conversion, if required
      call mf4lab2cm(lct,awr,awi,awp,q,ei,u,w,dinv)
      f=f4tab(e1,u1,f1,np1,nbt1,ibt1,nr1,e2,u2,f2,np2,nbt2,ibt2,nr2,ilaw,ei,w)
!     multiply by Jacobian, if required
      f4(ie,ju)=f*dinv
    enddo
  enddo
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  subroutine mf6_get_law1(eu,neu,epu,nepu,uu,nuu,&
                          awr,awi,awp,za,zai,zap,lct,lang,lep,lei, &
                          e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                          f6dis,f6con)
!
! Description:
! Calculate the discrete f6dis(e,ep,u) and the continuum f6con(e,ep,u)
! contributions to the energy-angle distribution from MF6/LAW1 at a set of NEU
! incident energies eu(ie) for NEPU outgoing energies epu(je) and NUU outgoing
! cosines uu(ju) specified by the user.
!
! Input:
! eu:  user's incident energies. 1D-array [eu(neu)]
! neu: number of user's incident energies
! epu: user's outgoing energies. 1D-array [epu(nepu)]
! nepu: number of user's outgoing energies
! uu:  user's outgoing cosines. 1D-array [uu(nuu)]
! nuu: number of user's outgoing cosines
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the required outgoing particle in MF6
! za:  ZA number of the target (ZA=1000*Z+A)
! zai: ZA number of the incident particle
! zap: ZA number of the outgoing particle
! lct: reference system for energy-angular distribution
! lang: Angular representation flag:
!        lang=1, Legendre coefficients
!        lang=2, Kalbach-Mann systematics
!        lang=11-15, tabulated angular distribution
! lep: interpolation scheme for outgoing energies
! lei: interpolation scheme between incident energies e1 and e2
! e1:  incident energy of the lower panel
! nd1: Number of dicrete energies given at e1
! na1: number of angular parameters at e1
!       lang=1, na1=Legendre expansion order
!       lang=2, na1=1 r is given by the evaluator and a should be calculated
!               na1=2 r and a are given by the evaluator
!       lang=11-15, na1/2 pairs (u,p(u)) are given
!               na1=0, isotropic distribution for all representations
! nep1: number of outgoing energies given at e1
! ep1: outgoing energy values at e1. 1D-array [ep1(npe1)]
! b1:  outgoing energy-angle distribution at e1. 2D-array [b1(nep1,na1)]
! e2:  incident energy of the upper panel
! nd2: Number of dicrete energies given at e2
! na2: number of angular parameters at e2
!       lang=1, na2=Legendre expansion order
!       lang=2, na2=1 r is given by the evaluator and a should be calculated
!               na2=2 r and a are given by the evaluator
!       lang=11-15, na2/2 [u,p(u)] pairs are given
!               na2=0, isotropic distribution for all representations
! nep2: number of outgoing energies given at e2
! ep2: outgoing energy values at e2. 1D-array [ep2(npe2)]
! b2:  Outgoing energy-angle distribution at e2. 2D-array [b2(nep2,na2)]
!
! Output:
! f6dis: discrete contribution to energy-angle distribution at [eu,epu,uu].
! f6con: continuum contribution to energy-angle distribution at [eu,epu,uu].
!
! fdis and f6con are 3D-array with dimension (neu,nepu,nuu)
!
  implicit real*8 (a-h, o-z)
  dimension eu(*),epu(*),uu(*),ep1(*),b1(nep1,*),ep2(*),b2(nep2,*)
  dimension f6dis(neu,nepu,*),f6con(neu,nepu,*)
  do ie=1,neu
    e=eu(ie)
    do je=1,nepu
      ep=epu(je)
      do ju=1,nuu
        u=uu(ju)
        call mf6lab2cm(awr,awi,awp,lct,e,ep,u,tp,w,dinv)
        fdis=f6law1dis(e,tp,w,za,zai,zap,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2)
        fcon=f6law1con(e,tp,w,za,zai,zap,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2)
        f6dis(ie,je,ju)=fdis*dinv
        f6con(ie,je,ju)=fcon*dinv
      enddo
    enddo
  enddo
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  subroutine mf6_get_law2(awr,awi,awp,q,lct,lang,e1,a1,nl1,e2,a2,nl2,ilaw,e,ne,xmu,nmu,f6)
!
! Description:
! Get the angular distribution f(E,u) given by MF6/LAW2 (Discrete 2-body
! reaction) for a set of incident energies e(ne) at different cosines xmu(nmu)
! supplied by the user. The results are returned in the f6(ie,ju) array.
!
! Input:
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the outgoing particle
! q: reaction q value from MF3
! lct: reference system for angular distributions.(1=LAB, 2=CM)
! lang: MF6/LAW2 representation flag:
!       lang=0, Legendre expansion
!       lang=12,Tabulated data with p(u) linear in u (ENDF6/INT=2)
!       lang=14,Tabulated data with log(p(u)) linear in u (ENDF6/INT=4)
! e1: incident energy for the lower panel
! a1: for lang=0, Legendre coefficients at e1
!     for lang>0, the (u,p(u)) pairs for tabulated angular distribution at e1
! nl1: for lang=0, Legendre expansion order
!      for lang>0, Number of tabulated pairs (u,p(u)) at e1
! e2: incident energy for the upper panel
! a2: for lang=0, Legendre coefficients at e2
!     for lang>0, the (u,p(u)) pairs for tabulated angular distribution at e2
! nl2: for lang=0, Legendre expansion order at e2
!      for lang>0, Number of tabulated pairs (u,p(u)) at e2
! ilaw: interpolation law between e1 and e2
! e(ie): user's incident energy array
! ne: number of user's incident energies
! xmu: user's cosine array (in the LAB system)
! nmu: number of user's cosines
!
! Output:
! f6(ie,ju): f(E,u) angular distribution in the lab system at ne incident
!            energies and for nmu cosine values
!
  implicit real*8 (a-h,o-z)
! externals
  dimension a1(*),a2(*),e(*),xmu(*),f6(ne,*)
  do ie=1,ne
    ei=e(ie)
    do ju=1,nmu
      u=xmu(ju)
!     Reference system conversion, if required
      call mf4lab2cm(lct,awr,awi,awp,q,ei,u,w,dinv)
!     calculate the f(E,w) in the reference system of the evaluation
      f=f6law2(lang,e1,a1,nl1,e2,a2,nl2,ilaw,ei,w)
!     multiply by Jacobian
      f6(ie,ju)=f*dinv
    enddo
  enddo
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  subroutine mf6_get_law5(za,awr,zap,awp,spi,lidp,lei,ltp, &
                          e1,nl1,a1,e2,nl2,a2,eni,sni,np,nbt,ibt,nr, &
                          e,ne,xmu,nmu,f65)
  implicit real*8 (a-h, o-z)
  parameter (zero=0.0d0)
! external dimensions
  dimension a1(*),a2(*),eni(*),sni(*),nbt(*),ibt(*)
  dimension e(*),xmu(*),f65(ne,*)
  lct=2     ! data must be given in the CM system for MF6/LAW5
  awi=awp   ! incident charged particle must be equal to outgoing particle
  q=zero    ! reaction Q value should be zero for elastic scattering
  do ie=1,ne
    ei=e(ie)
    do ju=1,nmu
      u=xmu(ju)
!     Reference system conversion, if required
      call mf4lab2cm(lct,awr,awi,awp,q,ei,u,w,dinv)
      f=f6law5(ei,w,za,awr,zap,awp,spi,lidp,lei,ltp, &
               e1,nl1,a1,e2,nl2,a2, &
               eni,sni,np,nbt,ibt,nr)
!     Multiply by Jacobian
      f65(ie,ju)=f*dinv
    enddo
  enddo
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  subroutine mf6_get_law6(awr,awi,awp,q,apsx,npsx,e,ne,ep,nep,xmu,nmu,f6)
!
! Description:
! Get the angular distribution f(E,E',u) given by MF6/LAW6 (N-Body Phase-Space
! Distribution)  for a set of incident energies e(ne) at different cosines
! xmu(nmu) supplied by the user.
! The results are returned in the f6(i,j,k) array.
!
! Input:
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the outgoing particle
! q: reaction q value from MF3
! apsx: total mass in neutron units of the N particles treated by LAW6
! npsx: number of particles distributed according to LAW6 (N)
! e: user's incident energy array
! ne: number of user's incident energies
! ep: user's outgoing energy array
! nep: number of user's outgoing energies
! xmu: user's cosine array (in the LAB system)
! nmu: number of user's cosines
!
! Output:
! f6(i,j,k): f(E,E',u) angular distribution in the lab system at ne incident
!            energies, nep outgoing energies and for nmu cosine values
!
  implicit real*8 (a-h,o-z)
  dimension e(*),ep(*),xmu(*)
  dimension f6(ne,nep,*)
  do i=1,ne
    do j=1,nep
      do k=1,nmu
        f6(i,j,k)=f6law6(awr,awi,awp,q,apsx,npsx,e(i),ep(j),xmu(k))
      enddo
    enddo
  enddo
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  subroutine mf6_get_law7(e,ne,ep,nep,xmu,nmu,lei, &
                         e1,lmu1,u11,ep11,f11,np11,nbt11,ibt11,nr11, &
                                 u12,ep12,f12,np12,nbt12,ibt12,nr12, &
                         e2,lmu2,u21,ep21,f21,np21,nbt21,ibt21,nr21, &
                                 u22,ep22,f22,np22,nbt22,ibt22,nr22,f6)
!
! Description:
! Calculate the angle-energy distribution given by MF6/LAW7 at ne incident
! energies between e1 and e2, nmu outgoing cosines between min(u11,u21) and
! max(u12,u22) at nep outgoing energies.
!
! Input:
!  e: User's incident energy array
! ne: number of incident energies between e1 and e2
! ep: User's outgoing energy array
!nep: number of outgoing energies
!xmu: User's outgoing cosine array
!nmu: number of outgoing cosines
!lei: interpolation law between e1 and e2
! e1: incident energy of the lower energy panel
!lmu1: interpolation law between u11 and u12 at e1
!u11: outgoing cosine value of the lower cosine panel at e1
!ep11: outgoing energies at 2D-panel (u11,e1)
!f11:  outgoing energy distribution at 2D-panel (u11,e1)
!np11: number of outgoing energies at 2D-panel (u11,e1)
!nbt11: interpolation ranges for f11
!ibt11: interpolation law for f11
!nr11:  number of interpolation ranges for f11
!u12: outgoing cosine value of the upper cosine panel at e1
!ep12: outgoing energies at 2D-panel (u12,e1)
!f12:  outgoing energy distribution at 2D-panel (u12,e1)
!np12: number of outgoing energies at 2D-panel (u21,e1)
!nbt12: interpolation ranges for f12
!ibt12: interpolation law for f12
!nr12:  number of interpolation ranges for f12
! e2: incident energy of the upper energy panel
!lmu2: interpolation law between u21 and u22 at e2
!u21: outgoing cosine value of the lower cosine panel at e2
!ep21: outgoing energies at 2D-panel (u12,e2)
!f21:  outgoing energy distribution at 2D-panel (u12,e2)
!np21: number of outgoing energies at 2D-panel (u12,e2)
!nbt21: interpolation ranges for f21
!ibt21: interpolation law for f21
!nr21:  number of interpolation ranges for f21
!u22: outgoing cosine value of the upper cosine panel at e2
!ep22: outgoing energies at 2D-panel (u12,e2)
!f22:  outgoing energy distribution at 2D-panel (u22,e2)
!np22: number of outgoing energies at 2D-panel (u22,e2)
!nbt22: interpolation ranges for f22
!ibt22: interpolation law for f22
!nr22:  number of interpolation ranges for f22
!
!Output:
! f6(i,j,k): angle-energy distribution given by MF6/LAW7 at ne incident energy
!            points, nep outgoing energy points and nmu outgoing cosines
!
  implicit real*8 (a-h,o-z)
  dimension e(*),ep(*),xmu(*),f6(ne,nep,*)
  dimension ep11(*),f11(*),nbt11(*),ibt11(*),ep12(*),f12(*),nbt12(*),ibt12(*)
  dimension ep21(*),f21(*),nbt21(*),ibt21(*),ep22(*),f22(*),nbt22(*),ibt22(*)
  do i=1,ne
    do k=1,nmu
      do j=1,nep
        f6(i,j,k)=f6law7(e(i),ep(j),xmu(k),lei, &
                         e1,lmu1,u11,ep11,f11,np11,nbt11,ibt11,nr11, &
                                 u12,ep12,f12,np12,nbt12,ibt12,nr12, &
                         e2,lmu2,u21,ep21,f21,np21,nbt21,ibt21,nr21, &
                                 u22,ep22,f22,np22,nbt22,ibt22,nr22)
      enddo
    enddo
  enddo
  return
  end
! ==============================================================================================================================
!
! representation based procedures for MF4 and MF6
!
! ==============================================================================================================================
  real*8 function f4leg(e1,a1,nl1,e2,a2,nl2,ilaw,e,u)
!
! Descrption:
! Calculate the angular distribution f(E,u) given by Legendre expansion in MF4
! at (e,u)
!
! Input:
! e1: incident energy for the Legendre coefficients a1[l]
! a1(l): Legendre coefficients at e1 (a0=1, not supplied)
! nl1: order of the Legendre expansion at e1
! e2: incident energy for the Legendre coefficients a2[l]
! a2(l): Legendre coeffients at e2 (a0=1, not supplied)
! nl1: order of the Legendre expansion at e2
! ilaw: interpolation law between e1 and e2
! e: incident energy
! u: outgoing cosine value
!
! Output:
! f4leg: f(E,u) angular distribution at (e,u)
!
  implicit real*8 (a-h,o-z)
! external arrays
  dimension a1(*),a2(*)
! internal arrays
  allocatable b1(:),b2(:),a(:)
  if (e.lt.e1.or.e.gt.e2) then
    f4leg=0.0d0
  else
!   Prepare Legendre coefficient arrays
!   Add the zero order coefficient, which is equal 1 due to normalization
    nb1=nl1+1
    nb2=nl2+1
    allocate(b1(nb1),b2(nb2))
    do l=1,nl1
      b1(l+1)=a1(l)
    enddo
    b1(1)=1.0d0
    do l=1,nl2
      b2(l+1)=a2(l)
    enddo
    b2(1)=1.0d0
    if (e.eq.e1) then
      f4leg=yleg(u,b1,nl1)
    elseif (e.eq.e2) then
      f4leg=yleg(u,b2,nl2)
    else
!     Legendre coefficient interpolation
      na=max(nb1,nb2)
      allocate(a(na))
      call list_intp(e1,b1,nb1,e2,b2,nb2,ilaw,e,a,na)
      nla=na-1 ! Legendre expansion order at the intermediate energy e
!     calculate f(E,u)
      f4leg=yleg(u,a,nla)
      deallocate(a)
    endif
  endif
  deallocate(b1,b2)
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function f4tab(e1,u1,f1,np1,nbt1,ibt1,nr1, &
                        e2,u2,f2,np2,nbt2,ibt2,nr2,ilaw,e,u)
!
! Description:
!
! Descrption:
! Get the angular distribution f(E,u) given by tabulated probabilities in MF4
! at (e,u)
!
! Input:
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
! e: incident energy
! u: outgoing cosine value
!
! Output:
! f4tab: f(E,u) angular distribution at (e,u)
!
  implicit real*8 (a-h,o-z)
  dimension u1(*),f1(*),nbt1(*),ibt1(*)
  dimension u2(*),f2(*),nbt2(*),ibt2(*)
  if (e.lt.e1.or.e.gt.e2) then
    f4tab=0.0d0
  elseif (e.eq.e1) then
    f4tab=tab1intp(u1,f1,np1,nbt1,ibt1,nr1,u)
  elseif (e.eq.e2) then
    f4tab=tab1intp(u2,f2,np2,nbt2,ibt2,nr2,u)
  else
    law=mod(ilaw,10)
    y1=tab1intp(u1,f1,np1,nbt1,ibt1,nr1,u)
    y2=tab1intp(u2,f2,np2,nbt2,ibt2,nr2,u)
    f4tab=yintp(e1,y1,e2,y2,law,e)
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function f6law1dis(e,tp,w,za,zai,zap,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2)
! e:   incident energy
! tp:  outgoing particle energy
! w:   cosine value
! za:  ZA number of the target (ZA=1000*Z+A)
! zai: ZA number of the incident particle
! zap: ZA number of the outgoing particle
! lang: Angular representation flag:
!        lang=1, Legendre coefficients
!        lang=2, Kalbach-Mann systematics
!        lang=11-15, tabulated angular distribution
! lep: interpolation scheme for outgoing energies
! lei: interpolation scheme between incident energies e1 and e2
! e1:  incident energy of the lower panel
! nd1: Number of dicrete energies given at e1
! na1: number of angular parameters at e1
!       lang=1, na1=Legendre expansion order
!       lang=2, na1=1 r is given by the evaluator and a should be calculated
!               na1=2 r and a are given by the evaluator
!       lang=11-15, na1/2 pairs (u,p(u)) are tabulated
!               na1=0, isotropic distribution for all representations
! nep1: number of outgoing energies given at e1
! ep1: outgoing energy values at e1. 1D-array [ep1(npe1)]
! b1:  outgoing energy-angle distribution at e1. 2D-array [b1(nep1,na1)]
! e2:  incident energy of the upper panel
! nd2: Number of dicrete energies given at e2
! na2: number of angular parameters at e2
!       lang=1, na2=Legendre expansion order
!       lang=2, na2=1 r is given by the evaluator and a should be calculated
!               na2=2 r and a are given by the evaluator
!       lang=11-15, na2/2 [u,p(u)] pairs are given
!               na2=0, isotropic distribution for all representations
! nep2: number of outgoing energies given at e2
! ep2: outgoing energy values at e2. 1D-array [ep2(npe2)]
! b2:  Outgoing energy-angle distribution at e2. 2D-array [b2(nep2,na2)]
!
! Output:
!  f6law1dis: Discrete contribution to angle-energy distribution at e,tp,w
!
  implicit real*8 (a-h, o-z)
  dimension ep1(*),b1(nep1,*),ep2(*),b2(nep2,*)
  if (e.lt.e1.or.e.gt.e2.or.(nd1.le.0.and.nd2.le.0)) then
!
!   e is out of range nor discrete data
!
    f6law1dis=0.0d0
  else
!
!   discrete data processing
!
    law=mod(lei,10)
    if (nd1.le.0) then
      f1=0.0d0
    else
      f1=f6law1_dis(e1,tp,w,za,zai,zap,lang,nd1,na1,nep1,ep1,b1)
    endif
    if (nd2.le.0) then
      f2=0.0d0
    else
      f2=f6law1_dis(e2,tp,w,za,zai,zap,lang,nd2,na2,nep2,ep2,b2)
    endif
    f6law1dis=yintp(e1,f1,e2,f2,law,e)
  endif
  return
  end
!-------------------------------------------------------------------------------------------------------------------------------
  real*8 function f6law1_dis(e,tp,w,za,zai,zap,lang,nd,na,nep,ep,b)
!
! Description:
! Calculate the discrete contribution to the energy-angle distribution
! at (e,tp,w) represented by MF6/LAW1
!
! Input:
! e:   incident energy
! tp:  outgoing particle energy
! w:   cosine value
! za:  ZA number of the target (ZA=1000*Z+A)
! zai: ZA number of the incident particle
! zap: ZA number of the outgoing particle
! lang: Angular representation flag:
!        lang=1, Legendre coefficients
!        lang=2, Kalbach-Mann systematics
!        lang=11-15, tabulated angular distribution
! nd:  number of outgoing energies for the discrete part of the distribution
! na:  number of angular parameters at e
!       lang=1, na=Legendre expansion order
!       lang=2, na=1 r is given by the evaluator and a should be calculated
!               na=2 r and a are given by the evaluator
!       lang=11-15, na/2 pairs (u,p(u)) are tabulated
!               na=0, isotropic distribution for all representations
!       the total number of angular parameters is nt=na+1
! nep: total number of outgoing energies given at e
!      the number of continumm outgoing energies is nepc=nep-nd
! ep: outgoing energy values at e. 1D-array [ep(nep)]
! b: outgoing energy-angle distribution at e. 2D-array [b(nep,na)]
!
! Output:
!  f6law1_dis: discrete contribution to energy-angle distribution at (e,tp,w)
!
  implicit real*8 (a-h,o-z)
  dimension ep(*),b(nep,*)
  allocatable a(:),y(:)
  allocatable nbt(:),ibt(:)
  i=imatch(tp,ep,nd)
  if (i.gt.0) then
    if (lang.eq.1.or.lang.eq.2) then
      nt=na+1
      allocate(a(nt))
      do j=1,nt
        a(j)=b(i,j)
      enddo
      if (lang.eq.1) then
        f6law1_dis=yleg(w,a,na)
      else
        f6law1_dis=ykalbach(zai,zap,za,e,tp,w,a,na)
      endif
      deallocate(a)
    elseif (lang.ge.11.and.lang.le.15) then
      f0=b(i,1)
      if (na.gt.0) then
        nmu=na/2
        nr=1
        allocate(a(nmu),y(nmu))
        allocate(nbt(nr),ibt(nr))
        k=1
        do j=1,nmu
          k=k+1
          a(j)=b(i,k)
          k=k+1
          y(j)=2.0d0*f0*b(i,k)
        enddo
        lmu=lang-10
        nbt(1)=nmu
        ibt(1)=lmu
        f6law1_dis=tab1intp(a,y,nmu,nbt,ibt,nr,w)
        deallocate(a,y)
        deallocate(nbt,ibt)
      else
        f6law1_dis=0.5d0*f0
      endif
    endif
  else
    f6law1_dis=0.0d0
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function f6law1con(e,tp,w,za,zai,zap,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2)
! e:   incident energy
! tp:  outgoing particle energy
! w:   cosine value
! za:  ZA number of the target (ZA=1000*Z+A)
! zai: ZA number of the incident particle
! zap: ZA number of the outgoing particle
! lang: Angular representation flag:
!        lang=1, Legendre coefficients
!        lang=2, Kalbach-Mann systematics
!        lang=11-15, tabulated angular distribution
! lep: interpolation scheme for outgoing energies
! lei: interpolation scheme between incident energies e1 and e2
! e1:  incident energy of the lower panel
! nd1: Number of dicrete energies given at e1
! na1: number of angular parameters at e1
!       lang=1, na1=Legendre expansion order
!       lang=2, na1=1 r is given by the evaluator and a should be calculated
!               na1=2 r and a are given by the evaluator
!       lang=11-15, na1/2 pairs (u,p(u)) are tabulated
!               na1=0, isotropic distribution for all representations
! nep1: number of outgoing energies given at e1
! ep1: outgoing energy values at e1. 1D-array [ep1(npe1)]
! b1:  outgoing energy-angle distribution at e1. 2D-array [b1(nep1,na1)]
! e2:  incident energy of the upper panel
! nd2: Number of dicrete energies given at e2
! na2: number of angular parameters at e2
!       lang=1, na2=Legendre expansion order
!       lang=2, na2=1 r is given by the evaluator and a should be calculated
!               na2=2 r and a are given by the evaluator
!       lang=11-15, na2/2 [u,p(u)] pairs are given
!               na2=0, isotropic distribution for all representations
! nep2: number of outgoing energies given at e2
! ep2: outgoing energy values at e2. 1D-array [ep2(npe2)]
! b2:  Outgoing energy-angle distribution at e2. 2D-array [b2(nep2,na2)]
!
! Output:
!  f6law1con: Continuum contribution to angle-energy distribution at e,tp,w
!
  implicit real*8 (a-h, o-z)
  dimension ep1(*),b1(nep1,*),ep2(*),b2(nep2,*)
  if (e.lt.e1.or.e.gt.e2.or.(nep1.le.nd1.and.nep2.le.nd2)) then
!
!   e is out of range nor continuum data
!
    f6law1con=0.0d0
  else
!
!   continuum part processing
!
    law=mod(lei,10)
    if (nep1.le.nd1) then
      f1=0.0d0
      f2=f6law1_con(e2,tp,w,za,zai,zap,lang,lep,nd2,na2,nep2,ep2,b2)
    elseif (nep2.le.nd2) then
      f1=f6law1_con(e1,tp,w,za,zai,zap,lang,lep,nd1,na1,nep1,ep1,b1)
      f2=0.0d0
    else
      x1low=ep1(nd1+1)
      x1high=ep1(nep1)
      x1range=x1high-x1low
      x2low=ep2(nd2+1)
      x2high=ep2(nep2)
      x2range=x2high-x2low
      yslope=(e-e1)/(e2-e1)
      xlow=x1low+yslope*(x2low-x1low)
      xhigh=x1high+yslope*(x2high-x1high)
      xrange=xhigh-xlow
      xslope=(tp-xlow)/xrange
      x=x1low+xslope*x1range
      f1=f6law1_con(e1,x,w,za,zai,zap,lang,lep,nd1,na1,nep1,ep1,b1)
      f1=f1*x1range/xrange
      x=x2low+xslope*x2range
      f2=f6law1_con(e2,x,w,za,zai,zap,lang,lep,nd2,na2,nep2,ep2,b2)
      f2=f2*x2range/xrange
    endif
    f6law1con=yintp(e1,f1,e2,f2,law,e)
  endif
  return
  end
!-------------------------------------------------------------------------------------------------------------------------------
  real*8 function f6law1_con(e,tp,w,za,zai,zap,lang,lep,nd,na,nep,ep,b)
!
! Description:
! Calculate the continuum contribution to the energy-angle distribution
! at (e,tp,w) represented by MF6/LAW1
!
! Input:
! e:   incident energy
! tp:  outgoing particle energy
! w:   cosine value
! za:  ZA number of the target (ZA=1000*Z+A)
! zai: ZA number of the incident particle
! zap: ZA number of the outgoing particle
! lang: Angular representation flag:
!        lang=1, Legendre coefficients
!        lang=2, Kalbach-Mann systematics
!        lang=11-15, tabulated angular distribution
! lep: interpolation scheme for outgoing energies
! nd:  number of outgoing energies for the discrete part of the distribution
! na:  number of angular parameters at e
!       lang=1, na=Legendre expansion order
!       lang=2, na=1 r is given by the evaluator and a should be calculated
!               na=2 r and a are given by the evaluator
!       lang=11-15, na/2 pairs (u,p(u)) are tabulated
!               na=0, isotropic distribution for all representations
!       the total number of angular parameters is nt=na+1
! nep: total number of outgoing energies given at e
!      the number of continumm outgoing energies is nepc=nep-nd
! ep: outgoing energy values at e. 1D-array [ep(nep)]
! b: outgoing energy-angle distribution at e. 2D-array [b(nep,na)]
!
! Output:
!  f6law1_con: Continuum contribution to energy-angle distribution at (e,tp,w)
!
  implicit real*8 (a-h, o-z)
  dimension ep(*),b(nep,*)
  allocatable a1(:),a2(:),a(:),y1(:),y2(:)
  allocatable nbt1(:),ibt1(:),nbt2(:),ibt2(:)
  iep0=nd+1
  i2=ihigh(tp,ep,iep0,nep)
  i1=i2-1
  if (i1.gt.0) then
    if (lang.eq.1.or.lang.eq.2) then
      nt=na+1
      nt1=nt
      nt2=nt
      allocate(a1(nt1),a2(nt2),a(nt))
      do l=1,nt
         a1(l)=b(i1,l)
         a2(l)=b(i2,l)
      enddo
      call list_intp(ep(i1),a1,nt1,ep(i2),a2,nt2,lep,tp,a,nt)
      if (lang.eq.1) then
         f6law1_con=yleg(w,a,na)
      else
        f6law1_con=ykalbach(zai,zap,za,e,tp,w,a,na)
      endif
      deallocate (a1,a2,a)
    elseif (lang.ge.11.and.lang.le.15) then
      f01=b(i1,1)
      f02=b(i2,1)
      if (na.gt.0) then
        nmu1=na/2
        nmu2=nmu1
        nr1=1
        nr2=1
        allocate(a1(nmu1),y1(nmu1),a2(nmu2),y2(nmu2))
        allocate(nbt1(nr1),ibt1(nr1),nbt2(nr2),ibt2(nr2))
        l=1
        do j=1,nmu1
          l=l+1
          a1(j)=b(i1,l)
          a2(j)=b(i2,l)
          l=l+1
          y1(j)=2.0d0*f01*b(i1,l)
          y2(j)=2.0d0*f02*b(i2,l)
        enddo
        lmu=lang-10
        nbt1(1)=nmu1
        ibt1(1)=lmu
        nbt2(1)=nmu2
        ibt2(1)=lmu
        f6law1_con=unit_base_intp(ep(i1),a1,y1,nmu1,nbt1,ibt1,nr1, &
                                  ep(i2),a2,y2,nmu2,nbt2,ibt2,nr2,lep,tp,w)

        deallocate(a1,y1,a2,y2)
        deallocate(nbt1,ibt1,nbt2,ibt2)
      else
        f0=yintp(ep(i1),f01,ep(i2),f02,lep,tp)
        f6law1_con=0.5d0*f0
      endif
    endif
  else
    f6law1_con=0.0d0
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function f6law2(lang,e1,a1,nl1,e2,a2,nl2,ilaw,e,u)
!
! Description:
! Calculate the value of f(E,u) given by MF6/LAW2 at (e,u)
!
! Input:
! lang: MF6/LAW2 representation flag:
!       lang=0, Legendre expansion
!       lang=12,Tabulated data with p(u) linear in u (ENDF6/INT=2)
!       lang=14,Tabulated data with log(p(u)) linear in u (ENDF6/INT=4)
! e1: incident energy for the lower panel
! a1: for lang=0, Legendre coefficients at e1
!     for lang>0, the (u,p(u)) pairs for tabulated angular distribution at e1
! nl1: for lang=0, Legendre expansion order
!      for lang>0, Number of tabulated pairs (u,p(u)) at e1
! e2: incident energy for the upper panel
! a2: for lang=0, Legendre coefficients at e2
!     for lang>0, the (u,p(u)) pairs for tabulated angular distribution at e2
! nl2: for lang=0, Legendre expansion order at e2
!      for lang>0, Number of tabulated pairs (u,p(u)) at e2
! ilaw: interpolation law between e1 and e2
! e: incident energy
! u: outgoing cosine
!
! Output:
! f6law2: f(E,u) angular distribution at (e,u)
!
  implicit real*8 (a-h,o-z)
! externals
  dimension a1(*),a2(*)
  allocatable u1(:),u2(:),f1(:),f2(:)
  allocatable nbt1(:),ibt1(:),nbt2(:),ibt2(:)
  if (lang.eq.0) then
    f6law2=f4leg(e1,a1,nl1,e2,a2,nl2,ilaw,e,u)
  elseif (lang.eq.12.or.lang.eq.14) then
    nr1=1
    nr2=1
    allocate (u1(nl1),f1(nl1),u2(nl2),f2(nl2))
    allocate (nbt1(nr1),ibt1(nr1),nbt2(nr2),ibt2(nr2))
    j=0
    do l=1,nl1
      j=j+1
      u1(l)=a1(j)
      j=j+1
      f1(l)=a1(j)
    enddo
    j=0
    do l=1,nl2
      j=j+1
      u2(l)=a2(j)
      j=j+1
      f2(l)=a2(j)
    enddo
    lmu=lang-10
    nbt1(1)=nl1
    ibt1(1)=lmu
    nbt2(1)=nl2
    ibt2(1)=lmu
    f6law2=f4tab(e1,u1,f1,nl1,nbt1,ibt1,nr1,e2,u2,f2,nl2,nbt2,ibt2,nr2,ilaw,e,u)
    deallocate(u1,f1,u2,f2)
    deallocate(nbt1,ibt1,nbt2,ibt2)
  else
    write(*,*)' Fatal error: LAW=2/LANG=',lang,' not allowed'
    stop
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function f6law5(e,u,za,awr,zap,awp,spi,lidp,lei,ltp, &
                         e1,nl1,a1,e2,nl2,a2, &
                         eni,sni,np,nbt,ibt,nr)
!
! Charged-particle Elastic Scattering (MF6/LAW5)
! Reaction MT2: z+Y=z+Y q=0
!
!  Description:
!  Calculate the charged-particles scattering cross section
!  in barn/(cosine_units)
!
! Input:
! e: energy of incident charged particle in the LAB system [eV]
! u: cosine of the scattering angle in the CM system
! za: ZA number of the target
! awr: relative atomic mass of the target
! zap: ZA number of the charged-particle
! awp: relative nuclear mass of the charged-particle
! spi: spin of the charged particle (spi=0, 1/2, 1, ...)
! lidp: flag for identical particles (lidp=1 for identical particles)
! lei: interpolation law between incident energies e1 and e2
! ltp: flag for elastic scattering representation
!      ltp=1 nuclear amplitude expansion
!      ltp=2 residual cross section expansion
!      ltp=12 nuclear + interference representation with pni linear in u
!      ltp=14 nuclear + interference representation with log(pni) linear in u
! e1: incident energy of the lower data panel
! nl1: angular parameter at e1:
!       nl1= higest Legendre order of the nuclear partial waves for LTP=1 or 2
!       nl1= number for tabulated cosines for LTP=12 or 14
! a1: List of parameters acccording to the representation at e1
!       coeficients b(i), [ar(i),ai(i)] for LTP=1
!       coeficients c(i) for LTP=2
!       pairs [u(i),pni(i)] for LTP=12 or 14
! e2: incident energy of the upper data panel
! nl2: angular parameter at e2:
!       nl2= higest Legendre order of the nuclear partial waves for LTP=1 or 2
!       nl2= number for tabulated cosines for LTP=12 or 14
! a2: List of parameters acccording to the representation at e2
!       coeficients b(i), [ar(i),ai(i)] for LTP=1
!       coeficients c(i) for LTP=2
!       pairs [u(i),pni(i)] for LTP=12 or 14
! eni: incident energy grid for MF3/MT2 data
! sni: MF3/MT2 data according to the representation
!       equal 1 for LTP=1 or LTP=2
!       equal to nuclear+interference cross-section(sni) for LTP=12 or 14
! np: number of incident energies
!nbt: interpolation ranges for sni
!ibt: interpolation law for echa range
!nr:  number of interpolation ranges
!
! Output:
! f6law5: elastic cross section in unit of barn/(cosine_unit) at (e,u)
!
  implicit real*8 (a-h, o-z)
  parameter (one=1.0d0)
  parameter (two=2.0d0)
  parameter (pi=3.141592653589793d0)
  parameter (umax=9.999999999999999d-1)
! external dimension
  dimension a1(*),a2(*),eni(*),sni(*),nbt(*),ibt(*)
  allocatable b1(:),b2(:),b(:),c1(:),c2(:),c(:)
  allocatable nbt1(:),ibt1(:),nbt2(:),ibt2(:)
  if (e.lt.e1.or.e.gt.e2) then ! out of range
    sige=0.0d0
  else
    if (u.ge.one) then ! check for singularities
      u=umax
    elseif (u.le.-one) then
      if (lidp.eq.1) then
        u=-umax
      else
        u=-one
      endif
    endif
    if (ltp.eq.1) then ! LTP=1: nuclear amplitude expansion
      if (e.eq.e1) then
        if (lidp.eq.1) then
          n1=nl1+1
          nn1=n1+n1
        else
          n1=nl1+nl1+1
          nn1=n1+1
        endif
        allocate (b1(n1),c1(nn1))
        do i=1,n1
          b1(i)=a1(i)
        enddo
        do i=1,nn1
          c1(i)=a1(n1+i)
        enddo
        sige=sctnae(e1,u,za,awr,zap,awp,spi,lidp,c1,b1,nl1)
        deallocate(b1,c1)
      elseif (e.eq.e2) then
        if (lidp.eq.1) then
          n2=nl2+1
          nn2=n2+n2
        else
          n2=nl2+nl2+1
          nn2=n2+1
        endif
        allocate (b2(n2),c2(nn2))
        do i=1,n2
          b2(i)=a2(i)
        enddo
        do i=1,nn2
          c2(i)=a2(n2+i)
        enddo
        sige=sctnae(e2,u,za,awr,zap,awp,spi,lidp,c2,b2,nl2)
        deallocate(b2,c2)
      else
        ilaw=mod(lei,10)
        if (lidp.eq.1) then
          n1=nl1+1
          nn1=n1+n1
          n2=nl2+1
          nn2=n2+n2
        else
          n1=nl1+nl1+1
          nn1=n1+1
          n2=nl2+nl2+1
          nn2=n2+1
        endif
        allocate (b1(n1),b2(n2))
        do i=1,n1
          b1(i)=a1(i)
        enddo
        do i=1,n2
          b2(i)=a2(i)
        enddo
        nb=max(n1,n2)
        allocate(b(nb))
        call list_intp(e1,b1,n1,e2,b2,n2,ilaw,e,b,nb)
        deallocate(b1,b2)
        allocate (c1(nn1),c2(nn2))
        do i=1,nn1
          c1(i)=a1(n1+i)
        enddo
        do i=1,nn2
          c2(i)=a2(n2+i)
        enddo
        nc=max(nn1,nn2)
        allocate (c(nc))
        call list_intp(e1,c1,nn1,e2,c2,nn2,ilaw,e,c,nc)
        deallocate(c1,c2)
        nl=max(nl1,nl2)
        sige=sctnae(e,u,za,awr,zap,awp,spi,lidp,c,b,nl)
        deallocate(b,c)
      endif
    elseif (ltp.eq.2) then ! LTP=2: residual cross section expansion
      if (e.eq.e1) then
        sige=sctrxe(e1,u,za,awr,zap,awp,spi,lidp,a1,nl1)
      elseif (e.eq.e2) then
        sige=sctrxe(e1,u,za,awr,zap,awp,spi,lidp,a2,nl2)
      else
        ilaw=mod(lei,10)
        nl=max(nl1,nl2)
        n1=nl1+1
        n2=nl2+1
        nc=nl+1
        allocate(c(nc))
        call list_intp(e1,a1,n1,e2,a2,n2,ilaw,e,c,nc)
        sige=sctrxe(e,u,za,awr,zap,awp,spi,lidp,c,nl)
        deallocate(c)
      endif
    else ! LTP>2: nuclear + interference representation
      if (lidp.eq.1) then
        uu=abs(u)
      else
        uu=u
      endif
      if(e.eq.e1) then
        allocate(b1(nl1),c1(nl1),nbt1(1),ibt1(1))
        do i=1,nl1
          ii=i+i
          b1(i)=a1(ii-1)
          c1(i)=a1(ii)
        enddo
        nr1=1
        nbt1(1)=nl1
        ibt1(1)=ltp-10
        signi=tab1intp(eni,sni,np,nbt,ibt,nr,e1)
        pni=tab1intp(b1,c1,nl1,nbt1,ibt1,nr1,uu)
        sige=sctnpi(e1,u,za,awr,zap,awp,spi,lidp,signi,pni)
        deallocate(b1,c1,nbt1,ibt1)
      elseif(e.eq.e2) then
        allocate(b2(nl2),c2(nl2),nbt2(1),ibt2(1))
        do i=1,nl2
          ii=i+i
          b2(i)=a2(ii-1)
          c2(i)=a2(ii)
        enddo
        nr2=1
        nbt2(1)=nl2
        ibt2(1)=ltp-10
        signi=tab1intp(eni,sni,np,nbt,ibt,nr,e2)
        pni=tab1intp(b2,c2,nl2,nbt2,ibt2,nr2,uu)
        sige=sctnpi(e2,u,za,awr,zap,awp,spi,lidp,signi,pni)
        deallocate(b2,c2,nbt2,ibt2)
      else
        ilaw=mod(lei,10)
        allocate(b1(nl1),c1(nl1),nbt1(1),ibt1(1))
        do i=1,nl1
          ii=i+i
          b1(i)=a1(ii-1)
          c1(i)=a1(ii)
        enddo
        nr1=1
        nbt1(1)=nl1
        ibt1(1)=ltp-10
        allocate(b2(nl2),c2(nl2),nbt2(1),ibt2(1))
        do i=1,nl2
          ii=i+i
          b2(i)=a2(ii-1)
          c2(i)=a2(ii)
        enddo
        nr2=1
        nbt2(1)=nl2
        ibt2(1)=ltp-10
        signi=tab1intp(eni,sni,np,nbt,ibt,nr,e)
        pni=unit_base_intp(e1,b1,c1,nl1,nbt1,ibt1,nr1, &
                           e2,b2,c2,nl2,nbt2,ibt2,nr2,ilaw,e,uu)
        sige=sctnpi(e,u,za,awr,zap,awp,spi,lidp,signi,pni)
        deallocate(b1,c1,nbt1,ibt1)
        deallocate(b2,c2,nbt2,ibt2)
      endif
    endif
  endif
  f6law5=two*pi*sige ! to convert barn/sr to barn/(unit_cosine)
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function f6law6(awr,awi,awp,q,apsx,npsx,e,ep,u)
!
! Description:
! Calculate the angular distribution f(E,E',u) given by MF6/LAW6
!(N-Body Phase-Space Distribution) at (e,ep,u)
!
! Input:
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the outgoing particle
! q: reaction q value from MF3
! apsx: total mass in neutron units of the N particles treated by LAW6
! npsx: number of particles distributed according to LAW6 (N)
! e: incident energy in the LAB system
! ep: outgoing energy in the LAB system
! u:  outgoing cosine value in the LAB system
!
! Output:
! f6law6: f(E,E',u) angular distribution in the lab system at (e,ep,u)
!
  implicit real*8 (a-h,o-z)
  parameter(pi=3.141592653589793d0)
  parameter(c3=4.0d0/pi, c4=105.0d0/32.0d0, c5=256.0d0/(14.0d0*pi))
  awc=awi+awr
  ea=awr/awc*e+q
  eimax=(apsx-awp)/apsx*ea
  f6law6=0.0d0
  if (eimax.gt.0.0d0) then
    es=awi*awp/(awc*awc)*e
    epc=es+ep-2.0d0*u*sqrt(es*ep)
    if (epc.lt.eimax) then
      if (npsx.eq.3) then
        cn=c3/(eimax*eimax)
      elseif(npsx.eq.4) then
        cn=c4/(eimax**3.5d0)
      elseif(npsx.eq.5) then
        cn=c5/(eimax**5.0d0)
      else
        cn=0.0d0
      endif
      f6law6=cn*sqrt(ep)*((eimax-epc)**(1.5d0*dble(npsx)-4.0d0))
    endif
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function f6law7(e,ep,u,lei, &
                         e1,lmu1,u11,ep11,f11,np11,nbt11,ibt11,nr11, &
                                 u12,ep12,f12,np12,nbt12,ibt12,nr12, &
                         e2,lmu2,u21,ep21,f21,np21,nbt21,ibt21,nr21, &
                                 u22,ep22,f22,np22,nbt22,ibt22,nr22)
!
! Description:
! Calculate the angle-energy distribution given by MF6/LAW7 at (e,ep,u)
!
! Input:
!  e: incident energy in the LAB system
! ep: outgoing energy in the LAB system
!  u: outgoing cosine in the LAB system
!lei: interpolation law between e1 and e2
! e1: incident energy of the lower energy panel
!lmu1: interpolation law between u11 and u12 at e1
!u11: outgoing cosine value of the lower cosine panel at e1
!ep11: outgoing energies at 2D-panel (u11,e1)
!f11:  outgoing energy distribution at 2D-panel (u11,e1)
!np11: number of outgoing energies at 2D-panel (u11,e1)
!nbt11: interpolation ranges for f11
!ibt11: interpolation law for f11
!nr11:  number of interpolation ranges for f11
!u12: outgoing cosine value of the upper cosine panel at e1
!ep12: outgoing energies at 2D-panel (u12,e1)
!f12:  outgoing energy distribution at 2D-panel (u12,e1)
!np12: number of outgoing energies at 2D-panel (u21,e1)
!nbt12: interpolation ranges for f12
!ibt12: interpolation law for f12
!nr12:  number of interpolation ranges for f12
! e2: incident energy of the upper energy panel
!lmu2: interpolation law between u21 and u22 at e2
!u21: outgoing cosine value of the lower cosine panel at e2
!ep21: outgoing energies at 2D-panel (u12,e2)
!f21:  outgoing energy distribution at 2D-panel (u12,e2)
!np21: number of outgoing energies at 2D-panel (u12,e2)
!nbt21: interpolation ranges for f21
!ibt21: interpolation law for f21
!nr21:  number of interpolation ranges for f21
!u22: outgoing cosine value of the upper cosine panel at e2
!ep22: outgoing energies at 2D-panel (u12,e2)
!f22:  outgoing energy distribution at 2D-panel (u22,e2)
!np22: number of outgoing energies at 2D-panel (u22,e2)
!nbt22: interpolation ranges for f22
!ibt22: interpolation law for f22
!nr22:  number of interpolation ranges for f22
!
!Output:
! f6law7: angle-energy distribution given by MF6/LAW7 at (e,ep,u)
!
  implicit real*8 (a-h,o-z)
  dimension ep11(*),f11(*),nbt11(*),ibt11(*),ep12(*),f12(*),nbt12(*),ibt12(*)
  dimension ep21(*),f21(*),nbt21(*),ibt21(*),ep22(*),f22(*),nbt22(*),ibt22(*)
  if (e.lt.e1.or.e.gt.e2) then
     f6law7=0.0d0
  else
    if (u.lt.u11.or.u.gt.u12) then
      f1=0.0d0
    else
      f1=unit_base_intp(u11,ep11,f11,np11,nbt11,ibt11,nr11, &
                        u12,ep12,f12,np12,nbt12,ibt12,nr12,lmu1,u,ep)
    endif
    if (u.lt.u21.or.u.gt.u22) then
      f2=0.0d0
    else
      f2=unit_base_intp(u21,ep21,f21,np21,nbt21,ibt21,nr21, &
                        u22,ep22,f22,np22,nbt22,ibt22,nr22,lmu2,u,ep)
    endif
    law=mod(lei,10)
    f6law7=yintp(e1,f1,e2,f2,law,e)
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
! basic procedures for law formalism
! ------------------------------------------------------------------------------------------------------------------------------
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
! ------------------------------------------------------------------------------------------------------------------------------
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
!        p dimension: nl+1
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
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function ykalbach(zai,zap,zat,e,ep,u,b,na)
!
! Description:
! Compute the kalbach-mann angular distribution at outgoing cosine u
!
! f(u)=a*f0*(cosh(a*u)+r*sinh(a*u))/(2*sinh(a)
!
! where f0=f0(e,ep) is the total emission probability
!        r=r(e,ep)  is the pre-compound fraction
!        a=a(e,ep)  is the slope, a simple parameterized function
!
! The incident energy e should be in the LAB system, and the outgoing energy ep
! and the outgoing cosine u should be given in the CM system
!
! Input:
! zai: ZA number of the incident particle (ZA=1000*Z+A)
! zap: ZA number of the outgoing particle
! zat: ZA number of the target
! e: incident energy in the LAB system [eV]
! ep: outgoing energy in the CM system [eV]
! u: outgoing cosine value in the CM system
! b: array of Kalbach-Mann parameters (dimension b(na+1)):
!      b(1)=f0
!      b(2)=r, if na=1 or na=2
!      b(3)=a, if na=2
! na: number of kalbach-mann parameters na=[0,1,2]
!
! Output:
!  ykalbach: f(e,ep,u)=a*f0*(cosh(a*u)+r*sinh(a*u))/(2*sinh(a)
!
  implicit real*8 (a-h,o-z)
  parameter (zero=0.0d0, amin=1.0d-38)
  dimension b(*)
  f0=b(1)
  if (na.eq.1) then
    r=b(2)
    a=bachaa(zai,zap,zat,e,ep)
  elseif (na.eq.2) then
    r=b(2)
    a=b(3)
  else
    r=zero
    a=zero
  endif
  if (abs(a).gt.amin) then
    au=a*u
    ykalbach=0.5d0*a*f0*(cosh(au)+r*sinh(au))/sinh(a)
  else
    ykalbach=0.5d0*f0
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
   real*8 function bachaa(zai,zap,zat,ee,epe)
!
!  Description:
!  compute the parameter a=a(e,ep) for Kalbach-Mann systematics:
!    f(u)=a*f0*(cosh(a*u)+r*sinh(a*u))/(2*sinh(a))
!    (adapted from NJOY2016 by D. Lopez Aldama)
!
!  Input:
!  zai:  incident particle ZA number
!  zap:  outgoing particle ZA number
!  zat:  target ZA number
!  ee:   incident energy of particle zai [eV]
!  epe:  outgoing energy of particle zap [eV]
!
!  Output:
!  bachaa: Kalbach-Mann parameter a=a(ee,epe)
!
   implicit real*8 (a-h,o-z)
   real*8 nc,nb
   parameter(third=.333333333d0, twoth=.666666667d0, fourth=1.33333333d0)
   parameter(c1=15.68d0, c2=-28.07d0, c3=-18.56d0)
   parameter(c4=33.22d0, c5=-0.717d0, c6=1.211d0)
   parameter(s2=2.22d0, s3=8.48d0, s4=7.72d0, s5=28.3d0)
   parameter(b1=0.04d0, b2=1.8d-6, b3=6.7d-7)
   parameter(d1=9.3d0)
   parameter(ea1=41.d0, ea2=130.d0)
   parameter(emc2=939.56542052539d0, emev=1.0d6)
   parameter(eps=1.0d-3)
!
   iza1i=int(zai+eps)
   iza2=int(zap+eps)
   izat=int(zat+eps)
   e=ee/emev
   ep=epe/emev
   iza1=iza1i
   if (iza1i.eq.0) iza1=1
   iza=izat
   if (iza.eq.6000) iza=6012
   if (iza.eq.12000) iza=12024
   if (iza.eq.14000) iza=14028
   if (iza.eq.16000) iza=16032
   if (iza.eq.17000) iza=17035
   if (iza.eq.19000) iza=19039
   if (iza.eq.20000) iza=20040
   if (iza.eq.22000) iza=22048
   if (iza.eq.23000) iza=23051
   if (iza.eq.24000) iza=24052
   if (iza.eq.26000) iza=26056
   if (iza.eq.28000) iza=28058
   if (iza.eq.29000) iza=29063
   if (iza.eq.31000) iza=31069
   if (iza.eq.40000) iza=40090
   if (iza.eq.42000) iza=42096
   if (iza.eq.48000) iza=48112
   if (iza.eq.49000) iza=49115
   if (iza.eq.50000) iza=50120
   if (iza.eq.63000) iza=63151
   if (iza.eq.72000) iza=72178
   if (iza.eq.74000) iza=74184
   if (iza.eq.82000) iza=82208
   aa=mod(iza,1000)
   if (aa.eq.0.) then
      write(*,*)' Fatal error in bachaa: Dominant isotope not known for ',iza
      stop
   endif
   za=int(iza/1000)
   ac=aa+mod(iza1,1000)
   zc=za+int(iza1/1000)
   ab=ac-mod(iza2,1000)
   zb=zc-int(iza2/1000)
   na=nint(aa-za)
   nb=nint(ab-zb)
   nc=nint(ac-zc)
   sa=c1*(ac-aa)+c2*((nc-zc)**2/ac-(na-za)**2/aa) &
     +c3*(ac**twoth-aa**twoth)+c4*((nc-zc)**2/ac**fourth-(na-za)**2/aa**fourth)&
     +c5*(zc**2/ac**third-za**2/aa**third)+c6*(zc**2/ac-za**2/aa)
   if (iza1.eq.1002) sa=sa-s2
   if (iza1.eq.1003) sa=sa-s3
   if (iza1.eq.2003) sa=sa-s4
   if (iza1.eq.2004) sa=sa-s5
   sb=c1*(ac-ab)+c2*((nc-zc)**2/ac-(nb-zb)**2/ab) &
     +c3*(ac**twoth-ab**twoth)+c4*((nc-zc)**2/ac**fourth-(nb-zb)**2/ab**fourth)&
     +c5*(zc**2/ac**third-zb**2/ab**third)+c6*(zc**2/ac-zb**2/ab)
   if (iza2.eq.1002) sb=sb-s2
   if (iza2.eq.1003) sb=sb-s3
   if (iza2.eq.2003) sb=sb-s4
   if (iza2.eq.2004) sb=sb-s5
   ecm=aa*e/ac
   ea=ecm+sa
   eb=ep*ac/ab+sb
   x1=eb
   if (ea.gt.ea2) x1=ea2*eb/ea
   x3=eb
   if (ea.gt.ea1) x3=ea1*eb/ea
   fa=1
   if (iza1.eq.2004) fa=0
   fb=1
   if (iza2.eq.1) fb=fb/2
   if (iza2.eq.2004) fb=2
   bb=b1*x1+b2*x1**3+b3*fa*fb*x3**4
   if (iza1i.eq.0) then
      fact=d1
      if (ep.ne.0.) fact=fact/sqrt(ep)
      test=1
      if (fact.lt.test) fact=test
      test=4
      if (fact.gt.test) fact=test
      bb=bb*sqrt(e/(2*emc2))*fact
   endif
   bachaa=bb
   return
   end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function sctnae(e,u,za,awr,zap,awp,spi,lidp,a,b,nl)
!
! Description:
! Compute elastic cross section for the nuclear amplitude representation (LTP=1)
!
! Input:
! e: energy of incident charged particle in the LAB system [eV]
! u: cosine of the scattering angle in the CM system
! za: ZA number of the target
! awr: relative atomic mass of the target
! zap: ZA number of the charged-particle
! awp: relative nuclear mass of the charged-particle
! spi: spin of the charged particle (spi=0, 1/2, 1, ...)
! lidp: flag for identical particles (lidp=1 for identical particles)
! a: list of coefficients (ar(i),ai(i)) for the interference expansion
! b: list of coefficient b(i) for the nuclear cross section expansion
! nl: higest Legendre order of the nuclear partial waves
!
! Output:
! sctnae: elastic cross section in units of barn/sr at (e,u)
!
  implicit real*8 (a-h, o-z)
  parameter(zero=0.0d0)
  parameter(half=0.5d0)
  parameter(one=1.0d0)
  parameter(two=2.0d0)
  complex(kind(1.0d0))ciu,sum1,sum2,csl,arg1,arg2
! external dimension
  dimension a(*),b(*)
! internal dimension
  allocatable p(:)
  nb=2*nl
  allocate(p(nb+1))
  call legndr(u,p,nb)
  ciu=dcmplx(zero,one)
  sigc=coul(e,u,za,awr,zap,awp,spi,lidp,eta)
  if (lidp.eq.1) then ! identical particles
    sigb=half*b(1)
    do l=1,nl
      ll=l+l
      sigb=sigb+(dble(ll)+half)*b(l+1)*p(ll+1)
    enddo
    sum1=half*dcmplx(a(1),a(2))
    sum2=sum1
    sgn=-one
    do l=1,nl
      ll=l+l+1
      csl=(dble(l)+half)*p(l+1)*dcmplx(a(ll),a(ll+1))
      sum1=sum1+csl
      sum2=sum2+sgn*csl
      sgn=-sgn
    enddo
    arg1=eta*log((one-u)*half)*ciu
    arg2=eta*log((one+u)*half)*ciu
    sigi=-two*eta/(one-u*u)*dble((one+u)*exp(arg1)*sum1+(one-u)*exp(arg2)*sum2)
  else ! distinguishable particles
    sigb=half*b(1)
    do l=1,nb
      l1=l+1
      sigb=sigb+(dble(l)+half)*b(l1)*p(l1)
    enddo
    sum1=half*dcmplx(a(1),a(2))
    do l=1,nl
      ll=l+l+1
      sum1=sum1+(dble(l)+half)*p(l+1)*dcmplx(a(ll),a(ll+1))
    enddo
    arg1=eta*log((one-u)*half)*ciu
    sigi=-two*eta/(one-u)*dble(exp(arg1)*sum1)
  endif
  sctnae=sigc+sigi+sigb ! Coulomb+interference+nuclear
  deallocate(p)
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function sctrxe(e,u,za,awr,zap,awp,spi,lidp,c,nl)
!
! Description:
! Compute elastic cross section for the residual cross section expansion
! representation (LTP=2)
!
! Input:
! e: energy of incident charged particle in the LAB system [eV]
! u: cosine of the scattering angle in the CM system
! za: ZA number of the target
! awr: relative atomic mass of the target
! zap: ZA number of the charged-particle
! awp: relative nuclear mass of the charged-particle
! spi: spin of the charged particle (spi=0, 1/2, 1, ...)
! lidp: flag for identical particles (lidp=1 for identical particles)
! c: list of coefficients c(i) for the residual cross section expansion
! nl: higest Legendre order of the nuclear partial waves
!
! Output:
! sctrxe: elastic cross section in units of barn/sr at (e,u)
!
  implicit real*8 (a-h, o-z)
  parameter(half=0.5d0)
  parameter(one=1.0d0)
! external dimension
  dimension c(*)
! internal dimension
  allocatable p(:)
  sigc=coul(e,u,za,awr,zap,awp,spi,lidp,eta)
  if (lidp.eq.1) then ! identical particles
    nc=2*nl
    allocate(p(nc+1))
    call legndr(u,p,nc)
    sigr=half*c(1)
    do l=1,nl
      ll=l+l
      sigr=sigr+(dble(ll)+half)*c(l+1)*p(ll+1)
    enddo
    sigr=sigr/(one-u*u)
  else ! distinguishable particles
    allocate(p(nl+1))
    call legndr(u,p,nl)
    sigr=half*c(1)
    do l=1,nl
      l1=l+1
      sigr=sigr+(dble(l)+half)*c(l1)*p(l1)
    enddo
    sigr=sigr/(one-u)
  endif
  sctrxe=sigc+sigr ! Coulomb + (residual contribution)
  deallocate(p)
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function sctnpi(e,u,za,awr,zap,awp,spi,lidp,sni,pni)
!
! Description:
! Compute elastic cross section for the nuclear + interference
! representation (LTP=12 or LTP=14)
!
! Input:
! e: energy of incident charged particle in the LAB system [eV]
! u: cosine of the scattering angle in the CM system
! za: ZA number of the target
! awr: relative atomic mass of the target
! zap: ZA number of the charged-particle
! awp: relative nuclear mass of the charged-particle
! spi: spin of the charged particle (spi=0, 1/2, 1, ...)
! lidp: flag for identical particles (lidp=1 for identical particles)
! sni: nuclear + interference cross section at e from MF3/MT2 data
! pni: nuclear + interference distribution at (e,u) from MF6/MT2 data
!
! Output:
! sctnpi: elastic cross section in units of barn/sr at (e,u)
!
  implicit real*8 (a-h, o-z)
  sigc=coul(e,u,za,awr,zap,awp,spi,lidp,eta)
  sctnpi=sigc+sni*pni ! Coulomb + (nuclear+interference contribution)
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function coul(e,u,za,awr,zap,awp,spi,lidp,eta)
!
! Description:
! Compute the Coulomb component of the elastic cross section
!
! Input:
! e: energy of incident charged particle in the LAB system [eV]
! u: cosine of the scattering angle in the CM system
! za: ZA number of the target
! awr: relative atomic mass of the target
! zap: ZA number of the charged-particle
! awp: relative nuclear mass of the charged-particle
! spi: spin of the charged particle (spi=0, 1/2, 1, ...)
! lidp: flag for identical particles (lidp=1 for identical particles)
!
! Output:
! coul: Coulomb component in units of barn/sr at (e,u)
! eta: dimensionless Coulomb parameter (needed for LTP=1)
!
  implicit real*8 (a-h, o-z)
  parameter(amn=1.00866491595d0)          ! neutron mass in amu
  parameter(ev=1.602176634E-12)           ! erg/eV
  parameter(amu=9.3149410242d+8)          ! atomic mas unit in ev/amu
  parameter(hbar=6.582119569d-16)         ! reduced Planck's constant in eV*s
  parameter(clight=2.99792458d+10)        ! speed of light in vacuum  in cm/s
  parameter(barn=1.0d-24)                 ! 1 barn=1.0e-24 cm**2)
  parameter(alpha=1.0d-16*ev*clight/hbar) ! fine-structure constant
  parameter(zero=0.0d0)
  parameter(one=1.0d0)
  parameter(two=2.0d0)
  parameter(c1=two*amu/(hbar*hbar*clight*clight)*barn)
  parameter(c2=alpha*alpha*amu/two)
  at=awr*amn
  ap=awp*amn
  izt=nint(za)
  izp=nint(zap)
  zt=int(izt/1000)
  zp=int(izp/1000)
  eta=zp*zt*sqrt(c2*ap/e)
  wk=at*sqrt(c1*ap*e)/(ap+at)
  if (lidp.eq.1) then ! identical particles
    u2=u*u
    r2s=two*spi
    i2s=nint(r2s)
    coul=two*eta*eta/(wk*wk*(one-u2))*((one+u2)/(one-u2) + &
         ((-1)**i2s)/(r2s+one)*cos(eta*log((one+u)/(one-u))))
  else ! distinguishable particles
    coul=eta*eta/(wk*wk*(one-u)*(one-u))
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
! procedures for LAB to CM conversion
! ------------------------------------------------------------------------------------------------------------------------------
  subroutine mf4lab2cm(lct,awr,awi,awp,q,e,u,w,dinv)
!
! Description:
! Convert the cosine value given in the LAB system (u) to the CM system (w)
! and compute the CM to LAB Jacobian (dinv), if the evaluated angle distribution
! is given in the CM system (lct=2).If lct is not equal 2, no transformation is
! applied (w=u and dinv=1).
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
! w: cosine value in the reference system of the original evaluated data
! dinv: Jacobian from CM to LAB for LCT=2, 1 otherwise
!
  implicit real*8 (a-h,o-z)
  parameter (zero=0.0d0)
  parameter (eps=1.0d-12)
  parameter (one=1.0d0)
  parameter (emin=1.0d-5)
  if (lct.eq.2.and.(awp*awi).ne.zero) then
!   distribution is given in the CM system for massive particles
!   convert cosine value from LAB to CM using non relativistic
!   two-body kinematic formulae
    ee=(awr+awi)/awr*q
    ethr=max(-ee,emin)
    rth=max(ee/e,-one)
    r2=awr*(awr+awi-awp)/(awi*awp)*(one+rth)
    r=sqrt(r2)
    if (r.le.one) then
      umin=cos(asin(r))
    else
      umin=-one
    endif
    if (e.ge.ethr.and.u.ge.umin) then
      u2=u*u
      w=(one-u2-r2*u2)/(r*(u2-one-u*sqrt(u2+r2-one)))
      if (w.gt.one) then
        w=one
      elseif (w.lt.-one) then
        w=-one
      endif
      rpw=abs(r+w)
      if (r.le.one.and.w.lt.zero.and.rpw.lt.eps) then
         rpw=eps
         if (-w.lt.r) then
           w=-r-eps
         else
           w=-r+eps
         endif
      endif
      xw=1.0d0+2.0d0*r*w+r2
      dinv=xw*sqrt(xw)/(r2*rpw)
    else
!     forbidden value of e or u or both
!     assigning a set of values to point out the issue
      u=one
      w=u
      dinv=zero
    endif
  else
!   distribution is in the LAB system or no conversion is required
!   no transformation is applied. the Jacobian dinv is set equal 1
    w=u
    dinv=one
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  subroutine mf6lab2cm(awr,awi,awp,lct,e,ep,u,tp,w,dinv)
!
! Description:
!  Make the reference system transformation if required
!
! Input:
!  awr: relative atomic mass of the target
!  awi: relative nuclear mass of the incident particle
!  awp: relative nuclear mass of the required outgoing particle in MF6
!  lct: reference system for angular distribution
!  e:   inciden energy in the LAB system
!  ep:  secondary energy in the LAB system
!  u:   cosine value in the LAB system
!
! Output:
!  tp: secondary energy in the reference system of the evaluation data
!  w:  cosine value in the reference system of the evaluation data
!  dinv: Jacobian determinant from evaluated data to user data
!  tp=ep, w=u and dinv=1.0 if input data are in the LAB system
!
  implicit real*8 (a-h, o-z)
  parameter (d2min=1.0d-38, cmin=1.0d-19)
  c0=sqrt(awi*awp)/(awi+awr)
  if ((lct.eq.2.or.(lct.eq.3.and.awp.lt.4.0d0)).and.e*c0.gt.0.0d0) then
    if (ep.gt.0.0d0) then
      c=c0*sqrt(e/ep)
      d2=1.0d0+c*c-2.0d0*c*u
      if (d2.lt.d2min) then
        d2=d2min
        c=u-cmin
      endif
      tp=ep*d2
      dinv=1.0d0/sqrt(d2)
      w=dinv*(u-c)
      if (w.gt.1.0d0) then
        w=1.0d0
      elseif (w.lt.-1.0d0) then
        w=-1.0d0
      endif
    else
      tp=c0*c0*e
      w=-1.0d0
      dinv=0.0d0
    endif
  else
    tp=ep
    w=u
    dinv=1.0d0
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
! auxiliary functions
! ------------------------------------------------------------------------------------------------------------------------------
  function ihigh(x0,x,i0,n)
!
! Description
! Return the index i of the first element in array x that fulfil
! the condition x(i)>x0. Array x is assumed to be in ascending order.
!
! Input:
! x0: input x value
! x:  array of x values
! i0: index for starting the search (i0<n)
! n:  total number of x values
!
! Output:
! ihigh: i index for x(i)>x0, 0 if x0<x(1) or x0>x(n)
!
  implicit real*8 (a-h,o-z)
  dimension x(*)
  if (x0.lt.x(i0).or.x0.gt.x(n).or.i0.ge.n) then
    ihigh=0
  else
    i=i0+1
    do while (x(i).lt.x0)
      i=i+1
    enddo
    ihigh=i
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  function imatch(x0,x,n)
!
! Description
! Return the index i if the array x contains an element abs(x(i))=x0
! within a relative fractional error of eps, otherwise return 0
!
! Input:
! x0: require x value
! x:  array of x values
! n:  number of x values
!
! Output:
! imatch: i value if x(i)=x0, 0 otherwise
!
  implicit real*8 (a-h,o-z)
  parameter (eps=1.0d-6)
  dimension x(*)
  imatch=0
  do i=1,n
    xi=abs(x(i))
    if (abs(xi-x0).le.abs(x0*eps)) then
      imatch=i
      exit
    endif
  enddo
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
!   interpolation functions
! ------------------------------------------------------------------------------------------------------------------------------
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
  parameter (zero=0.0d0, small=1.0d-38)
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
    stop
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
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
! ------------------------------------------------------------------------------------------------------------------------------
  subroutine list_intp(e1,a1,n1,e2,a2,n2,ilaw,e,a,na)
!
! Description:
! Interpolate a list of parameters as Legendre coefficients or
! Kalbach-Mann parameters for MF6/LAW1 among others.
! The parameters must be in the same order.
!
! Input:
! e1: value of the variable e at panel 1
! a1: list of parameters at e1
! n1: number of parameters in the array a1
! e2: value of the variable e at panel 2
! a2: list of parameters at e2
! n2: number of parameters in the array a2
! ilaw: interpolation law between e1 and e2
! e: desired value of e
!
! Output:
! a: list of interpolated parameters at e
! na: number of the parameters in the array a
!
  implicit real*8 (a-h, o-z)
  dimension a1(*),a2(*),a(*)
  if (e.eq.e1) then
!   case e equal to e1
    na=n1
    do l=1,n1
      a(l)=a1(l)
    enddo
  elseif (e.eq.e2) then
!   case e equal to e2
    na=n2
    do l=1,n2
      a(l)=a2(l)
    enddo
  else
!   case e1<e<e2
    law=mod(ilaw,10)
    n0=min(n1,n2)
    na=max(n1,n2)
    do l=1,n0
      a(l)=yintp(e1,a1(l),e2,a2(l),law,e)
    enddo
    if (na.gt.n0) then
      zero=0.0d0
      do l=n0+1,na
        if (l.gt.n1) then
          a(l)=yintp(e1,zero,e2,a2(l),law,e)
        else
          a(l)=yintp(e1,a1(l),e2,zero,law,e)
        endif
      enddo
    endif
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
  real*8 function unit_base_intp(y1,x1,f1,np1,nbt1,ibt1,nr1, &
                          y2,x2,f2,np2,nbt2,ibt2,nr2,inty,y0,x0)
! Description:
! Return the value of the function at (y0,x0) using unit-base interpolation
! between panels f1(x1) at y1 and f2(x2) at y2
!
! Input:
! y1: value of independent variable y for panel 1
! x1: values of independent variable x at y1
! f1: values of the function f(x1) at y1
! np1: number of pairs (x1,f1(x1)) given at y1
! nbt1: interpolation nodes at y1
! ibt1: interpolation law at y1
! nr1: number of interpolation ranges at y1
! y2: value of independent variable y for panel 2
! x2: values of independent variable x at y2
! f2: values of the function f(x2) at y2
! np2: number of pairs (x2,f2(x2)) given at y2
! nbt2: interpolation nodes at y2
! ibt2: interpolation law at y2
! nr2: number of interpolation ranges at y2
! y0: value of the independent variable y where the function f is calculated
! x0: value of the independent variable x where the function f is calculated
!
! Output:
!  value of the function f at (y0,x0)
!
  implicit real*8 (a-h,o-z)
  dimension x1(*),f1(*),x2(*),f2(*)
  dimension nbt1(*),ibt1(*),nbt2(*),ibt2(*)
  if (y0.lt.y1.or.y0.gt.y2) then
    unit_base_intp=0.0d0
  else
    law=mod(inty,10)
    x1low=x1(1)
    x1high=x1(np1)
    x1range=x1high-x1low
    x2low=x2(1)
    x2high=x2(np2)
    x2range=x2high-x2low
    yslope=(y0-y1)/(y2-y1)
    xlow=x1low+yslope*(x2low-x1low)
    xhigh=x1high+yslope*(x2high-x1high)
    xrange=xhigh-xlow
    xslope=(x0-xlow)/xrange
    x=x1low+xslope*x1range
    f1x=tab1intp(x1,f1,np1,nbt1,ibt1,nr1,x)
    f1x=f1x*x1range/xrange
    x=x2low+xslope*x2range
    f2x=tab1intp(x2,f2,np2,nbt2,ibt2,nr2,x)
    f2x=f2x*x2range/xrange
    unit_base_intp=yintp(y1,f1x,y2,f2x,law,y0)
  endif
  return
  end
! ------------------------------------------------------------------------------------------------------------------------------
! Set of procedures for analytical integration of tabulated ENDF-6 data
!-------------------------------------------------------------------------------------------------------------------------------
subroutine tabint(x1,y1,x2,y2,u1,u2,intlaw,yint,uint)
!
! Description:
! tabint analytically calculates the integral between x1 and x2 of the function
! y(x) given by tabulated data (x1,y1), (x2,y2) with interpolation law intlaw.
! It also estimates the absolute deviation of the definite integral considering
! the uncertainties given at the interval boundaries
! Defining yint=integral(x1,y1,x2,y2) as the definite integral in the
! interval [x1,x2] considering y(x1)=y1 and y(x2)=y2, then the value of
! uint is estimated as:
!  dmin=abs(integral(x1,y1-u1,x2,y2-u2)-integral(x1,y1,x2,y2))
!  dplus=abs(integral(x1,y1+u1,x2,y2+u2)-integral(x1,y1,x2,y2))
!  uint=max(dmin,dplus)
! The range of yint is [yint-uint,yint+uint]
!
! Input parameters:
! x1: lower boundary of the interval
! y1: value of the function y at x=x1
! x2: upper boundary of the interval
! y2: value of the function y at x=x2
! u1: uncertainty of y at x=x1
! u2: uncertainty of y at x=x2
! intlaw: interpolation law between x1 and x2 (intlaw=1,2,3,4,5)
!         intlaw=1: histogram interpolation (y constant)
!         intlaw=2: lin-lin (y is linear in x)
!         intlaw=3: lin-log (y is linear in log(x))
!         intlaw=4: log-lin (log(y) is linear in x)
!         intlaw=5: log-log (log(y) is linear in log(x))
!         Note: for ay other value lin-lin interpolation is assumed
!
! Output parameters:
! yint: integral of y(x) between x1 and x2
! uint: absolute deviation of yint estimated from u1 and u2
!
implicit real*8 (a-h,o-z)
law=mod(intlaw,10)
h=x2-x1
if (h.le.0.0d0) then    ! x2-x1=0.0
  yint=0.0d0
  uint=0.0d0
elseif (law.eq.1.or.y2.eq.y1) then ! histogram interpolation or y2=y1
  yint=h*y1
  uint=h*max(abs(u1),abs(u2))
elseif (law.eq.2) then             ! lin-lin interpolation
  h2=0.5d0*h
  yint=h2*(y1+y2)
  uint=h2*(abs(u1)+abs(u2))
elseif (law.eq.3) then             ! lin-log interpolation
  hl=h/log(x2/x1)
  yint=x2*y2-x1*y1-(y2-y1)*hl
  du1=x1*u1
  du2=x2*u2
  du3=(u2-u1)*hl
  uint=max(abs(du1-du2+du3),abs(du2-du1-du3))
elseif (law.eq.4) then             ! log-lin interpolation
  yint=h*(y2-y1)/log(y2/y1)
  ya=y1-u1
  yb=y2-u2
  dymin=abs(h*(yb-ya)/log(yb/ya)-yint)
  ya=y1+u1
  yb=y2+u2
  dyplus=abs(h*(yb-ya)/log(yb/ya)-yint)
  uint=max(dymin,dyplus)
elseif (law.eq.5) then            ! log-log interpolation
  dxl=log(x2/x1)
  d1=log(y2/y1)/dxl+1.0d0
  yint=x1*y1/d1*(exp(d1*dxl)-1.0d0)
  ya=y1-u1
  yb=y2-u2
  d1=log(yb/ya)/dxl+1.0d0
  dymin=abs(x1*ya/d1*(exp(d1*dxl)-1.0d0)-yint)
  ya=y1+u1
  yb=y2+u2
  d1=log(yb/ya)/dxl+1.0d0
  dyplus=abs(x1*ya/d1*(exp(d1*dxl)-1.0d0)-yint)
  uint=max(dymin,dyplus)
else ! unknown law
  write(*,*) ' Interpolation law: ',law,' not allowed'
  stop
endif
return
end
! ------------------------------------------------------------------------------------------------------------------------------
subroutine tablin(x1,y1,x2,y2,u1,u2,intlaw,epsy,epsx,nmax,x,y,u,ulin,n,iconv)
!
! Description:
! Subroutine tablin linearizes the function y(x) given by tabulated data (x1,y1)
! and (x2,y2) and the interpolation law intlaw between x1 and x2. The
! uncertainties at the interval endpoints u1 and u2 are used to estimate the
! maximum absolute deviations at intermediate points.
!
! Input parameters:
! x1: lower boundary of the interval
! y1: value of the function y at x=x1
! x2: upper boundary of the interval
! y2: value of the function y at x=x2
! u1: uncertainty of y(x) at x=x1
! u2: uncertainty of y(x) at x=x2
! intlaw: interpolation law between x1 and x2 (intlaw=1,2,3,4,5)
!         intlaw=1: histogram interpolation (y constant)
!         intlaw=2: lin-lin (y is linear in x)
!         intlaw=3: lin-log (y is linear in log(x))
!         intlaw=4: log-lig (log(y) is linear in x)
!         intlaw=5: log-log (log(y) is linear in log(x))
! epsy: fractional tolerance in y for linearization
! epsx: fractional tolerance in x for linearization
! nmax: Max. number of points allowed in the interval [x1,x2]
!       Note: nmax is the dimension of arrays x,y,u and its minimum value
!             should be 3. The dimension of ulin is equal to nmax-1
!
! Output parameters:
! x: array containing the values of the abscissa x after linearization in the
!    interval [x1,x2]
! y: array containing the values of y after linearization y(i)=y(x(i)) in
!    the interval [x1,x2]
! u: array containing the pointwise maximum absolute deviations estimated from
!    the uncertainties u1 and u2 at the interval endpoints
! ulin: array containing maximum relative differnce due to linearization in the
!    subinterval [x(i),x(i+1)]
! n: total number of points after linearization.
!    Actual size of arrays x, y and u.
!    the actual size of ulin is n-1 (n-1 = number of subintervals)
!    Note: 2 <= n <= nmax
! iconv: convergence trigger
!       iconv=1 converged
!       iconv=2 maximum number of point (nmax) reached without convergence
!
implicit real*8 (a-h,o-z)
parameter (ymin=1.0d-30)
dimension x(*),y(*),u(*),ulin(*)
allocatable xs(:),ys(:)
law=mod(intlaw,10)
h=x2-x1
if (h.le.0.0d0) then                 ! x1=x2
  x(1)=x1
  y(1)=y1
  u(1)=u1
  x(2)=x1
  y(2)=y2
  u(2)=u2
  n=2
  ulin(1)=0.0d0
  iconv=1
elseif (law.eq.1.and.y1.ne.y2) then ! law=1 (y=constant)
  x(1)=x1
  y(1)=y1
  u(1)=u1
  x(2)=x2
  y(2)=y1
  u(2)=u1
  x(3)=x2
  y(3)=y2
  u(3)=u2
  n=3
  ulin(1)=0.0d0
  ulin(2)=0.0d0
  iconv=1
elseif (law.eq.2.or.y1.eq.y2) then ! law=2 (lin-lin)
  x(1)=x1
  y(1)=y1
  u(1)=u1
  x(2)=x2
  y(2)=y2
  u(2)=u2
  n=2
  ulin(1)=0.0d0
  iconv=1
else ! law=3,4,5 (iterative procedure for linearization)
  kmax=nmax-1
  allocate(xs(kmax),ys(kmax))
  n=1
  x(n)=x1
  y(n)=y1
  xl=x1
  yl=y1
  xh=x2
  yh=y2
  k=0
  do while (iconv.eq.0) ! interval halving technique
    if (law.eq.4.or.xh*xl.le.0.0d0) then
      xm=0.5d0*(xl+xh)
    else
      xm=sqrt(xl*xh)
    endif
    ym=yintp(xl,yl,xh,yh,law,xm)
    erm=errlin(xl,yl,xh,yh,law)
    ki=n+k
    if (erm.le.abs(epsy*ym).or.(xh-xm).le.abs(epsx*xh).or.ki.eq.kmax.or.(abs(ym).eq.0.0d0.and.erm.le.epsy*ymin)) then
      if (ym.ne.0.0d0) then
        ulin(n)=abs(erm/ym)
      else
        ulin(n)=erm/ymin
      endif
      n=n+1
      x(n)=xh
      y(n)=yh
      if (k.eq.0) then
        iconv=1
      elseif (ki.eq.kmax) then
        do j=k,1,-1
          xl=x(n)
          yl=y(n)
          xh=xs(j)
          yh=ys(j)
          if (law.eq.4.or.xh*xl.le.0.0d0) then
            xm=0.5d0*(xl+xh)
          else
            xm=sqrt(xl*xh)
          endif
          ym=yintp(xl,yl,xh,yh,law,xm)
          erm=errlin(xl,yl,xh,yh,law)
          if (ym.ne.0.0d0) then
            ulin(n)=abs(erm/ym)
          else
            ulin(n)=erm/ymin
          endif
          n=n+1
          x(n)=xh
          y(n)=yh
        enddo
        iconv=2
      else
        xl=xh
        yl=yh
        xh=xs(k)
        yh=ys(k)
        k=k-1
      endif
    else
      k=k+1
      xs(k)=xh
      ys(k)=yh
      xh=xm
      yh=ym
    endif
  enddo
  deallocate(xs,ys)
  u(1)=u1
  u(n)=u2
  do i=2,n-1
    u(i)=absdev(x1,y1,u1,x2,y2,u2,law,x(i),y(i)) ! absolute deviation estimation
  enddo
endif
return
end
! ------------------------------------------------------------------------------------------------------------------------------
real*8 function errlin(x1,y1,x2,y2,law)
!
! Description:
! the procedure errlin estimates the maximum absolute difference between lin-lin
! interpolation and the actual interpolation law (law=3,4,5) in the interval
! [x1,x2]
!
! Input:
! x1: value of abscissa at the lower boundary
! y1: value of the function y at x=x1
! x2: value of abscissa at the upper boundary
! y2: value of the function y at x=x2
! law: non lin-lin ENDF-6 interpolation law between x1 and x2
!        law=1: histogram interpolation (y constant)
!        law=2: lin-lin (y is linear in x)
!        law=3: lin-log (y is linear in log(x))
!        law=4: log-lig (log(y) is linear in x)
!        law=5: log-log (log(y) is linear in log(x))
!
! Output:
! errlin: Maximum absolute difference between lin-lin interpolation and the
!         the actual interpolation law (law=3,4,5) in the interval [x1,x2]
!
implicit real*8 (a-h,o-z)
if (x2.eq.x1.or.y2.eq.y1.or.law.eq.1.or.law.eq.2) then ! x2=x1,y2=y1,law=1 or 2
  errlin=0.0d0
elseif (law.eq.3) then ! y is linear in log(x)
  dy=y2-y1
  a=dy/(x2-x1)
  b=dy/log(x2/x1)
  errlin=abs(b*(1-log(b/(a*x1)))-a*x1)
elseif (law.eq.4) then ! log(y) is linear in x
  a=(y2-y1)/log(y2/y1)
  b=a/y1
  errlin=abs(y1*(1.0d0-b)+a*log(b))
elseif (law.eq.5) then ! log(y) is linear in log(x)
  a=(y2-y1)/(x2-x1)
  b=log(y2/y1)/log(x2/x1)
  c=exp(log(a*x1/(b*y1))/(b-1.0d0))
  errlin=abs(y1*(1.0d0-exp(b*log(c)))+a*x1*(c-1.0d0))
else
  write(*,*) ' Fatal error: Interpolation law: ',law,' not allowed'
  stop
endif
return
end
! ------------------------------------------------------------------------------------------------------------------------------
real*8 function absdev(x1,y1,u1,x2,y2,u2,law,x,y)
!
! Description:
! absdev is a function designed to estimate the maximum absolute deviation
! for y=y(x) considering the uncertainties u1 and u2 at the endpoints of
! the interval. y(x) is given by a non-linear ENDF-6 interpolation law
! between x1 and x2.
! Defining,
!   y=yintp(x1,y1,x2,y2,law,x)
!   ymin=yintp(x1,y1-u1,x2,y2-u2,law,x)
!   yplus=yintp(x1,y1-u1,x2,y2-u2,law,x)
! then,
!   absdev=max(abs(y-ymin),abs(yplus-y))
!
! Input:
! x1: value of abscissa at the lower boundary
! y1: value of the function y at x1
! u1: uncertainty of y1 at x1
! x2: value of abscissa at the upper boundary
! y2: value of the function y at x2
! u2: uncertainty of y2 at x2
! law: non lin-lin ENDF-6 interpolation law between x1 and x2
!        law=1: histogram interpolation (y constant)
!        law=2: lin-lin (y is linear in x)
!        law=3: lin-log (y is linear in log(x))
!        law=4: log-log (log(y) is linear in x)
!        law=5: log-log (log(y) is linear in log(x))
! x: value of the abscissa x
! y: value of the function y(x)
!
! Output:
! absdev: maximum absolute deviation of y(x)
!
  implicit real*8 (a-h, o-z)
  ya=y1-u1
  yb=y2-u2
  ymin=yintp(x1,ya,x2,yb,law,x)
  ya=y1+u1
  yb=y2+u2
  yplus=yintp(x1,ya,x2,yb,law,x)
  absdev=max(abs(y-ymin),abs(yplus-y))
return
end
!-------------------------------------------------------------------------------------------------------------------------------
! Procedures to convert transition probabilities in MF12 to yields
!-------------------------------------------------------------------------------------------------------------------------------
subroutine init_trans2yield(elis,maxlevel,nlevel,qm,qi,ee,r,a)
!
!  Description:
!   The subroutine init_trans2yield initializes the transition data to be used by
!   subroutine trans2yield. Particularly, the level energy and probability matrices
!   are initialized for further use.
!
!  Input parameters:
!   elis: Excitation energy of the target nucleus relative to 0.0 for the ground state
!   maxlevel: Maximum number of levels for inelastic reactions (z,x') x=n,p,d,t,he-3,he-4
!             Dimension of arrays ee, a and r.
!   nlevels: Number of the discrete inelastic reactions (z,x') found in the evaluated data file
!   qm(i): Qm value of the inelastic reaction i ordered from the first to the last excited level
!          [qm(i), i=1,nlevel] i=1 for first level, i=nlevel for the last discrete inelastic level found
!   qi(i): Qi value of the inelastic reaction i ordered from the first to the last excited level
!          [qi(i), i=1,nlevel] i=1 for first level, i=nlevel for the last discrete inelastic level found
!
!  Output parameters:
!   ee(i):  Energy of the excited level i. [ee(i),i=1,maxlevel]
!   a(i,j): Probability that a gamma ray of energy eg is emitted in the transition from
!           level j to i, taken as the gamma-ray branching ratio, [a(i,j),i=1,maxlevel,j=1,maxlevel]
!   r(i,j): Probability that the nucleus initially excited at level i de-excites
!           at level j through all possible transitions [r(i,j),i=1,maxlevel,j=1,maxlevel]
!
  implicit real*8 (a-h, o-z)
  dimension qm(*),qi(*)                       ! input arrays of q values
  dimension ee(*),r(maxlevel,*),a(maxlevel,*) ! transitions matrices and level energies
  do i=1,maxlevel
    ee(i)=0.0d0
    do k=1,maxlevel
      if (i.eq.k) then
        r(i,k)=1.0d0
      else
        r(i,k)=0.0d0
      endif
      a(i,k)=0.0d0
    enddo
  enddo
  do i=1,nlevel
    ee(i+1)=qm(i)+elis-qi(i)
  enddo
return
end
!
subroutine trans2yield(mt,esns,nt,esi,tp,gp,maxlevel,ee,r,a,maxnk,nk,es,eg,y)
!
! Description:
!  The subroutine trans2yield converts transition probability arrays given for inelastic
!  discrete reactions in file MF12 to the corresponding yields. The routine should be
!  successively called from the first to the last excited level presented in the evaluation
!  to produce the yields at each level.
!
!  The array ee, and the matrices a and r contain the transition data up to the level NS-1
!  The data corresponding to the current level NS is added and yield data is computed.
!
! Input parameters:
!   mt: MT number of the inelastic reaction
!   esns: Energy of the residual excited state level for reaction mt (excited level NS=1,2,3,...)
!         Below the level numbered as NS there are NS levels including the ground level.
!         NS=mt-mt0, where mt0=50 for (z,n'),  mt0=600 for (z,p'), mt0=650 for (z,d'),
!                          mt0=700 for (z,t'), mt0=750 for (z,he3'), mt0=800 for (z,he4')
!   nt: Number of direct transitions for which data are given
!       (i.e., number of non-zero transition probabilities), nt <= NS
!   esi(i): energy of the i-th level to wich direct transitions are possible from level NS
!           esi(i) = 0.0 implies the ground state. [esi(i), i=1,nt]
!   tp(i): probability of a direct transition from level NS to a lower level i [tp(i),i=1,nt]
!   gp(i): probability that, given a transition from level NS to level i, the transition is
!          a photon transition (i.e., the conditional probability of photon emission)
!          gp(i)= 1.0 for pure photon transitions [gp(i),i=1,nt]
!   maxlevel: Maximum number of levels for inelastic reactions (z,x') x=n,p,d,t,he-3,he-4
!   ee(i):  Energy of the excited level i. [ee(i),i=1,maxlevel]
!   r(i,j): Probability that the nucleus initially excited at level i de-excites
!           at level j through all possible transitions [a(i,j),i=1,maxlevel, j=1,maxlevel]
!   a(i,j): Probability that a gamma ray of energy eg is emitted in the transition from
!           level j to i, taken as the gamma-ray branching ratio,
!           [a(i,j),i=1,maxlevel, j=1,maxlevel]
!   maxnk: maximum number of photons. Pre-dimension of arrays es, eg and y
!
! Output:
!   nk: Actual number of emitted photons for the inelastic reaction mt
!   es(k): energy of the level from which the k-th photon originates. [es(k),k=1,nk]
!   eg(k): k-th photon energy [eg(k), k=1, nk]
!   y(k):  yield of k-th photon [y(k), k=1, nk]
!
  implicit real*8 (a-h, o-z)
  dimension esi(*),tp(*),gp(*),ee(*),r(maxlevel,*),a(maxlevel,*)  ! input arrays (transition data)
  dimension es(*),eg(*),y(*) ! output arrays (yield data)
  ! check discrete reaction and assign first level mt1 according to reaction mt
  if (mt.gt.50.and.mt.lt.91) then       ! mt0=50    mtc=91
    mt0=50
  elseif (mt.gt.600.and.mt.lt.649) then ! mt0=600  mtc=649
    mt0=600
  elseif (mt.gt.650.and.mt.lt.699) then ! mt0=650  mtc=699
    mt0=650
  elseif (mt.gt.700.and.mt.lt.749) then ! mt0=700  mtc=749
    mt0=700
  elseif (mt.gt.750.and.mt.lt.799) then ! mt0=750  mtc=799
    mt0=750
  elseif (mt.gt.800.and.mt.lt.849) then ! mt0=800  mtc=849
    mt0=800
  elseif (mt.gt.875.and.mt.lt.891) then ! mt0=875  mtc=891
    mt0=875
  else
    write(*,*)' Fatal error: Transition probabilities given for a non discrete reaction MT=',mt
    stop
  endif
  !  generating/updating the transition probability matrices
  j=mt-mt0+1
  ee(j)=esns
  jm1=j-1
  do kk=1,jm1
    k=jm1-kk+1
    eek=ee(k)
    ii=0
    do i=1,nt
      esii=esi(i)
      if ((esii.eq.0.0d0.and.eek.eq.0.0d0).or.((esii.ne.0.0d0).and.(abs(esii-eek).le.0.0001d0*esii))) then
        ii=i
        exit
      endif
    enddo
    if (ii.eq.0) then
      r(k,j)=0.0d0
      a(k,j)=0.0d0
    else
      p=tp(ii)
      r(k,j)=p
      a(k,j)=p*gp(ii)
    endif
    if (k.ne.jm1) then
      kp1=k+1
      do i=kp1,jm1
        r(k,j)=r(k,j)+a(k,i)*r(i,j)
      enddo
    endif
  enddo
! calculating the yields for reaction mt
  nk=0
  do i=2,j
    j1=j+2-i
    ej1=ee(j1)
    do ii=1,jm1
      j2=j-ii
      ej2=ee(j2)
      yld=a(j2,j1)*r(j1,j)
      if (yld.gt.0.0d0) then
        nk=nk+1
        if (nk.gt.maxnk) then
          write(*,*)' Fatal error: too many photons from transition probabilities, increase maxnk=',maxnk
          stop
        endif
        eg(nk)=ej1-ej2
        es(nk)=ej1
        y(nk)=yld
      endif
    enddo
  enddo
  ! re-ordering data in descending order of gamma energies for reaction mt
  if (nk.gt.1) then
    lm1=nk-1
    do i=1,lm1
      ip1=i+1
      do ii=ip1,nk
        if (eg(i).lt.eg(ii)) then
          temp=eg(i)
          eg(i)=eg(ii)
          eg(ii)=temp
          temp=y(i)
          y(i)=y(ii)
          y(ii)=temp
          temp=es(i)
          es(i)=es(ii)
          es(ii)=temp
        endif
      enddo
    enddo
  endif
return
end
!-------------------------------------------------------------------------------------------------------------------------------
! Procedures for integration of MF6/LAW=1 (ddxs integration)
!-------------------------------------------------------------------------------------------------------------------------------
subroutine feep_full_law1con(e,awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                             tol,nepu,epu,nepmax,ep,feep,fdev,nep)
!
! Description:
!   The subroutine feep_full_law1con computes the spectrum of the emitted particle f(e,ep) given by mf6/law1,
!   ensuring that the values are linearly interpolable within a tolerance of tol.
!   Subroutines initial_epgrid and feep_law1con are called.
!   The values of the spectrum are in the laboratory system
!   The user can specify a set of output energy values of interest.
!   If nepmax<0 the linearization is not applied and the spectrum is calculated at the np0 points of
!   an internally computed initial grid, if abs(nepmax) >= np0
!
! Input parameters:
! e: incident energy in the laboratory system e1<= e <= e2
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the required outgoing particle in MF6
! za:  ZA number of the target (ZA=1000*Z+A)
! zai: ZA number of the incident particle
! zap: ZA number of the outgoing particle
! lct: Reference system for the energy-angular distribution
! lang: Angular representation flag:
!        lang=1, Legendre coefficients
!        lang=2, Kalbach-Mann systematics
!        lang=11-15, tabulated angular distribution
! lep: interpolation scheme for outgoing energies
! lei: interpolation scheme between incident energies e1 and e2
! e1:  incident energy of the lower panel
! nd1: Number of dicrete energies given at e1
! na1: number of angular parameters at e1
!       lang=1, na1=Legendre expansion order
!       lang=2, na1=1 r is given by the evaluator and a should be calculated
!               na1=2 r and a are given by the evaluator
!       lang=11-15, na1/2 pairs (u,p(u)) are tabulated
!               na1=0, isotropic distribution for all representations
! nep1: number of outgoing energies given at e1
! ep1: outgoing energy values at e1. 1D-array [ep1(i),i=1...npe1]
! b1:  outgoing energy-angle distribution at e1. 2D-array [b1(i,j),i=1...nep1,j=1...na1+1)]
! e2:  incident energy of the upper panel
! nd2: Number of dicrete energies given at e2
! na2: number of angular parameters at e2
!       lang=1, na2=Legendre expansion order
!       lang=2, na2=1 r is given by the evaluator and a should be calculated
!               na2=2 r and a are given by the evaluator
!       lang=11-15, na2/2 [u,p(u)] pairs are given
!               na2=0, isotropic distribution for all representations
! nep2: number of outgoing energies given at e2
! ep2: outgoing energy values at e2. 1D-array [ep2(i),i=1...npe2]
! b2:  Outgoing energy-angle distribution at e2. 2D-array [b2(i,j),i=1...nep2,j=1...na2+1)]
! tol: Relative tolerance for integration and linearization
! nepu: number of fixed outgoing energies provided by user
! epu:  outgoing energy values provided by the user. 1D array [epu(i),i=1..nepu)]
! nepmax: maximum expected number of outgoing energies. Dimension of arrays ep,feep,fdev
!
! Output parameters:
! ep: outgoing energies. 1D array: [ep(i), i=1...nep]
! feep: outgoing spectrum f(e,e')=f(e,ep). 1D array: [feep(i),i=1...nep] nep
! fdev: outgoing spectrum absolute deviation due to cosine integration. 1D array: [fdev(i),i=1...nep]
! nep: actual number of values of outgoing energies at which f(e,ep) is provided
!
implicit real*8 (a-h, o-z)
parameter (ksmax=20, tolx=1.0d-6)
dimension ep1(*),b1(nep1,*),ep2(*),b2(nep2,*)
dimension epu(*),ep(*),feep(*),fdev(*)
dimension xs(ksmax),ys(ksmax),dys(ksmax)
allocatable ep0(:)
np0=max(nep1-nd1,0)+max(nep2-nd2,0)+max(nepu,0)+5
allocate(ep0(np0))
call initial_epgrid(e,awr,awi,awp,lct,e1,nd1,nep1,ep1,e2,nd2,nep2,ep2,nepu,epu,ep0,np0)
if (nepmax.le.0) then
  nn=abs(nepmax)
  if (nn.ge.np0) then
    nep=np0
    do i=1,nep
      ep(i)=ep0(i)
    enddo
    call feep_points_law1con(e,awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                             tol,nep,ep,feep,fdev)
  else
    write(*,*)' Fatal error: increase the maximum number of outgoing energy points nepmax=',nn
    stop
  endif
else
  x1=ep0(1)
  call feep_law1con(e,x1,awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                    tol,y1,dy1)
  ep(1)=x1
  feep(1)=y1
  fdev(1)=dy1
  j=1
  do i=2,np0
    x2=ep0(i)
    call feep_law1con(e,x2,awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                      tol,y2,dy2)
    k=0
    istop=0
    do while (istop.eq.0)
      yl=0.5d0*(y1+y2)
      xm=0.5d0*(x1+x2)
      hm=xm-x1
      call feep_law1con(e,xm,awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                        tol,ym,dym)
      slope1=(ym-y1)/hm
      slope2=(y2-ym)/hm
      if ((abs(ym-yl).le.abs(tol*ym).and.slope1*slope2.gt.0.0d0).or.(y2.eq.ym.and.y1.eq.ym).or. &
            abs(x2-x1).le.abs(tolx*x2).or.k.eq.ksmax) then
        j=j+1
        if (j.gt.nepmax) then
          write(*,*)' Fatal error: increase the maximum number of outgoing energy points nepmax=',nepmax
          stop
        endif
        ep(j)=x2
        feep(j)=y2
        fdev(j)=dy2
        if (k.eq.0) then
          istop=1
        else
          x1=x2
          y1=y2
          dy1=dy2
          x2=xs(k)
          y2=ys(k)
          dy2=dys(k)
          k=k-1
        endif
      else
        k=k+1
        xs(k)=x2
        ys(k)=y2
        dys(k)=dy2
        x2=xm
        y2=ym
        dy2=dym
      endif
    enddo
    x1=x2
    y1=y2
    dy1=dy2
  enddo
  nep=j
endif
deallocate(ep0)
return
end
!-------------------------------------------------------------------------------------------------------------------------------
subroutine feep_points_law1con(e,awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                           tol,nepu,epu,fepu,fdevu)
!
! Description:
!   The subroutine feep_points_law1con computes the spectrum of the emitted particle f(e,ep) given by mf6/law1
!   at nepu outgoing energies provided by the user.
!   Subroutine feep_law1con is called.
!   The values of the spectrum are in the laboratory system
!
! Input parameters:
! e: incident energy in the laboratory system e1<= e <= e2
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the required outgoing particle in MF6
! za:  ZA number of the target (ZA=1000*Z+A)
! zai: ZA number of the incident particle
! zap: ZA number of the outgoing particle
! lct: Reference system for the energy-angular distribution
! lang: Angular representation flag:
!        lang=1, Legendre coefficients
!        lang=2, Kalbach-Mann systematics
!        lang=11-15, tabulated angular distribution
! lep: interpolation scheme for outgoing energies
! lei: interpolation scheme between incident energies e1 and e2
! e1:  incident energy of the lower panel
! nd1: Number of dicrete energies given at e1
! na1: number of angular parameters at e1
!       lang=1, na1=Legendre expansion order
!       lang=2, na1=1 r is given by the evaluator and a should be calculated
!               na1=2 r and a are given by the evaluator
!       lang=11-15, na1/2 pairs (u,p(u)) are tabulated
!               na1=0, isotropic distribution for all representations
! nep1: number of outgoing energies given at e1
! ep1: outgoing energy values at e1. 1D-array [ep1(i),i=1...npe1]
! b1:  outgoing energy-angle distribution at e1. 2D-array [b1(i,j),i=1...nep1,j=1...na1+1)]
! e2:  incident energy of the upper panel
! nd2: Number of dicrete energies given at e2
! na2: number of angular parameters at e2
!       lang=1, na2=Legendre expansion order
!       lang=2, na2=1 r is given by the evaluator and a should be calculated
!               na2=2 r and a are given by the evaluator
!       lang=11-15, na2/2 [u,p(u)] pairs are given
!               na2=0, isotropic distribution for all representations
! nep2: number of outgoing energies given at e2
! ep2: outgoing energy values at e2. 1D-array [ep2(i),i=1...npe2]
! b2:  Outgoing energy-angle distribution at e2. 2D-array [b2(i,j),i=1...nep2,j=1...na2+1)]
! tol: Relative tolerance for cosine integration
! nepu: number of fixed outgoing energies provided by user
! epu:  outgoing energy values provided by user. 1D array [epu(i),i=1..nepu)]
!
! Output parameters:
! fepu: outgoing particle spectrum f(e,e')=f(e,ep). 1D array: [fepu(i),i=1...nep] nepu
! fdevu: outgoing particle spectrum absolute deviation due to numerical integration. 1D array: [fdevu(i),i=1...nepu]
!
implicit real*8 (a-h, o-z)
dimension ep1(*),b1(nep1,*),ep2(*),b2(nep2,*)
dimension epu(*),fepu(*),fdevu(*)
do i=1,nepu
  call feep_law1con(e,epu(i),awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                    tol,fepu(i),fdevu(i))
enddo
return
end
!-------------------------------------------------------------------------------------------------------------------------------
subroutine feep_law1con(e,ep,awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                       tol,fep,fdev)
!
! Description:
!  The subroutine feep_law1con computes the spectrum of the emitted particle f(e,ep) given by mf6/law1 by
!  calling subroutines get_griddata and feep_uint_law1con
!  The value of the spectrum is returned in the laboratory system
!
! Input parameters:
! e: incident energy in the laboratory system e1<= e <= e2
! ep: outgoing energy value in the laboratory system ep >= 0.0
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the required outgoing particle in MF6
! za:  ZA number of the target (ZA=1000*Z+A)
! zai: ZA number of the incident particle
! zap: ZA number of the outgoing particle
! lct: Reference system for the energy-angular distribution
! lang: Angular representation flag:
!        lang=1, Legendre coefficients
!        lang=2, Kalbach-Mann systematics
!        lang=11-15, tabulated angular distribution
! lep: interpolation scheme for outgoing energies
! lei: interpolation scheme between incident energies e1 and e2
! e1:  incident energy of the lower panel
! nd1: Number of dicrete energies given at e1
! na1: number of angular parameters at e1
!       lang=1, na1=Legendre expansion order
!       lang=2, na1=1 r is given by the evaluator and a should be calculated
!               na1=2 r and a are given by the evaluator
!       lang=11-15, na1/2 pairs (u,p(u)) are tabulated
!               na1=0, isotropic distribution for all representations
! nep1: number of outgoing energies given at e1
! ep1: outgoing energy values at e1. 1D-array [ep1(i),i=1...npe1]
! b1:  outgoing energy-angle distribution at e1. 2D-array [b1(i,j),i=1...nep1,j=1...na1+1)]
! e2:  incident energy of the upper panel
! nd2: Number of dicrete energies given at e2
! na2: number of angular parameters at e2
!       lang=1, na2=Legendre expansion order
!       lang=2, na2=1 r is given by the evaluator and a should be calculated
!               na2=2 r and a are given by the evaluator
!       lang=11-15, na2/2 [u,p(u)] pairs are given
!               na2=0, isotropic distribution for all representations
! nep2: number of outgoing energies given at e2
! ep2: outgoing energy values at e2. 1D-array [ep2(i),i=1...npe2]
! b2:  Outgoing energy-angle distribution at e2. 2D-array [b2(i,j),i=1...nep2,j=1...na2+1)]
! tol: Relative tolerance for integration and linearization
!
! Output parameters:
! fep: outgoing particle spectrum f(e,e')=f(e,ep).
! fdev: outgoing particle spectrum absolute deviation due to numerical integration
!
implicit real*8 (a-h, o-z)
parameter (hz=0.25d0)
dimension ep1(*),b1(nep1,*),ep2(*),b2(nep2,*)
call get_griddata(e,e1,nd1,nep1,ep1,e2,nd2,nep2,ep2,ep1min,ep1range,ep2min,ep2range,epmin,eprange,epmax)
awip=awi*awp
umax=1.0d0
if ((lct.eq.2.or.(lct.eq.3.and.awp.lt.4.0d0)).and.(awip*e.gt.0.0d0)) then
  c0=sqrt(awip)/(awi+awr)
  if (ep.gt.0.0d0) then
    umin=(ep+c0*c0*e-epmax)/(2.0d0*c0*sqrt(ep*e))
    umin=max(umin,-1.0d0)
    umin=min(umin,1.0d0)
  else
    umin=1.0d0
  endif
else
  umin=-1.0d0
endif
if (umax.gt.umin) then
  zmin=acos(umin)
  nu=int(zmin/hz)
  nu=max(nu,3)
  dz=zmin/dble(nu)
  z1=zmin-dz
  u1=cos(z1)
  call feep_uint_law1con(e,ep,umin,u1,awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                         tol,f,err)
  fep=f
  fdev=err
  do i=2,nu-1
    z2=z1-dz
    u2=cos(z2)
    call feep_uint_law1con(e,ep,u1,u2,awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                          tol,f,err)
    fep=fep+f
    fdev=fdev+err
    z1=z2
    u1=u2
  enddo
  call feep_uint_law1con(e,ep,u1,umax,awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                        tol,f,err)
  fep=fep+f
  fdev=fdev+err
else
  fep=0.0d0
  fdev=0.0d0
endif
return
end
!-------------------------------------------------------------------------------------------------------------------------------
subroutine feep_uint_law1con(e,ep,u1,u2,awr,awi,awp,za,zai,zap,lct,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                            tol,fep,err)
!
! Description:
!  The subroutine feep_uint_law1con computes the spectrum of the emitted particle f(e,ep) given by mf6/law1
!  in the cosine interval (u1,u2).
!  The value of the spectrum is in the laboratory system
!  Integration is performed using the Romberg-Richarson numerical method
!
! Input parameters:
! e: incident energy in the laboratory system e1<= e <= e2
! ep: outgoing energy value in the laboratory system ep >= 0.0
! u1: lower cosine boundary in the laboratory system
! u2: upper cosine boundary in the laboratory system
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the required outgoing particle in MF6
! za:  ZA number of the target (ZA=1000*Z+A)
! zai: ZA number of the incident particle
! zap: ZA number of the outgoing particle
! lct: Reference system for the energy-angular distribution
! lang: Angular representation flag:
!        lang=1, Legendre coefficients
!        lang=2, Kalbach-Mann systematics
!        lang=11-15, tabulated angular distribution
! lep: interpolation scheme for outgoing energies
! lei: interpolation scheme between incident energies e1 and e2
! e1:  incident energy of the lower panel
! nd1: Number of dicrete energies given at e1
! na1: number of angular parameters at e1
!       lang=1, na1=Legendre expansion order
!       lang=2, na1=1 r is given by the evaluator and a should be calculated
!               na1=2 r and a are given by the evaluator
!       lang=11-15, na1/2 pairs (u,p(u)) are tabulated
!               na1=0, isotropic distribution for all representations
! nep1: number of outgoing energies given at e1
! ep1: outgoing energy values at e1. 1D-array [ep1(i),i=1...npe1]
! b1:  outgoing energy-angle distribution at e1. 2D-array [b1(i,j),i=1...nep1,j=1...na1+1)]
! e2:  incident energy of the upper panel
! nd2: Number of dicrete energies given at e2
! na2: number of angular parameters at e2
!       lang=1, na2=Legendre expansion order
!       lang=2, na2=1 r is given by the evaluator and a should be calculated
!               na2=2 r and a are given by the evaluator
!       lang=11-15, na2/2 [u,p(u)] pairs are given
!               na2=0, isotropic distribution for all representations
! nep2: number of outgoing energies given at e2
! ep2: outgoing energy values at e2. 1D-array [ep2(i),i=1...npe2]
! b2:  Outgoing energy-angle distribution at e2. 2D-array [b2(i,j),i=1...nep2,j=1...na2+1)]
! tol: Relative tolerance for integration and linearization
!
! Output parameters:
! fep: the integrated outgoing spectrum f(e,e')=f(e,ep) between the cosines u1 and u2.
! err: outgoing spectrum absolute deviation due to numerical integration in the interval (u1,u2)
!
implicit real*8 (a-h, o-z)
parameter (nmax=12)
dimension ep1(*),b1(nep1,*),ep2(*),b2(nep2,*)
dimension r0(nmax),r1(nmax)
if (u2.gt.u1.and.ep.gt.0.0d0) then
  call mf6lab2cm(awr,awi,awp,lct,e,ep,u1,tp,w,dinv)
  y1=f6law1con(e,tp,w,za,zai,zap,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2)*dinv
  call mf6lab2cm(awr,awi,awp,lct,e,ep,u2,tp,w,dinv)
  y2=f6law1con(e,tp,w,za,zai,zap,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2)*dinv
  call romberg_init(i,jmax,u1,y1,u2,y2,h,r0)
  do while (i.lt.nmax)
    sum=0.0d0
    do j=1,jmax
      u=u1+dble(2*j-1)*h
      call mf6lab2cm(awr,awi,awp,lct,e,ep,u,tp,w,dinv)
      y=f6law1con(e,tp,w,za,zai,zap,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2)*dinv
      sum=sum+y
    enddo
    call richardson(i,jmax,h,sum,r0,r1,nmax,tol,n)
  enddo
  fep=r1(n)
  err=abs(r1(n)-r1(n-1))
else
 fep=0.0d0
 err=0.0d0
endif
return
end
!-------------------------------------------------------------------------------------------------------------------------------
subroutine romberg_init(i,jmax,u1,y1,u2,y2,h,r0)
!
! Description:
!  Initializes Romberg-Richarson method
!
! Input parameters:
!  u1: value of the lower boundary of the independent variable
!  y1: value of the function at u1
!  u2: value of the upper boundary of the independent variable
!  y2: value of the function at u2
!
! Output parameters:
! i: Romberg-Richarson main index (layer index)
! jmax: number of points to added at next layer i
! h: half interval
! r0(1): integral of the function y between (u1,u2) using trapezoidal rule
!
implicit real*8 (a-h, o-z)
dimension r0(*)
i=1
jmax=1
h=0.5d0*(u2-u1)
r0(1)=h*(y1+y2)
return
end
!-------------------------------------------------------------------------------------------------------------------------------
subroutine richardson(i,jmax,h,sum,r0,r1,nmax,tol,n)
!
! Description:
!  Computes Richarson extrapolation
!
! Input/Output parameters:
! i: Romberg-Richarson main index (layer index)
! jmax: number of points to added at next layer i
! h: integration interval
! r0 and r1: Arrays to compute the Richarson extrapolation
! nmax: maximum number of extrapolation layers
! tol: relative tolerance
! n: current index of the computed integral r1(n)=integral
!
implicit real*8 (a-h, o-z)
dimension r0(*),r1(*)
r1(1)=0.5d0*r0(1)+h*sum
pow=4.0d0
do k=1,i
  r1(k+1)=r1(k)+(r1(k)-r0(k))/(pow-1.0d0)
  pow=4.0d0*pow
enddo
n=i+1
if (abs(r1(n)-r1(i)).le.abs(tol*r1(n)).or.n.eq.nmax) then
  i=nmax
else
  h=0.5d0*h
  jmax=2*jmax
  do k=1,n
    r0(k)=r1(k)
  enddo
  i=n
endif
return
end
! ------------------------------------------------------------------------------------------------------------------------------
subroutine get_griddata(e,e1,nd1,nep1,ep1,e2,nd2,nep2,ep2,ep1min,ep1range,ep2min,ep2range,epmin,eprange,epmax)
!
! Description:
!  Extract outgoing energy range for mf6/law1 from panels e1 and e2 and compute values at e
!
! Input parameters:
! e: incident energy in the laboratory system e1<= e <= e2
! e1:  incident energy of the lower panel
! nd1: Number of dicrete energies given at e1
! na1: number of angular parameters at e1
! ep1: outgoing energy values at e1. 1D-array [ep1(i),i=1...npe1]
! e2:  incident energy of the upper panel
! nd2: Number of dicrete energies given at e2
! na2: number of angular parameters at e2
!       lang=1, na2=Legendre expansion order
!       lang=2, na2=1 r is given by the evaluator and a should be calculated
!               na2=2 r and a are given by the evaluator
!       lang=11-15, na2/2 [u,p(u)] pairs are given
!               na2=0, isotropic distribution for all representations
! nep2: number of outgoing energies given at e2
! ep1: outgoing energy values at e2. 1D-array [ep2(i),i=1...npe2]
!
! Output parameters:
! ep1min: minimum value of the outgoing energy at e1
! ep1range: range of the outgoing energy values at e1
! ep2min: minimum value of the outgoing energy at e2
! ep2range: range of the outgoing energy values at e2
! epmin: Computed minumun value of the outgoing energy at e
! eprange: Computed outgoing energy range at e
! epmax: Computed maximum outgoing energy value at e
!
implicit real*8 (a-h, o-z)
dimension ep1(*),ep2(*)
n1=nd1+1
n2=nd2+1
if (nep1.gt.n1) then
  ep1min=ep1(n1)
  ep1max=ep1(nep1)
else
  ep1min=0.0d0
  ep1max=0.0d0
endif
ep1range=ep1max-ep1min
if (nep2.gt.n2) then
  ep2min=ep2(n2)
  ep2max=ep2(nep2)
else
  ep2min=0.0d0
  ep2max=0.0d0
endif
ep2range=ep2max-ep2min
slope=(e-e1)/(e2-e1)
epmin=ep1min+(ep2min-ep1min)*slope
epmax=ep1max+(ep2max-ep1max)*slope
eprange=epmax-epmin
return
end
!-------------------------------------------------------------------------------------------------------------------------------
subroutine initial_epgrid(e,awr,awi,awp,lct,e1,nd1,nep1,ep1,e2,nd2,nep2,ep2,nepu,epu,ep0,np0)
!
! Description:
!  Prepare initial outgoing energy grid for integration of mf6/law1
!
! Input parameters:
! e: incident energy in the laboratory system e1<= e <= e2
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the required outgoing particle in MF6
! za:  ZA number of the target (ZA=1000*Z+A)
! zai: ZA number of the incident particle
! zap: ZA number of the outgoing particle
! lct: Reference system for the energy-angular distribution
! e1:  incident energy of the lower panel
! nd1: Number of dicrete energies given at e1
! na1: number of angular parameters at e1
!       lang=1, na1=Legendre expansion order
!       lang=2, na1=1 r is given by the evaluator and a should be calculated
!               na1=2 r and a are given by the evaluator
!       lang=11-15, na1/2 pairs (u,p(u)) are tabulated
!               na1=0, isotropic distribution for all representations
! nep1: number of outgoing energies given at e1
! ep1: outgoing energy values at e1. 1D-array [ep1(i),i=1...npe1]
! e2:  incident energy of the upper panel
! nd2: Number of dicrete energies given at e2
! na2: number of angular parameters at e2
! nep2: number of outgoing energies given at e2
! ep2: outgoing energy values at e2. 1D-array [ep2(i),i=1...npe2]
! nepu: number of fixed outgoing energies provided by user
! epu:  outgoing energy values provided by the user. 1D array [epu(i),i=1..nepu)]
!
! Output parameters
! ep0: outgoing energy values in the initial grid [ep0(i),i=1,np0]
! np0: integer variable.
!      At enter, the dimension of the ep0 array.
!      Upon return, the actual number of values in the initial outgoing energy grid
!
implicit real*8 (a-h, o-z)
parameter (tol0=1.0d-6)
dimension ep1(*),ep2(*),epu(*),ep0(*)
n1=nd1+1
n2=nd2+1
call get_griddata(e,e1,nd1,nep1,ep1,e2,nd2,nep2,ep2,ep1min,ep1range,ep2min,ep2range,epmin,eprange,epmax)
k=1
ep0(k)=0.0d0
if (lct.eq.2.or.(lct.eq.3.and.awp.lt.4.0d0)) then
  c0=sqrt(awi*awp)/(awr+awi)
  c=c0*c0*e
  k=k+1
  ep0(k)=c
  el=sqrt(epmax)+c0*sqrt(e)
  k=k+1
  ep0(k)=el*el
  if (nep1.gt.n1.and.e.ne.e2) then
    do i=n1,nep1
      ep=epmin+(ep1(i)-ep1min)*eprange/ep1range
      k=k+1
      ep0(k)=ep+c
    enddo
    el1=sqrt(ep1min+ep1range)+c0*sqrt(e1)
    k=k+1
    ep0(k)=el1*el1
  endif
  if (nep2.gt.n2.and.e.ne.e1) then
    do i=n2,nep2
      ep=epmin+(ep2(i)-ep2min)*eprange/ep2range
      k=k+1
      ep0(k)=ep+c
    enddo
    el2=sqrt(ep2min+ep2range)+c0*sqrt(e2)
    k=k+1
    ep0(k)=el2*el2
  endif
else
  if (nep1.gt.n1.and.e.ne.e2) then
    do i=n1,nep1
      k=k+1
      ep0(k)=epmin+(ep1(i)-ep1min)*eprange/ep1range
    enddo
    k=k+1
    ep0(k)=ep1(nep1)
  endif
  if (nep2.gt.n2.and.e.ne.e1) then
    do i=n2,nep2
      k=k+1
      ep0(k)=epmin+(ep2(i)-ep2min)*eprange/ep2range
    enddo
  endif
  k=k+1
  ep0(k)=ep2(nep2)
endif
do i=1,nepu
  k=k+1
  ep0(k)=epu(i)
enddo
irem=1
call orderx(ep0,k,tol0,irem)
np0=k
return
end
!-------------------------------------------------------------------------------------------------------------------------------
subroutine orderx(x,n,tol,irem)
!
! the n elements of array x are ordered in ascending order
! if irem is greater than 0, then elements within relative
! tolerance tol are removed
!
implicit real*8 (a-h, o-z)
dimension x(*)
if (n.gt.1) then
  m=n
  i=0
  do while (i.lt.m-1)
    i=i+1
    j=i
    do while (j.lt.m)
      j=j+1
      if (x(j).lt.x(i)) then
        temp=x(j)
        x(j)=x(i)
        x(i)=temp
      endif
    enddo
    if (i.gt.1) then
      if (irem.gt.0.and.abs(x(i)-x(i-1)).le.abs(tol*x(i))) then
        m=m-1
        if (i.gt.m) then
           n=m
           return
        endif
         do k=i,m
           x(k)=x(k+1)
         enddo
         i=i-1
      endif
    endif
  enddo
  if (irem.gt.0.and.abs(x(m)-x(m-1)).le.abs(tol*x(m))) m=m-1
  n=m
endif
return
end
!-------------------------------------------------------------------------------------------------------------------------------
! Procedures for integration of MF6/LAW=6 (ddxs integration)
!-------------------------------------------------------------------------------------------------------------------------------
subroutine feep_full_law6(e,awr,awi,awp,q,apsx,npsx,nepu,epu,tol,nepmax,ep,feep,fdev,nep)
!
! Description:
!   The subroutine feep_full_law6 computes the spectrum of the emitted particle f(e,ep) given by mf6/law6,
!   ensuring that the values are linearly interpolable within a tolerance of tol.
!   Subroutine feep_law6 is called.
!   The values of the spectrum are in the laboratory system
!   The user can specify a set of outgoing energy values of interest.
!   If nepmax<0 the linearization is not applied and the spectrum is calculated at the np0 points of
!   an internally computed initial grid, if abs(nepmax) >= np0
!
! Input parameters:
! e: incident energy in the LAB system
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the outgoing particle
! q: reaction q value from MF3
! apsx: total mass in neutron units of the N particles treated by LAW6
! npsx: number of particles distributed according to LAW6 (N)
! nepu: number of fixed outgoing energies provided by user
! epu:  outgoing energy values provided by the user. 1D array [epu(i),i=1..nepu)]
! tol: tol for linearization
! nepmax: maximum expected number of outgoing energies. Dimension of arrays ep,feep,fdev
!
! Output parameters:
! ep: outgoing energies. 1D array: [ep(i), i=1...nep]
! feep: outgoing particle spectrum f(e,e')=f(e,ep). 1D array: [feep(i),i=1...nep] nep
! fdev: outgoing particle spectrum absolute deviation. 1D array: [fdev(i),i=1...nep]
! nep: actual number of values of outgoing energies at which f(e,ep) is provided
!
implicit real*8 (a-h, o-z)
parameter (ksmax=20, tolx=1.0d-6, nep0=10)
dimension epu(*),ep(*),feep(*),fdev(*)
dimension xs(ksmax),ys(ksmax),dys(ksmax)
allocatable ep0(:)
np0=nep0+max(nepu,0)
allocate(ep0(np0))
! initial grid computation
k=1
ep0(k)=0.0d0
awc=awi+awr
es=awi*awp*e/(awc*awc)
k=k+1
ep0(k)=es
eimax=(apsx-awp)/apsx*(awr/awc*e+q)
el=sqrt(eimax)+sqrt(es)
elmax=el*el
nn=nep0-k
h=(elmax-es)/dble(nn)
do i=1,nn-1
  k=k+1
  ep0(k)=ep0(k-1)+h
enddo
k=k+1
ep0(k)=elmax
do i=1,nepu
  k=k+1
  ep0(k)=epu(i)
enddo
irem=1
call orderx(ep0,k,tolx,irem)
np0=k
if (nepmax.le.0) then
  nn=abs(nepmax)
  if (nn.ge.np0) then
    nep=np0
    do i=1,nep
      ep(i)=ep0(i)
    enddo
    call feep_points_law6(e,awr,awi,awp,q,apsx,npsx,nep,ep,feep,fdev)
  else
    write(*,*)' Fatal error: increase the maximum number of outgoing energy points nepmax=',nn
    stop
  endif
else
  x1=ep0(1)
  call feep_law6(e,x1,awr,awi,awp,q,apsx,npsx,y1,dy1)
  ep(1)=x1
  feep(1)=y1
  fdev(1)=dy1
  j=1
  do i=2,np0
    x2=ep0(i)
    call feep_law6(e,x2,awr,awi,awp,q,apsx,npsx,y2,dy2)
    k=0
    istop=0
    do while (istop.eq.0)
      yl=0.5d0*(y1+y2)
      xm=0.5d0*(x1+x2)
      hm=xm-x1
      call feep_law6(e,xm,awr,awi,awp,q,apsx,npsx,ym,dym)
      slope1=(ym-y1)/hm
      slope2=(y2-ym)/hm
      if ((abs(ym-yl).le.abs(tol*ym).and.slope1*slope2.gt.0.0d0).or.(y1.eq.ym.and.y2.eq.ym).or. &
            abs(x2-x1).le.abs(tolx*x2).or.k.eq.ksmax) then
        j=j+1
        if (j.gt.nepmax) then
          write(*,*)' Fatal error: increase the maximum number of outgoing energy points nepmax=',nepmax
          stop
        endif
        ep(j)=x2
        feep(j)=y2
        fdev(j)=dy2
        if (k.eq.0) then
          istop=1
        else
          x1=x2
          y1=y2
          dy1=dy2
          x2=xs(k)
          y2=ys(k)
          dy2=dys(k)
          k=k-1
        endif
      else
        k=k+1
        xs(k)=x2
        ys(k)=y2
        dys(k)=dy2
        x2=xm
        y2=ym
        dy2=dym
      endif
    enddo
    x1=x2
    y1=y2
    dy1=dy2
  enddo
  nep=j
endif
deallocate(ep0)
return
end
!-------------------------------------------------------------------------------------------------------------------------------
subroutine feep_points_law6(e,awr,awi,awp,q,apsx,npsx,nepu,epu,fepu,fdevu)
!
! Description:
!   The subroutine feep_points_law1con computes the
!   spectrum of the emitted particle f(e,ep) given by mf6/law6
!   at nepu outgoing energies provided by the user.
!   The subroutine feep_law6 is called.
!   The values of the spectrum are given in the laboratory system
!
! Input parameters:
! e: incident energy in the LAB system
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the outgoing particle
! q: reaction q value from MF3
! apsx: total mass in neutron units of the N particles treated by LAW6
! npsx: number of particles distributed according to LAW6 (N)
! nepu: number of fixed outgoing energies provided by user
! epu:  outgoing energy values provided by the user. 1D array [epu(i),i=1..nepu)]
!
! Output parameters:
! fepu: outgoing particle spectrum f(e,e')=f(e,ep). 1D array: [fepu(i),i=1...nepu] nep
! fdevu: outgoing particle spectrum absolute deviation. 1D array: [fdevu(i),i=1...nepu]
!        (For analytical integration fdevu(i)=0.0)
!
implicit real*8 (a-h, o-z)
dimension epu(*),fepu(*),fdevu(*)
do i=1,nepu
  call feep_law6(e,epu(i),awr,awi,awp,q,apsx,npsx,fepu(i),fdevu(i))
enddo
return
end
!-------------------------------------------------------------------------------------------------------------------------------
subroutine feep_law6(e,ep,awr,awi,awp,q,apsx,npsx,feep,fdev)
!
! Description:
! The subroutine feep_law6 computes the spectrum of the emitted particle f(e,ep)
! given by mf6/law6 by calling the subroutine feep_uint_law6
! The value of the spectrum is in the laboratory system
!
! Input parameters:
! nepu: number of fixed outgoing energies provided by user
! Input parameters:
! e: incident energy in the LAB system
! ep: outgoing particle energy in the LAB system
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the outgoing particle
! q: reaction q value from MF3
! apsx: total mass in neutron units of the N particles treated by LAW6
! npsx: number of particles distributed according to LAW6 (N)
!
! Output parameters:
! fep: outgoing particle spectrum f(e,e')=f(e,ep).
! fdev: outgoing particle spectrum absolute deviation
!
implicit real*8 (a-h,o-z)
parameter (small=1.0d-10)
fep=0.0d0
fdev=0.0d0
awc=awi+awr
eimax=(apsx-awp)/apsx*(awr/awc*e+q)
es=awi*awp/(awc*awc)*e
if (ep.gt.0.0d0.and.es.gt.0.0d0.and.eimax.gt.small) then
  umin=0.5d0*(ep+es-eimax)/sqrt(es*ep)
  umin=max(umin,-1.0d0)
  umin=min(umin,1.0d0)+small
  umax=1.0d0
  call feep_uint_law6(e,ep,umin,umax,awr,awi,awp,q,apsx,npsx,feep,fdev)
endif
return
end
!-------------------------------------------------------------------------------------------------------------------------------
subroutine feep_uint_law6(e,ep,u1,u2,awr,awi,awp,q,apsx,npsx,feep,fdev)
!
! Description:
! The subroutine feep_uint_law6 computes the spectrum of the emitted particle f(e,ep)
! given by mf6/law6 in the cosine interval (u1,u2).
! The value of the spectrum is in the laboratory system
! Analytical integration is applied, therefore fdev=0.0
!
! Input parameters:
! e: incident energy in the LAB system
! ep: outgoing particle energy in the LAB system
! u1: lower cosine boundary
! u2: upper cosine boundary
! awr: relative atomic mass of the target
! awi: relative nuclear mass of the incident particle
! awp: relative nuclear mass of the outgoing particle
! q: reaction q value from MF3
! apsx: total mass in neutron units of the N particles treated by LAW6
! npsx: number of particles distributed according to LAW6 (N)
!
! Output parameters:
! feep: the integrated outgoing spectrum f(e,e')=f(e,ep) in the cosine interval (u1,u2).
! fdev: outgoing spectrum absolute deviation due to cosine integration (fdev = 0.0)
!
implicit real*8 (a-h,o-z)
parameter(pi=3.141592653589793d0, small=1.0d-10)
parameter(c3=4.0d0/pi, c4=105.0d0/32.0d0, c5=256.0d0/(14.0d0*pi))
feep=0.0d0
fdev=0.0d0
if (u2.gt.u1) then
  awc=awi+awr
  eimax=(apsx-awp)/apsx*(awr/awc*e+q)
  es=awi*awp/(awc*awc)*e
  b=2.0d0*sqrt(ep*es)
  if (eimax.gt.small.and.b.gt.0.0d0) then
    a=eimax-ep-es
    decm1=max(a+b*u1,small)
    decm2=max(a+b*u2,decm1)
    if (npsx.eq.3) then
      cn=c3/(eimax*eimax)
    elseif(npsx.eq.4) then
      cn=c4/(eimax**3.5d0)
    elseif(npsx.eq.5) then
      cn=c5/(eimax**5.0d0)
    else
      cn=0.0d0
    endif
    rn1=1.5d0*dble(npsx)-3.0d0
    feep=cn*sqrt(ep)*(decm2**rn1-decm1**rn1)/(rn1*b)
  endif
endif
return
end
!-------------------------------------------------------------------------------------------------------------------------------