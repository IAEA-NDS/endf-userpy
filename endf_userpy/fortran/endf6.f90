! ----------------------------------------------------------------------------------
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
! externals
  dimension a1(*),a2(*),e(*),xmu(*),f4(ne,*)
! internals
  allocatable b1(:),b2(:),a(:)
! allocate space for internals arrays
  nb1=nl1+1
  nb2=nl2+1
  na=max(nb1,nb2)
  allocate(b1(nb1),b2(nb2),a(na))
! Prepare Legendre coefficient arrays for parameter interpolation
! Add zero order coefficient, which should be 1 for MF4 data
  do l=1,nl1
    b1(l+1)=a1(l)
  enddo
  b1(1)=1.0d0
  do l=1,nl2
    b2(l+1)=a2(l)
  enddo
  b2(1)=1.0d0
! Cycle for incident energies
  do ie=1,ne
    ei=e(ie)
!   Legendre coefficient interpolation
    call list_intp(e1,b1,nb1,e2,b2,nb2,ilaw,ei,a,na)
    nla=na-1 ! Legendre expansion order at the intermediate energy ei
    do ju=1,nmu
      u=xmu(ju)
!     Reference system conversion, if required
      call mf4lab2cm(lct,awr,awi,awp,q,ei,u,w,dinv)
!     calculate the f(E,w) in the reference system of the evaluation
      feu=yleg(w,a,nla)
!     multiply by Jacobian
      f4(ie,ju)=feu*dinv
    enddo
  enddo
  deallocate(b1,b2,a)
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
!     Reference system conversion, if required
      call mf4lab2cm(lct,awr,awi,awp,q,ei,u,w,dinv)
      if (ei.eq.e1) then
        feu=tab1intp(u1,f1,np1,nbt1,ibt1,nr1,w)
      elseif (ei.eq.e2) then
        feu=tab1intp(u2,f2,np2,nbt2,ibt2,nr2,w)
      else
        law=mod(ilaw,10)
        feu1=tab1intp(u1,f1,np1,nbt1,ibt1,nr1,w)
        feu2=tab1intp(u2,f2,np2,nbt2,ibt2,nr2,w)
        feu=yintp(e1,feu1,e2,feu2,law,ei)
      endif
!     multiply by Jacobian, if required
      f4(ie,ju)=feu*dinv
    enddo
  enddo
  return
  end
! ------------------------------------------------------------------------------
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
! Output
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
        call f6law1(e,tp,w,za,zai,zap,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1,&
                    e2,nd2,na2,nep2,ep2,b2,fdis,fcon)
        f6dis(ie,je,ju)=fdis*dinv
        f6con(ie,je,ju)=fcon*dinv
      enddo
    enddo
  enddo
  return
  end
! ------------------------------------------------------------------------------
  subroutine f6law1(e,tp,w,za,zai,zap,lang,lep,lei,e1,nd1,na1,nep1,ep1,b1, &
                    e2,nd2,na2,nep2,ep2,b2,f6dis,f6con)
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
! Output
!  f6dis: Discrete contribution to angle-energy distribution at e,tp,w
!  f6con: Continuum contribution to angle-energy distribution at e,tp,w
!
  implicit real*8 (a-h, o-z)
  dimension ep1(*),b1(nep1,*),ep2(*),b2(nep2,*)
  if (e.lt.e1.or.e.gt.e2) then
!
!   e is out of range
!
    f6dis=0.0d0
    f6con=0.0d0
  else
    law=mod(lei,10)
!
!   process discrete part, if any
!
    if (nd1.le.0.and.nd2.le.0) then
      f6dis=0.0d0
    else
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
      f6dis=yintp(e1,f1,e2,f2,law,e)
    endif
!
!   process continuum part
!
    if (nep1.le.nd1.and.nep2.le.nd2) then
      f6con=0.0d0
    else
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
      f6con=yintp(e1,f1,e2,f2,law,e)
    endif
  endif
  return
  end
!-------------------------------------------------------------------------------
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
! ------------------------------------------------------------------------------
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
! ------------------------------------------------------------------------------
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
! internals
  allocatable u1(:),u2(:),f1(:),f2(:)
  allocatable nbt1(:),ibt1(:),nbt2(:),ibt2(:)
  if (lang.eq.0) then
    call mf4_get_leg(awr,awi,awp,q,lct,e1,a1,nl1,e2,a2,nl2,ilaw,e,ne,xmu,nmu,f6)
  else
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
    law=mod(ilaw,10)
    call mf4_get_tab(awr,awi,awp,q,lct,e1,u1,f1,nl1,nbt1,ibt1,nr1, &
                     e2,u2,f2,nl2,nbt2,ibt2,nr2,law,e,ne,xmu,nmu,f6)
    deallocate(u1,f1,u2,f2)
    deallocate(nbt1,ibt1,nbt2,ibt2)
  endif
  return
  end
! ------------------------------------------------------------------------------
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
  parameter(pi=3.141592653589793d0)
  parameter(c3=4.0d0/pi, c4=105.0d0/32.0d0, c5=256.0d0/(14.0d0*pi))
  dimension e(*),ep(*),xmu(*)
  dimension f6(ne,nep,*)
  awc=awi+awr
  c0=awr/awc
  c1=(apsx-awp)/apsx
  c2=awi*awp/(awc*awc)
  r=1.5d0*dble(npsx)-4.0d0
  cn=0.0d0
  do i=1,ne
    ei=e(i)
    ea=c0*ei+q
    eimax=c1*ea
    es=c2*ei
    if (npsx.eq.3) then
      cn=c3/(eimax*eimax)
    elseif(npsx.eq.4) then
      cn=c4/(eimax**3.5d0)
    elseif(npsx.eq.5) then
      cn=c5/(eimax**5.0d0)
    endif
    do j=1,nep
       epj=ep(j)
       do k=1,nmu
         u=xmu(k)
         epc=es+epj-2.0d0*u*sqrt(es*epj)
         if (epc.lt.eimax) then
           f6(i,j,k)=cn*sqrt(epj)*(eimax-epc)**r
         else
           f6(i,j,k)=0.0d0
         endif
       enddo
    enddo
  enddo
  return
  end
! ------------------------------------------------------------------------------
!     auxiliary functions
! ------------------------------------------------------------------------------
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
! ------------------------------------------------------------------------------
  function imatch(x0,x,n)
!
! Description
! Return the index i if the array x contains an element x(i)=x0
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
    if (abs(x(i)-x0).le.abs(x0*eps)) then
      imatch=i
      exit
    endif
  enddo
  return
  end
! ------------------------------------------------------------------------------
!   interpolation functions
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
! ------------------------------------------------------------------------------
  function unit_base_intp(y1,x1,f1,np1,nbt1,ibt1,nr1, &
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
! ------------------------------------------------------------------------------
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
  parameter (rthmin=-0.999999d0)
  if (lct.eq.2) then
!   distribution is in the CM system
!   convert input cosine from LAB to CM using two-body kinematic formulae
    rth=(awr+awi)/awr*q/e
    if (rth.lt.rthmin) rth=rthmin
    r2=awr*(awr+awi-awp)/(awi*awp)*(1.0d0+rth)
    r=sqrt(r2)
    u2=u*u
    w=(1.0d0-u2-r2*u2)/(r*(u2-1.0d0-u*sqrt(u2+r2-1.0d0)))
    if (w.gt.1.0d0) then
      w=1.0d0
    elseif (w.lt.-1.0d0) then
      w=-1.0d0
    endif
    xw=1.0d0+2.0d0*r*w+r2
    dinv=xw*sqrt(xw)/(r2*(r+w))
  else
!   distribution is in the LAB system or no conversion is required
!   no transformation is applied. the Jacobian dinv is set equal 1
    w=u
    dinv=1.0d0
  endif
  return
  end
! ------------------------------------------------------------------------------
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
  if (lct.eq.2.or.(lct.eq.3.and.awp.le.4.0d0)) then
    c=sqrt(awi*awp*e/ep)/(awi+awr)
    d2=1.0d0+c*c-2.0d0*c*u
    if (d2.lt.d2min) then
      d2=d2min
      c=u-cmin
    endif
    tp=ep*d2
    dinv=1/sqrt(d2)
    w=dinv*(u-c)
    if (w.gt.1.0d0) then
      w=1.0d0
    elseif (w.lt.-1.0d0) then
      w=-1.0d0
    endif
  else
    tp=ep
    w=u
    dinv=1.0d0
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
! ------------------------------------------------------------------------------
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
! ------------------------------------------------------------------------------
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
! -----------------------------------------------------------------------------