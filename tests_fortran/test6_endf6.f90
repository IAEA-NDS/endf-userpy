      Program test6_endf6
      implicit real*8(a-h, o-z)
      parameter (nbmax=50000, nnx=450,nnsni=2000)
      parameter (nedim=60,nadim=51,nepdim=200)
      parameter (neu=5,nepu=5,nuu=5)
      character*66 line
      character*120 fin,fout
      dimension nbt(20),ibt(20),llaw(nepdim)
      dimension x(nbmax),y(nbmax)
      dimension eni(nnsni),sni(nnsni)
      dimension ep11(nepdim),f11(nepdim),ep12(nepdim),f12(nepdim)
      dimension ep21(nepdim),f21(nepdim),ep22(nepdim),f22(nepdim)
      dimension nbt11(20),ibt11(20),nbt12(20),ibt12(20)
      dimension nbt21(20),ibt21(20),nbt22(20),ibt22(20)
      dimension nbtsni(20),ibtsni(20)
      dimension b01(nbmax),b02(nbmax)
      dimension qi(nnx),mf6(nnx)
      dimension eu(neu),epu(nepu),uu(nuu)
      dimension f6dis(neu,nepu,nuu),f6con(neu,nepu,nuu)
      dimension f64(neu,nuu),f67(1,nepu,1)
      dimension e(nedim),ile(nedim)
      dimension nmu(nedim),ilmu(nedim,nadim),xu(nedim,nadim),nep(nedim,nadim)
      dimension ep(nedim,nadim,nepdim),f(nedim,nadim,nepdim),ilef(nedim,nadim,nepdim)
      allocatable ep1(:),b1(:,:),ep2(:),b2(:,:)
      data nin/3/,nou/12/
      write(*,*)' Input endf-6 formatted file:'
      read(*,*)fin
      write(*,*)fin
      write(*,*)' Output file:'
      read(*,*)fout
      write(*,*)fout
      write(*,*)' MAT number:'
      read(*,*)mati
      write(*,*)mati
      open(nin,file=fin)
      call readtext(nin,line,mat,mf,mt,nsi)
      call findmat(nin,mati,icod)
      if (icod.ne.0) then
        write(*,*)
        write(*,*)' Fatal error: MAT ',mati,' not found on tape ',fin
        close(nin)
        stop
      endif
      call readcont(nin,c1,c2,l1,l2,n1,n2,mat,mf,mt,ns)
      call readcont(nin,c1,c2,l1,l2,n1,n2,mat,mf,mt,ns)
      call readcont(nin,awi,emax,l1,l2,nsub,n2,mat,mf,mt,ns)
      izai=nsub/10
      zai=dble(izai)
      itype=mod(nsub,10)
      write(*,*)
      write(*,*)' NSUB=',nsub,' IPART=',izai,' itype=',itype
      write(*,*)' EMAX=',emax,' AWI=',awi
      write(*,*)' MAT ',mati
      if (itype.ne.0) then
         write(*,*)
         write(*,*)' Fatal error: Sublibrary not allowed NSUB=',nsub
         close(nin)
         stop
      endif
      open(nou,file=fout)
      write(nou,*)' NSUB=',nsub,' IPART=',izai,' itype=',itype
      write(nou,*)' EMAX=',emax,' AWI=',awi
      write(nou,*)' MAT ',mati
      nmf6=0
      call findmf(nin,mati,6,icod)
      if (icod.eq.0) then
        backspace(nin)
        mt=1000
        do while (mt.gt.0)
          call findnextmt(nin,6,mt)
          if (mt.gt.0) then
            nmf6=nmf6+1
            mf6(nmf6)=mt
          endif
        enddo
        write(nou,*)' Sections found on MF6: ',nmf6
      else
        write(nou,*)' MF6 not found'
        close(nin)
        stop
      endif
      rewind(nin)
      call readtext(nin,line,mat,mf,mt,nsi)
      call findmf(nin,mati,3,icod)
      do i=1,nmf6
        call findmt(nin,mati,3,mf6(i),icod)
        if (icod.ne.0) then
          write(nou,*)' mt=',mf6(i),' not found on MF3'
          write(*,*)' mt=',mf6(i),' not found on MF3'
          qi(i)=0.0d0
        else
          call readcont(nin,za,awr,l1,l2,n1,n2,mat,mf,mt,ns)
          call readtab1(nin,qm,qi(i),l1,lr,nr,np,nbt,ibt,x,y)
          if (mt.eq.2.and.izai.gt.1) then
            npsni=np
            do j=1,np
              eni(j)=x(j)
              sni(j)=y(j)
            enddo
            nrsni=nr
            do j=1,nr
              nbtsni(j)=nbt(j)
              ibtsni(j)=ibt(j)
            enddo
          endif
        endif
      enddo
      call findmf(nin,mati,6,icod)
      backspace(nin)
      mt=1000
      imt=0
      do while (mt.gt.0)
        call findnextmt(nin,6,mt)
        if (mt.gt.0) then
          imt=imt+1
!
!         processing MF6 by sections
!
          call readcont(nin,za,awr,jp,lct0,nk,n2,mat,mf,mt,nsi)
          if (jp.eq.0.and.lct0.ne.4) then
            q=qi(imt)
            write(nou,*)
            write(nou,*)
            write(nou,'(a,i4,a,i4,a,i5,a,1pe13.6)')' imt=',imt,' MT=',mt,' MAT=',mat,' QI=',q
            write(nou,'(a,1p,e13.6,a,e13.6,a,i2,a,i4)')' ZA=',za,' AWR=',awr,' LCT=',lct0,' NK=',nk
            write(*,'(a,i4,a,i4,a,i5,a,1pe13.6,a,i4)')' imt=',imt,' MT=',mt,' MAT=',mat,' QI=',q,' NK=',nk
            do kk=1,nk
              call readtab1(nin,zap,awp,lip,law,nr,np,nbt,ibt,x,y)
              eprim=0.0d0
              if (law.eq.2.and.zap.eq.0.0d0.and.awp.ne.0.0d0) then
                eprim=awp
                awp=0.0d0
              endif
              if (law.eq.6) then
                lct=1
              elseif (lct0.eq.3.and.awp.gt.4.0d0) then
                lct=1
              elseif (lct0.eq.3.and.awp.le.4.0d0) then
                lct=2
              else
                lct=lct0
              endif
              write(nou,'(110a1)')('*',i=1,110)
              write(nou,'(a,i4)')' Particle ',kk
              if (eprim.eq.0.0d0) then
                write(nou,'(a,1p,e13.6,a,e13.6,a,i2)')' ZAP=',zap,' AWP=',awp,' LCT=',lct
              else
                write(nou,'(a,1p,e13.6,a,e13.6,a,i2,a,e13.6)')' ZAP=',zap,' AWP=',awp,' LCT=',lct,' E_prim=',eprim
              endif
              if (law.eq.1) then
                call readtab2(nin,c1,c2,lang,lep,nr,ne,nbt,ibt)
                write(nou,'(a,i3,a,i3,a,i3,a,i5)')' LAW=',law,' LANG=',lang,' LEP=',lep,' NE=',ne
                write(nou,'(110a1)')('=',i=1,110)
                call readlist(nin,c1,e1,nd1,na1,nw1,nep1,b01)
                do ie=2, ne
                  allocate(ep1(nep1),b1(nep1,na1+1))
                  k=0
                  do i=1,nep1
                    k=k+1
                    ep1(i)=b01(k)
                    do j=1,na1+1
                      k=k+1
                      b1(i,j)=b01(k)
                    enddo
                  enddo
                  call readlist(nin,c1,e2,nd2,na2,nw2,nep2,b02)
                  lei=intlaw(ie,nbt,ibt,nr)
                  write(nou,'(a,i5,a,1p,e13.6,a,e13.6,a,i3)')' IE=',ie-1,' E1=',e1,' E2=',e2,' LEIN=',lei
                  write(nou,'(6(a,i4))')' nd1=',nd1,' na1=',na1,' nep1=',nep1,' nd2=',nd2,' na2=',na2,' nep2=',nep2
                  write(nou,'(110a1)')('-',i=1,110)
                  allocate(ep2(nep2),b2(nep2,na2+1))
                  k=0
                  do i=1,nep2
                    k=k+1
                    ep2(i)=b02(k)
                    do j=1,na2+1
                      k=k+1
                      b2(i,j)=b02(k)
                    enddo
                  enddo
                  e0=(awr+awi)/awr*(-q)*1.00001d0
                  if (e1.gt.e0) e0=e1*1.00001d0
                  h=(e2-e0)/dble(neu-1)
                  do i=1,neu
                    eu(i)=e0+h*dble(i-1)
                  enddo
                  if (nep1.le.nd1.and.nep2.le.nd2) then
                     nnn=min(nepu,nd1,nd2)
                     do i=1,nnn
                       epu(i)=ep1(nd1-i+1)
                     enddo
                     do i=nnn+1,neu
                       epu(i)=epu(nnn)*(1.0d0+dble(0.1d0*i))
                     enddo
                  else
                    if (nep1.gt.nd1.and.nep2.gt.nd2) then
                      tp1=max(ep1(nd1+1),ep2(nd2+1),1.0d-6)
                      tp2=min(ep1(nep1),ep2(nep2))
                    elseif (nep1.gt.nd1) then
                      tp1=max(ep1(nd1+1),ep2(nd2),1.0d-6)
                      tp2=min(ep1(nep1),ep2(1))
                    else
                      tp1=max(ep2(nd2+1),ep1(nd1),1.0d-6)
                      tp2=min(ep2(nep2),ep1(1))
                    endif
                    h=(tp2-tp1)/(nepu-1)
                    do i=1,nepu
                      epu(i)=tp1+h*dble(i-1)
                    enddo
                  endif
                  h=1.98d0/dble(nuu-1)
                  do i=1,nuu
                    uu(i)=-0.99+h*dble(i-1)
                  enddo
                  call mf6_get_law1(eu,neu,epu,nepu,uu,nuu,&
                        awr,awi,awp,za,zai,zap,lct,lang,lep,lei, &
                        e1,nd1,na1,nep1,ep1,b1,e2,nd2,na2,nep2,ep2,b2, &
                        f6dis,f6con)
                  write(nou,'(7a16)')'ei','ep','u','tp','w','f6dis','f6con'
                  do i=1,neu
                    do j=1,nepu
                      do k=1,nuu
                        call mf6lab2cm(awr,awi,awp,lct,eu(i),epu(j),uu(k),tp,w,dinv)
                        write(nou,'(1p7e16.8)')eu(i),epu(j),uu(k),tp,w,f6dis(i,j,k),f6con(i,j,k)
                      enddo
                    enddo
                  enddo
                  deallocate(ep1,b1,ep2,b2)
                  e1=e2
                  nd1=nd2
                  na1=na2
                  nw1=nw2
                  nep1=nep2
                  do ll=1,nw2
                    b01(ll)=b02(ll)
                  enddo
                enddo
              elseif (law.eq.2) then
                call readtab2(nin,c1,c2,n1,n2,nr,ne,nbt,ibt)
                write(nou,'(a,i3,a,i5)')' LAW=',law,' lct=',lct,' NE=',ne
                write(nou,'(110a1)')('=',i=1,110)
                call readlist(nin,c1,e1,lang1,n2,nw1,nl1,b01)
                do ie=2,ne
                  call readlist(nin,c1,e2,lang2,n2,nw2,nl2,b02)
                  if (e1.ne.e2.and.lang1.eq.lang2) then
                    lei=intlaw(ie,nbt,ibt,nr)
                    write(nou,'(a,i5,a,1p,e13.6,a,e13.6,a,i3)')' IE=',ie-1,' E1=',e1,' E2=',e2,' LEIN=',lei
                    write(nou,'(6(a,i4))')' lang=',lang1,' nl1=',nl1,' nl2=',nl2
                    write(nou,'(110a1)')('-',i=1,110)
                    e0=(awr+awi)/awr*(-q)*1.00001d0
                    if (e1.gt.e0) e0=e1*1.00001d0
                    h=(e2-e0)/dble(neu-1)
                    do i=1,neu
                      eu(i)=e0+h*dble(i-1)
                    enddo
                    rth=(awr+awi)/awr*q/e0
                    if (awi*awp.ne.0.0d0) then
                      r2=awr*(awr+awi-awp)/(awi*awp)*(1.0d0+rth)
                    else
                      r2=1.0d38
                    endif
                    if (r2.gt.1.0d0) then
                       umin=-1.0d0
                    else
                       umin=sqrt(1.0d0-r2)+1.0d-5
                    endif
                    h=(1.0d0-umin)/(nuu-1)
                    do i=1,nuu
                      uu(i)=umin+h*dble(i-1)
                    enddo
                    if (uu(1).lt.-1.0d0) uu(1)=-1.0d0
                    if (uu(nuu).gt.1.0d0)uu(nuu)=1.0d0
                    call mf6_get_law2(awr,awi,awp,q,lct,lang1,e1,b01,nl1,e2,b02,nl2,lei,eu,neu,uu,nuu,f64)
                    write(nou,'(4a16)')'ei','u','w','f6law2'
                    do i=1,neu
                      do k=1,nuu
                        call mf4lab2cm(lct,awr,awi,awp,q,eu(i),uu(k),w,dinv)
                        write(nou,'(1p4e16.8)')eu(i),uu(k),w,f64(i,k)
                      enddo
                    enddo
                  endif
                  e1=e2
                  lang1=lang2
                  nw1=nw2
                  nl1=nl2
                  do ll=1,nw2
                    b01(ll)=b02(ll)
                  enddo
                enddo
              elseif(law.eq.5) then
                call readtab2(nin,spi,c2,lidp,n2,nr,ne,nbt,ibt)
                write(nou,'(a,i3,a,i3,a,f5.1,a,i5)')' LAW=',law,' lct=',lct,' spi=',spi,' NE=',ne
                write(nou,'(110a1)')('=',i=1,110)
                call readlist(nin,c1,e1,ltp,n2,nw1,nl1,b01)
                do ie=2,ne
                  call readlist(nin,c1,e2,ltp2,n2,nw2,nl2,b02)
                  if (e1.ne.e2.and.ltp.eq.ltp2) then
                    lei=intlaw(ie,nbt,ibt,nr)
                    write(nou,'(a,i5,a,1p,e13.6,a,e13.6,a,i3)')' IE=',ie-1,' E1=',e1,' E2=',e2,' LEIN=',lei
                    write(nou,'(6(a,i4))')' LTP=',ltp,' lidp=',lidp,' nl1=',nl1,' nl2=',nl2
                    write(nou,'(110a1)')('-',i=1,110)
                    e0=1.0d-5
                    if (e1.gt.e0) e0=e1*1.000001d0
                    h=(e2-e0)/dble(neu-1)
                    do i=1,neu
                      eu(i)=e0+h*dble(i-1)
                    enddo
                    if (awi*awp.ne.0.0d0) then
                      r2=awr*awr/(awp*awp)
                    else
                      r2=1.0d38
                    endif
                    if (r2.gt.1.0d0) then
                       umin=-9.999999999999999d-1
                    else
                       umin=sqrt(1.0d0-r2)+1.0d-5
                    endif
                    h=(1.0d0-umin)/(nuu-1)
                    do i=1,nuu
                      uu(i)=umin+h*dble(i-1)
                    enddo
                    if (uu(1).le.-1.0d0) uu(1)=-9.999999999999999d-1
                    if (uu(nuu).ge.1.0d0)uu(nuu)=9.999999999999999d-1
                    call  mf6_get_law5(za,awr,zap,awp,spi,lidp,lei,ltp, &
                          e1,nl1,b01,e2,nl2,b02,eni,sni,npsni,nbtsni,ibtsni,nrsni, &
                          eu,neu,uu,nuu,f64)
                    write(nou,'(4a16)')'ei','u','w','f6law5'
                    do i=1,neu
                      do k=1,nuu
                        call mf4lab2cm(lct,awr,awi,awp,q,eu(i),uu(k),w,dinv)
                        if (w.ge.1.0d0) then
                          w=9.999999999999999d-1
                        elseif (w.le.-1.0d0) then
                          if (lidp.eq.1) then
                            w=-9.999999999999999d-1
                          else
                            w=-1.0d0
                          endif
                        endif
                        write(nou,'(1p4e16.8)')eu(i),uu(k),w,f64(i,k)
                      enddo
                    enddo
                  endif
                  e1=e2
                  ltp=ltp2
                  nw1=nw2
                  nl1=nl2
                  do ll=1,nw2
                    b01(ll)=b02(ll)
                  enddo
                enddo
              elseif(law.eq.6) then
                call readcont(nin,apsx,c1,l1,l2,n1,npsx,mat,mf,mt,nsi)
                write(nou,'(a,i2,a,1pe13.6,a,i2)')' LAW=',law,' APSX=',apsx,' NPSX=',npsx
                write(nou,'(110a1)')('=',i=1,110)
                 if (q.lt.0.0d0) then
                   e0=(awr+awi)/awr*(-q)*1.00001d0
                 else
                   e0=1.0d-5
                 endif
                 e2=emax
                 h=(e2-e0)/dble(neu-1)
                 do i=1,neu
                   eu(i)=e0+h*dble(i-1)
                 enddo
                 tp1=1.0d-5
                 tp2=min(e2-e0,(e0+e2)/2.0d0)
                 h=(tp2-tp1)/(nepu-1)
                 do i=1,nepu
                    epu(i)=tp1+h*dble(i-1)
                 enddo
                 h=1.98d0/dble(nuu-1)
                 do i=1,nuu
                   uu(i)=-0.99+h*dble(i-1)
                 enddo
                 call mf6_get_law6(awr,awi,awp,q,apsx,npsx,eu,neu,epu,nepu,uu,nuu,f6con)
                 write(nou,'(4a16)')'ei','ep','u','f6law6'
                 do i=1,neu
                   do j=1,nepu
                     do k=1,nuu
                       write(nou,'(1p4e16.8)')eu(i),epu(j),uu(k),f6con(i,j,k)
                     enddo
                   enddo
                 enddo
              elseif (law.eq.7) then
                call readmf6_law7_lab(nin,nedim,ne,e,ile,nadim,nmu,xu,ilmu,nepdim,nep,ep,f,ilef)
                write(nou,'(a,i2,a,i4,a,i4,a,i4)')' LAW=',law,' ne=',ne,' numax=',maxval(nmu),' nepmax=',maxval(nep)
                write(nou,'(110a1)')('=',i=1,110)
                write(nou,'(4a16)')'ei','ep','u','f6law7'
                do i=1,ne-1
                   ip=i+1
                   e1=e(i)
                   e2=e(ip)
                   lei=ile(i)
                   e0=(awr+awi)/awr*(-q)*1.000001d0
                   if (e1.gt.e0) e0=e1*1.000001d0
                   he=(e2-e0)/dble(neu-1)
                   do ie=1,neu
                      eu(ie)=e0+he*dble(ie-1)
                      nmu1=nmu(i)
                      nmu2=nmu(ip)
                      umin=max(xu(i,1),xu(ip,1))
                      umax=min(xu(i,nmu1),xu(ip,nmu2))
                      hu=(umax-umin)/dble(nuu-1)
                      do ju=1,nuu
                        uu(ju)=umin+hu*dble(ju-1)
                        do j=1,nmu1
                          x(j)=xu(i,j)
                        enddo
                        i12=ihigh(uu(ju),x,1,nmu1)
                        i11=i12-1
                        u11=xu(i,i11)
                        u12=xu(i,i12)
                        lmu1=ilmu(i,i11)
                        nep11=nep(i,i11)
                        nep12=nep(i,i12)
                        do k=1,nep11
                          ep11(k)=ep(i,i11,k)
                          f11(k)=f(i,i11,k)
                          llaw(k)=ilef(i,i11,k)
                        enddo
                        call packibt(nep11,llaw,nr11,nbt11,ibt11)
                        do k=1,nep12
                          ep12(k)=ep(i,i12,k)
                          f12(k)=f(i,i12,k)
                          llaw(k)=ilef(i,i12,k)
                        enddo
                        call packibt(nep12,llaw,nr12,nbt12,ibt12)
                        do j=1,nmu2
                          x(j)=xu(ip,j)
                        enddo
                        i22=ihigh(uu(ju),x,1,nmu2)
                        i21=i22-1
                        u21=xu(ip,i21)
                        u22=xu(ip,i22)
                        lmu2=ilmu(ip,i21)
                        nep21=nep(ip,i21)
                        nep22=nep(ip,i22)
                        do k=1,nep21
                          ep21(k)=ep(ip,i21,k)
                          f21(k)=f(ip,i21,k)
                          llaw(k)=ilef(ip,i21,k)
                        enddo
                        call packibt(nep21,llaw,nr21,nbt21,ibt21)
                        do k=1,nep22
                          ep22(k)=ep(ip,i22,k)
                          f22(k)=f(ip,i22,k)
                          llaw(k)=ilef(ip,i22,k)
                        enddo
                        call packibt(nep22,llaw,nr22,nbt22,ibt22)
                        epmin=max(ep11(1),ep12(1),ep21(1),ep22(1),5.0d-6)
                        epmax=min(ep11(nep11),ep12(nep12),ep21(nep21),ep22(nep22))
                        hep=(epmax-epmin)/dble(nepu-1)
                        do je=1,nepu
                          epu(je)=epmin+hep*dble(je-1)
                        enddo
                        ii=1
                        jj=1
                        call mf6_get_law7(eu(ie),ii,epu,nepu,uu(ju),jj,lei, &
                                  e1,lmu1,u11,ep11,f11,nep11,nbt11,ibt11,nr11, &
                                          u12,ep12,f12,nep12,nbt12,ibt12,nr12, &
                                  e2,lmu2,u21,ep21,f21,nep21,nbt21,ibt21,nr21, &
                                          u22,ep22,f22,nep22,nbt22,ibt22,nr22,f67)
                        do je=1,nepu
                          write(nou,'(1p4e16.8)')eu(ie),epu(je),uu(ju),f67(ii,je,jj)
                        enddo
                      enddo
                   enddo
                enddo
              else
                write(nou,'(a,i2)')' LAW=',law
                if (law.eq.0) then
                  write(nou,*)' no distribution, just yields are given'
                elseif (law.eq.3) then
                  write(nou,*)' Isotropic distribution at all incident energies'
                elseif (law.eq.4) then
                  write(nou,*)' Discrete two-body recoil distribution'
                endif
                write(nou,'(110a1)')('=',i=1,110)
                call nextsub6(nin,law,nbt,ibt,x,y)
              endif
            enddo
          else
            if (jp.ne.0) then
              write(nou,*)' Section MT=',mt,' with JP= ',jp,' was skipped'
              write(*,*)' Section MT=',mt,' with JP= ',jp,' was skipped'
            elseif (lct0.eq.4) then
              write(nou,*)' Section MT=',mt,' with LCT= ',lct0,' was skipped'
              write(*,*)' Section MT=',mt,' with LCT= ',lct0,' was skipped'
            endif
          endif
        endif
      enddo ! end processing MF6
      end
!==============================================================================
!      General routines for ENDF-6 formatted files
!==============================================================================
      subroutine findnextmat(nin)
!
!     Find next material from cursor position
!
      character*66 line
      mat=10000
      do while (mat.ne.0.and.mat.ne.-1)
        read(nin,'(a66,i4,i2,i3,i5)')line,mat,mf,mt,ns
      enddo
      if (mat.eq.-1) backspace nin
      end
! ----------------------------------------------------------------------
      subroutine findmat(nin,mat,icod)
!
!      Find material mat on endf6 formatted tape
!      on return if icod=0, material found
!                if icod=1, material not found
!
      character*66 line
      read(nin,'(a66,i4,i2,i3,i5)',iostat=iosnin)line,mat0,mf,mt,ns
      if (iosnin.lt.0.or.mat0.eq.-1.or.mat0.ge.mat) then
        rewind(nin)
        read(nin,*)
        mat0=0
      elseif (iosnin.gt.0) then
        icod=1
        return
      elseif (mat0.eq.0) then
        backspace nin
        backspace nin
        read(nin,'(a66,i4,i2,i3,i5)',err=10,end=10)line,mat0,mf,mt,ns
        if (mat0.ge.mat) then
          rewind(nin)
          read(nin,*)
          mat0=0
        endif
      endif
      do while (mat0.lt.mat.and.mat0.ne.-1)
        read(nin,'(a66,i4,i2,i3,i5)',err=10,end=10)line,mat0,mf,mt,ns
      enddo
      if (mat0.eq.mat)then
        icod=0
        backspace nin
      else
        icod=1
      endif
      return
   10 icod=1
      return
      end
!==============================================================================
      subroutine findmf(nin,mat,mf,icod)
!
!       Find file mf for material mat on endf6 formatted tape
!       on return if icod=0, mat/mf found
!                 if icod=1, mat/mf not found
!
      character*66 line
      read(nin,'(a66,i4,i2,i3,i5)',iostat=iosnin)line,mat0,mf0,mt,ns
      if (mat0.eq.0) then
        read(nin,'(a66,i4,i2,i3,i5)',iostat=iosnin)line,mat0,mf0,mt,ns
      endif
      if (iosnin.ne.0.or.mat0.eq.-1.or.mat0.gt.mat.or. &
        (mat0.eq.mat.and.mf0.gt.mf)) then
        call findmat(nin,mat,icod)
      elseif (mat0.eq.mat) then
        if (mf0.eq.0.or.mf0.eq.mf) then
          backspace nin
          backspace nin
          read(nin,'(a66,i4,i2,i3,i5)',err=10,end=10)line,mat0,mf0,mt,ns
          if (mf0.ge.mf) then
            call findmat(nin,mat,icod)
          else
            icod=0
          endif
        else
          icod=0
        endif
      elseif (mat0.lt.mat) then
        do while (mat0.lt.mat.and.mat0.ne.-1)
          read(nin,'(a66,i4,i2,i3,i5)',err=10,end=10)line,mat0,mf0,mt,ns
        enddo
        if (mat0.eq.mat)then
          icod=0
          backspace nin
        else
          icod=1
        endif
      endif
      if (icod.eq.0) then
        mat0=mat
        mf0=-1
        do while (mat0.eq.mat.and.mf0.lt.mf)
          read(nin,'(a66,i4,i2,i3,i5)',err=10,end=10)line,mat0,mf0,mt,ns
        enddo
        if (mat0.eq.mat.and.mf0.eq.mf) then
          icod=0
          backspace nin
        else
          icod=1
        endif
      endif
      return
   10 icod=1
      return
      end
!==============================================================================
      subroutine findmt(nin,mat,mf,mt,icod)
!
!      Find reaction mt on file mf for material mat on endf6 formatted
!      tape, on return if icod=0, mat/mf/mt found
!                      if icod=1, mat/mf/mt not found
!
      character*66 line
      read(nin,'(a66,i4,i2,i3,i5)',iostat=iosnin)line,mat0,mf0,mt0,ns
      if (iosnin.ne.0.or.mat0.ne.mat.or. &
        (mat0.eq.mat.and.mf0.ne.mf).or. &
        (mat0.eq.mat.and.mf0.eq.mf.and.mt0.gt.mt)) then
        call findmf(nin,mat,mf,icod)
      elseif (mat0.eq.mat.and.mf0.eq.mf) then
        if (mt0.eq.0.or.mt0.eq.mt) then
         backspace nin
         backspace nin
         read(nin,'(a66,i4,i2,i3,i5)',err=10,end=10)line,mat0,mf0,mt0,ns
         if (mt0.ge.mt) then
           call findmf(nin,mat,mf,icod)
         else
           icod=0
         endif
        else
         icod=0
        endif
      endif
      if (icod.eq.0) then
        mf0=mf
        mt0=-1
        do while (mt0.lt.mt.and.mf0.eq.mf)
         read(nin,'(a66,i4,i2,i3,i5)',err=10,end=10)line,mat0,mf0,mt0,ns
        enddo
        if (mf0.eq.mf.and.mt0.eq.mt) then
          icod=0
          backspace nin
        else
          icod=1
        endif
      endif
      return
   10 icod=1
      return
      end
!==============================================================================
      subroutine findnextmt(nin,mf,mt)
!
!     find next mt on mf file for current material
!
      character*66 line
      mt0=-1
      do while (mt0.ne.0)
        read(nin,'(a66,i4,i2,i3,i5)',err=10,end=10)line,mat0,mf0,mt0,ns
      enddo
      read(nin,'(a66,i4,i2,i3,i5)',err=10,end=10)line,mat0,mf0,mt,ns
      if (mt.ne.0.and.mf0.eq.mf) then
        backspace nin
      else
        mt=-1
      endif
      return
   10 mt=-2
      return
      end
!==============================================================================
      subroutine nextsub6(nin,law,nbt,ibt,x,b)
!
!     find next subsection on MF6 section
!
      implicit real*8 (a-h,o-z)
      dimension nbt(*),ibt(*),x(*),b(*)
      if (law.eq.1.or.law.eq.2.or.law.eq.5) then
        call readtab2(nin,c1,c2,l1,l2,n1,ne,nbt,ibt)
        do i=1,ne
          call readlist(nin,c1,c2,l1,l2,n1,n2,b)
        enddo
      elseif (law.eq.6) then
        call readcont(nin,c1,c2,l1,l2,n1,n2,mat,mf,mt,ns)
      elseif (law.eq.7) then
        call readtab2(nin,c1,c2,l1,l2,n1,ne,nbt,ibt)
        do i=1,ne
          call readtab2(nin,c1,c2,l1,l2,n1,nmu,nbt,ibt)
          do j=1,nmu
            call readtab1(nin,c1,c2,l1,l2,n1,n2,nbt,ibt,x,b)
          enddo
        enddo
      elseif (law.lt.0.or.law.gt.7) then
       write(*,*)' ERROR: unknown LAW=',law,' on MF6'
       stop
      endif
      return
      end
      subroutine readtext(nin,line,mat,mf,mt,ns)
!
!     read a TEXT record
!
      character*66 line
      read(nin,'(a66,i4,i2,i3,i5)')line,mat,mf,mt,ns
      return
      end
!==============================================================================
      subroutine readcont(nin,c1,c2,l1,l2,n1,n2,mat,mf,mt,ns)
!
!     read a CONT record
!
      implicit real*8 (a-h, o-z)
      read(nin,'(2e11.0,4i11,i4,i2,i3,i5)')c1,c2,l1,l2,n1,n2,mat,mf,mt,ns
      return
      end
!==============================================================================
      subroutine readlist(nin,c1,c2,l1,l2,npl,n2,b)
!
!     read a LIST record
!
      implicit real*8 (a-h, o-z)
      dimension b(*)
      read(nin,'(2e11.0,4i11,i4,i2,i3,i5)')c1,c2,l1,l2,npl,n2,mat,mf,mt,ns
      read(nin,'(6e11.0)')(b(n),n=1,npl)
      return
      end
!==============================================================================
      subroutine readtab1(nin,c1,c2,l1,l2,nr,np,nbt,intp,x,y)
!
!     read TAB1 record
!
      implicit real*8 (a-h, o-z)
      dimension nbt(*),intp(*),x(*), y(*)
      read(nin,'(2e11.0,4i11,i4,i2,i3,i5)')c1,c2,l1,l2,nr,np,mat,mf,mt,ns
      read(nin,'(6i11)')(nbt(n),intp(n),n=1,nr)
      read(nin,'(6e11.0)')(x(n),y(n),n=1,np)
      return
      end
!==============================================================================
      subroutine readtab2(nin,c1,z,l1,l2,nr,nz,nbt,intp)
!
!     read TAB2 record
!
      implicit real*8 (a-h, o-z)
      dimension nbt(*),intp(*)
      read(nin,'(2e11.0,4i11,i4,i2,i3,i5)')c1,z,l1,l2,nr,nz,mat,mf,mt,ns
      read(nin,'(6i11)')(nbt(n),intp(n),n=1,nr)
      return
      end
!-------------------------------------------------------------------------------
      function intlaw(i,nbt,ibt,nr)
!
!     return the interpolation law for the interval (x(i-1),x(i)]
!
      dimension nbt(*),ibt(*)
      j=1
      do while(nbt(j).lt.i.and.j.le.nr)
       j=j+1
      enddo
      if (j.le.nr) then
        intlaw=ibt(j)
      else
        intlaw=ibt(nr)
      endif
      return
      end
! =====================================================================================================
! subroutine readmf6_law7_lab: Read one MF6 subsection for law=7 (Laboratory Angle-Energy Distribution)
! =====================================================================================================
!
!    nedim: maximum number of incident (integer)
!       ne: actual number of incident energy points for law7 (integer)
!        e: incident energy points
!           real*8 array e=[e(i), i=1..ne]
!      ile: incident energy interpolation law by intervals
!           integer array ile=[ile(i), i=1..ne-1]
!   nadim: maximum number of outgoing cosines (integer)
!      nmu: actual number of tabulated outgoing cosines
!           integer array nmu=[nmu(i), i=1..ne],  1 < nmu(i) <= nedim
!      xmu: cosines values by incident energy (ranges from -1.0 to 1.0)
!           real*8 array xmu=[xmu(i,j), i=1..ne, j=1..nmu(i) ]
!     ilmu: cosine interpolation law by interval
!           real*8 array ilmu=[ilmu(i,j), i=1..ne, j=1..nmu(i)-1 ]
!   nepdim: maximum number of outgoing energies (integer)
!      nep: Number of secondary energies by cosine and incident energy
!           integer array nep=[nep(i,j), i=1..ne, j=1..nmu(i) ]
!       ep: Secondary/outgoing energies
!           real*8 array ep=[ep(i,j,k), i=1..ne, j=1..nmu(i), k=1..nep(j,i)]
!        f: Angle-energy distribution
!           real*8 array f=[f(i,j,k), ]
!     ilef: secondary energy interpolation law
!           integer array ilef=[ilef(i,j,k),i=1..ne, j=1..nmu(i), k=1..nep(j,i)]
!
  subroutine readmf6_law7_lab(nin,nedim,ne,e,ile,nadim,nmu,xmu,ilmu,nepdim,nep,ep,f,ilef)
    implicit real*8 (a-h,o-z)
    parameter (nrmax=20)
!   external arrays
    dimension e(*),ile(*),nmu(*),xmu(nedim,*),ilmu(nedim,*),nep(nedim,*)
    dimension ep(nedim,nadim,*),f(nedim,nadim,*),ilef(nedim,nadim,*)
!   internal arrays
    dimension nbt(nrmax),ibt(nrmax),jbtep(nepdim),jbtna(nadim)
    dimension x1(nepdim),y1(nepdim)
    call readtab2(nin,c1,c2,l1,l2,nr,ne,nbt,ibt)
    if (ne.gt.nedim) then
      write(*,*)' Fatal Error: too many incident energies for MF6/LAW7'
      write(*,*)' ne=',ne,' nemax=',nedim
      stop
    endif
    call unpackibt(nr,nbt,ibt,ne,ile)
    do i=1,ne
      call readtab2(nin,c1,e(i),l1,l2,nr,nmui,nbt,ibt)
      if (nmui.gt.nadim) then
         write(*,*)' Fatal Error: too many outgoing cosine for MF6/LAW7'
         write(*,*)' nmu=',nmui,' nmumax=',nadim
        stop
      endif
      nmu(i)=nmui
      call unpackibt(nr,nbt,ibt,nmui,jbtna)
      do j=1,nmui-1
        ilmu(i,j)=jbtna(j)
      enddo
      do j=1,nmui
        call readtab1(nin,c1,xmuij,l1,l2,nr,nepij,nbt,ibt,x1,y1)
        if (nepij.gt.nepdim) then
          write(*,*)' Fatal Error: too many outgoing energies for MF6/LAW7'
          write(*,*)' nep=',nepij,' nepmax=',nepdim
          stop
        endif
        xmu(i,j)=xmuij
        nep(i,j)=nepij
        call unpackibt(nr,nbt,ibt,nepij,jbtep)
        do k=1,nepij-1
          ep(i,j,k)=x1(k)
          f(i,j,k)=y1(k)
          ilef(i,j,k)=jbtep(k)
        enddo
        ep(i,j,nepij)=x1(nepij)
        f(i,j,nepij)=y1(nepij)
      enddo
    enddo
    return
  end subroutine readmf6_law7_lab
!
! ==============================================================================================================
! subroutine unpackibt: Unpack interpolation table from TAB1 and TAB2 records
! ==============================================================================================================
!   nr,nbt,ibt = Packed interpolation table from TAB1 and TAB2 records (nr > 20 is not allowed in ENDF-6 format)
!   np         = np number of points of the X-Y table (np-1=number of intervals)
!   ibtu       = array containing the interpolation law between [x(i),x(i+1)]
!
  subroutine unpackibt(nr,nbt,ibt,np,ibtu)
    parameter (nrmax=20)
    dimension nbt(*),ibt(*),ibtu(*)
    if (nr.gt.nrmax) then
      write(*,*)' Error: Too many ENDF-6 interpolation ranges nr=',nr,' nrmax=',nrmax
      stop
    else
      jf=0
      do i=1,nr
        j0=jf+1
        jf=nbt(i)-1
        ilaw=ibt(i)
        do j=j0,jf
          ibtu(j)=ilaw
        enddo
      enddo
      np1=np-1
      if (jf.lt.np1) then
        write(*,*)' Warning: Interpolation law could be incorrent NR<(NP-1) ',nr,' < ',np1
        do i=jf+1,np1
          ibt(i)=ibt(jf)
        enddo
      elseif (jf.gt.np1) then
        write(*,*)' Warning: Interpolation law could be incorrent NR>(NP-1) ',nr,' > ',np1
      endif
      return
    endif
  end subroutine unpackibt
!
! =================================================================================================================================
! subroutine packibt: Pack interpolation table for TAB1 and TAB2 records
! =================================================================================================================================
!   np         = number of points of the X-Y table (np-1=number of intervals)
!   ibtu       = array containing the interpolation law between [x(i),x(i+1)]
!   nr,nbt,ibt = Packed interpolation table for TAB1 and TAB2 records
!
  subroutine packibt(np,ibtu,nr,nbt,ibt)
    parameter (nrmax=20)
    dimension ibtu(*),nbt(*),ibt(*)
    nr=0
    ilaw=ibtu(1)
    nru=np-1
    do i=2,nru
      if (ilaw.ne.ibtu(i)) then
        nr=nr+1
        nbt(nr)=i
        ibt(nr)=ilaw
        ilaw=ibtu(i)
      endif
    enddo
    nr=nr+1
    nbt(nr)=np
    ibt(nr)=ibtu(nru)
    if (nr.gt.nrmax) then
      write(*,*)' Error: Too many interpolation ranges nr=',nr,' nrmax=',nrmax
      stop
    endif
    return
  end subroutine packibt
! ==================================================================================================================================