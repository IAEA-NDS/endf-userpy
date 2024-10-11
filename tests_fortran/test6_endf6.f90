      Program test6_endf6
      implicit real*8(a-h, o-z)
      parameter (nbmax=500000, nnx=450)
      parameter (neu=5,nepu=5,nuu=5)
      character*66 line
      character*120 fin,fout
      dimension nbt(20),ibt(20)
      dimension x(nbmax),y(nbmax)
      dimension b01(nbmax),b02(nbmax)
      dimension qi(nnx),mf6(nnx)
      dimension eu(neu),epu(nepu),uu(nuu)
      dimension f6dis(neu,nepu,nuu),f6con(neu,nepu,nuu)
      dimension f64(neu,nuu)
      allocatable ep1(:),b1(:,:),ep2(:),b2(:,:)
      data nin/3/,nou/12/
      data awi/1.0d0/,zai/1.0d0/,emax/2.0d7/ ! can be read or calculate from MF1/451
      write(*,*)' Input endf-6 formatted file:'
      read(*,*)fin
      write(*,*)' Read endf-6 formatted filename:',fin
      write(*,*)' Output file:'
      read(*,*)fout
      write(*,*)' Read output filename:',fout
      write(*,*)' MAT number:'
      read(*,*)mati
      write(*,*)' Read MAT number:',mati
      open(nin,file=fin)
      call readtext(nin,line,mat,mf,mt,nsi)
      call findmat(nin,mati,icod)
      if (icod.ne.0) then
        write(*,*)' MAT ',mati,' not found on tape ',fin
        close(nin)
        stop
      endif
      open(nou,file=fout)
      write(*,*)' MAT ',mati
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
        write(nou,*)' ',nmf6,' sections found on MF6'
      else
        write(nou,*)' MF6 not found'
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
          call readcont(nin,qm,qi(i),l1,lr,nr,np,mat,mf,mt,ns)
!         call readtab1(nin,qm,qi(i),l1,lr,nr,np,nbt,ibt,x,y)
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
            write(*,'(a,i4,a,i4,a,i5,a,1pe13.6)')' imt=',imt,' MT=',mt,' MAT=',mat,' QI=',q
            write(nou,*)
            write(nou,*)
            write(nou,'(a,i4,a,i4,a,i5,a,1pe13.6)')' imt=',imt,' MT=',mt,' MAT=',mat,' QI=',q
            write(nou,'(a,1p,e13.6,a,e13.6,a,i2,a,i4)')' ZA=',za,' AWR=',awr,' LCT=',lct0,' NK=',nk
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
              write(nou,'(110a1)')('*',i=1,110)
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
                  write(nou,'(7a15)')'ei','ep','u','tp','w','f6dis','f6con'
                  do i=1,neu
                    do j=1,nepu
                      do k=1,nuu
                        call mf6lab2cm(awr,awi,awp,lct,eu(i),epu(j),uu(k),tp,w,dinv)
                        write(nou,'(1p7e15.6)')eu(i),epu(j),uu(k),tp,w,f6dis(i,j,k),f6con(i,j,k)
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
                    If (uu(nuu).gt.1.0d0)uu(nuu)=1.0d0                                                          
                    call mf6_get_law2(awr,awi,awp,q,lct,lang1,e1,b01,nl1,e2,b02,nl2,lei,eu,neu,uu,nuu,f64)
                    write(nou,'(4a15)')'ei','u','w','f6law2'
                    do i=1,neu
                      do k=1,nuu
                        call mf4lab2cm(lct,awr,awi,awp,q,eu(i),uu(k),w,dinv)
                        write(nou,'(1p4e15.6)')eu(i),uu(k),w,f64(i,k)
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
                 write(nou,'(4a15)')'ei','ep','u','f6law6'
                 do i=1,neu
                   do j=1,nepu
                     do k=1,nuu
                       write(nou,'(1p4e15.6)')eu(i),epu(j),uu(k),f6con(i,j,k)
                     enddo
                   enddo
                 enddo                                               
              else
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
!-------------------------------------------------------------------------------
      function intlaw(i,nbt,ibt,nr)
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
!==============================================================================
