program test12
implicit real*8 (a-h, o-z)
character*66 line,fin
parameter (maxlevel=50,maxnk=5500,nbmax=2000000)
dimension mf3(450),qm(450),qi(450)
dimension esi(maxlevel),tp(maxlevel),gp(maxlevel)
dimension es(maxnk),eg(maxnk),y(maxnk)
dimension ee(maxlevel),r(maxlevel,maxlevel),a(maxlevel,maxlevel)
dimension b(nbmax)
data nin/1/
write(*,*)' Input endf-6 formatted file:'
read(*,*)fin
write(*,*)fin
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
call readcont(nin,elis,sta,l1,l2,n1,n2,mat,mf,mt,ns)
call readcont(nin,awi,emax,l1,l2,nsub,n2,mat,mf,mt,ns)
izai=nsub/10
zai=dble(izai)
itype=mod(nsub,10)
write(*,*)
if (itype.ne.0) then
   write(*,*)' Fatal error: Sublibrary not allowed NSUB=',nsub
   close(nin)
   stop
endif
write(*,*)' NSUB=',nsub,' IPART=',izai,' itype=',itype
write(*,*)' EMAX=',emax,' AWI=',awi,' ELIS=',elis
write(*,*)' MAT ',mati
nmf3=0
call findmf(nin,mati,3,icod)
if (icod.eq.0) then
  backspace(nin)
  mt=1000
  do while (mt.gt.0)
    call findnextmt(nin,3,mt)
    if ((mt.gt.50.and.mt.lt.91).or.(mt.gt.600.and.mt.lt.649).or.(mt.gt.650.and.mt.lt.699).or. &
        (mt.gt.700.and.mt.lt.749).or.(mt.gt.750.and.mt.lt.799).or.(mt.gt.800.and.mt.lt.850)) then
      nmf3=nmf3+1
      mf3(nmf3)=mt
    endif
  enddo
  write(*,*)' Sections found on MF3: ',nmf3
else
  write(*,*)' MF3 not found'
  close(nin)
  stop
endif
rewind(nin)
call readtext(nin,line,mat,mf,mt,nsi)
call findmf(nin,mati,3,icod)
i50=0
n50=0
i600=0
n600=0
i650=0
n650=0
i700=0
n700=0
i750=0
n750=0
i800=0
n800=0
do i=1,nmf3
  call findmt(nin,mati,3,mf3(i),icod)
  call readcont(nin,za,awr,l1,l2,n1,n2,mat,mf,mt,nsi)
  call readcont(nin,qm(i),qi(i),l1,l2,n1,n2,mat,mf,mt,nsi)
  if (mt.gt.50.and.mt.lt.91) then
    if (i50.eq.0) i50=i
    n50=n50+1
  elseif (mt.gt.600.and.mt.lt.649) then
    if (i600.eq.0) i600=i
    n600=n600+1
  elseif (mt.gt.650.and.mt.lt.699) then
    if (i650.eq.0) i650=i
    n650=n650+1
  elseif (mt.gt.700.and.mt.lt.749) then
    if (i700.eq.0) i700=i
    n700=n700+1
  elseif (mt.gt.750.and.mt.lt.799) then
    if (i750.eq.0) i750=i
    n750=n750+1
  elseif (mt.gt.800.and.mt.lt.849) then
    if (i800.eq.0) i800=i
    n800=n800+1
  endif
enddo
write(*,*)
write(*,*)' MF3/MT QM and QI values'
do i=1,nmf3
   write(*,*)i, mf3(i), qm(i),qi(i)
enddo
call findmf(nin,mati,12,icod)
do ii=1,nmf3
  mt=mf3(ii)
  call findmt(nin,mati,12,mt,icod)   ! read MF12 data for reaction mt
  call readcont(nin,za,awr,l0,lg,ns,n2,matx,mfx,mtx,nsi)
  if (l0.eq.2) then                ! convert if l0=2 transition probabilities
    call readlist(nin,esns,c2,lp,l2,nw,nt,b) ! read MF12 data for l0=2
    ! data interpretation
    k=0
    do i=1,nt
      k=k+1
      esi(i)=b(k)
      k=k+1
      tp(i)=b(k)
      if (lg.eq.2) then
        k=k+1
        gp(i)=b(k)
      else
        gp(i)=1.0d0
      endif
    enddo
    ! initialization called if mt=mt0+1 (first excited level)
    if (mt.eq.51) then
      i0=i50
      nlevel=n50
      call init_trans2yield(elis,maxlevel,nlevel,qm(i0),qi(i0),ee,r,a)
    elseif (mt.eq.601) then
      i0=i600
      nlevel=n600
      call init_trans2yield(elis,maxlevel,nlevel,qm(i0),qi(i0),ee,r,a)
    elseif (mt.eq.651) then
      i0=i650
      nlevel=n650
      call init_trans2yield(elis,maxlevel,nlevel,qm(i0),qi(i0),ee,r,a)
    elseif (mt.eq.701) then
      i0=i700
      nlevel=n700
      call init_trans2yield(elis,maxlevel,nlevel,qm(i0),qi(i0),ee,r,a)
    elseif (mt.eq.751) then
      i0=i750
      nlevel=n750
      call init_trans2yield(elis,maxlevel,nlevel,qm(i0),qi(i0),ee,r,a)
    elseif (mt.eq.801) then
      i0=i800
      nlevel=n800
      call init_trans2yield(elis,maxlevel,nlevel,qm(i0),qi(i0),ee,r,a)
    endif
    ! yields calculation for discrete reaction mt
    call trans2yield(mt,esns,nt,esi,tp,gp,maxlevel,ee,r,a,maxnk,nk,es,eg,y)
    ! printing yields
    write(*,*)
    write(*,*)' MT=',mt,' ESns=',esns,' LP=',lp,' NT=',nt,' NK=',nk
    write(*,'(a5,3a16)')' k',' es(k)',' eg(k)',' y(k)'
    sum=0.0d0
    do kk=1,nk
      write(*,'(i5,1p3e16.7)')kk,es(kk),eg(kk),y(kk)
      sum=sum+y(kk)
    enddo
    write(*,'(a5,32x,1pe16.7)')' sum',sum
  endif
enddo
stop
end
!
      subroutine readtext(nin,line,mat,mf,mt,ns)

      character*66 line
      read(nin,10)line,mat,mf,mt,ns
      return
   10 format(a66,i4,i2,i3,i5)
      end

      subroutine readcont(nin,c1,c2,l1,l2,n1,n2,mat,mf,mt,ns)

      implicit real*8 (a-h, o-z)
      read(nin,10)c1,c2,l1,l2,n1,n2,mat,mf,mt,ns
      return
   10 format(2e11.0,4i11,i4,i2,i3,i5)
      end

      subroutine readlist(nin,c1,c2,l1,l2,npl,n2,b)

      implicit real*8 (a-h, o-z)
      dimension b(*)
      read(nin,10)c1,c2,l1,l2,npl,n2,mat,mf,mt,ns
      read(nin,20)(b(n),n=1,npl)
      return
   10 format(2e11.0,4i11,i4,i2,i3,i5)
   20 format(6e11.0)
      end

      subroutine readtab1(nin,c1,c2,l1,l2,nr,np,nbt,intp,x,y)

      implicit real*8 (a-h, o-z)
      dimension nbt(*),intp(*),x(*), y(*)
      read(nin,10)c1,c2,l1,l2,nr,np,mat,mf,mt,ns
      read(nin,20)(nbt(n),intp(n),n=1,nr)
      read(nin,30)(x(n),y(n),n=1,np)
      return
   10 format(2e11.0,4i11,i4,i2,i3,i5)
   20 format(6i11)
   30 format(6e11.0)
      end

      subroutine findmat(nin,mat,icod)

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

      subroutine findmf(nin,mat,mf,icod)

      character*66 line
      read(nin,'(a66,i4,i2,i3,i5)',iostat=iosnin)line,mat0,mf0,mt,ns
      if (mat0.eq.0) then
        read(nin,'(a66,i4,i2,i3,i5)',iostat=iosnin)line,mat0,mf0,mt,ns
      endif
      if (iosnin.ne.0.or.mat0.eq.-1.or.mat0.gt.mat.or.(mat0.eq.mat.and.mf0.gt.mf)) then
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

      subroutine findmt(nin,mat,mf,mt,icod)

      character*66 line
      read(nin,'(a66,i4,i2,i3,i5)',iostat=iosnin)line,mat0,mf0,mt0,ns
      if (iosnin.ne.0.or.mat0.ne.mat.or.(mat0.eq.mat.and.mf0.ne.mf).or.&
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
      return
      end
