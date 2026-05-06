program testlaw
implicit real*8 (a-h, o-z)
parameter (nx0=5,ndim=2048+1)
dimension x(ndim),y(ndim),u(ndim),yint(ndim),uint(ndim),ulin(ndim-1)
dimension ylint(nx0),ulint(nx0),ullin(nx0),ultot(nx0)
! input data
x0=1.0d-5
xmax=20.0d+6
epy1=0.01d0
epy2=0.02d0
epsy=0.001d0
epsx=1.0d-10
sigth=1.0d0
xth=0.025d0
x1=x0
y1=sigth*sqrt(xth/x1)
u1=epy1*y1
x2=xmax
y2=sigth*sqrt(xth/x2)
u2=epy2*y2
intlaw=5
! intlaw=4
! intlaw=3
! intlaw=2
! intlaw=1
call tabint(x1,y1,x2,y2,u1,u2,intlaw,yint0,uint0)
write(*,*)' Integral 1 interval. Interpolation law:',intlaw,' hx=',x2-x1
write(*,'(1p,a6,3e20.11)')' total',yint0,uint0,uint0/yint0
write(*,*)
nx=nx0 ! number of subintervals (npoints=nx0+1)
hx=(xmax-x0)/float(nx)
write(*,*)' Integral multiple intervals. Interpolation law:',intlaw,' hx=',hx
write(*,'(a6,3a20)')' i','yint','abs. dev.',' rel. dev.'
xx0=x0
yy0=yinterp(x1,y1,x2,y2,intlaw,xx0)
uu0=absdev(x1,y1,u1,x2,y2,u2,intlaw,xx0,yy0)
sumyint=0.0d0
sumu=0.0d0
do i=1,nx
  xx1=xx0+hx
  yy1=yinterp(x1,y1,x2,y2,intlaw,xx1)
  uu1=absdev(x1,y1,u1,x2,y2,u2,intlaw,xx1,yy1)
  call tabint(xx0,yy0,xx1,yy1,uu0,uu1,intlaw,yint(i),uint(i))
  sumyint=sumyint+yint(i)
  sumu=sumu+uint(i)
  write(*,'(i6,1p,3e20.11)')i,yint(i),uint(i),uint(i)/yint(i)
  xx0=xx1
  yy0=yy1
  uu0=uu1
enddo
write(*,'(a6,1p,3e20.11)')' total',sumyint,sumu,sumu/sumyint
write(*,*)
! linearized data law=2
law2=2
write(*,*)' Linearized data by interval. Interpolation law:',law2,' hx=',hx
xx0=x0
yy0=yinterp(x1,y1,x2,y2,intlaw,xx0)
uu0=absdev(x1,y1,u1,x2,y2,u2,intlaw,xx0,yy0)
do i=1,nx
  xx1=xx0+hx
  yy1=yinterp(x1,y1,x2,y2,intlaw,xx1)
  uu1=absdev(x1,y1,u1,x2,y2,u2,intlaw,xx1,yy1)
  call tablin(xx0,yy0,xx1,yy1,uu0,uu1,intlaw,epsy,epsx,ndim,x,y,u,ulin,n)
  sumylin=0.0d0
  sumulin=0.0d0
  sumuint=0.0d0
  sumutot=0.0d0
  write(*,*)' Interval=',i,' np=',n,'nj=',n-1
  write(*,'(a6,11a20)')'j','x(j)','y(j)','rel. u(j)','x(j+1)','y(j+1)','rel. u(j+1)','yint(j)','abs. dev(j)','rel. dev(j)','abs. devlin(j)','rel. devlin(j)'
  do j=1,n-1
    call tabint(x(j),y(j),x(j+1),y(j+1),u(j),u(j+1),law2,yint(j),uint(j))
    write(*,'(i6,1p,11e20.11)')j,x(j),y(j),u(j)/y(j),x(j+1),y(j+1),u(j+1)/y(j+1),yint(j),uint(j),uint(j)/yint(j),yint(j)*ulin(j),ulin(j)
    sumylin=sumylin+yint(j)
    uintlin=abs(yint(j)*ulin(j))
    sumulin=sumulin+uintlin
    sumuint=sumuint+uint(j)
    sumutot=sumutot+uint(j)+uintlin
  enddo
  ylint(i)=sumylin
  ullin(i)=sumulin
  ulint(i)=sumuint
  ultot(i)=sumutot
  xx0=xx1
  yy0=yy1
  uu0=uu1
enddo
write(*,*)
write(*,*)' Integral multiple intervals. Interpolation law:',law2,' hx=',hx
write(*,'(a6,7a20)')' i','yint','abs. tot. dev.',' rel. tot. dev.','abs. unc. dev.',' rel. unc. dev.','abs. lin. dev.',' rel. lin. dev.'
sumyint=0.0d0
sumulin=0.0d0
sumuint=0.0d0
sumutot=0.0d0
do i=1,nx
  write(*,'(i6,1p,7e20.11)')i,ylint(i),ultot(i),ultot(i)/ylint(i),ulint(i),ulint(i)/ylint(i),ullin(i),ullin(i)/ylint(i)
  sumyint=sumyint+ylint(i)
  sumulin=sumulin+ullin(i)
  sumuint=sumuint+ulint(i)
  sumutot=sumutot+ullin(i)+ulint(i)
enddo
sumu2=sqrt(sumu2)
write(*,'(a6,1p,7e20.11)')' total',sumyint,sumutot,sumutot/sumyint,sumuint,sumuint/sumyint,sumulin,sumulin/sumyint
stop
end
!
