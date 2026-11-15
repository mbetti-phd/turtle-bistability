module system_equations

double precision :: PI = 4.0*atan(1.0)
complex :: I = (0.0, 1.0)

contains


subroutine vector_field(t,y,dy,n,K_sex,betta,k_pred,r,delta,c,m,phi,v,&
     & mu_mod)

implicit none

double precision :: P
double precision :: J
double precision :: A
double precision, intent(in) :: n
double precision, intent(in) :: K_sex
double precision, intent(in) :: betta
double precision, intent(in) :: k_pred
double precision, intent(in) :: r
double precision, intent(in) :: delta
double precision, intent(in) :: c
double precision, intent(in) :: m
double precision, intent(in) :: phi
double precision, intent(in) :: v
double precision, intent(in) :: mu_mod
double precision, intent(in) :: t
double precision, intent(in) :: y(3)
double precision, intent(inout) :: dy(3)


P = y(1)
J = y(2)
A = y(3)

dy(1) = J*P*phi/(J*v + 1.0) + P*betta/(P*k_pred + 1.0) - P*delta
dy(2) = A**2*c/(K_sex**2*(A**2*n/K_sex**2 + 1.0)) &
     & - J*P*r/(J*v + 1.0) - J*m
dy(3) = -A*mu_mod + J*m

end subroutine


end module


subroutine func(ndim,y,icp,args,ijac,dy,dfdu,dfdp)

use system_equations
implicit none
integer, intent(in) :: ndim, icp(*), ijac
double precision, intent(in) :: y(ndim), args(*)
double precision, intent(out) :: dy(ndim)
double precision, intent(inout) :: dfdu(ndim,ndim), dfdp(ndim,*)

call vector_field(args(14), y, dy, args(1), args(2), args(3), args(4), &
     & args(5), args(6), args(7), args(8), args(9), args(15), args(16))

if (ijac .eq. 0) return

dfdu(1,1) = -args(3)*args(4)*y(1)/(args(4)*y(1) + 1.0d0)&
     & **2 + args(3)/(args(4)*y(1) + 1.0d0) &
     & - args(6) + args(9)*y(2)/(args(15)* y(2) + 1.0d0)
dfdu(1,2) = -args(15)*args(9)*y(1)*y(2)/(args(15)* y(2) + 1.0d0)&
     & **2 + args(9)*y(1)/( args(15)*y(2) + 1.0d0)
dfdu(2,1) = -args(5)*y(2)/(args(15)*y(2) + 1.0d0)
dfdu(2,2) = args(15)*args(5)*y(1)*y(2)/(args(15)* y(2) + 1.0d0)**2 &
     & - args(5)*y(1)/( args(15)*y(2) + 1.0d0) - args(8)
dfdu(2,3) = -2*args(1)*args(7)*y(3)**3&
     & /(args(2)**4*( args(1)*y(3)**2/args(2)**2 &
     & + 1.0d0)**2) + 2* args(7)*y(3)&
     & /(args(2)**2*(args(1)* y(3)**2/args(2)**2 + 1.0d0))
dfdu(3,2) = args(8)
dfdu(3,3) = -args(16)

if (ijac .eq. 1) return

dfdp(1,3) = y(1)/(args(4)*y(1) + 1.0d0)
dfdp(1,4) = -args(3)*y(1)**2/(args(4)*y(1) + 1.0d0)**2
dfdp(1,6) = -y(1)
dfdp(1,9) = y(1)*y(2)/(args(15)*y(2) + 1.0d0)
dfdp(1,15) = -args(9)*y(1)*y(2)**2/(args(15)*y(2) + 1.0d0)**2
dfdp(2,1) = -args(7)*y(3)**4/(args(2)**4*(args(1)* y(3)**2/args(2)**2 &
     & + 1.0d0)**2)
dfdp(2,2) = 2*args(1)*args(7)*y(3)**4&
     & /(args(2)**5*( args(1)*y(3)**2/args(2)**2 + 1.0d0)**2) &
     & - 2* args(7)*y(3)**2/(args(2)**3*(args(1)* y(3)**2/args(2)**2 &
     & + 1.0d0))
dfdp(2,5) = -y(1)*y(2)/(args(15)*y(2) + 1.0d0)
dfdp(2,7) = y(3)**2/(args(2)**2*(args(1)*y(3)**2/ args(2)**2 + 1.0d0))
dfdp(2,8) = -y(2)
dfdp(2,15) = args(5)*y(1)*y(2)**2/(args(15)*y(2) + 1.0d0)**2
dfdp(3,8) = y(2)
dfdp(3,16) = -y(3)

end subroutine func


subroutine stpnt(ndim, y, args, t)

implicit None
integer, intent(in) :: ndim
double precision, intent(inout) :: y(ndim), args(*)
double precision, intent(in) :: t

args(1) = 1.0  ! n
args(2) = 1.0  ! K_sex
args(3) = 0.615  ! betta
args(4) = 1.0  ! k_pred
args(5) = 2.0  ! r
args(6) = 0.5  ! delta
args(7) = 17.0  ! c
args(8) = 0.06666666666666667  ! m
args(9) = 0.6  ! phi
args(15) = 1.0  ! v
args(16) = 0.01  ! mu_mod
y(1) = 11.18389971421499  ! P
y(2) = 2.9873443020221164  ! J
y(3) = 19.914507629338715  ! A

end subroutine stpnt



subroutine bcnd
end subroutine bcnd


subroutine icnd
end subroutine icnd


subroutine fopt
end subroutine fopt


subroutine pvls
end subroutine pvls
