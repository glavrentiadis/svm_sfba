clear
close all

%time average shear wave velocity
vs30 = 300;
%depth array
z_array = (0:.2:200)';

% scaling parameters
% n
s1 = 7.291;
s2 = 0.072;
s3 = 4.006;
% k
r1 = 3.426;
r2 = 2.32/10000;
r3 = 8.936;
r4 = 2.435;
%constant depth
z_star = 2.5;

fun_n = @(Vs30) 1. + s1 ./ (1 + s2 * Vs30.^-s3);
fun_k = @(Vs30) exp(-r4 + r1 ./ ( 1. + r2 * Vs30.^-r3) );
fun_Vs0 = @(Vs30,k,n) (k*(-n.^-1+1.)*z_star + (1.+k*(30.-z_star))^(-n.^-1+1.) - 1.) / (30.*(-n.^-1+1.)*k) * Vs30;

%shear wave veocity k/sec
fun_Vs = @ (z,Vs0,k,n) Vs0 * (1. + k * max(0., z-z_star) ).^(1./n);


%comptue parameters
n   = fun_n(vs30);
k   = fun_k(vs30);
vs0 = fun_Vs0(vs30,k,n);


%compute vel profile
vs = fun_Vs(z_array,vs0,k,n);

fig = figure;
plot(vs,z_array);
grid on
xlabel('V_{S} (m/sec)')
ylabel('z (m)')
set(gca, 'YDir', 'reverse');
