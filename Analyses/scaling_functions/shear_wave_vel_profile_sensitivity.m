clear
close all

%time average shear wave velocity
vs30 = 200;
%depth array
z_array = (0:.2:200)';

% scaling parameters (Hessam's model)
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

fun_n = @(Vs30) 1. + s1 ./ (1 + s2 * (0.001*Vs30).^-s3);
fun_k = @(Vs30) exp(-r4 + r1 ./ ( 1. + r2 * (0.001*Vs30).^-r3) );
fun_Vs0 = @(Vs30,k,n) (k*(-n.^-1+1.)*z_star + (1.+k*(30.-z_star))^(-n.^-1+1.) - 1.) / (30.*(-n.^-1+1.)*k) * Vs30;

%shear wave veocity k/sec
fun_Vs = @ (z,Vs0,k,n) Vs0 * (1. + k * max(0., z-z_star) ).^(1./n);

%comptue parameters
%seed parameter
n_seed   = fun_n(vs30);
k_seed   = fun_k(vs30);
vs0_seed = fun_Vs0(vs30,k_seed,n_seed);
%sensitivity on k
n_s1a   = n_seed;
k_s1a   = 0.5 * k_seed;
vs0_s1a = fun_Vs0(vs30,k_s1a,n_s1a);
n_s1b   = n_seed;
k_s1b   = 2.0 * k_seed;
vs0_s1b = fun_Vs0(vs30,k_s1b,n_s1b);
%sensitivity on n
n_s2a   = 0.5 * n_seed;
k_s2a   = k_seed;
vs0_s2a = fun_Vs0(vs30,k_s2a,n_s2a);
n_s2b   = 2.0 * n_seed;
k_s2b   = k_seed;
vs0_s2b = fun_Vs0(vs30,k_s2b,n_s2b);
%sensitivity on k and n
n_s3a   = 0.5 * n_seed;
k_s3a   = 0.5 * k_seed;
vs0_s3a = fun_Vs0(vs30,k_s3a,n_s3a);
n_s3b   = 2.0 * n_seed;
k_s3b   = 2.0 * k_seed;
vs0_s3b = fun_Vs0(vs30,k_s3b,n_s3b);


%compute vel profile
vs_seed = fun_Vs(z_array, vs0_seed, k_seed, n_seed);
vs_s1a  = fun_Vs(z_array, vs0_s1a,  k_s1a,  n_s1a);
vs_s1b  = fun_Vs(z_array, vs0_s1b,  k_s1b,  n_s1b);
vs_s2a  = fun_Vs(z_array, vs0_s2a,  k_s2a,  n_s2a);
vs_s2b  = fun_Vs(z_array, vs0_s2b,  k_s2b,  n_s2b);
vs_s3a  = fun_Vs(z_array, vs0_s3a,  k_s3a,  n_s3a);
vs_s3b  = fun_Vs(z_array, vs0_s3b,  k_s3b,  n_s3b);

fig = figure;
hl0 = plot(vs_seed, z_array, 'k', 'LineWidth',2); hold on
hl1 = plot(vs_s1a,  z_array, 'b');
plot(vs_s1b,  z_array, '--b');
hl2 = plot(vs_s2a,  z_array, 'g');
plot(vs_s2b,  z_array, '--g');
hl3 = plot(vs_s3a,  z_array, 'r');
plot(vs_s3b,  z_array, '--r');
grid on
xlabel('V_{S} (m/sec)')
ylabel('z (m)')
set(gca, 'YDir', 'reverse');
legend([hl0,hl1,hl2,hl3],{'Seed','Sensitivity on k','Sensitivity on n','Sensitivity on k & n'})