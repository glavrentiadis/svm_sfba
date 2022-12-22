
%% Input
%user functions
sigmoid = @(x) 1./(1+exp(-x));
sigmoid = @(x) exp(x)./(1+exp(x));

%vs30 range
vs30_array = linspace(200,1000);

%output directory
dir_out = '../../Data/scaling_functions/';

%% Original Model
% original scaling coefficients
% - - - - - - - - - - - 
z_star = 2.5;
z_1 = 30;

%k scaling (original scaling)
r1_orig =-59.67;
r2_orig =-0.2722;
r3_orig = 11.132;
%k scaling (proposed scaling)
r1_new =-0.461371631791187;
r2_new =-0.209145223381101;
r3_new = 0.361724149163279;

%original functional forms
fun_vs0_orig = @(Vs30,p1,p2,p3)     p1*Vs30.^2 + p2*Vs30 + p3;
fun_k_orig   = @(Vs30,r1,r2,r3)     exp( r1*Vs30.^r2 + r3 );

%% Ouptut
mkdir(dir_out)

% comparison figures
% ---   ---   ---   ---
% k scaling
% - - - - - - - - - - - 
%first model
fname = 'scaling_relationship_k_comparison';
fig = figure;
semilogx(vs30_array, fun_k_orig(vs30_array,r1_orig,r2_orig,r3_orig),'LineWidth',2); hold on
semilogx(vs30_array, fun_k_orig(vs30_array,r1_new,r2_new,r3_new),   'LineWidth',2)
grid on
xlabel('V_{S30}')
ylabel('k')
legend('Original Model','New Model','location','northwest')
title('Second Proposed Model for k')
saveas(fig, [dir_out,fname,'.png'])
