%% Test New Scaling Functional Forms

%% Input
%user functions
sigmoid = @(x) 1./(1+exp(-x));

%vs30 range
vs30_array = linspace(200,1000);

%output directory
dir_out = '../../Data/scaling_functions/';

%% Original Model
% original scaling coefficients
% - - - - - - - - - - - 
z_star = 2.5;
z_1 = 30;

%Vs0 scaling
p1 = -2.1688*10^(-4);
p2 = 0.5182;
p3 = 69.452;
%k scaling
r1 =-59.67;
r2 =-0.2722;
r3 = 11.132;
%n scaling
s1 = 4.110;
s2 =-1.0521*10^(-4);
s3 =-10.827;
s4 =-7.6187*10^(-3);

%%
z_array = linspace(0,30,100000);

vs30_calc = nan(size(vs30_array));
vs30_explicit = nan(size(vs30_array));

for j = 1:length(vs30_array)
    vs30 = vs30_array(j);
    %
    vs0 = p1*vs30.^2 + p2*vs30 + p3;
    k   = exp( r1*vs30.^r2 + r3 );
    n   = s1*exp(s2*vs30) + s3*exp(s4*vs30);

    %compute vel profile
    vs_array = vs0 * (1 + k*max(0,z_array-z_star)).^(1/n);

    %numerically compute vs30
    vs30_calc(j) = 30/sum(diff(z_array)./vs_array(2:end));

    %analytically compute vs30
    a = -1/n;
    vs30_term1 = 30*k*(a+1)*vs0;
    vs30_term2 = k*(a+1)*z_star;
    vs30_term3 = ( 1 + k*(z_1-z_star) ).^(a+1) -1;
    vs30_explicit(j) = vs30_term1 / (vs30_term2 + vs30_term3);
end

%% Ouptut

% comparison figures
% ---   ---   ---   ---
% vs30
% - - - - - - - - - - - 
%first model
fname = 'scaling_vs30';
fig = figure;
plot(vs30_array,  vs30_calc,     '-', 'LineWidth', 2); hold on
plot(vs30_array,  vs30_explicit, '--', 'LineWidth', 2);
grid on
xlabel('Input V_{S30}')
ylabel('Computed V_{S30}')
legend('Numerical','Explicit','location','southeast')
title('Profile Implied V_{S30}')
saveas(fig, [dir_out,fname,'.png'])

