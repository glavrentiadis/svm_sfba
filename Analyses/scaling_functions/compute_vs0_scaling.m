%% Scaling Vs0
% --------------------------------------------
clear all;
close all;

%% Input
%user functions
sigmoid = @(x) exp(x)./(1+exp(x));

%scaling relationnshps for k and n
fun_k = @(Vs30,r1,r2,r3,r4) exp(r1 + r2*sigmoid((log(Vs30)-r3)*r4));
fun_n = @(Vs30,s2,s3,s4)         1  + s2*sigmoid((log(Vs30)-s3)*s4);

%scaling coefficients
% k scaling
r1 =-2.9326;
r2 = 2.7602;
r3 = 6.0031;
r4 = 7.4509;
% n scaling
s2 = 7.89052;
s3 = 6.47206;
s4 = 2.87082;

%vs30 array
vs30_array = logspace(log10(50), log10(2000));

%% Processing
k_array = fun_k(vs30_array,r1,r2,r3,r4);
n_array = fun_n(vs30_array,   s2,s3,s4);

vs0_array = nan(size(vs30_array));
for j = 1:length(vs30_array)
    vs0_array(j) = fun_Vs0(vs30_array(j),k_array(j),n_array(j));
end


%% Plotting

%plot k scaling
figid = figure;
plot(vs30_array,k_array,'LineWidth',2)
set(gca,'XScale','log')
grid on
xlabel('V_{S30}')
ylabel('k')
title('n Scaling')

%plot n scaling
figid = figure;
plot(vs30_array,n_array,'LineWidth',2)
set(gca,'XScale','log')
set(gca,'YScale','log')
grid on
xlabel('V_{S30}')
ylabel('k')
title('k Scaling')

%plot vs0 scaling
figid = figure;
plot(vs30_array,vs0_array,'LineWidth',2)
set(gca,'XScale','log')
set(gca,'YScale','log')
grid on
xlabel('V_{S30}')
xlabel('V_{S0}')
title('V_{S0} Scaling')