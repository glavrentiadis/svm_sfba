%% Test New Scaling Functional Forms

%% Input
%user functions
sigmoid = @(x) exp(x)./(1+exp(x));

%vs30 range
vs30_reg  = linspace(200,1000);
vs30_test = linspace(50,2000);

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

%original functional forms
fun_vs0_orig = @(Vs30,p1,p2,p3)     p1*Vs30.^2 + p2*Vs30 + p3;
fun_k_orig   = @(Vs30,r1,r2,r3)     exp( r1*Vs30.^r2 + r3 );
fun_n_orig   = @(Vs30,s1,s2,s3,s4)  s1*exp(s2*Vs30) + s3*exp(s4*Vs30);

%evaluate original scaling relationships
vs0_orig_reg = fun_vs0_orig(vs30_reg,p1,p2,p3);
k_orig_reg   = fun_k_orig(vs30_reg,r1,r2,r3);
n_orig_reg   = fun_n_orig(vs30_reg,s1,s2,s3,s4);

%% Proposed Model
% n scaling
% - - - - - - - - - - - 
%proposed functional forms
fun_n_prop1   = @(Vs30,s1,s2,s3) 1      + s3*sigmoid((log(Vs30)-s1)*s2);
fun_n_prop2   = @(Vs30,s1,s2,s3) 1+pi/2 + s3*atan((log(Vs30)-s1)*s2);

%evaluate first alternative functional form
theta0_n_prop1 = [log(250),1,2];
[theta_n_prop1, res_n_prop1] = lsqcurvefit(@(theta,x) fun_n_prop1(x,theta(1),theta(2),theta(3)), ...
                                           theta0_n_prop1, vs30_reg, n_orig_reg);
%evaluate second alternative functional form
theta0_n_prop2 = [log(250),1,2];
[theta_n_prop2, res_n_prop2] = lsqcurvefit(@(theta,x) fun_n_prop2(x,theta(1),theta(2),theta(3)), ...
                                           theta0_n_prop2, vs30_reg, n_orig_reg);


% vs0 scaling
% - - - - - - - - - - - 
%proposed functional forms
fun_vs0_prop1   = @(Vs30,p1,p2)    p1*Vs30 + p2;
fun_vs0_prop2   = @(Vs30,p1,p2,p3) p3*sigmoid((log(Vs30)-p1)*p2);
fun_vs0_prop3   = @(Vs30,p1,p2,p3) p1*log(Vs30+1) + p2*Vs30 + p3;
fun_vs0_prop4   = @(Vs30,k,a) (k.*(a+1)*z_star + (1+k*(z_1-z_star)).^(a+1) - 1) ./ (30.*(a+1).*k) .* Vs30;

%evaluate first alternative functional form
theta0_vs0_prop1 = [p2,p3];
[theta_vs0_prop1, res_vs0_prop1] = lsqcurvefit(@(theta,x) fun_vs0_prop1(x,theta(1),theta(2)), ...
                                               theta0_vs0_prop1, vs30_reg, vs0_orig_reg);

%evaluate second alternative functional form
theta0_vs0_prop2 = [log(250),1,250];
[theta_vs0_prop2, res_vs0_prop2] = lsqcurvefit(@(theta,x) fun_vs0_prop2(x,theta(1),theta(2),theta(3)), ...
                                               theta0_vs0_prop2, vs30_reg, vs0_orig_reg);

%evaluate thrird alternative functional form
theta0_vs0_prop3 = [1,p2,p3];
[theta_vs0_prop3, res_vs0_prop3] = lsqcurvefit(@(theta,x) fun_vs0_prop3(x,theta(1),theta(2),theta(3)), ...
                                               theta0_vs0_prop3, vs30_reg, vs0_orig_reg);

%% Ouptut
% summary information
% ---   ---   ---   ---
fprintf('Vs0 Scalign\n\t Prop1: RMSE=%.1f\n\t Prop2: RMSE=%.1f\n\t Prop3: RMSE=%.1f\n', ...
        res_vs0_prop1,res_vs0_prop2,res_vs0_prop3)
fprintf('n Scalign\n\t Prop1: RMSE=%.1f\n\t Prop2: RMSE=%.1f\n', ...
        res_n_prop1,res_n_prop2)
mkdir(dir_out)

% comparison figures
% ---   ---   ---   ---
% k scaling
% - - - - - - - - - - - 
%first model
fname = 'scaling_relationship_k';
fig = figure;
semilogx(vs30_reg,  k_orig_reg, 'o', 'LineWidth', 2); hold on
semilogx(vs30_test, fun_k_orig(vs30_test,r1,r2,r3),'--','LineWidth',1,'Color','#0072BD')
grid on
xlabel('V_{S30}')
ylabel('k')
legend('Original Model','Original Model (extended)','location','northwest')
title('Second Proposed Model for k')
saveas(fig, [dir_out,fname,'.png'])

% n scaling
% - - - - - - - - - - - 
%first model
fname = 'scaling_relationship_n_1';
fig = figure;
semilogx(vs30_reg,  n_orig_reg, 'o', 'LineWidth', 2); hold on
semilogx(vs30_test, fun_n_orig(vs30_test,s1,s2,s3,s4),'--','LineWidth',1,'Color','#0072BD')
semilogx(vs30_test, fun_n_prop1(vs30_test,theta_n_prop1(1),theta_n_prop1(2),theta_n_prop1(3)),'LineWidth',2)
grid on
xlabel('V_{S30}')
ylabel('n')
legend('Original Model','Original Model (extended)','Proposed Relationship','location','southeast')
title('First Proposed Model for n')
saveas(fig, [dir_out,fname,'.png'])

%second model
fname = 'scaling_relationship_n_2';
fig = figure;
semilogx(vs30_reg,  n_orig_reg, 'o', 'LineWidth',2); hold on
semilogx(vs30_test, fun_n_orig(vs30_test,s1,s2,s3,s4),'--','LineWidth',1,'Color','#0072BD')
semilogx(vs30_test, fun_n_prop2(vs30_test,theta_n_prop2(1),theta_n_prop2(2),theta_n_prop2(3)),'LineWidth',2)
grid on
xlabel('V_{S30}')
ylabel('n')
legend('Original Model','Original Model (extended)','Proposed Relationship','location','southeast')
title('Second Proposed Model for n')
saveas(fig, [dir_out,fname,'.png'])

% vs0 scaling
% - - - - - - - - - - - 
%first model
fname = 'scaling_relationship_vs0_1';
fig = figure;
semilogx(vs30_reg,  vs0_orig_reg, 'o', 'LineWidth', 2); hold on
semilogx(vs30_test, fun_vs0_orig(vs30_test,p1,p2,p3),'--','LineWidth',1,'Color','#0072BD')
semilogx(vs30_test, fun_vs0_prop1(vs30_test,theta_vs0_prop1(1),theta_vs0_prop1(2)), 'LineWidth',2)
grid on
xlabel('V_{S30}')
ylabel('V_0')
legend('Original Model','Original Model (extended)','Proposed Relationship','location','southeast')
title('First Proposed Model for V_{S0}')
saveas(fig, [dir_out,fname,'.png'])

%second model
fname = 'scaling_relationship_vs0_2';
fig = figure;
semilogx(vs30_reg,  vs0_orig_reg, 'o', 'LineWidth', 2); hold on
semilogx(vs30_test, fun_vs0_orig(vs30_test,p1,p2,p3),'--','LineWidth',1,'Color','#0072BD')
semilogx(vs30_test, fun_vs0_prop2(vs30_test,theta_vs0_prop2(1),theta_vs0_prop2(2),theta_vs0_prop2(3)), 'LineWidth',2)
grid on
xlabel('V_{S30}')
ylabel('V_0')
legend('Original Model','Original Model (extended)','Proposed Relationship','location','southeast')
title('Second Proposed Model for V_{S0}')
saveas(fig, [dir_out,fname,'.png'])

%third model
fname = 'scaling_relationship_vs0_3';
fig = figure;
semilogx(vs30_reg,  vs0_orig_reg, 'o', 'LineWidth', 2); hold on
semilogx(vs30_test, fun_vs0_orig(vs30_test,p1,p2,p3),'--','LineWidth',1,'Color','#0072BD')
semilogx(vs30_test, fun_vs0_prop3(vs30_test,theta_vs0_prop3(1),theta_vs0_prop3(2),theta_vs0_prop3(3)), 'LineWidth',2)
grid on
xlabel('V_{S30}')
ylabel('V_0')
legend('Original Model','Original Model (extended)','Proposed Relationship','location','southeast')
title('Third Proposed Model for V_{S0}')
saveas(fig, [dir_out,fname,'.png'])

%implied vs0
fname = 'scaling_relationship_vs0_4';

k_array = fun_k_orig(vs30_test,r1,r2,r3);
a_array_orig  = -1./fun_n_orig(vs30_reg,s1,s2,s3,s4);
a_array_prop1 = -1./fun_n_prop1(vs30_test,theta_n_prop1(1),theta_n_prop1(2),theta_n_prop1(3));

fig = figure;
semilogx(vs30_reg,  vs0_orig_reg, 'o', 'LineWidth', 2); hold on
semilogx(vs30_test, fun_vs0_orig(vs30_test,p1,p2,p3),'--','LineWidth',1,'Color','#0072BD')
semilogx(vs30_reg, fun_vs0_prop4(vs30_reg,k_orig_reg,a_array_orig), 'LineWidth',2)
semilogx(vs30_test, fun_vs0_prop4(vs30_test,k_array,a_array_prop1), 'r', 'LineWidth',2)
grid on
xlabel('V_{S30}')
ylabel('V_0')
legend('Original Model','Original Model (extended)','Proposed Relationship (original n)','Proposed Relationship (proposed n)','location','southeast')
title('Fourth Proposed Model for V_{S0}')
saveas(fig, [dir_out,fname,'.png'])