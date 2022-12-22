%% Scaling Functional Form for r

%% Input
%user functions
sigmoid = @(x) exp(x)./(1+exp(x));

%vs30 range
vs30_reg  = linspace(200,1000);
vs30_test = linspace(50,2000);

%filename of Vs parameters
fname_vs_param  = '../../Data/global_reg/bayesian_fit_temp/JianFunUpd3dB_log_res/all_trunc/all_trunc_stan_parameters.csv';
fname_vs_hparam = '../../Data/global_reg/bayesian_fit_temp/JianFunUpd3dB_log_res/all_trunc/all_trunc_stan_hyperparameters.csv';

%output directory
dir_out = '../../Data/scaling_functions/';

%% Load Data
df_param  = readtable(fname_vs_param);
df_hparam = readtable(fname_vs_hparam);


%% Processing
%compute log of k
df_param{:,'param_logk_mean'} = log(df_param.param_k_mean);
df_param{:,'param_logk_med'}  = log(df_param.param_k_med);


%% Proposed Model
% k scaling
% - - - - - - - - - - - 
%proposed functional forms
fun_logk_prop1 = @(Vs30,r1,r2)    (r1 + r2*log(Vs30));
fun_logk_prop2 = @(Vs30,r1,r2,r3) (r1 + r2*log(Vs30) + r3*log(Vs30).^2);
fun_logk_prop3 = @(Vs30,r1,r2)    (r1 + r2*min(Vs30,250));
fun_logk_prop4 = @(Vs30,r1,r2)    (r1 + r2*min(Vs30,300));
fun_logk_prop5 = @(Vs30,r1,r2)    (r1 + r2*min(Vs30,350));
fun_logk_prop6 = @(Vs30,r1,r2)    (r1 + r2*min(Vs30,400));
fun_logk_prop7 = @(Vs30,r1,r2)    (r1 + r2*min(Vs30,500));
fun_logk_prop8 = @(Vs30,r1,r2,r3) (r1 + r2*min(Vs30,500) + r3*min(Vs30,500).^2);
fun_logk_prop9 = @(Vs30,r1,r2,r3) (r1 + r2*min(Vs30,400) + r3*min(Vs30,500));

%evaluate first alternative functional form
theta0_r_prop1 = [0,0];
[theta_r_prop1, rmse_r_prop1] = lsqcurvefit(@(theta,x) fun_logk_prop1(x,theta(1),theta(2)), ...
                                            theta0_r_prop1, df_param.Vs30, df_param.param_logk_med);

%evaluate first alternative functional form
theta0_r_prop2 = [0,0,0];
[theta_r_prop2, rmse_r_prop2] = lsqcurvefit(@(theta,x) fun_logk_prop2(x,theta(1),theta(2),theta(3)), ...
                                            theta0_r_prop2, df_param.Vs30, df_param.param_logk_med);

%evaluate thrid alternative functional form
theta0_r_prop3 = [0,0];
[theta_r_prop3, rmse_r_prop3] = lsqcurvefit(@(theta,x) fun_logk_prop3(x,theta(1),theta(2)), ...
                                            theta0_r_prop1, df_param.Vs30, df_param.param_logk_med);

%evaluate fourth alternative functional form
theta0_r_prop4 = [0,0];
[theta_r_prop4, rmse_r_prop4] = lsqcurvefit(@(theta,x) fun_logk_prop4(x,theta(1),theta(2)), ...
                                            theta0_r_prop1, df_param.Vs30, df_param.param_logk_med);

%evaluate fifth alternative functional form
theta0_r_prop5 = [0,0];
[theta_r_prop5, rmse_r_prop5] = lsqcurvefit(@(theta,x) fun_logk_prop5(x,theta(1),theta(2)), ...
                                            theta0_r_prop5, df_param.Vs30, df_param.param_logk_med);

%evaluate sith alternative functional form
theta0_r_prop6 = [0,0];
[theta_r_prop6, rmse_r_prop6] = lsqcurvefit(@(theta,x) fun_logk_prop6(x,theta(1),theta(2)), ...
                                            theta0_r_prop5, df_param.Vs30, df_param.param_logk_med);

%evaluate seventh alternative functional form
theta0_r_prop7 = [0,0];
[theta_r_prop7, rmse_r_prop7] = lsqcurvefit(@(theta,x) fun_logk_prop7(x,theta(1),theta(2)), ...
                                            theta0_r_prop7, df_param.Vs30, df_param.param_logk_med);


%evaluate eight alternative functional form
theta0_r_prop8 = [0,0,0];
[theta_r_prop8, rmse_r_prop8] = lsqcurvefit(@(theta,x) fun_logk_prop8(x,theta(1),theta(2),theta(3)), ...
                                            theta0_r_prop8, df_param.Vs30, df_param.param_logk_med);

%evaluate nineth alternative functional form
theta0_r_prop9 = [0,0,0];
[theta_r_prop9, rmse_r_prop9] = lsqcurvefit(@(theta,x) fun_logk_prop9(x,theta(1),theta(2),theta(3)), ...
                                            theta0_r_prop9, df_param.Vs30, df_param.param_logk_med);

%% Ouptut
% summary information
% ---   ---   ---   ---
fprintf(['k Scalign\n\t Prop1: RMSE=%.1f\n\t Prop2: RMSE=%.1f\n\t Prop3: RMSE=%.1f\n\t ',...
                       'Prop4: RMSE=%.1f\n\t Prop5: RMSE=%.1f\n\t Prop6: RMSE=%.1f\n\t ',...
                       'Prop7: RMSE=%.1f\n\t Prop8: RMSE=%.1f\n\t Prop9: RMSE=%.1f\n'], ...
        rmse_r_prop1,rmse_r_prop2,rmse_r_prop3,rmse_r_prop4,rmse_r_prop5, ...
        rmse_r_prop6,rmse_r_prop7,rmse_r_prop8,rmse_r_prop9)
mkdir(dir_out)

% comparison figures
% ---   ---   ---   ---
% k scaling (all models)
% - - - - - - - - - - - 
%k scaling
fname = 'scaling_relationship_k';
fig = figure;
plot(df_param.Vs30, df_param.param_k_med, 'o', 'LineWidth', 2); hold on
plot(vs30_test, exp(fun_logk_prop1(vs30_test,theta_r_prop1(1),theta_r_prop1(2))),                  'LineWidth',2)
plot(vs30_test, exp(fun_logk_prop2(vs30_test,theta_r_prop2(1),theta_r_prop2(2),theta_r_prop2(3))), 'LineWidth',2)
plot(vs30_test, exp(fun_logk_prop3(vs30_test,theta_r_prop3(1),theta_r_prop3(2))),                  'LineWidth',2)
plot(vs30_test, exp(fun_logk_prop4(vs30_test,theta_r_prop4(1),theta_r_prop4(2))),                  'LineWidth',2)
plot(vs30_test, exp(fun_logk_prop5(vs30_test,theta_r_prop5(1),theta_r_prop5(2))),                  'LineWidth',2)
plot(vs30_test, exp(fun_logk_prop6(vs30_test,theta_r_prop6(1),theta_r_prop6(2))),                  'LineWidth',2)
plot(vs30_test, exp(fun_logk_prop7(vs30_test,theta_r_prop7(1),theta_r_prop7(2))),                  'LineWidth',2)
plot(vs30_test, exp(fun_logk_prop8(vs30_test,theta_r_prop8(1),theta_r_prop8(2),theta_r_prop8(3))), 'LineWidth',2)
plot(vs30_test, exp(fun_logk_prop9(vs30_test,theta_r_prop9(1),theta_r_prop9(2),theta_r_prop9(3))), 'LineWidth',2)
grid on
xlabel('V_{S30}')
ylabel('k')
xlim([0,1400])
ylim([0,50])
legend('Regressed Data', ...
       'Model 1','Model 2','Model 3','Model 4', ...
       'Model 5','Model 6','Model 7','Model 8', ...
       'Model 9','location','northeast')
title('Scaling Relationships for k')
saveas(fig, [dir_out,fname,'.png'])

%log(k) scaling
fname = 'scaling_relationship_logk';
fig = figure;
plot(df_param.Vs30, df_param.param_logk_med, 'o', 'LineWidth', 2); hold on
plot(vs30_test, fun_logk_prop1(vs30_test,theta_r_prop1(1),theta_r_prop1(2)),                  'LineWidth',2)
plot(vs30_test, fun_logk_prop2(vs30_test,theta_r_prop2(1),theta_r_prop2(2),theta_r_prop2(3)), 'LineWidth',2)
plot(vs30_test, fun_logk_prop3(vs30_test,theta_r_prop3(1),theta_r_prop3(2)),                  'LineWidth',2)
plot(vs30_test, fun_logk_prop4(vs30_test,theta_r_prop4(1),theta_r_prop4(2)),                  'LineWidth',2)
plot(vs30_test, fun_logk_prop5(vs30_test,theta_r_prop5(1),theta_r_prop5(2)),                  'LineWidth',2)
plot(vs30_test, fun_logk_prop6(vs30_test,theta_r_prop6(1),theta_r_prop6(2)),                  'LineWidth',2)
plot(vs30_test, fun_logk_prop7(vs30_test,theta_r_prop7(1),theta_r_prop7(2)),                  'LineWidth',2)
plot(vs30_test, fun_logk_prop8(vs30_test,theta_r_prop8(1),theta_r_prop8(2),theta_r_prop8(3)), 'LineWidth',2)
plot(vs30_test, fun_logk_prop9(vs30_test,theta_r_prop9(1),theta_r_prop9(2),theta_r_prop9(3)), 'LineWidth',2)
grid on
xlabel('V_{S30}')
ylabel('log(k)')
xlim([0,1400])
ylim([-8,4])
legend('Regressed Data', ...
       'Model 1','Model 2','Model 3','Model 4', ...
       'Model 5','Model 6','Model 7','Model 8', ...
       'Model 9','location','southeast')
title('Scaling Relationships for log(k)')
saveas(fig, [dir_out,fname,'.png'])


% k scaling (proposed model)
% - - - - - - - - - - - 
%k scaling (proposed model)
fname = 'scaling_relationship_k_proposed';
fig = figure;
plot(df_param.Vs30, df_param.param_k_med, 'o', 'LineWidth', 2); hold on
plot(vs30_test, exp(fun_logk_prop7(vs30_test,theta_r_prop7(1),theta_r_prop7(2))), ...
    'color','k', 'LineWidth',2)
grid on
xlabel('V_{S30}')
ylabel('k')
xlim([0,1400])
ylim([0,50])
legend('Regressed Data','Proposed (Model 7)','location','northeast')
title('Scaling Relationships for k')
saveas(fig, [dir_out,fname,'.png'])

%log(k) scaling (proposed model)
fname = 'scaling_relationship_logk_proposed';
fig = figure;
plot(df_param.Vs30, df_param.param_logk_med, 'o', 'LineWidth', 2); hold on
plot(vs30_test, fun_logk_prop7(vs30_test,theta_r_prop7(1),theta_r_prop7(2)), ...
     'color','k', 'LineWidth',2)
grid on
xlabel('V_{S30}')
ylabel('log(k)')
xlim([0,1400])
ylim([-8,4])
legend('Regressed Data','Proposed (Model 7)','location','northeast')
title('Scaling Relationships for log(k)')
saveas(fig, [dir_out,fname,'.png'])


% k scaling residuals (proposed model)
% - - - - - - - - - - - 
res_r_prop7 = df_param.param_logk_med - fun_logk_prop7(df_param.Vs30,theta_r_prop7(1),theta_r_prop7(2));

%log(k) scaling (proposed model)
fname = 'scaling_relationship_logk_proposed_res';
fig = figure;
hl = plot(df_param.Vs30, res_r_prop7, 'o'); hold on
hl.MarkerFaceColor = hl.Color;
plot([0,1400], [0,0], 'color','k', 'linewidth',2);
grid on
xlabel('V_{S30}')
ylabel('residuals log(k)')
xlim([0,1400])
ylim([-5,5])
title('Residuals Proposed Relationship for log(k)')
saveas(fig, [dir_out,fname,'.png'])
