%% Determine scaling coefficients for k - Vs30 relationship

%% Input
%user functions
sigmoid = @(x) exp(x)./(1+exp(x));

%read flatfile
fname_flatfile = '../../Data/global_reg/bayesian_fit/JianFunUpd7.5GPdBr_log_res/all_trunc/all_trunc_stan_parameters.csv';
%column names for vs30 and k
col_vs30 = 'Vs30';
col_k    = 'param_k_med';

%functional form
fun_k =  @(Vs30,r1,r2,r3,r4) exp(r1 + r2*sigmoid((log(Vs30)-r3)*r4));
%seed r values
r_seed0 = [-2.51945, 1.97386, 6.47206, 2.87082];

%vs30 range
vs30_reg  = linspace(100,3000);

%% Load Files
%parameters' flatfile
df_flatfile = readtable(fname_flatfile,'VariableNamingRule','preserve');

%profiles to exclude
df_flatfile = df_flatfile(~and(df_flatfile.DSID==1, df_flatfile.VelID==9),  :);
df_flatfile = df_flatfile(~and(df_flatfile.DSID==3, df_flatfile.VelID==56), :);
df_flatfile = df_flatfile(~and(df_flatfile.DSID==3, df_flatfile.VelID==57), :);
df_flatfile = df_flatfile(~and(df_flatfile.DSID==3, df_flatfile.VelID==31), :);

%% Regression
%objective function 
fun_k_wrap = @(r_array,vs30) log(fun_k(vs30,r_array(1),r_array(2),r_array(3),r_array(4)));

%k regression parameters
r_fit   = fitnlm(df_flatfile{:,col_vs30},log(df_flatfile{:,col_k}),fun_k_wrap,r_seed0)
r_array = r_fit.Coefficients.Estimate;

%% Plotting
%plot regression fit
figid = figure;
scatter(df_flatfile.Vs30,df_flatfile.param_k_med,'filled'); hold on
plot(vs30_reg,fun_k(vs30_reg,r_array(1),r_array(2),r_array(3),r_array(4)),'LineWidth',2)
set(gca,'XScale','log')
set(gca,'YScale','log')
grid on
xlabel('V_{S30}')
ylabel('k')
title('Determine Scaling k Parameters')

%plot regression fit
figid = figure;
scatter(df_flatfile.Vs30,r_fit.Residuals.Raw,'filled'); hold on
set(gca,'XScale','log')
grid on
xlabel('V_{S30}')
ylabel('Residuals')
title('Residuals of Scaling k')
