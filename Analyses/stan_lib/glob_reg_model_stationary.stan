/*********************************************
Stan program for fitting a velocity model
with Shi and Asimaki (2018) functional form

Logistic and hinge function for k
Logistic function for n
 ********************************************/
data {
  int N; //number of points
  int NVEL; //number of velocity profiles
  
  //z^star parameter
  real z_star;
  
  //velocity array indices 
  array[N] int<lower=1,upper=NVEL> i_vel;
  
  //input arrays
  vector[NVEL] Vs30;  
  //observations
  vector[N] Z; //Depth array 
  vector[N] Y; //Velocity array
}

transformed data {
  real delta = 1e-9;
}

parameters {
  //model parameters
  real<lower=0.1, upper=11.9> logVs30mid;
  real<lower=0.1, upper=14.9> logVs30scl;
  //k scaling
  real<lower=-10., upper=10.> r1;
  real<lower= 0.0, upper=20.> r2;
  real<lower= 0.0, upper=1.0> r3;  
  //s scaling
  real<lower= 0.0, upper=40.> s2;
 
  //aleatory variability
  real<lower=0.0, upper=3.0> sigma_vel;
}

transformed parameters {
  //velocity model parameters
  vector[NVEL] n_p;
  vector[NVEL] a_p;
  vector<lower=0.0>[NVEL] k_p;
  vector<lower=0.0>[NVEL] Vs0_p;
  
  //vs30 scaling array
  vector[NVEL] logVs30 = (log(Vs30)-logVs30mid) / logVs30scl;

  //model parameters
  n_p   =      1. + s2 * inv_logit( logVs30 );
  k_p   = exp( r1 + r2 * inv_logit( logVs30 ) + r3 * logVs30scl * log( 1+exp(logVs30) ) );
  a_p   =-1. ./ n_p;
  for(i in 1:NVEL){
    if (abs(n_p[i]-1) <= delta)
      Vs0_p[i] = (z_star + 1/k_p[i] * log(1.+k_p[i]*(30.-z_star)))/30. * Vs30[i];  
    else
      Vs0_p[i] = (k_p[i]*(a_p[i]+1.)*z_star + (1.+k_p[i]*(30.-z_star))^(a_p[i]+1.) - 1.) / (30.*(a_p[i]+1.)*k_p[i]) * Vs30[i];
  }
}

model {
  //velocity model
  vector[N] VEL_M;

  //prior distributions
  //---   ---   ---   ---   ---   ---
  //model coefficients
  //Vs30 scaling
  //logVs30mid ~ normal(5.7, 0.1);
  logVs30mid ~ normal(5.7, 0.5);
  logVs30scl ~ gamma(2.0, 2.0);
  // n_p scaling
  s2 ~ lognormal(2.0, 0.3);
  // k_p scaling
  r1 ~ normal(0.0, 5.0);
  r2 ~ lognormal(0.5, 0.5);
  r3 ~ exponential(2);
  
  //aleatory variability
  sigma_vel ~ lognormal(-1.,0.6);
 
  //functional model
  //---   ---   ---   ---   ---   ---
  //evaluate velocity model
  for(i in 1:N){
    int i_v = i_vel[i];
    VEL_M[i] =  log(Vs0_p[i_v]) + log( (1. + k_p[i_v] * max([0., Z[i]-z_star]) )^(1./n_p[i_v]) );
  }
   
  //likelihood function
  //---   ---   ---   ---   ---   ---
  //model likelihood
  Y ~ normal(VEL_M,sigma_vel);
}
