/*********************************************
Stan program for fitting a velocity model
with Jian's functional form
Version 5.0 without between profile terms
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
  real            r1;
  real<lower=0.0> r2;
  //s scaling
  real<lower=0.0> s2;
 
  //aleatory variability
  real<lower=0.0> sigma_vel;
  //between event std
  real<lower=0.0> tau_r;

  //between profile term
  vector[NVEL] rdB;
}

transformed parameters {
  //velocity model parameters
  vector[NVEL] n_p;
  vector[NVEL] a_p;
  vector<lower=0.0>[NVEL] k_p;
  vector<lower=0.0>[NVEL] Vs0_p;

  //model parameters
  n_p   =      1. + s2 * inv_logit( (log(Vs30)-logVs30mid) * logVs30scl );
  k_p   = exp( r1 + r2 * inv_logit( (log(Vs30)-logVs30mid) * logVs30scl ) + rdB );
  a_p   =-1. ./ n_p;
  for(i in 1:NVEL){
      // Vs0 = (k_p*(a_p+1.)*z_star + (1.+k_p*(30.-z_star))^(a_p+1.) - 1.) / (30.*(a_p+1.)*k_p) * Vs30;
      Vs0_p[i] = (k_p[i]*(a_p[i]+1.)*z_star + (1.+k_p[i]*(30.-z_star))^(a_p[i]+1.) - 1.) / (30.*(a_p[i]+1.)*k_p[i]) * Vs30[i];
      //print("logVs30mid=",logVs30mid," logVs30scl=",logVs30scl);
      //print("r1=",r1," r2=",r2," s2=",s2);
      //print("i=",i," Vs30=",Vs30[i]," k_p=",k_p[i]," n_p=",n_p[i]," a_p=",a_p[i]," Vs0[i]=",Vs0_p[i]);
  }
}

model {
  //velocity model
  vector[N] VEL_M;

  //prior distributions
  //---   ---   ---   ---   ---   ---
  //model coefficients
  //Vs30 scaling
  logVs30mid ~ normal(6.0, 0.5);
  logVs30scl ~ gamma(2.0, 0.5);
  // n_p scaling
  s2 ~ lognormal(2.0, 0.3);
  // k_p scaling
  r1 ~ normal(0.0, 2.0);
  r2 ~ lognormal(0.5, 0.5);
 
  //aleatory variability
  sigma_vel ~ lognormal(-1.,0.6);
  tau_r     ~ exponential(10.);  
 
  //between event term r scaling
  rdB ~ normal(0,tau_r);
 
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