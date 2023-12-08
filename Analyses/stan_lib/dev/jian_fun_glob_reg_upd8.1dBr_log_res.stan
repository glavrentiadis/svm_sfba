/*********************************************
Stan program for fitting a velocity model
with Jian's functional form
Version 8.1 without between profile terms
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
  
  //fixed model parameters
  real logVs30mid_fxd;
  real logVs30scl_fxd;
}

transformed data {
  real delta = 1e-9;
}

parameters {
  //k scaling
  real<lower=-10.0, upper= 10.0> r1;
  real<lower=  0.0, upper= 20.0> r2;
  real<lower=  0.0, upper= 20.0> r3;  
  //s scaling
  real<lower= 0.0, upper= 40.0> s2;
 
  //aleatory variability
  real<lower= 0.0, upper= 3.0> sigma_vel;
  //between event std
  real<lower=0.0, upper=10.0> tau_rdB;

  //between profile term
  vector[NVEL] rdB;
}

transformed parameters {
  //velocity model parameters
  vector[NVEL] n_p;
  vector[NVEL] a_p;
  vector<lower=0.0>[NVEL] k_p;
  vector<lower=0.0>[NVEL] Vs0_p;
  //fixed model coefficinets   
  real logVs30mid = logVs30mid_fxd;
  real logVs30scl = logVs30scl_fxd;
  
  //vs30 scaling array
  vector[NVEL] logVs30 = (log(Vs30)-logVs30mid) / logVs30scl;

  //model parameters
  n_p   =      1. + s2 * inv_logit( logVs30 );
  k_p   = exp( r1 + r2 * inv_logit( logVs30 ) + r3 * logVs30scl * log( 1+exp(logVs30) ) + rdB );
  a_p   =-1. ./ n_p;
  for(i in 1:NVEL){
    // for n_p==0; Vs0 = (z_star + 1/k_p * log(1.+k_p*(30.-z_star))) * Vs30;
    // for n_p!=0; Vs0 = (k_p*(a_p+1.)*z_star + (1.+k_p*(30.-z_star))^(a_p+1.) - 1.) / (30.*(a_p+1.)*k_p) * Vs30;
    if (abs(n_p[i]-1) <= delta)
      Vs0_p[i] = (z_star + 1/k_p[i] * log(1.+k_p[i]*(30.-z_star)))/30. * Vs30[i];  
    else
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
  // n_p scaling
  s2 ~ lognormal(2.0, 0.3);
  // k_p scaling
  r1 ~ normal(0.0, 2.0);
  r2 ~ lognormal(0.5, 0.5);
  r3 ~ lognormal(0.5, 0.5);
 
  //aleatory variability
  sigma_vel ~ lognormal(-1.,0.6);
  tau_rdB   ~ exponential(2.);  
 
  //aleatory variability
  //sigma_vel ~ lognormal(-0.3,0.6);
  sigma_vel ~ lognormal(-1.,0.6);
 
  //between event term r scaling
  rdB ~ normal(0,tau_rdB);
  
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
