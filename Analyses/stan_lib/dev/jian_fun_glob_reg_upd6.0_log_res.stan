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
  //k scaling
  real<lower=-10., upper=10.> r1;
  real<lower= 0.0, upper=20.> r2;
  real<lower=0.1, upper=11.9> r3;
  real<lower=0.1, upper=14.9> r4;
  //s scaling
  real<lower= 0.0, upper=40.> s2;
  real<lower=0.1, upper=11.9> s3;
  real<lower=0.1, upper=14.9> s4;
 
  //aleatory variability
  real<lower=0.0, upper=3.0> sigma_vel;
}

transformed parameters {
  //velocity model parameters
  vector[NVEL] n_p;
  vector[NVEL] a_p;
  vector<lower=0.0>[NVEL] k_p;
  vector<lower=0.0>[NVEL] Vs0_p;

  //model parameters
  n_p   =      1. + s2 * inv_logit( (log(Vs30)-s3) * s4 );
  k_p   = exp( r1 + r2 * inv_logit( (log(Vs30)-r3) * r4 ) );
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
  //n_p scaling
  s1 ~ lognormal(2.0, 0.3);
  s3 ~ normal(6.0, 0.5);
  s4 ~ gamma(2.0, 0.5);
  // k_p scaling
  r2 ~ lognormal(0.5, 0.5);
  r1 ~ normal(0.0, 2.0);
  r3 ~ normal(6.0, 0.5);
  r4 ~ gamma(2.0, 0.5);
 
  //aleatory variability
  //sigma_vel ~ lognormal(-0.3,0.6);
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
