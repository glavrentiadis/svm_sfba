/*********************************************
Stan program for fitting a velocity model
with Jian's functional form
 ********************************************/
data {

  int N; //number of points
  int NVEL; //number of velocity profiles
  
  //z^star parameter
  real z_star;
  
  //velocity array indices 
  int<lower=1,upper=NVEL> i_vel[N];
  
  //input arrays
  vector[N] Vs30;  
  //observations
  vector[N] Z; //Depth array 
  vector[N] Y; //Velocity array

}

transformed data {
  real delta = 1e-9;
}

parameters {
  //model parameters
  //Vs0 scaling
  real            p1;
  real<lower=0.0> p2;
  real<lower=0.0> p3;
  //k scaling
  real r1;
  real r2;
  real r3;
  //s scaling
  real<lower=0.0> s1;
  real<upper=1.0> s2;
  real<upper=1.0> s3;
  real<upper=1.0> s4; 
 
  //aleatory variability
  real<lower=0.0> sigma_vel;
  real<lower=0.0> sigma_Vs0;
  //between velocity terms
  vector[N] dB_Vs0;
}

transformed parameters {
  //velocity model parameters
  vector<lower=0.0>[N] k;
  vector<lower=0.0, upper=1.0>[N] n;
  vector<lower=0.0>[N] Vs0;

  //model parameters
  // Vs0 = p1*Vs30^2 + p2*Vs30 + p3 + dB_Vs0[i_vel];
  // k   = exp(r1*Vs30^r2 + r3);
  // n   = 1 / ( s1*exp(s2*Vs30) + s3*exp(s4*Vs0) );
  Vs0 = p1*Vs30^2 + p2*Vs30 + p3 + dB_Vs0[i_vel];
  k   = exp(r1*Vs30^r2 + r3);
  n   = 1 / ( s1*exp(s2*Vs30) + s3*exp(s4*Vs30) );
}

model {
  //velocity model
  vector[N] VEL_M;

  //prior distributions
  //---   ---   ---   ---   ---   ---
  //model coefficients
  p1 ~ normal(0,0.01);
  p2 ~ normal(0,2.0);
  p3 ~ normal(50,50.);
  r1 ~ normal(0,100.);
  r2 ~ normal(0,1.0);
  r3 ~ normal(0,25.);
  s1 ~ normal(0,25.);
  s2 ~ normal(0,0.001);
  s3 ~ normal(0,25.);
  s4 ~ normal(0,0.01);
  
  //aleatory variability
  sigma_vel ~ lognormal(-1.20,0.3);
  sigma_Vs0 ~ exponential(5);
 
  //Vs0 random term
  dB_Vs0 ~ normal(0, sigma_Vs0);
 
  //functional model
  //---   ---   ---   ---   ---   ---
  for(i in 1:N){
      VEL_M[i] =  Vs0[i] * max([1., (1 + k[i] * (Z[i]-z_star))^n[i]]');
  }
  //evaluate velocity model
  //VEL_M[i] =  Vs0 * max([1., (1 + k * (Z-z_star))^n]');

  //likelihood function
  //---   ---   ---   ---   ---   ---
  //model likelihood
  Y ~ normal(VEL_M,sigma_vel);
}
