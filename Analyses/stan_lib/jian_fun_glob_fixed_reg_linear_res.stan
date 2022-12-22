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
  
  //scaling parameters
  real p1 = - 2.1688*10^(-4);
  real p2 =   0.5182;
  real p3 =  69.4520;
  real r1 = -59.6700;
  real r2 = - 0.2722;
  real r3 =  11.1320;
  real s1 =   4.1100;
  real s2 = - 1.0521*10^(-4);
  real s3 = -10.8270;
  real s4 = - 7.6187*10^(-3);
}

parameters {
  //model parameters
 
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
  Vs0 = p1*Vs30^2 + p2*Vs30 + p3 + dB_Vs0[i_vel];
  k   = exp(r1*Vs30^r2 + r3);
  n   = 1. / ( s1*exp(s2*Vs30) + s3*exp(s4*Vs30) );
  print("n[1] = ", n[1]);
  //print("Vs0[1] = ", Vs0[1]);
  for(i in 1:N){
    print("Vs30[",i,"]=",Vs30[i],",  n[",i,"]=",n[i]);
  }
}

model {
  //velocity model
  vector[N] VEL_M;

  //aleatory variability
  sigma_vel ~ lognormal(-1.20,0.3);
  sigma_Vs0 ~ exponential(5);
 
  //Vs0 random term
  dB_Vs0 ~ normal(0, sigma_Vs0);
 
  //functional model
  //---   ---   ---   ---   ---   ---
  for(i in 1:N){
      VEL_M[i] =  Vs0[i] * (1 + k[i] * max([0., Z[i]-z_star]') )^n[i];
      //print("k[",i,"]=",k[i],",  n[",i,"]=",n[i],",  Z[",i,"]=",Z[i],",  VEL_M[",i,"] = ", VEL_M[i]);
  }
  //evaluate velocity model
  //VEL_M[i] =  Vs0 * max([1., (1 + k * (Z-z_star))^n]');

  //likelihood function
  //---   ---   ---   ---   ---   ---
  //model likelihood
  Y ~ normal(VEL_M,sigma_vel);
}
