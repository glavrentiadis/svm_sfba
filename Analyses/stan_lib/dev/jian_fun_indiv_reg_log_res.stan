/*********************************************
Stan program for fitting a velocity model
with Jian's functional form
 ********************************************/
data {

  int N; //number of points
  
  //z^star parameter
  real z_star;
  
  //input arrays
  vector[N] Z; //Depth array 
  vector[N] Y; //Velocity array

}

transformed data {
  real delta = 1e-9;
}

parameters {
 //model parameters
 real<lower=0.0> k;
 real<lower=0.0, upper=1.0> n;
 real logVs0;
 
 //noise std
 real<lower=0.0> sigma;
}

transformed parameters{
 real Vs0;

 Vs0 = exp(logVs0);
}

model {
  //velocity model
  vector [N] VEL_M;

  //prior distributions
  k      ~ exponential(0.5);
  n      ~ beta(1.0,3.0);
  logVs0 ~ normal(6.05,0.4);
  sigma  ~ lognormal(-0.3,0.6);
  
  //evaluate velocity model
  for(i in 1:N){
      VEL_M[i] =  logVs0 + log( max([1., (1 + k * (Z[i]-z_star))^n]') );
  }
   
  //likelihood
  Y ~ normal(VEL_M,sigma);
}
