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
 real<lower=0.0> Vs0;
 
 //noise std
 real<lower=0.0> sigma;
}

model {
  //velocity model
  vector [N] VEL_M;

  //prior distributions
  k     ~ exponential(0.5);
  n     ~ beta(1.0,3.0);
  Vs0   ~ lognormal(6.05,0.4);
  sigma ~ lognormal(-1.20,0.3);
  
  //evaluate velocity model
  for(i in 1:N){
      VEL_M[i] =  Vs0 * max([1., (1 + k * (Z[i]-z_star))^n]');
  }
   
  //likelihood
  Y ~ normal(VEL_M,sigma);
}
