/*********************************************
Stan program for fitting a velocity model
with Shi and Asimaki (2018) functional form

Logistic and hinge function with between event 
profile variability for k
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

  //coordinates
  matrix[NVEL,2] X; //velocity profile coordinates
  
  //fixed model parameters
  real logVs30mid_fxd; 
  real logVs30scl_fxd; 
  real s2_fxd;
  real r1_fxd;
  real r2_fxd;
  real r3_fxd;
}

transformed data {
  //prewidening
  real delta = 1e-9;
  
  //compute distances
  matrix[NVEL, NVEL] dist_vel;
          
  //compute distance between vel profiles
  for(i in 1:NVEL) {
    for(j in i:NVEL) {
      real d_v = distance(X[i,:],X[j,:]);
      dist_vel[i,j] = d_v;
      dist_vel[j,i] = d_v;
    }
  }
}

parameters {
  // //model parameters
  //k scaling (adjustment)
  real<lower=-1., upper=1.> dr1;
  real<lower=-1., upper=1.> dr2;

  //aleatory variability
  real<lower=0.0, upper=3.0> sigma_vel;
  //between event std
  real<lower=0.0, upper=500.0> ell_rdB;
  real<lower=0.0, upper=3.0> omega_rdB;
    
  //standardized between profile term
  vector[NVEL] z_rdB;
}

transformed parameters {
  //velocity model parameters
  vector[NVEL] n_p;
  vector[NVEL] a_p;
  vector<lower=0.0>[NVEL] k_p;
  vector<lower=0.0>[NVEL] Vs0_p;
  //between profile random term
  vector[NVEL] rdB; 
  
  //fixed model coefficinets
  real logVs30mid = logVs30mid_fxd;
  real logVs30scl = logVs30scl_fxd;
  real s2         = s2_fxd;
  real r1         = r1_fxd + dr1;
  real r2         = r2_fxd + dr2;
  real r3         = r3_fxd;
  
  //vs30 scaling array
  vector[NVEL] logVs30 = (log(Vs30)-logVs30mid) / logVs30scl;

  //spatillay latent variable for event contributions to GP
  {
    matrix[NVEL,NVEL] COV_rdB;
    matrix[NVEL,NVEL] L_rdB;

    for(i in 1:NVEL) {
      //diagonal terms
      COV_rdB[i,i] = omega_rdB^2 + delta;
      //off-diagonal terms
      for(j in (i+1):NVEL) {
        real C_rdB = (omega_rdB^2 * exp(-dist_vel[i,j]/ell_rdB));
        COV_rdB[i,j] = C_rdB;
        COV_rdB[j,i] = C_rdB;
      }
    }
    L_rdB = cholesky_decompose(COV_rdB);
    rdB = L_rdB * z_rdB;
  }

  //model parameters
  n_p   =      1. + s2 * inv_logit( logVs30 );
  k_p   = exp( r1 + r2 * inv_logit( logVs30 ) + r3 * logVs30scl * log( 1+exp(logVs30) ) + rdB );
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
  // k scaling
  dr1 ~ normal(0.0, 0.2);
  dr2 ~ normal(0.0, 0.2);
  //v3
  //dr1 ~ normal(0.0, 0.4);
  //dr2 ~ normal(0.0, 0.4);
      
  //aleatory variability
  sigma_vel ~ lognormal(-0.3,0.6);
  //GP parameters
  ell_rdB   ~ inv_gamma(1.,50);
  //ell_rdB   ~ inv_gamma(0.5,50);
  //omega_rdB ~ normal(0,0.05);  //part 1
  //omega_rdB ~ normal(0,0.025); //part 2
  //omega_rdB ~ normal(0,0.01);  //part 3
  //omega_rdB ~ normal(0,0.02); //part 4
  //omega_rdB ~ normal(0,0.015); //part 5
  omega_rdB ~ normal(0,0.025);
  
  //standardized between prof contributions to GP
  z_rdB ~ std_normal();
 
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
