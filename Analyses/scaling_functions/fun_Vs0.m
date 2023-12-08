function [Vs0] = fun_Vs0(Vs30, k, n, z_star)
%fun_Vs0 scaling Vs0 relationship

if nargin < 4; z_star=2.5; end

%exponent
a = -1/n;

%vs0 scaling
if abs(n-1.0) < 1e-9
    Vs0 = (z_star + 1/k * log(1.+ k*(30.-z_star)))/30. * Vs30;  
else
    Vs0 = (k*(a+1.)*z_star + (1.+k*(30.-z_star))^(a+1.) - 1.) / (30.*(a+1.)*k) * Vs30;
end

end