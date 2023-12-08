Filename					Description
--------------					--------------
fit_glob_vel_models_Jian_upd1.0_log_res.py	Global regression with Vs30 fixed V0 and logistic scaling for n.
fit_glob_vel_models_Jian_upd2.0_log_res.py	Global regression based on fit_glob_vel_models_Jian_upd_log_res.py
						log(Vs30) scaling for k "exp(r1 * log(Vs30) + r2)".
fit_glob_vel_models_Jian_upd3.0_log_res.py	Global regression based on fit_glob_vel_models_Jian_upd_log_res.py
						Constant k "exp(r1)".
fit_glob_vel_models_Jian_upd3.0dB_log_res.py	Global regression based on fit_glob_vel_models_Jian_upd_log_res.py
						Constant k with within profile effects "exp(r1 + dBr)".
fit_glob_vel_models_Jian_upd3.1_log_res.py	Updated based on fit_glob_vel_models_Jian_upd3.0_log_res.py
						with free intercept for n_p "s0 + s3 * inv_logit((log(Vs30)-s1)*s2)".
fit_glob_vel_models_Jian_upd3.1dB_log_res.py	Based on fit_glob_vel_models_Jian_upd3.1_log_res.py including random
                                                between profile effects for k.
fit_glob_vel_models_Jian_upd4.0_log_res.py	Global regression based on fit_glob_vel_models_Jian_upd3.0_log_res.py
                                                with bilinear scaling for k with Vs30 break point.
fit_glob_vel_models_Jian_upd4.0dB_log_res.py	Based on fit_glob_vel_models_Jian_upd4.0dB_log_res.py with random 
                                                between profile effects "exp(r1 + r2(Vs30)*Vs30 + dBr)".
fit_glob_vel_models_Jian_upd5.0_log_res.py      Based on fit_glob_vel_models_Jian_upd3.0_log_res.py with a smoothly 
                                                vayring k scaling "exp(r1/(1+(r2*Vs30^-r3))+r4)".
fit_glob_vel_models_Jian_upd5.0dB_log_res.py    Based on fit_glob_vel_models_Jian_upd5.0_log_res.py with a between profile
                                                random term.
fit_glob_vel_model_Jian_upd6.0_log_res.py	Global regression with logistic scaling for k and n as a function of Vs30
fit_glob_vel_model_Jian_upd6.1_log_res.py	Global regression with logistic scaling for k and n as a function of Vs30
						reformulated equations
fit_glob_vel_model_Jian_upd7.0_log_res.py       Global regression with same logistic scaling for k and n same midpoint and 
                                                scale, different amplitude
fit_glob_vel_model_Jian_upd7.0dBr_log_res.py	Based on fit_glob_vel_model_Jian_upd7.0_log_res.py with k random term
fit_glob_vel_model_Jian_upd7.0GPdBr_log_res.py	Based on fit_glob_vel_model_Jian_upd7.0_log_res.py with k spatially varying 
						term
fit_glob_vel_model_Jian_upd7.1dBr_log_res.py	Based on fit_glob_vel_model_Jian_upd7.0_log_res.py with k random term having
						logVs30mid and logVs30scl fixed
fit_glob_vel_model_Jian_upd7.1GPdBr_log_res.py	Based on fit_glob_vel_model_Jian_upd7.0_log_res.py with k spatially varying 
						term having logVs30mid and logVs30scl fixed
fit_glob_vel_model_Jian_upd7.2dBr_log_res.py 	Based on fit_glob_vel_model_Jian_upd7.0_log_res.py with k random term having
						logVs30mid, logVs30scl, r1, r2, and s2 fixed
fit_glob_vel_model_Jian_upd7.2GPdBr_log_res.py 	Based on fit_glob_vel_model_Jian_upd7.0_log_res.py with k spatially varying 
						term having logVs30mid, logVs30scl, r1, r2, and s2 fixed
fit_glob_vel_model_Jian_upd7.3dBr_log_res.py	Based on fit_glob_vel_model_Jian_upd7.0_log_res.py with k random term having
						logVs30mid, logVs30scl, r1, r2, and s2 fixed, but inlude delta_r1 and delta_r2 
						adustments
fit_glob_vel_model_Jian_upd7.3GPdBr_log_res.py	Based on fit_glob_vel_model_Jian_upd7.0_log_res.py with k spatially varying 
						term having logVs30mid, logVs30scl, r1, r2, and s2 fixed, but inlude delta_r1 
						and delta_r2 adustments
fit_glob_vel_model_Jian_upd7.4dBr_log_res.py	Based on fit_glob_vel_model_Jian_upd7.0_log_res.py with k random term having
						logVs30mid, logVs30scl and s2 fixed, redetermining r1 and r2
fit_glob_vel_model_Jian_upd7.4GPdBr_log_res.py	Based on fit_glob_vel_model_Jian_upd7.0_log_res.py with k spatially varying 
						term having logVs30mid, logVs30scl, redetermining r1 and r2
fit_glob_vel_model_Jian_upd7.5GPdBr_log_res.py	Based on fit_glob_vel_model_Jian_upd7.0_log_res.py with fixed n and spatially 
						varying k through dBr, redetermining r1, r2, r3, and r4
fit_glob_vel_model_Jian_upd8.0_log_res.py       Global regression with same logistic scaling for k and n same midpoint and 
                                                scale, different amplitude					
