Filename					Description
--------------					--------------
fit_glob_vel_models_Jian_upd1_log_res.py	Global regression with Vs30 fixed V0 and logistic scaling for n.
fit_glob_vel_models_Jian_upd2_log_res.py	Global regression based on fit_glob_vel_models_Jian_upd_log_res.py
						log(Vs30) scaling for k "exp(r1 * log(Vs30) + r2)".
fit_glob_vel_models_Jian_upd3_log_res.py	Global regression based on fit_glob_vel_models_Jian_upd_log_res.py
						Constant k "exp(r1)".
fit_glob_vel_models_Jian_upd3dB_log_res.py	Global regression based on fit_glob_vel_models_Jian_upd_log_res.py
						Constant k with within profile effects "exp(r1 + dBr)".
fit_glob_vel_models_Jian_upd4_log_res.py	Global regression with bilinear scaling for k.
fit_glob_vel_models_Jian_upd4dB_log_res.py	Global regression with bilinear scaling for k and between profile 
						effects "exp(r1 + r2(Vs30)*Vs30 + dBr)".

