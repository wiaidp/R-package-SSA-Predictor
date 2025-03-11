# We here introduce to the M-SSA. see paper provide link???

# Example 1: we replicate the univariate SSA by M-SSA









# We can replicate univariate models by specifying `diagonal' xi and Sigma as below
if (F)
{
  xi<-xi_mat_uni_bi
  Sigma<-Sigma_uni_bi
  rho0<-rho0_uni_vec
  ht_ssa_vec<-compute_holding_time_from_rho_func(rho0)$ht
  
}






# Example 2: we illustrate a bivariate M-SSA based on a `bogus' design 
#  -In principle the same as example 1 but we add an irrelevant (cross-sectionally uncorrelated white noise) series 



