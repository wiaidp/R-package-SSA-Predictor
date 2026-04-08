# This function computes bk  in non-stationary I(1) case by imposing the cointegration constraint
# For given Lagrangian multiplier lambda the function derives bk based on cointegration solution, see formula 30 in Wildi 2026a
# Target(s):  
#   1. gamma_tilde is the MSE target as applied to epsilon_t 
#     this is not the proper target since it is a stationary finite MA filter applied to epsilon_t
#     its purpose is being the target for I-SSA optimization under the cointegration constraint (the constraint then includes the unit-root within the predictor)
#   2. gamma_mse: this is the MSE target as applied to original data in levels 
#     this is used for computing Gamma(0) in the cointegration constraint only
# For lambda=0 the SSA transformation is an identity, i.e., bk=gamma_tilde
# Xi is the Wold decomposition, Sigma is the summation (integration) operator (it is only used for computing target correlation); 
# M is the autocovariance generating matrix; B is used for setting-up cointegration constraint
bk_int_func<-function(lambda,gamma_mse,Xi,Sigma,Xi_tilde,M,B,gamma_tilde,ht_constraint)
{ #lambda1<-lambda_opt   lower_limit_nu<-lower_limit_nu_triangulation
  
  # the HT contsraint is interreted either as lag-one ACF or as effective HT depending on its magnitude  
  
  if (ht_constraint<(-1))
  {
    print("ht_constraint must be larger than -1")
    return()
  }
  # The HT constraint can be expressed either in terms of holding time or 
  # lag-one ACF. If ht_constraint>1 we interrpet this as the holding time. 
  # Otherwise it is interpreted as lag-one ACF
  if (ht_constraint>1)
  {
    print("ht_constraint is interpreted as a holding time")
    # I-SSA needs the lag-one ACF for implementing the HT constraint
    rho1<-compute_rho_from_ht(ht_constraint)$rho
  } else
  {
    print("ht_constraint is interpreted as lag-one ACF")
    rho1<-ht_constraint
  }
  rho1<-as.double(rho1)
  
  # Dimension checks  
  
  L<-dim(M)[1]
  if (length(gamma_mse)!=L)
  {
    print(paste("length of gamma differs from L=",L,sep=""))
  }
  if (dim(B)[1]!=L)
  {
    print(paste("dim(B) differs from L=",L,sep=""))
  }
  if (dim(Sigma)[1]!=L)
  {
    print(paste("dim(Sigma) differs from L=",L,sep=""))
  }
  if (dim(Xi_tilde)[1]!=L)
  {
    print(paste("dim(Xi_tilde) differs from L=",L,sep=""))
  }
  # First Cartesian basis vector for cointegration constraint, see section 5.3 technical paper  
  e1<-c(1,rep(0,L-1))
  e1
  # Cointegration constraint: sum of SSA coefficients should equal sum of MSE coefficients, see section 5.3 technical paper    
  Gamma0<-sum(gamma_mse)
  # Formula for bk given lambda under cointegration constraint, see section 5.3 technical paper    
  b_tilde<-solve(t(B)%*%t(Xi_tilde)%*%Xi_tilde%*%B+
                   lambda*t(B)%*%t(Xi)%*%(M-rho1*diag(rep(1,L)))%*%Xi%*%B)%*%
    (t(B)%*%t(Xi_tilde)%*%(gamma_tilde-Gamma0*Xi_tilde%*%e1)
     -lambda*Gamma0*t(B)%*%t(Xi)%*%(M-rho1*diag(rep(1,L)))%*%
       Xi%*%e1)
  # Derive b_x applied to original data from b_tilde, see section 5.3 technical paper  
  # This is also the same parameter which is applied to first differences of data in the synthetic stationary processes, see section 5.3 technical paper    
  b_x<-Gamma0*e1+B%*%b_tilde
  # Derive b_eps as applied epsilon: this is used to compute lag-one ACF rho_yy and MSE mse_yz of SSA below
  # It is not relevant for filtering!
  b_eps<-Xi%*%b_x
  
  # Compute lag-one acf      
  rho_yy<-as.double(t(b_eps)%*%M%*%b_eps/(t(b_eps)%*%b_eps))
  # The HT constraint can be expressed either in terms of holding time or 
  # lag-one ACF. If ht_constraint>1 we interrpet this as the holding time. 
  # Otherwise it is interpreted as lag-one ACF
  if (ht_constraint>1)
  {
    # I-SSA needs the lag-one ACF for implementing the HT constraint
    ht_issa<-compute_holding_time_from_rho_func(rho_yy)$ht
  } else
  {
    ht_issa<-rho_yy
  }
  
  
  # Compute criterion (objective function): two different targets: rho_yz (virtual target correlation based on synthetic stationary series) and mse_yz
  # We use mse_yz because formula of cointegrated SSA is simpler when relying on MSE objective  
  # 1. target correlation based on synthetic stationary series
  rho_yz<-as.double(t(Sigma%*%b_eps)%*%gamma_tilde)/(sqrt(t(Sigma%*%b_eps)%*%(Sigma%*%b_eps))*sqrt(t(gamma_tilde)%*%gamma_tilde))
  # 2. MSE  
  mse_yz<-as.double(t(gamma_tilde-Sigma%*%b_eps)%*%(gamma_tilde-Sigma%*%b_eps))
  return(list(b_x=b_x,b_eps=b_eps,ht_issa=ht_issa,rho_yz=rho_yz,mse_yz=mse_yz))
}




# This function is used in numerical optimization of Lagrangian lambda such that solution bk conforms with HT constraint
# It uses bk_int_func above and returns the absolute difference between desired rho1 and effective rho_yy
# Numerical optimization then minimizes this gap
b_optim<-function(lambda,gamma,Xi,Sigma,Xi_tilde,M,B,gamma_tilde,ht_constraint)
{
  ht_constraint<-as.double(ht_constraint)
  return(abs(bk_int_func(lambda,gamma,Xi,Sigma,Xi_tilde,M,B,gamma_tilde,ht_constraint)$ht_issa-ht_constraint))
}







# This function computes system matrices for I-SSA. 
# The target gamma_target can be of larger length than L, to improve accuracy of finite length MA-inversions when computing the MSE optimal filter
compute_issa_system_filters_func<-function(L,gamma_target,symmetric_target,delta,a1,b1)
{
# delta<-20
# We use a longer length for convolutions to exploit fully the finite-length target 
  if (symmetric_target)
  {
    L_conv<-L+max((delta+1+(L-1)/2),0)-1
#    L_conv<-500
  } else
  {
    L_conv<-L
  }
# The above is not useful: there is no more information to extract
  L_conv<-L
  
  
  gamma_target_sys<-gamma_target
  # Wold decomposition of ARMA (MA inversion)  
  xi<-c(1,ARMAtoMA(ar=a1,ma=b1,lag.max=L_conv-1))
  
  
  # Compute all system matrices: all matrices are specified in Wildi (2026a)
  Xi<-NULL
  for (i in 1:L_conv)
    Xi<-rbind(Xi,c(xi[i:1],rep(0,L_conv-i)))#c(1,0,0),c(1,1,0),c(1,1,1)))

  
  Sigma<-NULL
  for (i in 1:L_conv)
    Sigma<-rbind(Sigma,c(rep(1,i),rep(0,L_conv-i)))#c(1,0,0),c(1,1,0),c(1,1,1)))

  
  # Invert: gives Delta
  Delta<-solve(Sigma)

  # Convolve integration operator and Wold decomposition (section 5.3 in Wildi 2026a)  
  Xi_tilde<-(Sigma)%*%Xi
  
  # Compute gamma_mse: the optimal MSE filter applied to x_tilde the non stationary data
  # 1. Compute weights of MA-inversion of process i.e. x_tilde: this is applied to two-sided filter and it must be of sufficient length for the MA-inversions to hold  
  xi_int<-conv_two_filt_func(rep(1,L_conv),xi)$conv
  # 2. Convolution target filter and MA-inversion  
  hp_xi<-conv_two_filt_func(gamma_target_sys,xi_int)$conv
  # 3. Extract causal part i.e. remove future epsilon (which are replaced by forecast 0)
  # Distinguish symmetric and one-sided targets  
  if (symmetric_target)
  {
    if (delta<(-(L-1)/2))
    {
      print("The backcast does not change for delta<=-(L-1)/2")
      print("delta is set to -(L-1)/2")
      delta<--(L-1)/2
    }
    if ((delta+1+(L-1)/2)<1)
      print("(delta+1+(L-1)/2)<1: delta is corrected")
# Must add  (L-1)/2) to delta to be about the intended lag delta  
    hp_xi_causal<-hp_xi[max((delta+1+(L-1)/2),0):(length(hp_xi))]
  } else
  {
#    if (delta<0)
#    {
#      print("The backcast does not change for delta<=0")
#      print("delta is set to 0")
#      delta<-0
#    }
    if (delta<0)
    {
# Backcasting a causal target without HT constraint is not meaningful (it's just the shifted target)
# With an imposed HT the problem is non trivial though
# But this would need adding padding with zeroes which is not done yet      
      print("Currently we do not treat backcasts for causal targets")
      print("delta < 0: delta is corrected (delta=0: nowcast)")
    }
    hp_xi_causal<-hp_xi[max((delta+1),1):(length(hp_xi))]
  }
  

# 4. Deconvolute filt2 from filt1: filt1 is the convolution
# 4.1 Ensure that the filters have identical length: otherwise the 
# function deconvolute_func() pads the shorter filter with 0s which is 
# not correct in the case of integrated processes (because coefficients do not decay to zero) 
  common_len<-min(length(hp_xi_causal),length(xi_int))
  hp_xi_causal<-hp_xi_causal[1:common_len]
  xi_int<-xi_int[1:common_len]
  gamma_mse<-deconvolute_func(hp_xi_causal,xi_int)$dec_filt
# Restrict to filter length L is not exact when backcasting (but OK for now/forecasting)
# When backcasting (delta<0) the correct filter length increases by -delta=abs(delta): (L+ifelse(delta<0,abs(delta),0))
# This is because the length of the symmetric target is 2*L-1: this is also larger than L
  if (length(gamma_mse)<L)
    gamma_mse<-c(gamma_mse,rep(0,L-length(gamma_mse)))

  ts.plot(gamma_mse)
# Cointegration constraint: the difference of summed coefficients (transfer functio in frequency zero) should vanish  
  sum(gamma_mse)-sum(gamma_target_sys)
  # Compute transformed MSE filter: convolution of MSE (applied to non-stationary x_tilde) with Xi_tilde
  # This is used as the target in the optimization criterion
  # It is the finite MA approaximation of the MSE filter: it is applied to epsilon_t
  # However, the finite MA representation cannot be used as predictor (the predictor is non-stationary, whereas the finite MA is stationary)
  # Its sole purpose is to serve as target in the constrained optimization, assuming the cointegration constraint is imposed (see formula 30 in Wildi 2026a).
  gamma_tilde<-Xi_tilde%*%gamma_mse[1:ncol(Xi_tilde)]
  # Autocovariance generating matrix, see theorem 1  
  M<-matrix(nrow=L,ncol=L)
  M[L,]<-rep(0,L)
  M[L-1,]<-c(rep(0,L-1),0.5)
  for (i in 1:(L-2))
    M[i,]<-c(rep(0,i),0.5,rep(0,L-1-i))
  M<-M+t(M)
  M
  # Cointegration matrix, see section 5.3 technical paper  
  B<-rbind(rep(-1,L-1),diag(rep(1,L-1)))
  B
  return(list(B=B,M=M,gamma_tilde=gamma_tilde,gamma_mse=gamma_mse,Xi_tilde=Xi_tilde,Sigma=Sigma,Delta=Delta,Xi=Xi))
}




# This is the main function for I-SSA it calls the above functions 
# to compute the optimal I-SSA predictor that tracks gamma_target optimally while
# complying with a HT constraint expressed in first differences (ht_constraint).
# A symmetric target can be handled by supplying the left tail and setting symmetric_target==T. In this case the target will be mirrored at the center point.
# A symmetric target can also be handled by suypplying the casual form of the target and adding (L-1)/2+1 to delta (forecast, nowcast or backcast) 
ISSA_func<-function(ht_constraint,L,delta,gamma_target,symmetric_target=F,a1=0,b1=0,lambda_start=0)
{
  
  # Compute system matrices and filters
  filter_obj <- compute_issa_system_filters_func(L,gamma_target,symmetric_target,delta,a1,b1)
  
  B            <- filter_obj$B            # Cointegration matrix, see cited literature
  M            <- filter_obj$M            # Autocovariance generator
  gamma_mse    <- filter_obj$gamma_mse    # MSE-optimal filter (one-sided estimate of HP target)
  gamma_tilde  <- filter_obj$gamma_tilde  # Finite MA inversion of MSE predictor
  Sigma        <- filter_obj$Sigma        # Integration operator (see section 5.3)
  Delta        <- filter_obj$Delta        # Differencing operator: Delta%*%Sigma=I
  Xi           <- filter_obj$Xi           # Wold MA representation in matrix notation, see equation 22 in Wildi 2026a
  Xi_tilde     <- filter_obj$Xi_tilde     # Sigma%*%Xi (see section 5.3: Wold-decomposition of first differences convolved with integration operator)
  
  # Shortly:
  #   gamma_mse   : Benchmark 1
  #                 the MSE-optimal nowcast of the two-sided HP trend under 
  #                 the AR(1) model assumption: no other nowcast outperfroms
  #                 gamma_mse in terms of mean-squared filter error.
  #   hp_trend    : Benchmark 2 (suited for signaling economic phases in first differences)
  #
  #   gamma_tilde : used for optimization only (not relevant for filtering)
  #
  # Note:
  #   1. A cointegration constraint between the target z_t and I-SSA y_t ensures a 
  #   finite MSE for non-stationary I(1) processes. Wildi (2026a) also proposes 
  #   extensions to I(2) processes. The constraint is refelected in B above.
  
  lambda <- lambda_start
  
  # Classical numerical optimization procedure optim in R
  opt_obj <- optim(
    lambda,
    b_optim,
    lambda,
    gamma_mse,
    Xi,
    Sigma,
    Xi_tilde,
    M,
    B,
    gamma_tilde,
    ht_constraint
  )
  
  # Optimal Lagrange multiplier
  lambda_opt <- opt_obj$par
  # Negative values of the Lagrange multiplier indicate stronger smoothing of 
  # I-SSA compared to the MSE benchmark.

  # Compute I(1) cointegrated I-SSA solution based on lambda_opt
  bk_obj <- bk_int_func(
    lambda_opt,
    gamma_mse,
    Xi,
    Sigma,
    Xi_tilde,
    M,
    B,
    gamma_tilde,
    ht_constraint
  )
  
  if (F)
  {
    b_optim(lambda_opt,
            gamma_mse,
            Xi,
            Sigma,
            Xi_tilde,
            M,
            B,
            gamma_tilde,
            ht_constraint)
  }
  # Optimal I-SSA filter  
  b_x<-bk_obj$b_x
  
  return(list(b_x=b_x,bk_obj=bk_obj,lambda_opt=lambda_opt,gamma_mse=gamma_mse))
  
}



#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# The following two functions are used for the I-SSA trend introduced in 2026
# The target is the identity shifted by delta
# This is a special case of the general ISSA above
compute_issa_trend_system_filters_func<-function(L,a_vec,b_vec)
{
  
  
  # Wold decomposition (MA inversion)  
  xi<-c(1,ARMAtoMA(ar=a_vec,ma=b_vec,lag.max=L-1))
  
  
  # Compute all system matrices: all matrices are specified in Wildi (2026a)
  Xi<-NULL
  for (i in 1:L)
    Xi<-rbind(Xi,c(xi[i:1],rep(0,L-i)))#c(1,0,0),c(1,1,0),c(1,1,1)))
  Xi
  
  
  Sigma<-NULL
  for (i in 1:L)
    Sigma<-rbind(Sigma,c(rep(1,i),rep(0,L-i)))#c(1,0,0),c(1,1,0),c(1,1,1)))
  Sigma
  
  
  # Invert: gives Delta
  Delta<-solve(Sigma)
  Delta
  
  # Convolve integration operator and Wold decomposition (section 5.3 in Wildi 2026a)  
  Xi_tilde<-(Sigma)%*%Xi
  
  # Compute gamma_mse: the optimal MSE filter applied to x_tilde the non stationary data
  # 1. Compute weights of MA-inversion of process i.e. x_tilde: this is applied to two-sided filter and it must be of the same length as that filter  
  xi_int<-conv_two_filt_func(rep(1,L),xi)$conv
  
  
  # Autocovariance generating matrix, see theorem 1  
  M<-matrix(nrow=L,ncol=L)
  M[L,]<-rep(0,L)
  M[L-1,]<-c(rep(0,L-1),0.5)
  for (i in 1:(L-2))
    M[i,]<-c(rep(0,i),0.5,rep(0,L-1-i))
  M<-M+t(M)
  M
  # Cointegration matrix, see section 5.3 technical paper  
  B<-rbind(rep(-1,L-1),diag(rep(1,L-1)))
  B
  return(list(B=B,M=M,Xi_tilde=Xi_tilde,Sigma=Sigma,Delta=Delta,Xi=Xi))
}


# New 2026 I-SSA trend function
# This is based on I-SSA applied to an I(1) series
# The target is x_{t+delta}, with -T<=delta<= 0: a now cast or backcast of 
# the original I(1) data.
# It imposes a HT constraint in stationary first differences of the process.
# The HT constraint is interperetd as effective HT or as lag-one ACF depending 
# on the value of ht_constraint.
# A length L I-SSA trend is obtained, assuming the differenced data conforms
# to an ARMA with parameters a1 (vector of length p) and b1 (vector of length q).
# Default values are white noise.
# One can supply a better initial value for the Lagrangian: lambda_start=0 
# assumes an MSE nowcast, i.e., the identity (in smoothing).
ISSA_Trend_func<-function(ht_constraint,L,delta,a1=0,b1=0,lambda_start=0)
{
  
  if (ht_constraint<(-1))
  {
    print("ht_constraint must be larger than -1")
    return()
  }
  # The HT constraint can be expressed either in terms of holding time or 
  # lag-one ACF. If ht_constraint>1 we interrpet this as the holding time. 
  # Otherwise it is interpreted as lag-one ACF
  if (ht_constraint>1)
  {
    print("ht_constraint is interpreted as a holding time")
    # I-SSA needs the lag-one ACF for implementing the HT constraint
    rho1<-compute_rho_from_ht(ht_constraint)$rho
  } else
  {
    print("ht_constraint is interpreted as lag-one ACF")
    rho1<-ht_constraint
  }
  rho1<-as.double(rho1)
  
  # ─────────────────────────────────────────────────────────────────────────────
  # 1.1  Model Setup for I-SSA
  # ─────────────────────────────────────────────────────────────────────────────
  # Compute system matrices and filters
  filter_obj <- compute_issa_trend_system_filters_func(L, a1, b1)
  
  B           <- filter_obj$B            # Cointegration matrix (see cited literature)
  M           <- filter_obj$M            # Lag-one autocovariance generating matrix
  Xi_tilde    <- filter_obj$Xi_tilde     # Convolution operator (see Section 5.3: Wold
  #   decomposition of first differences convolved
  #   with the integration operator)
  Sigma       <- filter_obj$Sigma        # Integration operator (see Section 5.3)
  Delta       <- filter_obj$Delta        # Differencing operator
  Xi          <- filter_obj$Xi           # Wold MA representation in matrix form
  
  # ─────────────────────────────────────────────────────────────────────────────
  # 1.2  I-SSA Settings
  # ─────────────────────────────────────────────────────────────────────────────
  
  # Target in levels: we want to track the random walk x_t directly,
  # so the target filter is the identity (a unit spike at lag -delta).
  # This differs from Exercise 6, where the target was the one-sided HP of x_t.
  if (delta>0)
  {
    print("delta must by smaller or equal 0")
    return()
  }
  target_filter <- c(rep(0, -delta), 1, rep(0, L + delta - 1))
  
  # Target in first differences: apply the summation operator to the level
  # target filter. This yields a finite-length proxy of the effective
  # first-difference target used in optimisation. The proxy converges to the
  # true target as filter length L grows, because MA coefficients decay
  # sufficiently fast under the cointegration constraint (see Section 5.3 in
  # Wildi 2026a for details).
  Xi_tilde <- (Sigma) %*% Xi
  target_filter_diff <- Xi_tilde %*% target_filter
  
  
  # The targets do not hint at HP: HP enters into I-SSA through its HT only
  
  # ─────────────────────────────────────────────────────────────────────────────
  # 1.3  Compute I-SSA Solution (Wildi 2026a, Sections 5.3–5.4)
  # ─────────────────────────────────────────────────────────────────────────────
  # Numerical optimisation (optim) determines the optimal Lagrange multiplier
  # lambda ensuring compliance with the HT constraint.
  # Initialising at lambda = 0 corresponds to the unconstrained MSE benchmark.
  # Note: lambda here is the SSA Lagrange multiplier and is unrelated to the HP
  # regularisation parameter lambda_hp.
  
  # 1.3.1  I-SSA Optimisation
  # ─────────────────────────────────────────────────────────────────────────────
  lambda <- 0
  
  opt_obj <- optim(
    lambda,
    b_optim,
    lambda,
    target_filter,
    Xi,
    Sigma,
    Xi_tilde,
    M,
    B,
    target_filter_diff,
    rho1
  )
  
  # Optimal Lagrange multiplier
  lambda_opt <- opt_obj$par
  
  # Compute the I(1) cointegrated I-SSA solution based on lambda_opt
  bk_obj <- bk_int_func(
    lambda_opt,
    target_filter,
    Xi,
    Sigma,
    Xi_tilde,
    M,
    B,
    target_filter_diff,
    rho1
  )
  b_x<-bk_obj$b_x
  return(list(b_x=b_x,bk_obj=bk_obj,lambda_opt=lambda_opt,rho1=rho1,ht1=ht1,target_filter=target_filter))
}



#-------------------------------------------------------------------------
#-------------------------------------------------------------------------
# The following two functions are used for I-SSA tracking HP
# The target is the two-sided HP
# The target is computed internally to shorten the exercise in the tutorial

# This function computes system matrices for I-SSA when applied to HP
compute_issa_hp_system_filters_func<-function(L,gamma_target,a_vec,b_vec)
{
  # Compute the two-sided filter of double length 2*(L-1)+1: the double length 
  # increases accuracy of finite MA-inversions when computing acausal MSE predictor
  
  
  # Wold decomposition of ARMA (MA inversion)  
  xi<-c(1,ARMAtoMA(ar=a_vec,ma=b_vec,lag.max=L-1))
  
  
  # Compute all system matrices: all matrices are specified in Wildi (2026a)
  Xi<-NULL
  for (i in 1:L)
    Xi<-rbind(Xi,c(xi[i:1],rep(0,L-i)))#c(1,0,0),c(1,1,0),c(1,1,1)))
  Xi
  
  
  Sigma<-NULL
  for (i in 1:L)
    Sigma<-rbind(Sigma,c(rep(1,i),rep(0,L-i)))#c(1,0,0),c(1,1,0),c(1,1,1)))
  Sigma
  
  
  # Invert: gives Delta
  Delta<-solve(Sigma)
  Delta
  
  # Convolve integration operator and Wold decomposition (section 5.3 in Wildi 2026a)  
  Xi_tilde<-(Sigma)%*%Xi
  
  # Compute gamma_mse: the optimal MSE filter applied to x_tilde the non stationary data
  # 1. Compute weights of MA-inversion of process i.e. x_tilde: this is applied to two-sided filter and it must be of the same length as that filter  
  xi_int<-conv_two_filt_func(rep(1,length(hp_two)),xi)$conv
  # 2. Convolution target filter and MA-inversion  
  hp_xi<-conv_two_filt_func(hp_two,xi_int)$conv
  # 3. Extract causal part i.e. remove future epsilon (which are replaced by forecast 0)
  hp_xi_causal<-hp_xi[L:(length(hp_xi))]
  # 4. Deconvolute filt2 from filt1: filt1 is the convolution
  gamma_mse<-deconvolute_func(hp_xi_causal,xi_int[1:L])$dec_filt
  # Coefficients add to one (unit-root constraint inherited from hp_two)  
  sum(gamma_mse)
  
  if (F)
  {
    # MSE filter for random-walk: the above specification is more general since it accounts for Wold decomposition of first differences too    
    gamma_mse<-hp_two[L:length(hp_two)]+c(sum(hp_two[(L+1):length(hp_two)]),rep(0,L-1))
    ts.plot(gamma_mse)
  }
  
  # Compute transformed MSE filter: convolution of MSE (applied to non-stationary x_tilde) with Xi_tilde
  # This is used as the target in the optimization criterion
  # It is the finite MA approaximation of the MSE filter: it is applied to epsilon_t
  # However, the finite MA representation cannot be used as predictor (the predictor is non-stationary, whereas the finite MA is stationary)
  # Its sole purpose is to serve as target in the constrained optimization, assuming the cointegration constraint is imposed (see formula 30 in Wildi 2026a).
  gamma_tilde<-Xi_tilde%*%gamma_mse
  # Autocovariance generating matrix, see theorem 1  
  M<-matrix(nrow=L,ncol=L)
  M[L,]<-rep(0,L)
  M[L-1,]<-c(rep(0,L-1),0.5)
  for (i in 1:(L-2))
    M[i,]<-c(rep(0,i),0.5,rep(0,L-1-i))
  M<-M+t(M)
  M
  # Cointegration matrix, see section 5.3 technical paper  
  B<-rbind(rep(-1,L-1),diag(rep(1,L-1)))
  B
  return(list(B=B,M=M,gamma_tilde=gamma_tilde,gamma_mse=gamma_mse,Xi_tilde=Xi_tilde,Sigma=Sigma,Delta=Delta,Xi=Xi,hp_two=hp_two,hp_trend=hp_trend))
}




# This is the main function for I-SSA when tracking HP: it calls the above functions 
# to compute the optimal I-SSA predictor that tracks the two-sided HP optimally while
# complying with a HT constraint expressed in first differences (ht_constraint)
ISSA_HP_func<-function(ht_constraint,L,delta,lambda_hp,a1=0,b1=0,lambda_start=0)
{
  
  # Compute system matrices and filters
  filter_obj <- compute_issa_hp_system_filters_func(L, lambda_hp, a1, b1)
  
  B            <- filter_obj$B            # Cointegration matrix, see cited literature
  M            <- filter_obj$M            # Autocovariance generator
  gamma_mse    <- filter_obj$gamma_mse    # MSE-optimal filter (one-sided estimate of HP target)
  gamma_tilde  <- filter_obj$gamma_tilde  # Finite MA inversion of MSE predictor
  Sigma        <- filter_obj$Sigma        # Integration operator (see section 5.3)
  Delta        <- filter_obj$Delta        # Differencing operator: Delta%*%Sigma=I
  Xi           <- filter_obj$Xi           # Wold MA representation in matrix notation, see equation 22 in Wildi 2026a
  Xi_tilde     <- filter_obj$Xi_tilde     # Sigma%*%Xi (see section 5.3: Wold-decomposition of first differences convolved with integration operator)
  hp_two       <- filter_obj$hp_two       # Two-sided HP target
  hp_trend     <- filter_obj$hp_trend     # Classic one-sided HP (HP-C): benchmark for I-SSA customization
  
  # Shortly:
  #   gamma_mse   : Benchmark 1
  #                 the MSE-optimal nowcast of the two-sided HP trend under 
  #                 the AR(1) model assumption: no other nowcast outperfroms
  #                 gamma_mse in terms of mean-squared filter error.
  #   hp_trend    : Benchmark 2 (suited for signaling economic phases in first differences)
  #
  #   gamma_tilde : used for optimization only (not relevant for filtering)
  #
  # Note:
  #   1. A cointegration constraint between the target z_t and I-SSA y_t ensures a 
  #   finite MSE for non-stationary I(1) processes. Wildi (2026a) also proposes 
  #   extensions to I(2) processes. The constraint is refelected in B above.
  
  lambda <- lambda_start
  
  # Classical numerical optimization procedure optim in R
  opt_obj <- optim(
    lambda,
    b_optim,
    lambda,
    gamma_mse,
    Xi,
    Sigma,
    Xi_tilde,
    M,
    B,
    gamma_tilde,
    ht_constraint
  )
  
  # Optimal Lagrange multiplier
  lambda_opt <- opt_obj$par
  # Negative values of the Lagrange multiplier indicate stronger smoothing of 
  # I-SSA compared to the MSE benchmark.
  lambda_opt
  
  # Compute I(1) cointegrated I-SSA solution based on lambda_opt
  bk_obj <- bk_int_func(
    lambda_opt,
    gamma_mse,
    Xi,
    Sigma,
    Xi_tilde,
    M,
    B,
    gamma_tilde,
    ht_constraint
  )
  # Optimal I-SSA filter  
  b_x<-bk_obj$b_x

  return(list(b_x=b_x,bk_obj=bk_obj,lambda_opt=lambda_opt,gamma_mse=gamma_mse,hp_two=hp_two,hp_one=hp_one))
  
}

