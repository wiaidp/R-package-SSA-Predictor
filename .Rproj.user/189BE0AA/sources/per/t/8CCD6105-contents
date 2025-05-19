# These are the main functions for the simple sign accuracy (SSA) criterion 
# This code is used in the JBCY-article
#   -JBCY relies on simple univariate designs 
#   -White noise data generating process
# The code can handle more complex problems
#   -Arbitrary stationary processes (extending the restrictive white noise assumption)
#   -Multivariate problems (article in preparation)
# Idea: the SSA grafts onto an extisting design (for example ARMA, state-space, HP, B-K, CF,...) and 
#   1. modifies characteristics in terms of timeliness and/or smoothness
#   2. stays as close as possible to the original design (optimality criterion)
# Smoothness can be addressed directly, by the hyperparameter rho1 or, equivalently, ht1 (holding-time constraint)
# Timeliness (lead or lag at zero-crossings) can be addressed indirectly, by modifying the forecast horizon (next step with formal timeliness applications is in preparation).
# Assumption: the user has pre-selected a particular filter/predictor which should be `enhanced' by SSA in terms of timeliness and/or smoothness


#----------------------------------------------------
# The following three functions compute holding-times and link lag-one acf and holding-times, see section 1 in JBCY paper
compute_holding_time_func<-function(b)
{
  if (length(b)>1)
  {  
    rho_ff1<-b[1:(length(b)-1)]%*%b[2:length(b)]/sum(b^2)
    # Mean holding-time
    ht<-1/(2*(0.25-asin(rho_ff1)/(2*pi)))
    # Alternative expression  
    if (F)
      ht<-pi/acos(rho_ff1)
  } else
  {
    ht<-2
    rho_ff1<-0
  }
  return(list(ht=ht,rho_ff1=rho_ff1))
}

compute_rho_from_ht<-function(ht)
{  
  rho<-sin(-((1/ht)/2-0.25)*2*pi)
  return(list(rho=rho))
}

compute_holding_time_from_rho_func<-function(rho_ff1)
{
  # Mean holding-time
  ht<-1/(2*(0.25-asin(rho_ff1)/(2*pi)))
  return(list(ht=ht))
}

rhomax_func<-function(L)
{
  rhomax<-max(M_func(L,NULL)$eigen_M_obj$values)
  return(rhomax=rhomax)
    
}  


compute_empirical_ht_func<-function(x)
{  
  len<-length(na.exclude(x))
  empirical_ht<-len/length(which(sign(x[2:len])!=sign(x[1:(len-1)])))
  return(list(empirical_ht=empirical_ht))
}

#--------------------------------
# Some useful convolution functions

# Convolution of two functions
conv_two_filt_func<-function(filt1,filt2)
{
  L<-max(length(filt1),length(filt2))
  if (length(filt1)<L)
    filt1<-c(filt1,rep(0,L-length(filt1)))
  if (length(filt2)<L)
    filt2<-c(filt2,rep(0,L-length(filt2)))
  conv<-filt1
  L<-length(filt1)
  for (i in 1:L)
  {
    conv[i]<-sum(filt1[1:i]*filt2[i:1])
  }  
  return(list(conv=conv))
}

# Convolution with summation filter (unit-root assumption)
conv_with_unitroot_func<-function(filt)
{
  conv<-filt
  L<-length(filt)
  for (i in 1:L)
  {
    conv[i]<-sum(filt[1:i])
  }  
  return(list(conv=conv))
}

# Deconvolute filt2 from filt1: filt1 is the convolution
# See section 2 of JBCY paper
deconvolute_func<-function(filt1,filt2)
{
  filt1<-as.vector(filt1)
  filt2<-as.vector(filt2)
  if (length(filt1)<length(filt2))
    filt1<-c(filt1,rep(0,length(filt2)-length(filt1)))
  if (length(filt2)<length(filt2))
    filt2<-c(filt2,rep(0,length(filt1)-length(filt2)))
  
  # filt1<-as.vector(gammak_mse)    filt2<-as.vector(hp_mse)
  L<-length(filt1)
  dec_filt<-filt1
  dec_filt[1]<-filt1[1]/filt2[1]
  for (i in 2:L)
  {
    dec_filt[i]<-(filt1[i]-sum(dec_filt[1:(i-1)]*filt2[i:2]))/filt2[1]
  }  
  return(list(dec_filt=dec_filt))
}


sa_from_rho_func<-function(rho)
{
  return(0.5+asin(rho)/pi)
}


#-------------------------------------------------------
# SSA functions
# This function computes the solution of corollary 1 in JBCY paper 
# It allows for arbitrary stationary processes as well as multivariate problems (the latter are an extension not discussed in the JBCY paper)
# The algorithm looks for an optimal nu based on ht (holding-time). Numerical optimization is based on either of two functions:
#   1. fast_halfway_triangulation_find_lambda1_subject_to_holding_time_constraint_func or on 
#   2. grid_search_find_lambda1_subject_to_holding_time_constraint_func
# The first function is very fast and precise but it requires unicity of nu as a function of ht, see (JTSE paper in preparation)
# The second is significantly slower (or less precise) but it can handle non-uniqueness
# Classic prediction or signal extraction problems can be handled by the first (fast/precise) function
# The second (slower) function tackles also 'exotic' prediction/signal extraction problems
#   Exotic here means: either singular designs or designs with very strong (excessive?) smoothing or holding-time requirements
# An older simpler version is included at the end of the file: the old function is much simpler (white noise or MA(1) and univariate problems only): it can address all issues in the JBCY-paper (but not the papers in preparation)
# Parameters
#   1. split_grid: integer; the search-interval for nu (unknown parameter) is splitted by successive halvings so that the resolution is 1/2^split_grid
#     Values >10 are generally large enough for determination of nu
#     This parameter is relevant for first (fast) numerical optimization function above
#   2. L: integer; filter length
#   3. forecast_horizon_vec: integer vector; a vector of deltas (parameter delta in article)
#     -delta=0: nowcast, delta>0: forecast; delta<0: backcast
#     -The function loops through all deltas in the vector and computes solutions for each one
#   4. grid_size: integer; size of grid for second (slow) optimization function: the interval is split into grid_size equidistant points
#     This parameter is relevant for second (slow) optimization function only
#   5. gammak_generic: real vector; target
#     -Could be the bi-infinite filter
#     -Could be the MSE-estimate of bi-infinite target
#     -Whatever should be enhanced by SSA in terms of smoothness/timeliness
#   6. rho1: real; holding-time constraint in terms of lag-one acf (could be easily converted into or from holding-time by relying on above functions)
#   7. with_negative_lambda: Boolean
#       -if true: allow for filters with lots of zero-crossings 
#       -if false: allow for smoothing filters (default setting)
#   8. xi: real vector;Wold-decomposition of data if the latter is not white noise
#     -MA-inversion of AR or ARMA- or VAR- or VARMA-models
#     -if NULL: data is assumed to be white noise. This is the default setting. It works well for first differences or log-returns of non-seasonal economic time series. This is the setting used in the JBCY-paper
#   9. lower_limit_nu: character; has three possible settings
#     a. lower_limit_nu="rhomax": this is the default setting. It activates the first (fast) optimization routine. Works for all classic (non-exotic) applications
#     b. lower_limit_nu="2": this setting restricts the search to 'non unit-root' cases i.e. filter weights decay to zero 'fast'. It activates the first (fast) optimization routine) 
#     c. lower_limit_nu="0": this setting extends the search to all possible cases, including exotic and/or singular cases (these are mostly of academic interest: they require second slow grid-search function)
#         In these 'exotic' cases the SSA-filter follows a unit-root specification i.e. weights do not decay in the usual way
#   10. Sigma: addresses the cross-correlation noise structure in multivariate applications. 
#     Default value NULL means: univariate case
# The function performs dimensions checks 
#   -In some cases it completes or corrects automatically incomplete settings: in such a case it always prints a warning
#   -In other 'faulty' cases the function stops with an error message 
set_hyper<-function()
{
  grid_size=10;with_negative_lambda=F;lower_limit_nu="rhomax";Sigma=NULL;forecast_horizon_vec<-forecast_horizon;split_grid<-20
}



SSA_func<-function(L,forecast_horizon_vec,gammak_generic,rho1,xi=NULL,Sigma=NULL,split_grid=20,grid_size=10,with_negative_lambda=F,lower_limit_nu="rhomax")
{ 
# Check dimensions  
  if (!is.null(Sigma))
  {  
# Multivariate case    
    if (dim(Sigma)[1]!=dim(Sigma)[2])
    {  
      print("Error: Sigma is not a rectangular matrix")
      return()
    }
    if (dim(Sigma)[1]!=nrow(gammak_generic))
    {  
      print("Error: dimension of Sigma does not agree with gammak_generic")
      return()
    }
    if (dim(gammak_generic)[1]!=dim(Sigma)[1])
    {
      print("Error: dim(gammak_generic)[1]!=dim(Sigma)[1]")
      return()
    }  
    if (dim(gammak_generic)[2]!=L*dim(Sigma)[1])
    {
      print("Error: dim(gammak_generic)[2]!=L*dim(Sigma)[1]")
      return()
    }  
    if (!is.null(xi))
    if (nrow(xi)<L)
    {
      xi<-cbind(xi,matrix(rep(0,(L-nrow(xi))*dim(Sigma)[1]),ncol=L-nrow(xi)))
    }  
    
  } else
  {
# Univariate case    
    if (length(gammak_generic)<L)
    {
      gammak_generic<-c(gammak_generic,rep(0,L-length(gammak_generic)))
      print("Warning: length(gammak_generic)<L will be extended with zeroes")
    }
    if (!is.null(xi)&is.vector(xi))
    {
      xi<-matrix(xi,nrow=1)
    } 
    if (!is.null(xi))
    if (ncol(xi)<max(L,L+max(forecast_horizon_vec)))
    {
      print("Warning: xi is shorter than L+forecast-horizon: it will be extended with zeroes")      
      xi<-cbind(xi,matrix(rep(0,(max(L,L+max(forecast_horizon_vec))-nrow(xi))),ncol=max(L,L+max(forecast_horizon_vec))-nrow(xi)))
    }  
    
    
  }  

# Initialize outputs   
  bk_mat<-bk_x_mat<-crit_rhoyz<-crit_rhoyy<-crit_rhoy_target<-gammak_target_target_mat<-gammak_target_target_symmetric_mat<-rho_mat<-NULL
# If xi is not specified in call then it does not exist: here we define it as NULL  
  if (is.null(xi))
    xi<-NULL
  if (is.null(Sigma))
    Sigma<-NULL
  rho_mat<-w_mat<-ssa_eps<-ssa_x<-NULL

# Optimization    
# Loop over all forecast horizons  
  for (i in 1:length(forecast_horizon_vec))#i<-1
  {  
    forecast_horizon<-forecast_horizon_vec[i]
# If lower_limit_nu!=0 we use fast triangulation
#   This procedure assumes that |nu|>2 or |nu|>2*rhomax so that solution is unique. It makes use of strong monotonicity of rho as a function of nu    
# Otherwise i.e. if lower_limit_nu=="0" then we use grid-search
#   Grid-search is then restricted to [0,2] or [-2,2]  
    if (lower_limit_nu!="0")
    {
# In practice fast_halfway_triangulation applies to most prediction applications: note that depending on the selection 
#     of lower_limit_nu (rhomax, 2, or 0) the search is over $|nu|>2$, $|nu>2rhomax|$ or all nu. However, in the latter case
#     the function returns only one (of possibly many) solutions to the holding-time constraint. In such a case one should
#     use 'brute force' grid search (in general this would require a pretty uncommon target with non-decaying coefficients)      
      opt_obj<-fast_halfway_triangulation_find_lambda1_subject_to_holding_time_constraint_func(split_grid,L,gammak_generic,rho1,forecast_horizon,xi,lower_limit_nu,Sigma)
      if (is.null(opt_obj))
        return()
# rho_mat is NULL since we do not have criterion values for all grid-points on equidistant grid
# It is assumed that solution is unique i.e. |nu|>2*rho_max      
      rho_mat<-NULL
    } else
    {
# Grid-search would be required if $|nu|<2*rho_max(L)$ because in this case the solution to the holding-time constraint is not unique anymore
#   -In this case grid-search computes all solutions and returns a matrix rho_mat with entries abs(rho_i-rho(y,y,1)) and rho(y,z,delta) i.e. the criterion value
#   -One then has to look for all entries with small or vanishing abs(rho_i-rho(y,y,1)) (holding-time constraint is met). 
#     The SSA-solution is that entry which maximizes the criterion value rho(y,z,delta)     
      opt_obj<-grid_search_find_lambda1_subject_to_holding_time_constraint_func(grid_size,L,gammak_generic,rho1,forecast_horizon,with_negative_lambda,xi,lower_limit_nu,Sigma)
      if (is.null(opt_obj))
        return()
# rho_mat collects criterion value (correlation SSA with target), lag-one acf (holding-time) and corresponding nu
# Ordering is such that first row corresponds to nu which 1. maximizes criterion value and 2. minimizes deviation from holding-time
# First row should be best overall solution in case of multiple solutions (if $|nu|<2*rho_max$) 
# This is not used for fast_halfway_triangulation_find_lambda1_subject_to_holding_time_constraint_func above because 
#   there it is assumed that solution is unique i.e. |nu|>02*rho_max      
      rho_mat<-cbind(rho_mat,opt_obj$rho_mat)
    }
    
# Coefficients as applied to xt: xt can be autocorrelated    
    bk_mat<-cbind(bk_mat,opt_obj$bk_best)
# Coefficients as applied to epsilont in Wold decomposition of xt (convolution): 
#   bk_x_mat is NULL if xi==NULL (white noise)     
    bk_x_mat=cbind(bk_x_mat,opt_obj$bk_x)
# Optimal lambda and nu: note that nu=c*(lambda+1/lambda) where c depends on the choice of lower_limit_nu, see function bk_func below for details      
#   Since we do not return c (at this stage) lambda_opt is not useful; but nu_opt is!    
    lambda_opt<-opt_obj$lambda_opt
    nu_opt=opt_obj$nu_opt
# MSE as applied to epsilont    
    gammak_mse=t(opt_obj$gammak_mse)
# Objective function of SSA-criterion: correlation with MSE    
    crit_rhoyz<-c(crit_rhoyz,opt_obj$crit_rhoyz)
# Lag-one acf of SSA-estimate (can be converted to holding-time: this should match the constraint)    
    crit_rhoyy<-c(crit_rhoyy,opt_obj$crit_rhoyy)
# Objective function of SSA-criterion: correlation with proper target 
# The difference between crit_rhoyz and crit_rhoy_target is addressed in proposition 4 of JBCY paper    
    crit_rhoy_target<-c(crit_rhoy_target,opt_obj$crit_rhoy_target)
# Spectral decomposition of (MSE-) target: see section 3 in JBCY paper    
    w_mat<-cbind(w_mat,opt_obj$w)
# Spectral Fourier vectors
    V<-opt_obj$V
# Provide optimal MSE scaling of SSA    
    scaling<-as.double(t(gammak_mse)%*%bk_mat[,i]/t(bk_mat[,i])%*%bk_mat[,i])
    ssa_eps<-cbind(ssa_eps,scaling*bk_mat[,i])
    if (!is.null(xi))
    {  
      xi<-t(xi[1:L])
      gammak_x_mse<-matrix(deconvolute_func(gammak_mse,xi)$dec_filt,ncol=1)
    } else
    {
      gammak_x_mse<-gammak_mse
    }
    scaling<-as.double(t(gammak_x_mse)%*%bk_x_mat[,i]/t(bk_x_mat[,i])%*%bk_x_mat[,i])
    ssa_x<-cbind(ssa_x,scaling*bk_x_mat[,i])
    mse_eps<-gammak_mse
    mse_x<-gammak_x_mse
    
  }

# Brief check: for a lowpass design preserving signs the sum of the filter coefficients should be strictly positive
# Local check: this is not returned and is invisible to the user  
  apply(bk_mat,2,sum)
  
# In some cases the deconvolution can become unstable: filter coefficients at high lags are larger than at early lags
# In such a case one could apply bk_mat to epsilont (instead of bk_x_mat to xt): the corresponding filter is easier to interpret   
  if (any(apply(abs(bk_x_mat)[(L-min(L/2,4)):L,,drop=F],2,mean)>apply(abs(bk_x_mat)[1:min(L/2,4),,drop=F],2,mean)))
  {
    print("Warning: deconvolution possibly unstable. Check ssa_x")
    print("One can use ssa_eps applied to epsilont instead: transform original data xt into epsilont and apply ssa_eps")
  } 
# Description of returned variables:
# 1. Filters  
# ssa_x: optimal SSA filter as applied to xt and scaled for MSE optimality  
# ssa_eps: optimal SSA filter as applied to epsilont and scaled for MSE optimality    
# mse_eps: MSE filter as applied to epsilont
# mse_x: optimal MSE filter as applied to xt
# 2. Criteria (performance measures)
# crit_rhoyy: lag-one acf of SSA filter (should match holding-time constraint)
# crit_rhoy_target: criterion value i.e. correlation of SSA with effective target (symmetric two-sided filter)
# crit_rhoyz: SSA criterion value i.e. correlation with one-sided MSE benchmark

# The following additional variables are typically irrelevant for standard applications 
# nu_opt: optimal nu in theorem 1 of JBCY paper
# lambda_opt: optimal lambda whereby nu=lambda+1/lambda and nu is defined in theorem 1 in JBCY paper 
# rho_mat (mostly irrelevant: has to do with spectral decomposition, see section 3 of JBCY paper)
# w_mat (mostly irrelevant: has to do with spectral decomposition)
  
  if (abs(lambda_opt)<0.001)
  {
    print(paste("Lambda_opt=",lambda_opt," is close to zero (MSE filter)",sep=""))
    print("SSA-filters are identified with MSE filters")
    ssa_eps<-mse_eps
    ssa_x<-mse_x
  }
  
  return(list(crit_rhoyy=crit_rhoyy,ssa_eps=ssa_eps,lambda_opt=lambda_opt,crit_rhoy_target=crit_rhoy_target,
              ssa_x=ssa_x,crit_rhoyz=crit_rhoyz,nu_opt=nu_opt,mse_eps=mse_eps,rho_mat=rho_mat,
              w_mat=w_mat,mse_x=mse_x,ssa_eps=ssa_eps,V=V))
} 





# This function is called by SSA_func above if lower_limit_nu!="0" (default setting is "rhomax") 
# It implements the fast numerical optimization, assuming that  |nu|>2*rhomax(L) where rhomax is the maximal lag-one acf of the filter, see proposition 3 in JBCY paper
# The variables are described above (see SSA_func)
# Additional information about lower_limit_nu
# Trick: the parametrization of nu in this function is  nu<-(lambda1+1/lambda1)*c where nu is the parameter in theorem 1 and the search is conducted for lambda (instead of nu) in [-1,0[+]0,1] i.e. [-1,1]-{0}. 
#   Depending on the character string lower_limit_nu, various c are possible in this equation
#   -lower_limit_nu=="rhomax": default value for c 
#     In this case |nu|>2*rhomax(L): solution is unique but unit-roots are possible
#   -lower_limit_nu=="0"
#     In this case sqrt(gridsize)>|nu|>2/sqrt(gridsize): all nu are allowed if gridsize is large
#   -lower_limit_nu=="2": in this case |nu|>2 i.e. uniqueness and no unit-root so that bk decay to zero 'fast' (classic application case if holding-time constraint is not too large)
# -split_grid: search interval is splitted by successive halvings so that resolution is 1/2^split_grid
#   -If split_grid=10 then resolution corresponds to grid-search in a grid of 2^10~1000 grid-points
#   -This is much faster than grid-search (or much more precise for the same computation time)
#   -Potential problem: it assumes |nu|>2 (i.e. nu=lambda+1/lambda and solution is unique)

fast_halfway_triangulation_find_lambda1_subject_to_holding_time_constraint_func<-function(split_grid,L,gammak_generic,rho1,forecast_horizon,xi=NULL,lower_limit_nu="rhomax",Sigma=NULL)
{
# Check lower_limit_nu: if is different from the three options one assumes standard setting rhomax: a Warning is printed  
  if (!lower_limit_nu%in%c("rhomax","2","0"))
  {
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Selection of lower_limit_nu should be either rhomax or 2 or 0")
    print("Default value rhomax is used")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  }
  
# Triangulation is meaningful if solution to holding-time equation is unique: therefore we skip the case lower_limit_nu=0  
# Note that the call to fast_halfway_triangulation_find_lambda1_subject_to_holding_time_constraint_func checks already that lower_limit_nu!="0"  
  if (lower_limit_nu=="0")
  {
    print("lower_limit_nu==0 is not used for fast triangulation")
    lower_limit_nu_triangulation<-"rhomax"
  } else
  {
    lower_limit_nu_triangulation<-lower_limit_nu
  }
# Univariate: n=1; multivariate: n=dim(Sigma)[1]  
  n<-ifelse(is.null(Sigma),1,dim(Sigma)[1])
# Compute M (see section 3 of JBCY paper) and V_M i.e. basis v of spectral decomposition of M, see section 3 in JBCY paper
# I_tilde is an identity and M_tilde=M in the case of univariate applications  
  M_obj<-M_func(L,Sigma)
  M=M_obj$M;M_tilde=M_obj$M_tilde;I_tilde=M_obj$I_tilde;eigen_M_obj=M_obj$eigen_M_obj;eigen_M_tilde_obj=M_obj$eigen_M_tilde_obj;eigen_I_tilde_obj=M_obj$eigen_I_tilde_obj
  V_M_tilde<-eigen(M_tilde)$vectors
  V_M<-eigen(M)$vectors
# Simplify if process is univariate (JBCY addresses univariate cases only)  
  if (is.null(Sigma))
  {
    V_Sigma<-eigen_Sigma<-NULL
  } else
  {
    V_Sigma<-eigen(Sigma)$vectors
    eigen_Sigma<-eigen(Sigma)$values
  }  
  eigen_M<-eigen(M)$values
  
# Specify eigenvalues and eigenvectors of M_tilde: in univariate case no problem. In multivariate case the problem is that eigenvectors of M_tilde are ordered according to increasing size   
  if (is.null(Sigma))
  {
# Univariate: here we just use original orderings    
    V<-eigen_M_tilde_obj$vectors
    eigen_M_tilde<-eigen_M_tilde_obj$values
  } else
  { 
# Multivariate case (not discussed in JBCY paper)    
# Recompute orthonormal basis of eigenvectors of M_tilde according to  kronecker(V_Sigma[,j],V_M[,1])
#   -in ordering of lambda_k*sigma_j 
#   -consistent for eigenvectors and all eigenvalues such that formula of bk based on diagonal-matrix is correct
#   -Otherwise orderings of eigenvalues do not match: false solution 
    V<-eigen_M_tilde<-NULL
    for (j in 1:dim(Sigma)[1])
      for (k in 1:dim(V_M)[1])
      { 
        eigen_M_tilde<-c(eigen_M_tilde,eigen_M[k])
        v<-kronecker(V_Sigma[,j],V_M[,k])
        V<-cbind(V,v)
# Check that image is multiple of vector     
        if (F)
        {  
          eigenv<-as.vector(I_tilde%*%v)/v
          # Print ratios for strictly positive components only    
          print(eigenv[which(abs(v)>0.00000001)])
        }
      }
  }
  
  
# Specify target: we rely on the MSE target (delta is used for deriving MSE; but once MSE is computed we can set delta=0 when deriving SSA)  
# -This is the MSE-target as applied to epsilont and used for SSA-estimation: 
#     -it corresponds to convolution of gammak and Wold decomposition xi: 
#     -if xi is NULL or identity then this is simply gammak_generic shifted by forecast_horizon

# Target MSE: this is with shift by forecast_horizon (=delta in JBCY paper) 
# Corresponds to convolution of Xi and gammak_generic with shift by forecast_horizon (=delta in JBCY paper) 
  target_obj_mse<-target_func_one_sided(xi,gammak_generic,forecast_horizon,Sigma)
  gammak_target_mse=target_obj_mse$gammak_target_mse
# Truncate target if longer than L  
  if (ncol(gammak_target_mse)>L)
  {
    print("Warning: length(gammak_generic)>L will be truncated")
    gammak_target_mse<-gammak_target_mse[,1:L,drop=FALSE]
  }
# The same as above but without shift: used for computing criterion value crit_rhoy_target (does not affect optimization)
  target_obj<-target_func_one_sided(xi,gammak_generic,0,Sigma)
  gammak_target_two=target_obj$gammak_target_mse
# Truncate target if longer than L  
  if (ncol(gammak_target_two)>L)
  {
    print("Warning: length(gammak_generic)>L will be truncated")
    gammak_target_two<-gammak_target_two[,(ncol(gammak_target_two)+1)/2+(-(L-1)/2):((L-1)/2),drop=FALSE]
  }
# Old odd functionality: we now just skip this  
  if (F)
  {  
  # The Boolean symmetric_target does not affect the optimization at all. It is just a scaling of criterion value:
  #   -If F: criterion refers to MSE between SSA and MSE or gammak_target_mse (shifted by forecast_horizon), as computed above
  #   -If T: criterion refers to MSE between SSA and bi-infinite target obtained by mirroring gammak_target (unshifted), as computed above
  #     In the latter case we don't need to shift the symmetric target because the effect is just a scaling, see the computation of crit_rhoy_target
    if (symmetric_target)
    { 
# Mirror left/right tails
# Note_ length must be the same as gammak_target_mse i.e. we mirror half-length
      gammak_target_two=cbind(gammak_target_two[,as.integer((1+ncol(gammak_target_two))/2):2,drop=FALSE],gammak_target_two[,1:as.integer((1+ncol(gammak_target_two))/2),drop=FALSE])
# For even length we just have to add a zero    
      if (ncol(gammak_target_two)<ncol(gammak_target_mse))
        gammak_target_two<-cbind(gammak_target_two,rep(0,nrow(gammak_target_two)))
    } 
  }
# Check that target does not vanish: otherwise SSA-criterion is singular (solution 0 does not have a properly defined holding-time)  
  if (sum(abs(gammak_target_mse))<1.e-20)
  {
    print("Error: the target is zero. Change the target specification or the forecast horizon")
    return()
  }

  
# Loop over all series (n=1 for a univariate design; n>1 for a multivariate design) 
  n<-ifelse(is.null(Sigma),1,n)
# Initialize criteria, parameters, filters  
  crit_rhoyy<-crit_rhoyz<-crit_rhoy_target<-lambda_opt<-nu_opt<-bk_best<-NULL
  for (m in 1:n)#m<-1
  { 
# Compute spectral weights based on reordered eigenvectors: ordering according to lambda_k*sigma_j in multivariate case    
# See section 3 in JBCY paper    
    w<-t(V)%*%gammak_target_mse[m,]
    
# Compute lag-one acf at lower right (positive) half: lambda just marginally larger than 0 (lambda=0 would lead to a singularity in lambda+1/lambda)    
    lambda_lower<-0.00000001
# Compute optimal filter based on theorem 1 when nu=lambda+1/lambda and lambda is marginally larger than 0    
    bk_obj<-bk_func(V,w,lower_limit_nu_triangulation,lambda_lower,eigen_M_tilde,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj,grid_size)
# Compute lag-one acf of corresponding solution: this is the smallest lag-one acf (lower bound) if lambda>0    
    rho_yy_lower=bk_obj$rho_yy
# If lower bound of lag-one acf at right (positive) half is smaller than rho1 (holding-time constraint) then the 
# solution will be found on positive half i.e. lambda is in ]0,1]; Otherwise search will be in ]-1,0]    
    sign<-ifelse(rho_yy_lower<rho1[m],1,-1)

    if (sign>0)
    { 
# Solution is on right positive half (smoothing): lambda is in ]0,1]      
# Compute lag-one acf at upper boundary (solution corresponding to highest possible lag-one acf)    
      lambda_upper<-1
# Compute filter corresponding to nu=lambda+1/lambda=1+1=2      
      bk_obj<-bk_func(V,w,lower_limit_nu_triangulation,lambda_upper,eigen_M_tilde,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj,grid_size)
# Lag-one acf of corresponding solution      
      rho_yy_upper=bk_obj$rho_yy
# Check: if maximal possible lag-one acf is smaller than holding-time constraint: print error message and return      
      if (rho_yy_upper<rho1[m])
      {
        if (lower_limit_nu_triangulation=="rhomax")
        {  
          print("Error: there does not exist a solution: ht is larger than rhomax. Either increase L or decrease ht")
          return()
        } else 
        {
          print("Error: there does not exist a solution for lower_limit_nu=2: try lower_limit_nu=rhomax instead or increase L or decrease ht")
          return()
        } 
      }  
    } else
    {
# If solution is on left negative half (unsmoothing) i.e. lambda in [1,0[ then we must compute new lower and upper boundaries      
# Optimal filter for lower boundary      
      lambda_lower<--1
      
      bk_obj<-bk_func(V,w,lower_limit_nu_triangulation,lambda_lower,eigen_M_tilde,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj,grid_size)
# Lag-one acf      
      rho_yy_lower=bk_obj$rho_yy
# If smallest possible lag-one acf is larger than holding-time constraint then print an error message and return       
      if (rho_yy_lower>rho1[m])
      {
        if (lower_limit_nu_triangulation=="rhomax")
        {  
          print("Error: there does not exist a solution: ht is smaller than -rhomax. Either increase L or increase ht")
          return()
        } else
        {
          print("Error: there does not exist a solution for lower_limit_nu=2: try lower_limit_nu=rhomax instead or increase L or increase ht")
          return()
        }
      }  
# Upper limit in [-1,0[     
      
      lambda_upper<--0.00000001
# Compute corresponding SSA filter      
      bk_obj<-bk_func(V,w,lower_limit_nu_triangulation,lambda_upper,eigen_M_tilde,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj,grid_size)
# Lag one of filter      
      rho_yy_upper=bk_obj$rho_yy
      
    }
# Numerical optimization: find optimal lambda by triangulation in ]0,1] or in [-1,0[      
    for (i in 1:split_grid)#split_grid<-10
    {
# Halve-split: this is possible because lag-one acf is strictly monotonic function of nu or lambda, at elast if nu>2 rhomax      
# This technical issues could not be addressed in JBCY paper (new paper in preparation)      
      lambda_middle<-(lambda_upper+lambda_lower)/2
# Compute SSA-filter      
      bk_obj<-bk_func(V,w,lower_limit_nu_triangulation,lambda_middle,eigen_M_tilde,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj,grid_size)
# Compute lag-one acf of SSA-filter: should come as close as possible to lag-one acf of holding-time constraint      
      rho_yy_middle=bk_obj$rho_yy
# New upper and lower limits for search of lambda: length is halved at each iteration step      
      if (rho_yy_middle>rho1[m])
      {
        lambda_upper<-lambda_middle
      } else
      {
        lambda_lower<-lambda_middle
      }  
      
    }
# Solution is determined up to sign, see theorem 1 in JBCY-paper: change sign of filterif necessary    
    if (bk_obj$rho_yz<0)
    {  
      bk<--bk_obj$bk
      rho_yz<--bk_obj$rho_yz
    } else
    {
      bk<-bk_obj$bk
      rho_yz<-bk_obj$rho_yz
    }
# Optimal solution
    bk_best_n<-bk
# Lag-one acf (should match holding-time constraint very closely)       
    crit_rhoyy<-c(crit_rhoyy,rho_yy_middle)
# Criterion value (correlation with MSE-target): according to theorem 1 this is maximized by above filter      
    crit_rhoyz<-c(crit_rhoyz,rho_yz)
# Compute also criterion with respect to effective target (for example bi-infinite filter) 
# Background:
#   crit_rhoyz is cor with respect to MSE i.e. gammak_target_mse
#   we want cor with respect to gammak_generic
#   Replace length of MSE by length of target, see proposition 4 in JBCY paper 
    if (is.null(Sigma))
    {
#      target_vec<-as.vector(gammak_target)
    } else
    {  
#      target_vec<-gammak_target[m,]
    }
    crit_rhoy_target<-c(crit_rhoy_target,crit_rhoyz[length(crit_rhoyz)]*as.vector(sqrt((gammak_target_mse[m,,drop=F])%*%I_tilde%*%t(gammak_target_mse[m,,drop=F])/
                                        (gammak_target_two[m,,drop=F])%*%I_tilde%*%t(gammak_target_two[m,,drop=F]))))
# Optimal lambda    
    lambda_opt<-c(lambda_opt,lambda_middle)
# Generate optimal nu: it is c*(lambda_middle+1/lambda_middle) whereby c depends on selection by lower_limit_nu_triangulation: 
#     2*rho_max(L) (|nu|>2rho_max: uniqueness but possibly unit-roots) and 2 i.e. $|nu|>2$ (uniqueness and no unit-roots)
# Fast half-way triangulation is effective in case of uniqueness. Therefore we skip the possibility lower_limit_nu=0.
    nu_middle<-bk_obj$c*(lambda_middle+1/lambda_middle)
# Append nus (in multivariate setting): this is a scalar in the univariate case    
    nu_opt<-c(nu_opt,nu_middle)
# Append targets (in multivariate setting)    
    bk_best<-cbind(bk_best,bk_best_n)

  }

# If the data xt is not white noise, then we have to proceed to deconvolution, see section 2 in JBCY paper
# Background
# -The above optimal solution is the filter as applied to epsilont: it is the convolution of SSA-filter and Wold-decomposition of xt
# -If xt=epsilont then the above filter is the final solution: one can skip the following code-lines
# -if xt!=epsilont  (xt is an autocorrelated stationary process) then the above filter must be deconvoluted prior to being applied to xt  
# Deconvolution: in use iff xi!=NULL        
  if (!is.null(xi))
  {
    bk_x<-matrix(rep(0,dim(bk_best)[1]*dim(bk_best)[2]),nrow=dim(bk_best)[1],ncol=dim(bk_best)[2])
    xi_0<-xi[,1+(0:(n-1))*L]
    xi_0_inv<-solve(xi_0)
# Invert convolution back to bk as applied to xt  
# Lag 0 is just bk_best*Xi_0^{-1}      
    bk_x[1+(0:(n-1))*L,]<-bk_best[1+(0:(n-1))*L,]%*%xi_0_inv
# Lag ijk (in fact ijk-1)      
    for (ijk in 2:L)#ijk<-2
    {
# Initialize with convolution        
      bk_x[ijk+(0:(n-1))*L,]<-bk_best[ijk+(0:(n-1))*L,]
      for (k in 1:min(ijk-1,L-1))#k<-1
      {
# dimensions of xi and b_mat are transposed
        B_k<-bk_x[k+L*(0:(n-1)),]
        xi_k<-xi[,ijk-(k-1)+(0:(n-1))*L]
# Ordering is important: B is applied to Xi (not commutative in multivariate setting)
# Must use transpose of B_k because rows of B_k correspond to eps1, eps2 and eps3 (instead of series 1, series2, series 3)         
        B_dot_xi<-t(B_k)%*%xi_k
# Use transpose of B_dot_xi again        
        bk_x[ijk+(0:(n-1))*L,]<- bk_x[ijk+(0:(n-1))*L,]-t(B_dot_xi)
      }
# Multiply with X_0^{-1}        
      bk_x[ijk+(0:(n-1))*L,]<- bk_x[ijk+(0:(n-1))*L,]%*%xi_0_inv
    }
    
    
# Check deconvolution: compute convolution of bk_x and xi: should give bk_best   
# This is a local internal check: the user doesn't see the outcome    
    b_xi<-matrix(rep(0,n^2*L),nrow=n*L,ncol=n)
    for (m in 1:L)#m<-4
    {
# Compute m-th term of convolution: sum over m summands: 
#   Set all terms to zero if lag is larger than L i.e. loop runs from 1 to min(m,L-delta-1)  
      for (k in 1:min(m,L))# k<-1
      {
# Shift target by forecast_horizon and extract matrix Gamma_{k+forecast_horizon} from gamma_target    
        b_deltak<-bk_x[k+(0:(n-1))*L,]
# Extract matrix xi_{m-k} from xi[1:n,m-(k-1)+(0:(n-1))*L]    
        xi_m_k<-xi[,m-(k-1)+(0:(n-1))*L]  #xi[,c(1,51,101)]<-rep(1,n)
# Add Gamma_{k+forecast_horizon}%*%xi_{m-k} to m-th term gamma_xi[1:n,m-(k-1)+(0:(n-1))*L]  of convolution  
        b_xi[m+(0:(n-1)*L),]<-b_xi[m+(0:(n-1)*L),]+t(t(b_deltak)%*%xi_m_k)
      }  
    }
# Check completed: difference should vanish (or be very small)  
    max(abs(b_xi-bk_best))
    
  } else
  {
# If xt=epsilont then bk_x is the same as bk_best    
    bk_x<-bk_best
  }
  gammak_mse<-gammak_target_mse
  
  return(list(bk_best=bk_best,crit_rhoyy=crit_rhoyy,
              crit_rhoyz=crit_rhoyz,lambda_opt=lambda_opt,bk_x=bk_x,
              nu_opt=nu_opt,gammak_mse=gammak_mse,crit_rhoy_target=crit_rhoy_target,w=w,V=V))
}







# This function is called by SSA_func above if lower_limit_nu="0" 
# It covers the entire solution space and can handle also new 'exotic' unit-root or singular cases which are not relevant in typical applications
# The nice monotonicity/convexity of nu as a function of ht breaks-down for these 'exotic' cases: numerical optimization must resort to crude grid-search of nu in theorem 1 (or lambda)
# The function finds optimal lambda (note that nu=lambda+1/lambda)) through grid search 
# For identical resolution the grid-search is much slower than the previous above fast optimization function fast_split_find_lambda1_subject_to_holding_time_constraint_func
#   -Fast split is always to be preferred if |nu|>2\rho_{max}(L) (uniqueness, convexity)
#   -Otherwise grid-search must be used to find all local solutions of the holding-time constraint and select the one with the largets criterion value
#     If $\nu<2rho_max$ then grid-search is mandatory
# The function implements solution in corollary 1 of BCY paper
# Meaning of parameters: see description for main function SSA_func
# Additional background:
# -lower_limit_nu
# Trick: the parametrization of nu in function is  nu<-(lambda1+1/lambda1)*c. 
#   Depending on the character string lower_limit_nu various c are possible
#   -lower_limit_nu=="rhomax": default value for c 
#     In this case |nu|>2*rhomax(L): solution is unique but unit-roots are possible
#   -lower_limit_nu=="0"
#     In this case sqrt(gridsize)>|nu|>2/sqrt(gridsize): all nu are allowed if gridsize is large
#   -lower_limit_nu=="2": in this case |nu|>2 i.e. uniqueness and no unit-root i.e. bk decay to zero
# -rho1 is holding-time constraint: vector in case of multivariate series (one constraint per series)
# -xi is MA-inversion of xt: default is xi=NULL (white noise: xt=epsilont) 
# -with_negative_lambda: 
#   default is F. 
#   Smoothing SSA-filters (with holding-times larger than target: typical in applications) nu>0 i.e. negative values can be discarded (computations 2-times faster)
#   Unsmoothing SSA-filters (with holding-times smaller than target: untypical application): set with_negative_lambda<-T
grid_search_find_lambda1_subject_to_holding_time_constraint_func<-function(grid_size,L,gammak_generic,rho1,forecast_horizon,with_negative_lambda=F,xi=NULL,lower_limit_nu="rhomax",Sigma=NULL)
{  #forecast_horizon<-2  rho1<-rho0
# Some checks  
  if (!lower_limit_nu%in%c("rhomax","2","0"))
  {
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Warning: selection of lower_limit_nu should be either rhomax or 2 or 0")
    print("Default value rhomax is used")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  }
  n<-ifelse(is.null(Sigma),1,dim(Sigma)[1])
# Grid for determining nu i.e. lambda  
  Lambda<-(1:grid_size)/(grid_size+1)
  if (with_negative_lambda)
  { 
# Allow negative lambda1 too    
    Lambda<-((-grid_size):grid_size)/(grid_size+1)
# remove zero: singularity  
    Lambda<-Lambda[-(grid_size+1)]
  }
# Compute M (see section 3 of JBCY paper) and V_M i.e. basis v of spectral decomposition of M, see section 3 in JBCY paper
# I_tilde is an identity and M_tilde=M in the case of univariate applications  
  M_obj<-M_func(L,Sigma)
  M=M_obj$M;M_tilde=M_obj$M_tilde;I_tilde=M_obj$I_tilde;eigen_M_obj=M_obj$eigen_M_obj;eigen_M_tilde_obj=M_obj$eigen_M_tilde_obj;eigen_I_tilde_obj=M_obj$eigen_I_tilde_obj
  V_M_tilde<-eigen(M_tilde)$vectors
  V_M<-eigen(M)$vectors
  if (is.null(Sigma))
  {
    V_Sigma<-eigen_Sigma<-NULL
  } else
  {
    V_Sigma<-eigen(Sigma)$vectors
    eigen_Sigma<-eigen(Sigma)$values
  }  
  eigen_M<-eigen(M)$values
  
# For multivariate applications only:  
# Recompute orthonormal basis of eigenvectors of M_tilde according to  kronecker(V_Sigma[,j],V_M[,1])
#   -in ordering of lambda_k*sigma_j 
#   -consistent for eigenvectors and all eigenvalues such that formula of bk based on diagonal-matrix is correct
#   -Otherwise orderings of eigenvalues do not match: false solution  
# Specify eigenvalues and eigenvectors of M_tilde: in univariate case no problem. In multivariate case the problem is that eigenvectors of M_tilde are ordered according to increasing size   
  if (is.null(Sigma))
  {
# Univariate: here we just use original orderings    
    V<-eigen_M_tilde_obj$vectors
    eigen_M_tilde<-eigen_M_tilde_obj$values
  } else
  { 
# Recompute orthonormal basis of eigenvectors of M_tilde according to  kronecker(V_Sigma[,j],V_M[,1])
#   -in ordering of lambda_k*sigma_j 
#   -consistent for eigenvectors and all eigenvalues such that formula of bk based on diagonal-matrix is correct
#   -Otherwise orderings of eigenvalues do not match: false solution 
    V<-eigen_M_tilde<-NULL
    for (j in 1:dim(Sigma)[1])
      for (k in 1:dim(V_M)[1])
      { 
        eigen_M_tilde<-c(eigen_M_tilde,eigen_M[k])
        v<-kronecker(V_Sigma[,j],V_M[,k])
        V<-cbind(V,v)
# Check that image is multiple of vector     
        if (F)
        {  
          eigenv<-as.vector(I_tilde%*%v)/v
# Print ratios for strictly positive components only    
          print(eigenv[which(abs(v)>0.00000001)])
        }
      }
  }
  
  # Specify target: we rely on the MSE target (delta is used for deriving MSE; but once MSE is computed we can set delta=0 when deriving SSA)  
  # -This is the MSE-target as applied to epsilont and used for SSA-estimation: 
  #     -it corresponds to convolution of gammak and Wold decomposition xi: 
  #     -if xi is NULL or identity then this is simply gammak_generic shifted by forecast_horizon
  
  # Target MSE: this is with shift by forecast_horizon (=delta in JBCY paper) 
  # Corresponds to convolution of Xi and gammak_generic with shift by forecast_horizon (=delta in JBCY paper) 
  target_obj_mse<-target_func_one_sided(xi,gammak_generic,forecast_horizon,Sigma)
  gammak_target_mse=target_obj_mse$gammak_target_mse
  # Truncate target if longer than L  
  if (ncol(gammak_target_mse)>L)
  {
    print("Warning: length(gammak_generic)>L will be truncated")
    gammak_target_mse<-gammak_target_mse[,1:L,drop=FALSE]
  }
  # The same as above but without shift: used for computing criterion value crit_rhoy_target (does not affect optimization)
  target_obj<-target_func_one_sided(xi,gammak_generic,0,Sigma)
  gammak_target_two=target_obj$gammak_target_mse
  # Truncate target if longer than L  
  if (ncol(gammak_target_two)>L)
  {
    print("Warning: length(gammak_generic)>L will be truncated")
    gammak_target_two<-gammak_target_two[,(ncol(gammak_target_two)+1)/2+(-(L-1)/2):((L-1)/2),drop=FALSE]
  }
  # Old odd functionality: we now just skip this  
  if (F)
  {  
    # The Boolean symmetric_target does not affect the optimization at all. It is just a scaling of criterion value:
    #   -If F: criterion refers to MSE between SSA and MSE or gammak_target_mse (shifted by forecast_horizon), as computed above
    #   -If T: criterion refers to MSE between SSA and bi-infinite target obtained by mirroring gammak_target (unshifted), as computed above
    #     In the latter case we don't need to shift the symmetric target because the effect is just a scaling, see the computation of crit_rhoy_target
    if (symmetric_target)
    { 
      # Mirror left/right tails
      # Note_ length must be the same as gammak_target_mse i.e. we mirror half-length
      gammak_target_two=cbind(gammak_target_two[,as.integer((1+ncol(gammak_target_two))/2):2,drop=FALSE],gammak_target_two[,1:as.integer((1+ncol(gammak_target_two))/2),drop=FALSE])
      # For even length we just have to add a zero    
      if (ncol(gammak_target_two)<ncol(gammak_target_mse))
        gammak_target_two<-cbind(gammak_target_two,rep(0,nrow(gammak_target_two)))
    } 
  }
  # Check that target does not vanish: otherwise SSA-criterion is singular (solution 0 does not have a properly defined holding-time)  
  if (sum(abs(gammak_target_mse))<1.e-20)
  {
    print("Error: the target is zero. Change the target specification or the forecast horizon")
    return()
  }
  
  
# Initializations
# Maximal eigenvalue i.e. rhomax(L) : is used in scaling nu in loop for one particular option 
  maxrho<-max(eigen_M_obj$values)
# Initialize all relevant variables  
  nu_vec<-NULL
  bk_best<-bk_x<-crit_rhoy_target<-NULL#rho_yy_best<-rho_mat<-crit_rhoyz<-lambda_opt<-nu_vec<-bk_x<-nu_opt<-NULL
  rho_mat<-NULL
  if (!is.null(Sigma))
  {
    crit_rhoyy<-rep(10000,n)
    crit_rhoyz<-rep(-2,n)
    lambda_opt<-nu_opt<-i_select<-rep(NA,n)
  } else
  {
    crit_rhoyy<-10000
    crit_rhoyz<--2
    lambda_opt<-nu_opt<-i_select<-NA
    
  }
  
# Loop over all series (multivariate design: n>1; univariate design: n=1) 
  n<-ifelse(is.null(Sigma),1,n)
  for (m in 1:n)#m<-1
  { 
# Compute spectral weights based on reordered eigenvectors: ordering according to lambda_k*sigma_j
# See section 3 in JBCY-paper    
    w<-t(V)%*%gammak_target_mse[m,]
    
# Loop over all equidistant grid-points
    rho_mat_n<-matrix(nrow=length(Lambda),ncol=4)
    for (i in 1:length(Lambda))#i<-1001  #i<-length(Lambda)-9 i<-1561
    {
# Next point ofn grid      
      lambda1<-Lambda[i]
# Compute corresponding SSA-filter      
      bk_obj<-bk_func(V,w,lower_limit_nu,lambda1,eigen_M_tilde,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj)
# Filter      
      bk_new<-bk_obj$bk
# Lag-one acf: should be as close as possible to lag-one acf in holding-time constraint      
      rho_yy_best<-bk_obj$rho_yy
# Criterion value: correlation with target      
      rho_yz_best<-bk_obj$rho_yz
      c<-bk_obj$c
      nu<-bk_obj$nu
# Keep tracking of statistics for all grid-points      
      rho_mat_n[i,]<-c(rho_yy_best,rho_yz_best,nu,lambda1)
# Change sign if correlation with target is negative, see theorem 1
      if (rho_yz_best<0)
      {  
        rho_mat_n[i,2]<--rho_mat_n[i,2]
        bk_new<--bk_new
        rho_yz_best<--rho_yz_best
      }
# Select grid-point which minimizes difference of lag-one acf to holding-time constraint    
      crit<-abs(rho_yy_best-rho1[m])
      if (crit<crit_rhoyy[m])
      {
        bk_best_n<-bk_new
        crit_rhoyy[m]<-crit
        crit_rhoyz[m]<-rho_yz_best
        lambda_opt[m]<-lambda1
        nu_opt[m]<-(lambda_opt[m]+1/lambda_opt[m])*c
        i_select[m]<-i
      }
      
    }
# Compute also criterion with respect to effective target (for example bi-infinite filter) 
# Background:
#   crit_rhoyz is cor with respect to MSE i.e. gammak_target_mse
#   we want cor with respect to gammak_generic
#   Replace length of MSE by length of target, see proposition 4 in JBCY paper 
    if (is.null(Sigma))
    {
      target_vec<-as.vector(gammak_target)
    } else
    {  
      target_vec<-gammak_target[m,]
    }
    crit_rhoy_target<-c(crit_rhoy_target,crit_rhoyz[length(crit_rhoyz)]*as.vector(sqrt(t(gammak_target[m,])%*%I_tilde%*%gammak_target[m,]/
                                                                                         t(target_vec[m,])%*%I_tilde%*%target_vec[m,])))
#---------------------
# Grid-search: compute solutions if |nu|<2*rhomax i.e. multiple solutions for ht-constraint    
# If solution is not unique i.e. if $|nu|<2\rho_{max}(L)$ then we here pick out that solution with the highest criterion value (highest correlation of SSA with target)
# Proceeding:
# 1. order from tightest to loosest ht-fit (smallest absolute ht-error to largest absolute ht-error)
    compute_ht_from_rho_func<-function(rho_ff1)
    {
# Mean holding-time
      ht<-1/(2*(0.25-asin(rho_ff1)/(2*pi)))
      ht
    }
    diff_ht<-abs(compute_ht_from_rho_func(rho1)-apply(matrix(rho_mat_n[,1],ncol=1),1,compute_ht_from_rho_func))
    order_ht<-order(diff_ht)
# 2. Select best ones: with a ht-error smaller 0.1    
    best_onesh<-which(diff_ht[order_ht]<0.1)
# 2.1 If all have error larger 0.1: take the first only    
    if (length(best_onesh)==0)
    {  
      best_ones<-best_onesh[length(best_onesh)]
      print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      print(paste("ht-error is ",min(abs(rho1-apply(matrix(rho_mat_n[,1],ncol=1),1,compute_ht_from_rho_func))),sep=""))
      print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
      print("Try lower_limit_nu=rhomax instead")
    }
    rho_mat_h<-rho_mat_n[order_ht[1:best_onesh[length(best_onesh)]],]
    
# 3. Select solution with largest criterion  
    rho_mat_best<-rho_mat_h[which(rho_mat_h[,2]==max(rho_mat_h[,2])),]
# 4. If solution which minimizes holding-time error (i.e. nu_opt as computed above) differs from best compromise 
#     then it makes sense to replace the former by the latter.  
    if (abs(nu_opt-rho_mat_best[3])>10^(-3))
    {
      print("Solution which minimizes holding-time error is not the same as best compromise")
      print("Solution which minimizes holding-time error is replaced by best compromise")
      nu_opt<-rho_mat_best[3]
      lambda_opt<-rho_mat_best[4]
      bk_best_n<-V%*%diag(1/(2*eigen_M_tilde-nu_opt))%*%w
      rho_yy_best<-compute_holding_time_func(bk_best_n)$rho_ff1[1,1]
      rho_yz_best<-(t(bk_best_n)%*%gammak_target_mse[m,1:length(bk_best_n)])[1,1]/(sqrt(t(bk_best_n)%*%bk_best_n)*sqrt(gammak_target_mse[m,1:length(bk_best_n)]%*%gammak_target_mse[m,1:length(bk_best_n)]))[1,1]
      if (rho_yz_best<0)
      {
        bk_best_n<--bk_best_n
        rho_yz_best<--rho_yz_best
      }
    }  
    
# Multivariate setting: append rho_mat over all targets   
    if (n==1)
    {  
      colnames(rho_mat_n)<-c("rho_yy_best","rho_yz_best","nu","lambda")
    } else
    {  
      colnames(rho_mat_n)<-paste(c("rho_yy_best_","rho_yz_best_","nu_","lambda_"),m,sep="")
    }
    rho_mat<-cbind(rho_mat,rho_mat_n) 
    bk_best<-cbind(bk_best,bk_best_n)
    ts.plot(bk_best)

  }
  
# If the data xt is not white noise, then we have to proceed to deconvolution, see section 2 in JBCY paper
# Background
# -The above optimal solution is the filter as applied to epsilont: it is the convolution of SSA-filter and Wold-decomposition of xt
# -If xt=epsilont then the above filter is the final solution: one can skip the following code-lines
# -if xt!=epsilont  (xt is an autocorrelated stationary process) then the above filter must be deconvoluted prior to being applied to xt  
# Deconvolution: in use iff xi!=NULL        
  if (!is.null(xi))
  {
    bk_x<-matrix(rep(0,dim(bk_best)[1]*dim(bk_best)[2]),nrow=dim(bk_best)[1],ncol=dim(bk_best)[2])
    xi_0<-xi[,1+(0:(n-1))*L]
    xi_0_inv<-solve(xi_0)
# Invert convolution back to bk as applied to xt  
# Lag 0 is just bk_best*Xi_0^{-1}      
    bk_x[1+(0:(n-1))*L,]<-bk_best[1+(0:(n-1))*L,]%*%xi_0_inv
# Lag ijk (in fact ijk-1)      
    for (ijk in 2:L)#ijk<-2
    {
# Initialize with convolution        
      bk_x[ijk+(0:(n-1))*L,]<-bk_best[ijk+(0:(n-1))*L,]
      for (k in 1:min(ijk-1,L-1))#k<-1
      {
# dimensions of xi and b_mat are transposed: should eventually change and match...          
        B_k<-bk_x[k+L*(0:(n-1)),]
        xi_k<-xi[,ijk-(k-1)+(0:(n-1))*L]
# Ordering is important: B is applied to Xi (not commutative)
# Must use transpose of B_k because rows of B_k correspond to eps1, eps2 and eps3 (instead of series 1, series2, series 3)         
        B_dot_xi<-t(B_k)%*%xi_k
# Use transpose of B_dot_xi again        
        bk_x[ijk+(0:(n-1))*L,]<- bk_x[ijk+(0:(n-1))*L,]-t(B_dot_xi)
      }
# Multiply with X_0^{-1}        
      bk_x[ijk+(0:(n-1))*L,]<- bk_x[ijk+(0:(n-1))*L,]%*%xi_0_inv
    }
    
    
# Check deconvolution: compute convolution of bk_x and xi: should give bk_best   
# This is a local internal check: the user doesn't see the outcome    
    b_xi<-matrix(rep(0,n^2*L),nrow=n*L,ncol=n)
    for (m in 1:L)#m<-4
    {
# Compute m-th term of convolution: sum over m summands: 
#   Set all terms to zero if lag is larger than L i.e. loop runs from 1 to min(m,L-delta-1)  
      for (k in 1:min(m,L))# k<-1
      {
# Shift target by forecast_horizon and extract matrix Gamma_{k+forecast_horizon} from gamma_target    
        b_deltak<-bk_x[k+(0:(n-1))*L,]
# Extract matrix xi_{m-k} from xi[1:n,m-(k-1)+(0:(n-1))*L]    
        xi_m_k<-xi[,m-(k-1)+(0:(n-1))*L]  #xi[,c(1,51,101)]<-rep(1,n)
# Add Gamma_{k+forecast_horizon}%*%xi_{m-k} to m-th term gamma_xi[1:n,m-(k-1)+(0:(n-1))*L]  of convolution  
        b_xi[m+(0:(n-1)*L),]<-b_xi[m+(0:(n-1)*L),]+t(t(b_deltak)%*%xi_m_k)
      }  
    }
# Check completed: difference should vanish (up to rounding errors)  
    which(abs(b_xi-bk_best)>10^{-12})
  } else
  {
    bk_x<-bk_best
  }
  
  gammak_mse<-gammak_target_mse

  return(list(rho_mat=rho_mat,bk_best=bk_best,crit_rhoyy=crit_rhoyy,
              crit_rhoyz=crit_rhoyz,lambda_opt=lambda_opt,nu_vec=nu_vec,bk_x=bk_x,
              nu_opt=nu_opt,gammak_mse=gammak_mse,crit_rhoy_target=crit_rhoy_target,w=w,V=V))
}


# This function computes the system matrices (autocovariance generating function, see section 3 in JBCY-paper) in the case of uni and multivariate SSA-designs
M_func<-function(L,Sigma)
{
# Specify M (lag-one autocovariance generating function) 
  M<-matrix(nrow=L,ncol=L)
  M[L,]<-rep(0,L)
  M[L-1,]<-c(rep(0,L-1),0.5)
  for (i in 1:(L-2))
    M[i,]<-c(rep(0,i),0.5,rep(0,L-1-i))
  M<-M+t(M)
  eigen_M_obj<-eigen(M)
# Univariate case (Sigma==NULL) or multivariate case (Sigma is cross-correlation of epsilons)  
  if (is.null(Sigma))
  {  
# Univariate: I_tilde is identity and M_tilde=M 
    I_tilde<-diag(rep(1,L))
    M_tilde<-M
    eigen_M_tilde_obj<-eigen_M_obj
    eigen_I_tilde_obj<-eigen(I_tilde)
  } else  
  {
# Multivariate    
    I<-diag(rep(1,L))
    M_tilde<-kronecker(Sigma,M)
    I_tilde<-kronecker(Sigma,I)
    eigen_M_tilde_obj<-eigen(M_tilde)
    eigen_I_tilde_obj<-eigen(I_tilde)
  }
  return(list(M=M,M_tilde=M_tilde,I_tilde=I_tilde,eigen_M_obj=eigen_M_obj,eigen_M_tilde_obj=eigen_M_tilde_obj,eigen_I_tilde_obj=eigen_I_tilde_obj))
}


# This function computes the transformed target if the data is not white noise
# Convolution of xi (Data generating process) and target gammak_generic
target_func_one_sided<-function(xi,gammak_generic,forecast_horizon,Sigma)
{
  n<-ifelse(is.null(Sigma),1,dim(Sigma)[1])
# If target is vector: transform into matrix with a single row  
  if (is.vector(gammak_generic))
    gammak_generic<-matrix(gammak_generic,nrow=n)
  L<-ncol(gammak_generic)/n
  if (!is.null(xi))
  {  
# Compute convolution for MSE
# Initialize convolution
    gamma_xi<-matrix(rep(0,n^2*L),nrow=n,ncol=L*n)
# Coefficients of lags 0 to L-1     
    for (m in 1:L)#m<-1
    {
# Compute weight of epsilon_{t-m} in z_{t+forecast_horizon}
#  z_{t+forecast_horizon}=gamma_0*x_{t+forecast_horizon}+gamma_1*x_{t+forecast_horizon-1}+...
# =gamma0*(xi_0 epsilon_{t+forecast_horizon}+xi_1*epsilon_{t+forecast_horizon-1}+...)+
#  gamma1*(                                 +xi_0*epsilon_{t+forecast_horizon-1}+...)+
#   ...
# So weight of epsilon_t is xi_0*gamma_{forecast_horizon}+xi_1*gamma_{forecast_horizon-1}+... 
# Trick: we run through all k and check if the lags are admissible i.e. smaller L and larger 1        
      for (k in 1:L)# k<-1
      {
# Extract matrix Gamma_k from gamma_target 
        if (k<=L)
        {  
          Gamma_deltak<-gammak_generic[,k+(0:(n-1))*L]
        } else
        {
          Gamma_deltak<-matrix(rep(0,n^2),nrow=n)
        }
# Extract matrix xi_{m+forecast_horizon-k} from xi[1:n,m-(k-1)+(0:(n-1))*L] 
#   Set all terms to zero if lag is larger than L or smaller than 1   
        if (m+forecast_horizon-(k-1)<=ncol(xi)&m+forecast_horizon-(k-1)>=1)
        {  
          xi_m_k<-xi[,m+forecast_horizon-(k-1)+(0:(n-1))*L]
        } else
        {
          xi_m_k<-matrix(rep(0,n^2),nrow=n)
        }
# Add Gamma_{k}%*%xi_{m+forecast_horizon-k} to m-th term of convolution  
        gamma_xi[,m+(0:(n-1))*L]<-gamma_xi[1:n,m+(0:(n-1))*L]+Gamma_deltak%*%xi_m_k
      }  
    }
    gammak_target_mse<-gamma_xi
    
  } else 
  {
# In this case Wold-decomposition is identity i.e. xt=epsilont is white noise
# We just shift the MSE-target by the forecast_horizon     
    gammak_target_target<-gammak_generic
    gammak_target_mse<-NULL
# Shift target by forecast_horizon   
    if (forecast_horizon>0)
    {
# Forecast: Shift target to the left by forecast_horizon      
      for (i in 1:n)#i<-1
        gammak_target_mse<-cbind(gammak_target_mse,cbind(gammak_generic[,forecast_horizon+1:(L-forecast_horizon)+(i-1)*L,drop=F],
                                                         matrix(rep(0,forecast_horizon*n),nrow=n,ncol=forecast_horizon)))
    } else
    {
      if (forecast_horizon<0)
      {
# Backcast: shift target to the right by forecast_horizon         
        for (i in 1:n)#i<-1
          gammak_target_mse<-cbind(gammak_target_mse,cbind(matrix(rep(0,abs(forecast_horizon)*n),nrow=n,ncol=abs(forecast_horizon)),
                                                           gammak_generic[,1:(L-abs(forecast_horizon))+(i-1)*L,drop=F]))
      } else
      {
# Nowcast: no shift applied        
        gammak_target_mse<-gammak_generic
      }  
    }
  }    
  return(list(gammak_target_mse=gammak_target_mse))
}  



# This function computes SSA-estimate, criterion value and lag-one acf for given nu or lambda
#  if (lower_limit_nu=="0") then we focus on |nu|<2 only (case of multiple solutions)
#     -this case can occur only if grid-search has been called
#  otherwise (lower_limit_nu!="0") we focus on |nu|>2 or |nu|>2*rho_max only (uniqueness).
#     -this case occurs only if fast triangulation has been called
bk_func<-function(V,w,lower_limit_nu,lambda1,eigen_M_tilde,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj,grid_size=NULL)
{ #lambda1<-lambda_opt   lower_limit_nu<-lower_limit_nu_triangulation
  
  
# If lower_limit_nu!="0" then we emphasize either |nu|>2 (no unit-roots, unique solution) or $|nu|>2*rho_max$ (unique solution with possible unit-roots). 
#   -In both cases nu<-(lambda1+1/lambda1)*c  
#   -But c depends on the choice rhomax or 2
# Fast optimization (triangulation) always assumes lower_limit_nu!="0"   
# Otherwise i.e. if lower_limit_nu!="0" then the call to bk_func must come from grid-search
#   -We then search for |nu|<2 only (multiple solutions)
#   -In this case nu is parameterized as nu<-2*lambda1 so that the interval [0,2] or [-2,2] is screened linearly 
  if (lower_limit_nu!="0")
  {  
# lower_limit_nu=="rhomax": nu should be slightly larger than rhomax (otherwise singular)  
    if (lower_limit_nu=="rhomax")
      c<-max(eigen_M_obj$values)+1.e-10
# lower_limit_nu=="2": do not scale nu i.e. |nu|>2 (no unit-roots i.e. bk decay to zero)    
    if (lower_limit_nu=="2")
      c<-1
    nu<-(lambda1+1/lambda1)*c
  } else
  {  
# lambda1 is in Lambda which are grid-points in [0,1] or [-1,1]. 
#   -One could scale with rhomax (not done yet)    
    nu<-2*lambda1
  }
  

# Implement formula of theorem 1
# Notes in the case of multivariate designs (not considered in JBCY paper)
#   -We use the reordered eigenvalues, eigenvectors and spectral weights
#   -We cannot use original ordering of eigenvectors because we need eigen_M_tilde divided by eigen_I_tilde and the 
#     two are ordered differently (both in ascending order of eigenvalues so that ordering of eigenvectors will not match)
#   -The most effective way is indeed to recompute everything from scratch in a consistent way: same ordering for 
#     eigenvectors and eigenvalues of M_tilde and of I_tilde!!!!  
  bk<-V%*%diag(1/(2*eigen_M_tilde-nu))%*%w
  
  gammak_n<-gammak_target_mse[m,]
  
# Check different equivalent solutions: the above is simplest and fastest 
  if (F)
  {
# Original solution with matrix inversion: more difficult/lengthy 
    nu_tilde_mat<-2*M_tilde-nu*I_tilde
    bkh<-solve(nu_tilde_mat)%*%I_tilde%*%gammak_target_mse[m,]
    max(abs(bk-bkh)) 
# Simpler solution which does not depend on Sigma    
    nu_mat<-2*kronecker(diag(rep(1,n)),M)-nu*diag(rep(1,n*L))
    bkhh<-solve(nu_mat)%*%gammak_target_mse[m,]
    max(abs(bk-bkhh))  
    
  }
# Lag-one acf: SSA solution corresponds to grid-point whose lag-one acf is closest to imposed holding-time        
  rho_yy<-t(bk)%*%M_tilde%*%bk/t(bk)%*%I_tilde%*%bk
  
# Criterion      
  rho_yz<-(t(bk)%*%I_tilde%*%gammak_n)[1,1]/(sqrt(t(bk)%*%I_tilde%*%bk)*sqrt(gammak_n%*%I_tilde%*%gammak_n))[1,1]
  return(list(bk=bk,rho_yy=rho_yy,rho_yz=rho_yz,c=c,nu=nu,w=w))
}






# This function computes SSA for given lambda
#   -if lower_limit_nu=="0" then nu=2*lambda in bk_func
#   -if !(lower_limit_nu=="0") then nu=(lambda+1/lambda) or nu=(lambda+1/lambda)*rho_max
Compute_SSA_for_given_lambda<-function(lambda,split_grid,L,gammak_generic,rho1,forecast_horizon,xi=NULL,lower_limit_nu="rhomax",Sigma=NULL)
{
  # Check lower_limit_nu: if is different from the three options one assumes standard setting rhomax: a Warning is printed  
  if (!lower_limit_nu%in%c("rhomax","2","0"))
  {
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Selection of lower_limit_nu should be either rhomax or 2 or 0")
    print("Default value rhomax is used")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
  }
  
  # Triangulation is meaningful if solution to holding-time equation is unique: therefore we skip the case lower_limit_nu=0  
  # Note that the call to fast_halfway_triangulation_find_lambda1_subject_to_holding_time_constraint_func checks already that lower_limit_nu!="0"  
  lower_limit_nu_triangulation<-lower_limit_nu
  # Univariate: n=1; multivariate: n=dim(Sigma)[1]  
  n<-ifelse(is.null(Sigma),1,dim(Sigma)[1])
  # Compute M (see section 3 of JBCY paper) and V_M i.e. basis v of spectral decomposition of M, see section 3 in JBCY paper
  # I_tilde is an identity and M_tilde=M in the case of univariate applications  
  M_obj<-M_func(L,Sigma)
  M=M_obj$M;M_tilde=M_obj$M_tilde;I_tilde=M_obj$I_tilde;eigen_M_obj=M_obj$eigen_M_obj;eigen_M_tilde_obj=M_obj$eigen_M_tilde_obj;eigen_I_tilde_obj=M_obj$eigen_I_tilde_obj
  V_M_tilde<-eigen(M_tilde)$vectors
  V_M<-eigen(M)$vectors
  # Simplify if process is univariate (JBCY addresses univariate cases only)  
  if (is.null(Sigma))
  {
    V_Sigma<-eigen_Sigma<-NULL
  } else
  {
    V_Sigma<-eigen(Sigma)$vectors
    eigen_Sigma<-eigen(Sigma)$values
  }  
  eigen_M<-eigen(M)$values
  
  # Specify eigenvalues and eigenvectors of M_tilde: in univariate case no problem. In multivariate case the problem is that eigenvectors of M_tilde are ordered according to increasing size   
  if (is.null(Sigma))
  {
    # Univariate: here we just use original orderings    
    V<-eigen_M_tilde_obj$vectors
    eigen_M_tilde<-eigen_M_tilde_obj$values
  } else
  { 
    # Multivariate case (not discussed in JBCY paper)    
    # Recompute orthonormal basis of eigenvectors of M_tilde according to  kronecker(V_Sigma[,j],V_M[,1])
    #   -in ordering of lambda_k*sigma_j 
    #   -consistent for eigenvectors and all eigenvalues such that formula of bk based on diagonal-matrix is correct
    #   -Otherwise orderings of eigenvalues do not match: false solution 
    V<-eigen_M_tilde<-NULL
    for (j in 1:dim(Sigma)[1])
      for (k in 1:dim(V_M)[1])
      { 
        eigen_M_tilde<-c(eigen_M_tilde,eigen_M[k])
        v<-kronecker(V_Sigma[,j],V_M[,k])
        V<-cbind(V,v)
        # Check that image is multiple of vector     
        if (F)
        {  
          eigenv<-as.vector(I_tilde%*%v)/v
          # Print ratios for strictly positive components only    
          print(eigenv[which(abs(v)>0.00000001)])
        }
      }
  }
  
  
  # Specify target: we rely on the MSE target (delta is used for deriving MSE; but once MSE is computed we can set delta=0 when deriving SSA)  
  # -This is the MSE-target as applied to epsilont and used for SSA-estimation: 
  #     -it corresponds to convolution of gammak and Wold decomposition xi: 
  #     -if xi is NULL or identity then this is simply gammak_generic shifted by forecast_horizon
  
  # Target MSE: this is with shift by forecast_horizon (=delta in JBCY paper) 
  # Corresponds to convolution of Xi and gammak_generic with shift by forecast_horizon (=delta in JBCY paper) 
  target_obj_mse<-target_func_one_sided(xi,gammak_generic,forecast_horizon,Sigma)
  gammak_target_mse=target_obj_mse$gammak_target_mse
  # Truncate target if longer than L  
  if (ncol(gammak_target_mse)>L)
  {
    print("Warning: length(gammak_generic)>L will be truncated")
    gammak_target_mse<-gammak_target_mse[,1:L,drop=FALSE]
  }
  # The same as above but without shift: used for computing criterion value crit_rhoy_target (does not affect optimization)
  target_obj<-target_func_one_sided(xi,gammak_generic,0,Sigma)
  gammak_target_two=target_obj$gammak_target_mse
  # Truncate target if longer than L  
  if (ncol(gammak_target_two)>L)
  {
    print("Warning: length(gammak_generic)>L will be truncated")
    gammak_target_two<-gammak_target_two[,(ncol(gammak_target_two)+1)/2+(-(L-1)/2):((L-1)/2),drop=FALSE]
  }
  # Old odd functionality: we now just skip this  
  if (F)
  {  
    # The Boolean symmetric_target does not affect the optimization at all. It is just a scaling of criterion value:
    #   -If F: criterion refers to MSE between SSA and MSE or gammak_target_mse (shifted by forecast_horizon), as computed above
    #   -If T: criterion refers to MSE between SSA and bi-infinite target obtained by mirroring gammak_target (unshifted), as computed above
    #     In the latter case we don't need to shift the symmetric target because the effect is just a scaling, see the computation of crit_rhoy_target
    if (symmetric_target)
    { 
      # Mirror left/right tails
      # Note_ length must be the same as gammak_target_mse i.e. we mirror half-length
      gammak_target_two=cbind(gammak_target_two[,as.integer((1+ncol(gammak_target_two))/2):2,drop=FALSE],gammak_target_two[,1:as.integer((1+ncol(gammak_target_two))/2),drop=FALSE])
      # For even length we just have to add a zero    
      if (ncol(gammak_target_two)<ncol(gammak_target_mse))
        gammak_target_two<-cbind(gammak_target_two,rep(0,nrow(gammak_target_two)))
    } 
  }
  # Check that target does not vanish: otherwise SSA-criterion is singular (solution 0 does not have a properly defined holding-time)  
  if (sum(abs(gammak_target_mse))<1.e-20)
  {
    print("Error: the target is zero. Change the target specification or the forecast horizon")
    return()
  }
  
  
  # Loop over all series (n=1 for a univariate design; n>1 for a multivariate design) 
  n<-ifelse(is.null(Sigma),1,n)
  # Initialize criteria, parameters, filters  
  crit_rhoyy<-crit_rhoyz<-crit_rhoy_target<-lambda_opt<-nu_opt<-bk_best<-NULL
  for (m in 1:n)#m<-1
  { 
    # Compute spectral weights based on reordered eigenvectors: ordering according to lambda_k*sigma_j in multivariate case    
    # See section 3 in JBCY paper    
    w<-t(V)%*%gammak_target_mse[m,]
    
    
    # Compute SSA-filter      
    bk_obj<-bk_func(V,w,lower_limit_nu_triangulation,lambda,eigen_M_tilde,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj,grid_size)
    # Compute lag-one acf of SSA-filter: should come as close as possible to lag-one acf of holding-time constraint      
    rho_yy=bk_obj$rho_yy
    # Solution is determined up to sign, see theorem 1 in JBCY-paper: change sign of filterif necessary    
    if (bk_obj$rho_yz<0)
    {  
      bk<--bk_obj$bk
      rho_yz<--bk_obj$rho_yz
    } else
    {
      bk<-bk_obj$bk
      rho_yz<-bk_obj$rho_yz
    }
    # Optimal solution
    bk_best<-bk
    # Lag-one acf (should match holding-time constraint very closely)       
    crit_rhoyy<-rho_yy
    # Criterion value (correlation with MSE-target): according to theorem 1 this is maximized by above filter      
    crit_rhoyz<-rho_yz
    # Compute also criterion with respect to effective target (for example bi-infinite filter) 
    # Background:
    #   crit_rhoyz is cor with respect to MSE i.e. gammak_target_mse
    #   we want cor with respect to gammak_generic
    #   Replace length of MSE by length of target, see proposition 4 in JBCY paper 
    if (is.null(Sigma))
    {
      #      target_vec<-as.vector(gammak_target)
    } else
    {  
      #      target_vec<-gammak_target[m,]
    }
    crit_rhoy_target<-crit_rhoyz[length(crit_rhoyz)]*as.vector(sqrt((gammak_target_mse[m,,drop=F])%*%I_tilde%*%t(gammak_target_mse[m,,drop=F])/
                                                                      (gammak_target_two[m,,drop=F])%*%I_tilde%*%t(gammak_target_two[m,,drop=F])))
    nu<-
      if (!lower_limit_nu=="0")
      {
        nu<-bk_obj$c*(lambda+1/lambda)
      } else
      {
        # lambda is in [-1,1] and therefore nu is in [-2,2]: this is how nu is parameterized in bk_func if lower_limit_nu=="0" 
        nu<-2*lambda
      }
    
  }
  
  # If the data xt is not white noise, then we have to proceed to deconvolution, see section 2 in JBCY paper
  # Background
  # -The above optimal solution is the filter as applied to epsilont: it is the convolution of SSA-filter and Wold-decomposition of xt
  # -If xt=epsilont then the above filter is the final solution: one can skip the following code-lines
  # -if xt!=epsilont  (xt is an autocorrelated stationary process) then the above filter must be deconvoluted prior to being applied to xt  
  # Deconvolution: in use iff xi!=NULL        
  if (!is.null(xi))
  {
    bk_x<-matrix(rep(0,dim(bk_best)[1]*dim(bk_best)[2]),nrow=dim(bk_best)[1],ncol=dim(bk_best)[2])
    xi_0<-xi[,1+(0:(n-1))*L]
    xi_0_inv<-solve(xi_0)
    # Invert convolution back to bk as applied to xt  
    # Lag 0 is just bk_best*Xi_0^{-1}      
    bk_x[1+(0:(n-1))*L,]<-bk_best[1+(0:(n-1))*L,]%*%xi_0_inv
    # Lag ijk (in fact ijk-1)      
    for (ijk in 2:L)#ijk<-2
    {
      # Initialize with convolution        
      bk_x[ijk+(0:(n-1))*L,]<-bk_best[ijk+(0:(n-1))*L,]
      for (k in 1:min(ijk-1,L-1))#k<-1
      {
        # dimensions of xi and b_mat are transposed
        B_k<-bk_x[k+L*(0:(n-1)),]
        xi_k<-xi[,ijk-(k-1)+(0:(n-1))*L]
        # Ordering is important: B is applied to Xi (not commutative in multivariate setting)
        # Must use transpose of B_k because rows of B_k correspond to eps1, eps2 and eps3 (instead of series 1, series2, series 3)         
        B_dot_xi<-t(B_k)%*%xi_k
        # Use transpose of B_dot_xi again        
        bk_x[ijk+(0:(n-1))*L,]<- bk_x[ijk+(0:(n-1))*L,]-t(B_dot_xi)
      }
      # Multiply with X_0^{-1}        
      bk_x[ijk+(0:(n-1))*L,]<- bk_x[ijk+(0:(n-1))*L,]%*%xi_0_inv
    }
    
    
    # Check deconvolution: compute convolution of bk_x and xi: should give bk_best   
    # This is a local internal check: the user doesn't see the outcome    
    b_xi<-matrix(rep(0,n^2*L),nrow=n*L,ncol=n)
    for (m in 1:L)#m<-4
    {
      # Compute m-th term of convolution: sum over m summands: 
      #   Set all terms to zero if lag is larger than L i.e. loop runs from 1 to min(m,L-delta-1)  
      for (k in 1:min(m,L))# k<-1
      {
        # Shift target by forecast_horizon and extract matrix Gamma_{k+forecast_horizon} from gamma_target    
        b_deltak<-bk_x[k+(0:(n-1))*L,]
        # Extract matrix xi_{m-k} from xi[1:n,m-(k-1)+(0:(n-1))*L]    
        xi_m_k<-xi[,m-(k-1)+(0:(n-1))*L]  #xi[,c(1,51,101)]<-rep(1,n)
        # Add Gamma_{k+forecast_horizon}%*%xi_{m-k} to m-th term gamma_xi[1:n,m-(k-1)+(0:(n-1))*L]  of convolution  
        b_xi[m+(0:(n-1)*L),]<-b_xi[m+(0:(n-1)*L),]+t(t(b_deltak)%*%xi_m_k)
      }  
    }
    # Check completed: difference should vanish (or be very small)  
    max(abs(b_xi-bk_best))
    
  } else
  {
    # If xt=epsilont then bk_x is the same as bk_best    
    bk_x<-bk_best
  }
  gammak_mse<-gammak_target_mse
  
  return(list(bk_best=bk_best,crit_rhoyy=crit_rhoyy,
              crit_rhoyz=crit_rhoyz,bk_x=bk_x,
              nu=nu,gammak_mse=gammak_mse,crit_rhoy_target=crit_rhoy_target,w=w))
}

# SSA_obj<-SSA_obj_pos  xi<-xi_pos  target<-gamma_target
# This function computes SSA trffkt and amplitudes and replicates estimates
spec_an_func<-function(SSA_obj,target,xi=NULL)
{  
# 1. Spectral weights of target    
  w_mat<-SSA_obj$w_mat
  ts.plot(SSA_obj$w_mat)
  L<-nrow(w_mat)
  # Replicate w_mat  
  V<-SSA_obj$V
  ts.plot(t(V)[1,])
  
  if (is.null((xi)))
  {  
    target_conv<-target
  }  else
  {
    target_conv<-conv_two_filt_func(xi,target)$conv
  }
# spectral weights of MSE as applied to epsilont  
  w_mse_eps<-t(V)%*%target_conv
# Same but when applied to xt  
  w_mse<-t(V)%*%target
# They match  
  w_mse_eps-SSA_obj$w_mat
  ts.plot(abs(w_mse))
  ts.plot(abs(w_mse_eps))
  nu<-SSA_obj$nu_opt
  
# 2. Compute amplitude functions with formulas in paper: K=L+1    
  trffkt_ar2<-rep(NA,L)
  for (i in 1:L)
  {
    omegai<-i*pi/(L+1)
    trffkt_ar2[i]<-1/(nu-2*cos(omegai))
  }
  ts.plot(trffkt_ar2)
  ts.plot(abs(trffkt_ar2))
  # 3. Replicate SSA predictor with frequency domain convolution formula 
  br_white_noise<-br_eps<-rep(0,L)
  for (i in 1:L)
  {  
    br_eps<-br_eps+w_mse_eps[i]*trffkt_ar2[i]*V[,i]
    br_white_noise<-br_white_noise+w_mse[i]*trffkt_ar2[i]*V[,i]
  }
# They match up to arbitrary scaling (and sign) 
  scale(cbind(SSA_obj$ssa_eps,br_eps))
  if (is.null((xi)))
  {  
    scale(cbind(SSA_obj$ssa_x,br_eps))
  }  else
  {
# Deconvolute xi from b_eps 
    scale(cbind(SSA_obj$ssa_x,deconvolute_func(br_eps,xi)$dec_filt))
  }
# This one doesn't match in general (if xi!=NULL)
  scale(cbind(SSA_obj$ssa_x,br_white_noise))

# 4. Spectral decomposition of SSA predictor  
  w_br_x<-t(V)%*%SSA_obj$ssa_x
  w_br_eps<-t(V)%*%br_eps
  w_br_white_noise<-t(V)%*%br_white_noise
  ts.plot(abs(w_br_white_noise))
  ts.plot(abs(w_br_x))
  ts.plot(abs(w_br_eps))
# 5. Convolution: they match  
  w_br_white_noise-trffkt_ar2*w_mse
  w_br_eps-trffkt_ar2*w_mse_eps
  # Convolution in absolute value  
  abs(w_br_white_noise)-abs(trffkt_ar2)*abs(w_mse)
  return(list(trffkt_ar2=trffkt_ar2,w_br_white_noise=w_br_white_noise,w_br_eps=w_br_eps,w_br_x=w_br_x,w_mse=w_mse,w_mse_eps=w_mse_eps))
}




# This function computes bk  in non-stationary I(1) case by imposing cointegration constraint
# For given Lagrangian multiplier lambda the function derives bk based on cointegration solution
# Target:  gamma_tilde is the MSE target as applied to epsilon_t; gamma_mse (this is the MSE target as applied to original data) is used for 
#   computing Gamma(0) in the cointegration constraint only
# For lambda=0 the SSA transformation is an identity, i.e., bk=gamma_tilde
# Xi is the Wold decomposition, Sigma is the summation (integration) operator (it is only used for computing target correlation); 
# M is the autocovariance generating matrix; B is used for setting-up cointegration constraint
bk_int_func<-function(lambda,gamma_mse,Xi,Sigma,Xi_tilde,M,B,gamma_tilde)
{ #lambda1<-lambda_opt   lower_limit_nu<-lower_limit_nu_triangulation
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
# First Cartesian basis vector for cointegration constraint, see section 6.2 technical paper  
  e1<-c(1,rep(0,L-1))
  e1
# Cointegration constraint: sum of SSA coefficients should equal sum of MSE coefficients, see section 6.2 technical paper    
  Gamma0<-sum(gamma_mse)
# Formula for bk given lambda under cointegration constraint, see section 6.2 technical paper    
  b_tilde<-solve(t(B)%*%t(Xi_tilde)%*%Xi_tilde%*%B+
                   lambda*t(B)%*%t(Xi)%*%(M-rho1*diag(rep(1,L)))%*%Xi%*%B)%*%
    (t(B)%*%t(Xi_tilde)%*%(gamma_tilde-Gamma0*Xi_tilde%*%e1)
     -lambda*Gamma0*t(B)%*%t(Xi)%*%(M-rho1*diag(rep(1,L)))%*%
       Xi%*%e1)
# Derive b_x applied to original data from b_tilde, see section 6.2 technical paper  
# This is also the same parameter which is applied to first differences of data in the synthetic stationary processes, see section 6.2 technical paper    
  b_x<-Gamma0*e1+B%*%b_tilde
# Derive b_eps as applied epsilon: this is used to compute lag-one ACF rho_yy and MSE mse_yz of SSA below  
  b_eps<-Xi%*%b_x
  
# Compute lag-one acf      
  rho_yy<-as.double(t(b_eps)%*%M%*%b_eps/(t(b_eps)%*%b_eps))
  
# Compute criterion (objective function): two different targets: rho_yz (virtual target correlation based on synthetic stationary series) and mse_yz
# We use mse_yz because formula of cointegrated SSA is simpler when relying on MSE objective  
# 1. target correlation based on synthetic stationary series
  rho_yz<-as.double(t(Sigma%*%b_eps)%*%gamma_tilde)/(sqrt(t(Sigma%*%b_eps)%*%(Sigma%*%b_eps))*sqrt(t(gamma_tilde)%*%gamma_tilde))
# 2. MSE  
  mse_yz<-as.double(t(gamma_tilde-Sigma%*%b_eps)%*%(gamma_tilde-Sigma%*%b_eps))
  return(list(b_x=b_x,b_eps=b_eps,rho_yy=rho_yy,rho_yz=rho_yz,mse_yz=mse_yz))
}

# This function is used in numerical optimization of Lagrangian lambda such that solution bk conforms with HT constraint
# It uses bk_int_func above and returns the absolute difference between desired rho1 and effective rho_yy
# Numerical optimization then minimizes this gap
b_optim<-function(lambda,gamma,Xi,Sigma,Xi_tilde,M,B,gamma_tilde,rho1)
{
  if (abs(rho1)>1)
  {
    print(paste("Lag-one ACF rho1 is larger one in absolute value; rho1=",rho1,sep=""))
    return()
  } else
  {
    return(abs(bk_int_func(lambda,gamma,Xi,Sigma,Xi_tilde,M,B,gamma_tilde)$rho_yy-rho1))
  }
}



# This function computes system matrices, in particular also for the I(1) case with cointegration
compute_system_filters_func<-function(L,lambda_hp,a_vec,b_vec)
{
  
  HP_obj<-HP_target_mse_modified_gap(2*(L-1)+1,lambda_hp)
  hp_two<-HP_obj$target
  
  
  HP_obj<-HP_target_mse_modified_gap(L,lambda_hp)
  # Concurrent MSE estimate of bi-infinite HP assuming white noise (truncate symmetric filter)
  #  HP_obj$hp_trend
  gamma<-HP_obj$hp_mse
  ts.plot(gamma)
  hp_trend<-HP_obj$hp_trend

# Wold decompoistion (MA inversion)  
  xi<-ARMAtoMA(ar=a_vec,ma=b_vec,lag.max=L)

  
# Compute all system matrices: all matrices are specified in Wildi (2025)
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

  if (F)
  {
# MSE filter for random-walk: the above specification is more general since it accounts for Wold decomposition of first differences too    
    gamma_mse<-hp_two[L:length(hp_two)]+c(sum(hp_two[(L+1):length(hp_two)]),rep(0,L-1))
    ts.plot(gamma_mse)
  }
  
# Compute transformed MSE filter, see section 6.2 technical SSA paper  
  gamma_tilde<-Xi_tilde%*%gamma_mse
# Autocovariance generating matrix, see theorem 1  
  M<-matrix(nrow=L,ncol=L)
  M[L,]<-rep(0,L)
  M[L-1,]<-c(rep(0,L-1),0.5)
  for (i in 1:(L-2))
    M[i,]<-c(rep(0,i),0.5,rep(0,L-1-i))
  M<-M+t(M)
  M
# Cointegration matrix, see section 6.2 technical paper  
  B<-rbind(rep(-1,L-1),diag(rep(1,L-1)))
  B
  return(list(B=B,M=M,gamma_tilde=gamma_tilde,gamma_mse=gamma_mse,Xi_tilde=Xi_tilde,Sigma=Sigma,Delta=Delta,Xi=Xi,hp_two=hp_two,hp_trend=hp_trend))
}
