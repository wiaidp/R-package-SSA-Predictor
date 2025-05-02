
#----------------------------------------------------
# The following three functions compute holding-times and link lag-one acf and holding-times, see section 1 in paper
compute_holding_time_func<-function(b)
{
  rho_ff1<-b[1:(length(b)-1)]%*%b[2:length(b)]/sum(b^2)
  # Mean holding-time
  ht<-1/(2*(0.25-asin(rho_ff1)/(2*pi)))
  # Alternative expression  
  if (F)
    ht<-pi/acos(rho_ff1)
  
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

compute_empirical_ht_func<-function(x)
{ 
  x<-na.exclude(x)
  len<-length(x)-1
  empirical_ht<-len/length(which(sign(x[2:len])!=sign(x[1:(len-1)])))
  return(list(empirical_ht=empirical_ht))
}


peak_cor_func<-function(x_mat,j,k,lag_max,plot_T=F)
{  
  cor_left<-NULL
  for (l in 2:lag_max)
    cor_left<-c(cor_left,cor(na.exclude(cbind(x_mat[l:nrow(x_mat),j],x_mat[1:(nrow(x_mat)-l+1),k])))[1,2])
  cor_right<-NULL
  for (l in 1:lag_max)
    cor_right<-c(cor_right,cor(na.exclude(cbind(x_mat[l:nrow(x_mat),k],x_mat[1:(nrow(x_mat)-l+1),j])))[1,2])
  peak_cor<-c(cor_right[lag_max:1],cor_left)
  if (plot_T)
  {
    par(mfrow=c(1,1))
    plot(-lag_max+1:length(peak_cor),peak_cor,main="CCF",type="l",xlab="Lag",ylab="CCF",lwd=1)
    abline(v=0)
    abline(v=-lag_max+which(peak_cor==max(peak_cor)),col="blue")
  }
  return(peak_cor)
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


# Multivariate convolution for M-SSA: 
# Care: ordering of filters is relevant since matrix multiplication is generally not commutative
#   -Ordering is irrelevant if one of the sequence is diagonal (so that matrix multiplication is commutative)
# It is assumed that filters are transposed i.e. nrow=n and ncol=n*L
M_conv_two_filt_func<-function(filt1,filt2)
{
# filt1<-t(bk_mat)  filt2<-xi
  n<-dim(filt1)[1]
  if (n!=dim(filt2)[1])
  {
    print("Filter dimension n differ")
    return()
  }
  if (ncol(filt1)!=ncol(filt2))
  {
    print("Filter lengths differ")
    return()
  }
# Filter length (for each series, i=1,...,n)  
  L<-ncol(filt1)/n
# Initialize convolution as a corresponding matrix with zeroes  
  conv<-0*filt1
  for (i in 1:L)#i<-2
  {
    for (j in 1:i)#j<-2
    {
      conv[,i+(0:(n-1))*L]<-conv[,i+(0:(n-1))*L]+filt1[,j+(0:(n-1))*L]%*%filt2[,i+1-j+(0:(n-1))*L]
#        filt1[1:i]*filt2[i:1]
    }
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


# Multivariate deconvolution for M-SSA: ordering is relevant
#   Deconvolute filt2 from filt1: filt1 is the convolution
M_deconvolute_func<-function(filt1,filt2)
{
# filt1<-t(bk_best)  gamma_target  filt1<-xi  filt2<-gamma_mse
  n<-dim(filt1)[1]
  if (n!=dim(filt2)[1])
  {
    print("Filter dimension n differ")
    return()
  }
  if (ncol(filt1)!=ncol(filt2))
  {
    print("Filter lengths differ")
    return()
  }
# Filter length (for each series, i=1,...,n)  
  L<-ncol(filt1)/n
# Initialize convolution as a corresponding matrix with zeroes  
  deconv<-0*filt1
# Compute first element of deconvolution: ig filt2 ist MA-inversion xi then f2inv is just the identity 
  f2inv<-solve(filt2[,1+(0:(n-1))*L])
  deconv[,1+(0:(n-1))*L]<-filt1[,1+(0:(n-1))*L]%*%f2inv
  for (i in 2:L)#i<-2
  {
    for (j in 1:(i-1))#j<-2
    {
      deconv[,i+(0:(n-1))*L]<-deconv[,i+(0:(n-1))*L]+deconv[,j+(0:(n-1))*L]%*%filt2[,i+1-j+(0:(n-1))*L]
    }
    deconv[,i+(0:(n-1))*L]<-(filt1[,i+(0:(n-1))*L]-deconv[,i+(0:(n-1))*L])%*%f2inv 
  }  
  return(list(deconv=deconv))
}







sa_from_rho_func<-function(rho)
{
  return(0.5+asin(rho)/pi)
}


#-------------------------------------------------------

# This function computes exact SSA-filter based on corollary in paper: it relies on find_lambda1_subject_to_holding_time_constraint_func 
#   -grid-search of lambda for given target gamma_target (target=MSE estimate i.e. not symmetric filter) 
#     rho1 and length L in grid with resolution grid_size
#   -For smoothing one needs positive values of lambda only i.e. with_negative_lambda==T
# It computes optimal estimates for various forecast horizons as specified in forecast_horizon_vec

# The function returns:
# crit_rhoyy: lag-one ACF of SSA
# bk_mat: SSA applied to WN
# lambda_opt: optimal lambda
# crit_rhoy_target: target correlation with acausal target (objective function)
# bk_x_mat: SSA applied to xt (obtained from deconvolution of xi from bk_mat)
# crit_rhoyz: target correlation with MSE (objective function)
# nu_opt: optimal nu
# gammak_mse: causal MSE applied to epsilont
# rho_mat: lag one ACFs
# w_mat: spectral decomposition of MSE
# opt_obj: optimization object
# gammak_x_mse: causal MSE applied to xt
# gammak_target: effective acausal target convolved with xi (MA-inversion)

MSSA_func<-function(split_grid,L,forecast_horizon_vec,grid_size,gammak_generic,rho1,with_negative_lambda=F,xi=NULL,lower_limit_nu="rhomax",Sigma=NULL,symmetric_target=F)
{ 
#forecast_horizon_vec<-delta  gammak_generic<-gamma_target L<-101  L<-L_long rho1<-rho0
  if (is.vector(gammak_generic))
    gammak_generic<-matrix(gammak_generic,nrow=1)
  if (is.vector(xi))
    xi<-matrix(xi,nrow=1)
  if (!is.matrix(Sigma)&!is.null(Sigma))
    Sigma<-matrix(Sigma)
  # Check dimensions  
  if (!is.null(Sigma))
  {  
# Multivariate case    
    if (dim(Sigma)[1]!=dim(Sigma)[2])
    {  
      print("Sigma is not a rectangular matrix")
      return()
    }
    if (dim(Sigma)[1]!=nrow(gammak_generic))
    {  
      print("dimension of Sigma does not agree with gammak_generic")
      return()
    }
    if (dim(gammak_generic)[1]!=dim(Sigma)[1])
    {
      print("dim(gammak_generic)[1]!=dim(Sigma)[1]")
      return()
    }  
    if (dim(gammak_generic)[2]!=L*dim(Sigma)[1])
    {
      print(dim(gammak_generic)[2])
      print(L)
      print(dim(Sigma))
      print("dim(gammak_generic)[2]!=L*dim(Sigma)[1]")
      return()
    }  
  } else
  {
# Univariate case    
    if (length(gammak_generic)<L)
    {
      gammak_generic<-c(gammak_generic,rep(0,L-length(gammak_generic)))
      print("Warning: length(gammak_generic)<L will be extended with zeroes")
      if (is.vector(gammak_generic))
        gammak_generic<-matrix(gammak_generic,nrow=1)
      
    }
    if (!is.null(xi)&is.vector(xi))
    {
      xi<-matrix(xi,nrow=1)
    }  
        
  }
  
# Transpose if necessary:   
  if (dim(gammak_generic)[2]<dim(gammak_generic)[1])
    gammak_generic<-t(gammak_generic)
  if (!is.null(xi))
  {
    if (dim(xi)[2]<dim(xi)[1])
      xi<-t(xi)
  }
  bk_mat<-bk_x_mat<-crit_rhoyz<-crit_rhoyy<-crit_rhoy_target<-gammak_target_target_mat<-gammak_target_target_symmetric_mat<-rho_mat<-NULL
# If xi is not specified in call then it does not exist: here we define it as NULL  
  if (is.null(xi))
    xi<-NULL
  if (is.null(Sigma))
    Sigma<-NULL
  rho_mat<-w_mat<-NULL
  # Loop over all forecast horizons  
  for (i in 1:length(forecast_horizon_vec))#i<-1
  {  
#    print(paste(round(100*i/(1+length(forecast_horizon_vec)),0),"%",sep=""))
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
      opt_obj<-fast_halfway_triangulation_find_lambda1_subject_to_holding_time_constraint_func(split_grid,L,gammak_generic,rho1,forecast_horizon,xi,lower_limit_nu,Sigma,symmetric_target)
# rho_mat is NULL since we do not have criterion values for all grid-points on equidistant grid
# It is assumed that solution is unique i.e. |nu|>02*rho_max      
      rho_mat<-NULL
    } else
    {
# Grid-search would be required if $|nu|<2*rho_max(L)$ because in this case the solution to the holding-time constraint is not unique anymore
#   -In this case grid-searcg computes all solutions and returns a matrix rho_mat with entries abs(rho_i-rho(y,y,1)) and rho(y,z,delta) i.e. the criterion value
#   -One then has to look for all entries with small or vanishing abs(rho_i-rho(y,y,1)) (holding-time constraint is met). 
#     The SSA-solution is that entry which maximizes the criterion avlue rho(y,z,delta)     
      opt_obj<-grid_search_find_lambda1_subject_to_holding_time_constraint_func(grid_size=NULL,L,gammak_generic,rho1,forecast_horizon,with_negative_lambda,xi,lower_limit_nu,Sigma,symmetric_target)
# rho_mat collects criterion value (correlation SSA with target), lag-one acf (holding-time) and corresponding nu
# Ordering is such that first row correponds to nu which 1. maximizes criterion value and 2. minimizes deviation from holding-time
# First row should be best overall solution in case of multiple solutions (if $|nu|<2*rho_max$) 
# This is not used for fast_halfway_triangulation_find_lambda1_subject_to_holding_time_constraint_func above because 
#   there it is assumed that solution is unique i.e. |nu|>02*rho_max      
      rho_mat<-cbind(rho_mat,opt_obj$rho_mat)
    }
    
# Coefficients as applied to xt: xt can be autocorrelated    
    bk_mat<-cbind(bk_mat,opt_obj$bk_best)
# Coefficients as applied to epsilont in Wold decomposition of xt: is NULL if xi==NULL (white noise)     
    bk_x_mat=cbind(bk_x_mat,opt_obj$bk_x)
# Optimal lambda and nu: note that nu=c*(lambda+1/lambda) where c depends on the choice of lower_limit_nu, see function bk_func below for details      
#   Since we do not return c (at this stage) lambda_opt is not useful; but nu_opt is!    
    lambda_opt<-opt_obj$lambda_opt
    nu_opt=opt_obj$nu_opt
# MSE as applied to epsilont    
    gammak_mse=opt_obj$gammak_mse
# MSE as applied to xt    
    gammak_x_mse=opt_obj$gammak_x_mse
    gammak_target<-opt_obj$gammak_target
# Objective function of SSA-criterion: correlation with MSE    
    crit_rhoyz<-c(crit_rhoyz,opt_obj$crit_rhoyz)
# Holding-time or, better, lag-one acf of SSA-estimate    
    crit_rhoyy<-c(crit_rhoyy,opt_obj$crit_rhoyy)
# Objective function of SSA-criterion: correlation with proper target    
    crit_rhoy_target<-c(crit_rhoy_target,opt_obj$crit_rhoy_target)
# Spectral decomposition of (MSE-) target    
    w_mat<-cbind(w_mat,opt_obj$w)
# Variance-covariance matrix of acausal target    
    var_target<-opt_obj$var_target
  }
  
# Check: for a lowpass design preserving signs the sum of the filter coefficients should be strictly positive
  apply(bk_mat,2,sum)

# Check theoretical holding-times of optimum: should match ht-constraint
  if (!is.null(bk_x_mat))
  { 
# If xt is not white noise    
    apply(bk_x_mat,2,compute_holding_time_func)
  } else
  {
 # If xt is white noise    
    apply(bk_mat,2,compute_holding_time_func)
  }
#  gammak_mse=t(gammak_mse)
#  gammak_x_mse=t(gammak_x_mse)
#  gammak_target=t(gammak_target)
  return(list(crit_rhoyy=crit_rhoyy,bk_mat=bk_mat,lambda_opt=lambda_opt,crit_rhoy_target=crit_rhoy_target,
              bk_x_mat=bk_x_mat,crit_rhoyz=crit_rhoyz,nu_opt=nu_opt,gammak_mse=gammak_mse,rho_mat=rho_mat,w_mat=w_mat,opt_obj=opt_obj,gammak_x_mse=gammak_x_mse,gammak_target=gammak_target,var_target=var_target))
} 






# This function finds optimal lambda (note that nu=lambda+1/lambda)) through fast hlaf-splits of unit interval for lambda
# It implements solution in corollary 1 of BCY paper and corollary 2 of JTSE paper. Multivariate is based on IJFOR-paper
# Meaning of parameters:
# -gammak_generic is gamma of target as applied to xt (not MSE)
#   Internally the function computes the MSE gamma from gammak_generic and uses MSE (not zt) as target
# -rho1 is holding-time constraint: vector in case of multivariate series (one constraint per series)
# -xi is MA-inversion of xt: default is white noise 
# -lower_limit_nu
# Trick: the parametrization of nu in function is  nu<-(lambda1+1/lambda1)*c. 
#   Depending on the character string lower_limit_nu various c are possible
#   -lower_limit_nu=="rhomax": default value for c 
#     In this case |nu|>2*rhomax(L): solution is unique but unit-roots are possible
#   -lower_limit_nu=="0"
#     In this case sqrt(gridsize)>|nu|>2/sqrt(gridsize): all nu are allowed if gridsize is large
#   -lower_limit_nu=="2": in this case |nu|>2 i.e. uniqueness and no unit-root i.e. bk decay to zero
# -split_grid:  interval will be splitted by successive halvings so that resolution is 1/2^split_grid
#   -If split_grid=10 then resolution corresponds to grid-search in a grid of 2split_grid^~1000 grid-points
#   -This is much faster than grid-search
#   -Potential problem: it assumes |nu|>2 (so nu=lambda+1/lambda and solution is unique)

# The function returns:
# gammak_mse: causal MSE filter as applied to WN    
# gammak_target: acausal target as applied to WN
# gammak_x_mse: causal MSE as applied to xt
# bk_best: SSA as applied to WN
# bk_x: SSA as applied to xt
# crit_rhoyy: lag-one ACF  
# crit_rhoyz: objective function: correlation with causal MSE
# crit_rhoy_target: objective function: correlation with acausal target
# w: spectral decomposition of target
# nu_opt and lambda_opt: optimal nu and lambda  
# var_target: variance-covariance of acausal target
fast_halfway_triangulation_find_lambda1_subject_to_holding_time_constraint_func<-function(split_grid,L,gammak_generic,rho1,forecast_horizon,xi=NULL,lower_limit_nu="rhomax",Sigma=NULL,symmetric_target=F)
{

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
  n<-ifelse(is.null(Sigma),1,dim(Sigma)[1])
# Specify M, M_tilde, I_tilde, eigenvalues and eigenvectors (all derived from lag-one autocovariance generating function)
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
  
# Specify eigenvalues and eigenvectors of M_tilde: in univariate case no problem. In multivariate case the problem is that eigenvectors of M_tilde are ordered according to increasing size   
  if (is.null(Sigma))
  {
# Univariate: here we just use original orderings    
    V<-eigen_M_tilde_obj$vectors
# In the univariate case eigen_M_tilde_adjusted=eigen_M_tilde   
    eigen_M_tilde_adjusted<-eigen_M_tilde<-eigen_M_tilde_obj$values
  } else
  { 
# Recompute orthonormal basis of eigenvectors of M_tilde according to  kronecker(V_Sigma[,j],V_M[,1])
#   -in ordering of lambda_k*sigma_j 
#   -consistent for eigenvectors and all eigenvalues such that formula of bk based on diagonal-matrix is correct
#   -Otherwise orderings of eigenvalues do not match: false solution 
    V<-eigen_M_tilde<-eigen_M_tilde_adjusted<-NULL
    for (j in 1:dim(Sigma)[1])
      for (k in 1:dim(V_M)[1])
      { 
# These are the eigenvalues of M_tilde reordered according to lambdak*sigma_tildej: these eigenvalues are irrelevant for optimization       
        eigen_M_tilde<-c(eigen_M_tilde,eigen_M[k]*eigen_Sigma[j])
# For the M-SSA solution we need the adjusted eigenvalues lambdak*sigma_tildej/sigma_tildej=lambdak because sigma_tildej cancels        
        eigen_M_tilde_adjusted<-c(eigen_M_tilde_adjusted,eigen_M[k])
# Here we compute the corresponding eigenvectors. 
# The ordering means that we append the eigenvector V_M[,k] of M n-times with weights corresponding 
#   to the i-th component of V_Sigma[,j], i=1,...,n
# This ordering is derived from the original Kronecker product kronecker(Sigma,M) which, in turn, is determined by the 
#   original arrangement of the random variables in the multivariate notation of section 1, ordering first according to 
#   lag k=0,...,L-1 and then to dimension i=1,...,n       
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


# Specify target: we rely on the MSE target (delta is used for deriving MSE; but once MSE is computed delta=0 when deriving SSA)  
# -This is the MSE-target as applied to epsilont and used for SSA-estimation: 
#     -it corresponds to convolution of gammak and Wold decomposition xi: 
#     -if xi is NULL or identity then this is simply gammak_generic shifted by forecast_horizon
# -Symmetric signal extraction filters are computed if symmetric_target==T: 
#     -in this case it is implicitly assumed that gammak_generic is the right tail 
#     -the left tail will be mirrored gammak_generic (without lag0)   
# -If gammak_generic is the effective target (no symmetrization needed) then symmetric_target==F 
#     -Default is symmetric_target==F: this is typically used for forecasting  
# Note: we do not compute the Wold decomposition of the target z_{t+delta} which would involve future epsilon_{t+k} 
#     -Instead we compute the Wold decomposition of its MSE-estimate which involves present and past epsilon_{t-k} only   
#     -Either 'target' could be used for the SSA-criterion
#     -The criterion value would be affected but the SSA-solution (maximizing either criterion) would be the same  
  MSE_obj<-M_MSE_target_func(gammak_generic,xi,forecast_horizon,symmetric_target,Sigma)
# MSE predictor: as applied to WN  
  gammak_target_mse=MSE_obj$gamma_mse
# Target: as applied to WN (is not used for derivation of SSA)  
  gammak_target=MSE_obj$gamma_target 
# Variance of target: is used to scale objective function such that the effective target correlation (not correlation with MSE) can be computed  
  var_target<-MSE_obj$var_target
# Compute MSE filter as applied to xt if xi!=NULL: not relevant for M-SSA but potentially useful for benchmarking
# In the VAR(1)-case and delta=1 this is just A (deconvolution of the MA-inversion delivers A and a negligible error at lag L due to finite-length MA-inversion)
  if (!is.null(xi))
  {
    gammak_x_mse<-t(M_deconvolute_func(gammak_target_mse,xi)$deconv)
  } else
  {
    gammak_x_mse<-t(gammak_target_mse)
  } 
    
# Loop over all series (multivariate design) 
  n<-ifelse(is.null(Sigma),1,n)
  crit_rhoyy<-crit_rhoyz<-crit_rhoy_target<-lambda_opt<-nu_opt<-bk_best<-NULL
  for (m in 1:n)#m<-1
  { 
# Compute spectral weights based on reordered eigenvectors: ordering according to lambda_k*sigma_j    
    w<-t(V)%*%gammak_target_mse[m,]
    
# Compute lag-one acf at lower right (positive) half    
    lambda_lower<-0.00000001
    
    bk_obj<-bk_func(V,w,lower_limit_nu_triangulation,lambda_lower,eigen_M_tilde_adjusted,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj)
    
    rho_yy_lower=bk_obj$rho_yy
    
# If lower bound of lag-one acf at right (positive) half is smaller than ht then the solution will be found on positive half: Otherwise on left negative half    
    sign<-ifelse(rho_yy_lower<rho1[m],1,-1)
    
    if (sign>0)
    { 
# Solution is on right positive half (smoothing)      
# Compute lag-one acf at upper boundary    
      lambda_upper<-1
      
      bk_obj<-bk_func(V,w,lower_limit_nu_triangulation,lambda_upper,eigen_M_tilde_adjusted,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj)
      
      rho_yy_upper=bk_obj$rho_yy
      if (rho_yy_upper<rho1[m])
      {
        if (lower_limit_nu_triangulation=="rhomax")
        {  
          print("There does not exist a solution: ht is larger than rhomax. Either increase L or decrease ht")
          return()
        } else 
        {
          print("There does not exist a solution for lower_limit_nu=2: try lower_limit_nu=rhomax instead")
          return()
        } 
      }  
    } else
    {
# If solution is on left negative half (unsmoothing) then we must compute new lower and upper boundaries      
      lambda_lower<--1
      
      bk_obj<-bk_func(V,w,lower_limit_nu_triangulation,lambda_lower,eigen_M_tilde_adjusted,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj)
      
      rho_yy_lower=bk_obj$rho_yy
      if (rho_yy_lower>rho1[m])
      {
        if (lower_limit_nu_triangulation=="rhomax")
        {  
          print("There does not exist a solution: ht is smaller than -rhomax. Either increase L or increase ht")
          return()
        } else
        {
          print("There does not exist a solution for lower_limit_nu=2: try lower_limit_nu=rhomax instead")
          return()
        }
      }  
      
      
      lambda_upper<--0.00000001
      
      bk_obj<-bk_func(V,w,lower_limit_nu_triangulation,lambda_upper,eigen_M_tilde_adjusted,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj)
      
      rho_yy_upper=bk_obj$rho_yy
      
    }
    for (i in 1:split_grid)#split_grid<-10
    {
      
      lambda_middle<-(lambda_upper+lambda_lower)/2

      bk_obj<-bk_func(V,w,lower_limit_nu_triangulation,lambda_middle,eigen_M_tilde_adjusted,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj)
      
      rho_yy_middle=bk_obj$rho_yy
      
      if (rho_yy_middle>rho1[m])
      {
        lambda_upper<-lambda_middle
      } else
      {
        lambda_lower<-lambda_middle
      }  
      
    }
# Change sign if necessary    
    if (bk_obj$rho_yz<0)
    {  
      bk<--bk_obj$bk
      rho_yz<--bk_obj$rho_yz
    } else
    {
      bk<-bk_obj$bk
      rho_yz<-bk_obj$rho_yz
    }
    
    bk_best_n<-bk
    crit_rhoyy<-c(crit_rhoyy,rho_yy_middle)
    crit_rhoyz<-c(crit_rhoyz,rho_yz)
# Compute criterion with respect to effective target 
# Rescale  crit_rhoyz with variance of effective target var_target computed above
    crit_rhoy_target<-c(crit_rhoy_target,crit_rhoyz[length(crit_rhoyz)]*as.vector(sqrt(t(gammak_target_mse[m,])%*%I_tilde%*%gammak_target_mse[m,]/var_target[m,m])))
    
    lambda_opt<-c(lambda_opt,lambda_middle)
# Generate nu: it is c*(lambda_middle+1/lambda_middle) whereby c depends on selection by lower_limit_nu_triangulation: 
#     2*rho_max(L) (|nu|>2rho_max: uniqueness but possibly unit-roots) and 2 i.e. $|nu|>2$ (uniqueness and no unit-roots)
# Fast half-way triangulation is effective in case of uniqueness. Therefore we skip the possibility lower_limit_nu=0.
    nu_middle<-bk_obj$c*(lambda_middle+1/lambda_middle)
    nu_opt<-c(nu_opt,nu_middle)
# Merge all targets (in multivariate setting)    
    bk_best<-cbind(bk_best,bk_best_n)
    
    
    
  }
  
  # Multivariate deconvolution for M-SSA
  # Deconvolute filt2 from filt1: filt1 is the convolution
  if (!is.null(xi))
  {
    bk_x<-t(M_deconvolute_func(t(bk_best),xi)$deconv)
# Check: should vanish    
    max(abs(M_conv_two_filt_func(t(bk_x),xi)$conv-t(bk_best)))
# Reversing ordering does not vanish (matrix multiplication is not commutative)    
    max(abs(M_conv_two_filt_func(xi,t(bk_x))$conv-t(bk_best)))
    
  } else
  {
    bk_x<-bk_best
  }
# Causal MSE filter as applied to WN    
  gammak_mse<-t(gammak_target_mse)
# Acausal target as applied to WN
  gammak_target<-t(gammak_target)
# Causal MSE as applied to xt: gammak_x_mse
# bk_best: SSA as applied to WN
# bk_x: SSA as applied to xt
# crit_rhoyy: lag-one ACF  
# crit_rhoyz: objective function: correlation with causal MSE
# crit_rhoy_target: objective function: correlation with acausal target
# w: spectral decomposition of target
# nu_opt and lambda_opt: optimal nu and lambda  
  return(list(bk_best=bk_best,crit_rhoyy=crit_rhoyy,
              crit_rhoyz=crit_rhoyz,lambda_opt=lambda_opt,bk_x=bk_x,
              nu_opt=nu_opt,gammak_mse=gammak_mse,crit_rhoy_target=crit_rhoy_target,w=w,gammak_x_mse=gammak_x_mse,
              gammak_target=gammak_target,var_target=var_target))
}




# This function computes SSA-estimate, criterion value and lag-one acf
#  if (lower_limit_nu=="0") then we focus on |nu|<2 only (case of multiple solutions)
#     -this case can occur only if grid-search has been called
#  otherwise (lower_limit_nu!="0") we focus on |nu|>2 or |nu|>2*rho_max only (uniqueness).
#     -this case occurs only if fast triangulation has been called
bk_func<-function(V,w,lower_limit_nu,lambda1,eigen_M_tilde_adjusted,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj)
{ #lambda1<-lambda_opt   lower_limit_nu<-lower_limit_nu_triangulation

  
# If lower_limit_nu!="0" then we emphasize either |nu|>2 (no unit-roots, unique solution) or $|nu|>2*rho_max$ (unique solution with possible unit-roots). 
#   -In both cases nu<-(lambda1+1/lambda1)*c  
#   -But c depends on the choice rhomax or 2
# Fast triangulation always assumes lower_limit_nu!="0" (is meaningful for unique solutuons only)  
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


# We use the reordered eigenvalues, eigenvectors and spectral weights
# Note: one cannot use original ordering of eigenvectors because we need eigen_M_tilde_adjusted=eigen_M_tilde/eigen_I_tilde and the 
#  two are ordered differently (both in ascending order of eigenvalues so that ordering of eigenvectors will not match)
# The most effective way is indeed to recompute everything from scratch in a consistent way: same ordering for 
#  eigenvectors and eigenvalues of M_tilde and of I_tilde!!!!  
  bk<-V%*%diag(1/(2*eigen_M_tilde_adjusted-nu))%*%w
  
  gammak_n<-gammak_target_mse[m,]
  
# Check different equivalent solutions: the above is simplest and fastest 
  if (F)
  {
# Original solution with matrix inversion
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
  
  if (F)
  {
# Lag one acf depends on Sigma: although SSA solution for given nu does not depend on Sigma (see above), 
#    the lag-one acf does depend on Sigma and therefore the optimal nu depends on Sigma and therefore 
#    effective SSA-solution (which matches holding-time) depends on Sigma too.    
    t(bk)%*%kronecker(diag(rep(1,n)),M)%*%bk/t(bk)%*%bk
    
  }  
  # Criterion      
  rho_yz<-(t(bk)%*%I_tilde%*%gammak_n)[1,1]/(sqrt(t(bk)%*%I_tilde%*%bk)*sqrt(gammak_n%*%I_tilde%*%gammak_n))[1,1]
# Apply optimal MSE scaling: covariance of bk with target divided by variance of bk  
  bk<-bk*abs(t(bk)%*%I_tilde%*%gammak_n)[1,1]/as.double(t(bk)%*%I_tilde%*%bk)  
  return(list(bk=bk,rho_yy=rho_yy,rho_yz=rho_yz,c=c,nu=nu,w=w))
}





# This function finds optimal lambda (note that nu=lambda+1/lambda)) through grid search based on function compute_bk_from_ma_expansion_with_boundary_constraint_func above
# For identical resolution the grid-search is much slower than the above  fast_split_find_lambda1_subject_to_holding_time_constraint_func
#   -Fast split is always to be preferred if |nu|>2\rho_{max}(L) (uniqueness, convexity)
#   -Otherwise grid-search must be used to find all local solutions of the holding-time constraint and select the one with the largets criterion value
#     If $\nu<2rho_max$ then grid-search is mandatory
# The function implements solution in corollary 1 of BCY paper and corollary 2 of JTSE paper. Multivariate is based on IJFOR-paper
# Meaning of parameters:
# -lower_limit_nu
# Trick: the parametrization of nu in function is  nu<-(lambda1+1/lambda1)*c. 
#   Depending on the character string lower_limit_nu various c are possible
#   -lower_limit_nu=="rhomax": default value for c 
#     In this case |nu|>2*rhomax(L): solution is unique but unit-roots are possible
#   -lower_limit_nu=="0"
#     In this case sqrt(gridsize)>|nu|>2/sqrt(gridsize): all nu are allowed if gridsize is large
#   -lower_limit_nu=="2": in this case |nu|>2 i.e. uniqueness and no unit-root i.e. bk decay to zero
# -gammak_generic is gamma of target as applied to xt (not MSE)
#   Internally the function computes the MSE gamma from gammak_generic and uses MSE (not zt) as target
# -rho1 is holding-time constraint: vector in case of multivariate series (one constraint per series)
# -xi is MA-inversion of xt: default is white noise (not tested for multivariate processes: will not work!!)
# -with_negative_lambda: 
#   default is F. 
#   For smoothing nu>0 i.e. negative values can be discarded (computations 2-times faster)
#   For unsmoothing (see JTSE-paper) set with_negative_lambda<-T
grid_search_find_lambda1_subject_to_holding_time_constraint_func<-function(grid_size,L,gammak_generic,rho1,forecast_horizon,with_negative_lambda=F,xi=NULL,lower_limit_nu="rhomax",Sigma=NULL,symmetric_target=F)
{  #forecast_horizon<-2  rho1<-rho0
  if (!lower_limit_nu%in%c("rhomax","2","0"))
  {
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("Selection of lower_limit_nu should be either rhomax or 2 or 0")
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
    # remove zero  
    Lambda<-Lambda[-(grid_size+1)]
  }
  # Specify M, M_tilde, I_tilde, eigenvalues and eigenvectors (all derived from lag-one autocovariance generating function)
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
  
  
  # Recompute orthonormal basis of eigenvectors of M_tilde according to  kronecker(V_Sigma[,j],V_M[,1])
  #   -in ordering of lambda_k*sigma_j 
  #   -consistent for eigenvectors and all eigenvalues such that formula of bk based on diagonal-matrix is correct
  #   -Otherwise orderings of eigenvalues do not match: false solution  
  # Specify eigenvalues and eigenvectors of M_tilde: in univariate case no problem. In multivariate case the problem is that eigenvectors of M_tilde are ordered according to increasing size   
  if (is.null(Sigma))
  {
    # Univariate: here we just use original orderings    
    V<-eigen_M_tilde_obj$vectors
    # In the univariate case eigen_M_tilde_adjusted=eigen_M_tilde   
    eigen_M_tilde_adjusted<-eigen_M_tilde<-eigen_M_tilde_obj$values
  } else
  { 
    # Recompute orthonormal basis of eigenvectors of M_tilde according to  kronecker(V_Sigma[,j],V_M[,1])
    #   -in ordering of lambda_k*sigma_j 
    #   -consistent for eigenvectors and all eigenvalues such that formula of bk based on diagonal-matrix is correct
    #   -Otherwise orderings of eigenvalues do not match: false solution 
    V<-eigen_M_tilde<-eigen_M_tilde_adjusted<-NULL
    for (j in 1:dim(Sigma)[1])
      for (k in 1:dim(V_M)[1])
      { 
        # These are the eigenvalues of M_tilde reordered according to lambdak*sigma_tildej: these eigenvalues are irrelevant for optimization       
        eigen_M_tilde<-c(eigen_M_tilde,eigen_M[k]*eigen_Sigma[j])
        # For the M-SSA solution we need the adjusted eigenvalues lambdak*sigma_tildej/sigma_tildej=lambdak because sigma_tildej cancels        
        eigen_M_tilde_adjusted<-c(eigen_M_tilde_adjusted,eigen_M[k])
        # Here we compute the corresponding eigenvectors. 
        # The ordering means that we append the eigenvector V_M[,k] of M n-times with weights corresponding 
        #   to the i-th component of V_Sigma[,j], i=1,...,n
        # This ordering is derived from the original Kronecker product kronecker(Sigma,M) which, in turn, is determined by the 
        #   original arrangement of the random variables in the multivariate notation of section 1, ordering first according to 
        #   lag k=0,...,L-1 and then to dimension i=1,...,n       
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
  
  # Specify target: we rely on the MSE target (delta is used for deriving MSE; but once MSE is computed delta=0 when deriving SSA)  
  # -This is the MSE-target as applied to epsilont and used for SSA-estimation: 
  #     -it corresponds to convolution of gammak and Wold decomposition xi: 
  #     -if xi is NULL or identity then this is simply gammak_generic shifted by forecast_horizon
  # -Symmetric signal extraction filters are computed if symmetric_target==T: 
  #     -in this case it is implicitly assumed that gammak_generic is the right tail 
  #     -the left tail will be mirrored gammak_generic (without lag0)   
  # -If gammak_generic is the effective target (no symmetrization needed) then symmetric_target==F 
  #     -Default is symmetric_target==F: this is typically used for forecasting  
  # Note: we do not compute the Wold decomposition of the target z_{t+delta} which would involve future epsilon_{t+k} 
  #     -Instead we compute the Wold decomposition of its MSE-estimate which involves present and past epsilon_{t-k} only   
  #     -Either 'target' could be used for the SSA-criterion
  #     -The criterion value would be affected but the SSA-solution (maximizing either criterion) would be the same  
  if (F)
  {
    # Old code: is false if forecast_horizon>0 because convolution picks out Xis which belong to other series       
    if (symmetric_target)
    {  
      target_obj_mse<-target_func_symmetric(xi,gammak_generic,forecast_horizon,Sigma)
      # The same as mse but with forecast_horizon=0: this corresponds to convolution of Xi and gammak_generic without shift by forecast_horizon    
      target_obj<-target_func_symmetric(xi,gammak_generic,0,Sigma)
    } else
    {
      target_obj_mse<-target_func_one_sided(xi,gammak_generic,forecast_horizon,Sigma)
      # The same as mse but with forecast_horizon=0: this corresponds to convolution of Xi and gammak_generic without shift by forecast_horizon    
      target_obj<-target_func_one_sided(xi,gammak_generic,0,Sigma)
      max(abs(target_obj$gammak_target_mse-M_conv_two_filt_func(xi,gammak_generic)$conv))
      MSE_obj<-M_MSE_target_func(gammak_generic,xi,forecast_horizon,symmetric_target,Sigma)
      max(abs(target_obj_mse$gammak_target_mse-MSE_obj$gamma_mse))
    }
  }
  
  # New code: formally correct and simpler: relies on new multivariate convolution function  
  MSE_obj<-M_MSE_target_func(gammak_generic,xi,forecast_horizon,symmetric_target,Sigma)
  # MSE predictor: as applied to WN  
  gammak_target_mse=MSE_obj$gamma_mse
  # Target: as applied to WN (is not used for derivation of M-SSA but for computation of correlations with effective acausal target)  
  gammak_target=MSE_obj$gamma_target
  # Variance of target: is used to scale objective function such that the effective target correlation (not correlation with MSE) can be computed  
  var_target<-MSE_obj$var_target
  # Compute MSE filter as applied to xt: not relevant for M-SSA but potentially useful for benchmarking
  # In the VAR(1)-case and delta=1 this is just A (deconvolution of the MA-inversion delivers A and a negligible error at lag L due to finite-length MA-inversion)
  gammak_x_mse<-t(M_deconvolute_func(gammak_mse,xi)$deconv)
  
  
  # Initializations
  # Maximal eigenvalue i.e. rhomax(L) : is used in scaling nu in loop for one particular option 
  maxrho<-max(eigen_M_obj$values)
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
  
  # Loop over all series (multivariate design) 
  n<-ifelse(is.null(Sigma),1,n)
  for (m in 1:n)#m<-1
  { 
    # Compute spectral weights based on reordered eigenvectors: ordering according to lambda_k*sigma_j    
    w<-t(V)%*%gammak_target_mse[m,]
    
    # Loop over all grid-points
    rho_mat_n<-matrix(nrow=length(Lambda),ncol=4)
    for (i in 1:length(Lambda))#i<-1001  #i<-length(Lambda)-9 i<-1561
    {
      lambda1<-Lambda[i]

      bk_obj<-bk_func(V,w,lower_limit_nu,lambda1,eigen_M_tilde_adjusted,gammak_target_mse,m,M_tilde,I_tilde,eigen_M_obj)
      
      bk_new<-bk_obj$bk
      rho_yy_best<-bk_obj$rho_yy
      rho_yz_best<-bk_obj$rho_yz
      c<-bk_obj$c
      nu<-bk_obj$nu
      
      rho_mat_n[i,]<-c(rho_yy_best,rho_yz_best,nu,lambda1)
      # Change sign if correlation with target is negative: the sign of the optimal solution bk below is changed accordingly   
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
# Compute criterion with respect to effective target 
# Rescale  crit_rhoyz with variance of effective target var_target computed above
    crit_rhoy_target<-c(crit_rhoy_target,crit_rhoyz[length(crit_rhoyz)]*as.vector(sqrt(t(gammak_target_mse[m,])%*%I_tilde%*%gammak_target_mse[m,]/var_target[m,m])))
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
#    diff_ht[order_ht]
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
      #  if (F)
    {
      print("Solution which minimizes holding-time error is not the same as best compromise")
      print("Solution which minimizes holding-time error is replaced by best compromise")
      nu_opt<-rho_mat_best[3]
      lambda_opt<-rho_mat_best[4]
      bk_best_n<-V%*%diag(1/(2*eigen_M_tilde_adjusted-nu_opt))%*%w
      rho_yy_best<-compute_holding_time_func(bk_best_n)$rho_ff1[1,1]
      rho_yz_best<-(t(bk_best_n)%*%gammak_target_mse[m,1:length(bk_best_n)])[1,1]/(sqrt(t(bk_best_n)%*%bk_best_n)*sqrt(gammak_target_mse[m,1:length(bk_best_n)]%*%gammak_target_mse[m,1:length(bk_best_n)]))[1,1]
      if (rho_yz_best<0)
      {
        bk_best_n<--bk_best_n
        rho_yz_best<--rho_yz_best
      }
    }  
    
# Merge rho_mat over all targets (in multivariate setting)  
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
    
    #    bk_x<-cbind(bk_x,bk_x_n)
    
  }
  
  # Multivariate deconvolution for M-SSA
  # Deconvolute filt2 from filt1: filt1 is the convolution
  if (!is.null(xi))
  {
    bk_x<-t(M_deconvolute_func(t(bk_best),xi)$deconv)
    # Check: should vanish    
    max(abs(M_conv_two_filt_func(t(bk_x),xi)$conv-t(bk_best)))
    # Reversing ordering does not vanish (matrix multiplication is not commutative)    
    max(abs(M_conv_two_filt_func(xi,t(bk_x))$conv-t(bk_best)))
    
  } else
  {
    bk_x<-bk_best
  }
  
  gammak_mse<-gammak_target_mse
  #  epsilon<-0.1
  if (F)
  {  
    if (!is.null(Sigma))
    {
      i_vec<-rep(NA,n)
      for (i in 1:n)#i<-2
      {  
        smaller_eps<-which(abs(rho1[m]-(rho_mat[order(abs(rho_mat[,(i-1)*2+1]-rho1[m])),(i-1)*2+1]))>epsilon)[1]-1
        max_crit_of_good_ht_match<-max((rho_mat[order(abs(rho_mat[,(i-1)*2+1]-rho1[m])),(i-1)*2+2])[1:smaller_eps])
        i_vec[i]<-which(rho_mat[,(i-1)*2+2]==max_crit_of_good_ht_match)
      }
    }  
  }
  return(list(rho_mat=rho_mat,bk_best=bk_best,crit_rhoyy=crit_rhoyy,
              crit_rhoyz=crit_rhoyz,lambda_opt=lambda_opt,nu_vec=nu_vec,bk_x=bk_x,
              nu_opt=nu_opt,gammak_mse=gammak_mse,crit_rhoy_target=crit_rhoy_target,w=w,gammak_x_mse=gammak_x_mse,var_target=var_target))
}
# Check why lambda*xi-nu
# Check why for series 2 the holding-time does not match max crit








# This function computes system matrices
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
  # Univariate case (Sigma==NULL) or multivariate case (SIgma is cross-correlation of epsilons)  
  if (is.null(Sigma))
  {  
    # Compute eigenvalues of M 
    I_tilde<-diag(rep(1,L))
    M_tilde<-M
    eigen_M_tilde_obj<-eigen_M_obj
    eigen_I_tilde_obj<-eigen(I_tilde)
  } else  
  {
    I<-diag(rep(1,L))
    M_tilde<-kronecker(Sigma,M)
    I_tilde<-kronecker(Sigma,I)
    eigen_M_tilde_obj<-eigen(M_tilde)
    eigen_I_tilde_obj<-eigen(I_tilde)
  }
  return(list(M=M,M_tilde=M_tilde,I_tilde=I_tilde,eigen_M_obj=eigen_M_obj,eigen_M_tilde_obj=eigen_M_tilde_obj,eigen_I_tilde_obj=eigen_I_tilde_obj))
}




# This function is used to compute the MSE filter
# If xi==NULL then gammak_generic is shifted by forecast_horizon
# Otherwise the filter computes the convolution of xi and shifted gammak_generic 
# If symmetric_target==T then right tail of gammak_generic is mirrored to the left: c(gammak_generic[,L:2],gammak_generic[,1:L])
# When provided, Sigma is used to compute the variance of the (acausal) target. 
M_MSE_target_func<-function(gammak_generic,xi,forecast_horizon,symmetric_target,Sigma)
{
  
#  gammak_generic<-gamma_target  forecast_horizon<-delta
  if (is.vector(gammak_generic))
  {
    n<-1
  } else
  {
    n<-dim(gammak_generic)[1]
  }
  if (!is.null(xi)&!is.vector(xi))
  {
    if (n!=dim(xi)[1])
    {
      print("Filter dimension n differ")
      return()
    }
  }
  if (!is.null(xi)&!is.vector(xi))
  {
    if(ncol(gammak_generic)!=ncol(xi))
    {
      print("Filter lengths differ")
      return()
    }
  }
  # If target is vector: transform into matrix with a single row  
  if (is.vector(gammak_generic))
    gammak_generic<-matrix(gammak_generic,nrow=1)
  if (is.vector(xi))
    xi<-matrix(xi,nrow=1)
  # Transpose if necessary:   
  if (dim(gammak_generic)[2]<dim(gammak_generic)[1])
    gammak_generic<-t(gammak_generic)
  if (!is.null(xi))
  {
    if (dim(xi)[2]<dim(xi)[1])
      xi<-t(xi)
  }
  
  
  L<-ncol(gammak_generic)/n
  
  
  # 1. Distinguish two- (symmetric) and one-sided targets: in the former case we must compute the convolution for the entire 
  #   two-sided filter and later replace future epsilon by zero
  if (symmetric_target)
  {
    # Mirror right tail to the left  
    xi_s<-gammak_generic_s<-NULL
    for (i in 1:n)
    {
      # Mirror right tail to the left      
      gammak_generic_s<-cbind(gammak_generic_s,cbind(gammak_generic[,(i-1)*L+L:2,drop=F],gammak_generic[,(i-1)*L+1:L,drop=F]))
      # Lengthen xi and append zeroes at end      
      if (!is.null(xi))
        xi_s<-cbind(xi_s,cbind(xi[,(i-1)*L+1:L,drop=F],0*gammak_generic[,(i-1)*L+L:2,drop=F]))
    }
  } else
  {
    xi_s<-xi
    gammak_generic_s<-gammak_generic
    
  }
  
  # 2. Convolution of target and xi (if the target is not symmetric then it has been shifted, otherwise it will be shifted further down)
  if (!is.null(xi))
  {
# The ordering is generally relevant, i.e., M_conv_two_filt_func(gammak_generic_s,xi_s)!=M_conv_two_filt_func(xi_s,gammak_generic_s) 
# However, if one of the two sequences is diagonal, then the ordering is irrelevant (typically, the target is diagonal)
# Otherwise, gamma_generic is applied to X-t i.e. the natural ordering is gammak_generic_s,xi_s    
    gamma_mse<-M_conv_two_filt_func(gammak_generic_s,xi_s)$conv
# This is the same as if the matrix product is commutative (for instance if one sequence is diagonal)   
#    gamma_mse<-M_conv_two_filt_func(xi_s,gammak_generic_s)$conv
  } else
  {
    gamma_mse<-gammak_generic_s
  }
  # Specify the effective target (not the MSE predictor), i.e., target generally involves future epsilons
  gamma_target<-gamma_mse
  
  
  # 3. # We must eliminate all future epsilons (which are replaced by zeroes)
  # Shift target by forecast_horizon: symmetric target must be shifted additionally by full right tail  
  if (symmetric_target)
  {
    shift<-L-1+forecast_horizon
  } else
  {
    shift<-forecast_horizon
  }
  gamma_mse_short<-NULL
  for (i in 1:n)#i<-1
  {
    # Forecast    
    if (forecast_horizon>0)
    {
      # If forecast_horizon>0 then we shift and append zeroes at end  
      if (symmetric_target)
      {
        gamma_mse_short<-cbind(gamma_mse_short,gamma_mse[,shift+(i-1)*(2*(L-1)+1)+1:(L-forecast_horizon),drop=F],0*gamma_mse[,1:forecast_horizon,drop=F])
      } else
      {
        gamma_mse_short<-cbind(gamma_mse_short,gamma_mse[,shift+(i-1)*L+1:(L-forecast_horizon),drop=F],0*gamma_mse[,1:forecast_horizon,drop=F])
      } 
    } 
    # Nowcast    
    if (forecast_horizon==0)
    {
      # If forecast_horizon=0 then there is either no shift (for one-sided target: shift=0) or a shift by L-1 (for symmetric target) 
      # In any case we can supply all computed values        
      if (symmetric_target)
      {
        gamma_mse_short<-cbind(gamma_mse_short,gamma_mse[,shift+(i-1)*(2*(L-1)+1)+1:L,drop=F])
      } else
      {
# Note that in this case shift=0 so that shift could be omitted on the following code line        
        gamma_mse_short<-cbind(gamma_mse_short,gamma_mse[,shift+(i-1)*L+1:L,drop=F])
      }
    } 
    # Backcast    
    if (forecast_horizon<0)
    {
      if (symmetric_target)
      {
        # If forecast_horizon<0 and target symmetric then we can supply all computed values from left and right tails        
        gamma_mse_short<-cbind(gamma_mse_short,gamma_mse[,shift+(i-1)*(2*(L-1)+1)+1:L,drop=F])
      } else
      {
        # If forecast_horizon<0 and target one-sided then we must append zeroes at start (there is no forward looking left tail in this case)     
        gamma_mse_short<-cbind(gamma_mse_short,0*gamma_mse[,(i-1)*L+((L-abs(shift)+1):L),drop=F],
                               gamma_mse[,(i-1)*L+1:(L-abs(shift)),drop=F])
      }
    }
  } 
  gamma_mse<-gamma_mse_short
  
  # 4. For a symmetric target we also compute the variance: the latter is used when scaling the objective function to obtain the effective traget correlation
  if (n>1)
  {
    if (symmetric_target)
    {
      # Length is 2*L-1   
      I<-diag(rep(1,2*L-1))
      I_tilde<-kronecker(Sigma,I)
    } else
    {
      # Length is L    
      I<-diag(rep(1,L))
      I_tilde<-kronecker(Sigma,I)
    }
  } else
  {
    if (is.null(Sigma))
    {
      I_tilde<-diag(rep(1,dim(gamma_target)[2]))
    } else
    {
      I_tilde<-diag(rep(1,dim(gamma_target)[2]))*as.double(Sigma)
    }
  }
  var_target<-(gamma_target)%*%I_tilde%*%t(gamma_target)

  return(list(gamma_mse=gamma_mse,gamma_target=gamma_target,var_target=var_target))
}



# This function computes the MA-inversion: it is based on MTS package from Tsay but discards the prompt
M_MA_inv<-function (Phi = NULL, Theta = NULL, Sigma = NULL, lag = 12, orth = TRUE) 
{
# Phi<-A  Theta<-NULL   lag<-L  
  q = 0
  p = 0
  k = 0
  if (length(Theta) > 0) {
    k = dim(Theta)[1]
    k1 = dim(Theta)[2]
    q = floor(k1/k)
  }
  if (length(Phi) > 0) {
    k = dim(Phi)[1]
    k1 = dim(Phi)[2]
    p = floor(k1/k)
  }
  if (is.null(Sigma)) {
    Sigma = diag(rep(1, k))
  }
  if (orth) {
    m1 = eigen(Sigma)
    v1 = sqrt(m1$values)
    vv = diag(v1)
    Pmtx = m1$vectors
    Sh = Pmtx %*% vv %*% t(Pmtx)
  }
  if (k < 1) 
    k = 1
  PSI = diag(rep(1, k))
  if (orth) {
    WGT = c(PSI %*% Sh)
  }
  else {
    WGT = c(PSI)
  }
  for (il in 1:lag) {
    ilk = il * k
    tmp = matrix(0, k, k)
    if ((q > 0) && (il <= q)) 
      tmp = -Theta[, (ilk - k + 1):ilk]
    if (p > 0) {
      iend = min(il, p)
      for (j in 1:iend) {
        jdx = (il - j)
        kdx = j * k
        tmp = tmp + Phi[, (kdx - k + 1):kdx] %*% PSI[, 
                                                     (jdx * k + 1):(jdx * k + k)]
      }
    }
    PSI = cbind(PSI, tmp)
    if (orth) {
      WGT = cbind(WGT, c(tmp %*% Sh))
    }
    else {
      WGT = cbind(WGT, c(tmp))
    }
  }
  wk1 = WGT
  for (i in 1:k^2) {
    wk1[i, ] = cumsum(WGT[i, ])
  }
  tdx = c(1:(lag + 1)) - 1
  par(mfcol = c(k, k), mai = c(0.3, 0.3, 0.3, 0.3))
  if (orth) {
    gmax = max(WGT)
    gmin = min(WGT)
    cx = (gmax - gmin)/10
    gmax = gmax + cx
    gmin = gmin - cx
    for (j in 1:k^2) {
      plot(tdx, WGT[j, ], type = "l", xlab = "lag", ylab = "IRF", 
           ylim = c(gmin, gmax), cex.axis = 0.8)
      points(tdx, WGT[j, ], pch = "*", cex = 0.8)
      title(main = "Orth. innovations")
    }
    #        cat("Press return to continue ", "\n")
    #        readline()
    gmax = max(wk1)
    gmin = min(wk1)
    cx = (gmax - gmin)/10
    gmax = gmax + cx
    gmin = gmin - cx
    for (j in 1:k^2) {
      plot(tdx, wk1[j, ], type = "l", xlab = "lag", ylab = "Acu-IRF", 
           ylim = c(gmin, gmax), cex.axis = 0.8)
      points(tdx, wk1[j, ], pch = "*", cex = 0.8)
      title(main = "Orth. innovations")
    }
  }
  else {
    gmax = max(WGT)
    gmin = min(WGT)
    cx = (gmax - gmin)/10
    gmax = gmax + cx
    gmin = gmin - cx
    for (j in 1:k^2) {
      plot(tdx, WGT[j, ], type = "l", xlab = "lag", ylab = "IRF", 
           ylim = c(gmin, gmax), cex.axis = 0.8)
      points(tdx, WGT[j, ], pch = "*", cex = 0.8)
      title(main = "Orig. innovations")
    }
    #        cat("Press return to continue ", "\n")
    #        readline()
    gmax = max(wk1)
    gmin = min(wk1)
    cx = (gmax - gmin)/10
    gmax = gmax + cx
    gmin = gmin - cx
    for (j in 1:k^2) {
      plot(tdx, wk1[j, ], type = "l", xlab = "lag", ylab = "Acu-IRF", 
           ylim = c(gmin, gmax), cex.axis = 0.8)
      points(tdx, wk1[j, ], pch = "*", cex = 0.8)
      title(main = "Orig. innovations")
    }
  }
  par(mfcol = c(1, 1))
# Ordering according to M-SSA parametrization
  M_MA_inv <- list(psi = PSI,irf=WGT)
}










