
# Perform simulations for Signal Extraction applications (univariate SSA and bivariate M-SSA): check convergence of sample performances to expected (true) expressions

# This function applies filters to data and computes expected and sample performances: MSE, SA, HT, target correlations
# It generates filterd data based on x_mat (original data) or epsilon (MA-inversion) if the latter is provided: both should be identical up to finte MA-inversion error
# It returns expected and sample performances and filtered series
# It also demonstrates the formula in the paper when deriving expected measures

# Inputs:
# -hp_classic_concurrent and hp_classic_concurrent_eps: classic concurrent HP applied to x and epsilon (the first is used for filtering and the second for computing theoretical performance measures)
# -gammak_mse,gammak_x_mse: MSE estimates of two-sided HP applied to x and epsilon
# -gamma_target: left tail of acausal HP target 
# -gamma_target_long: convolution of acausal target with MA inversion: used for computing MSEs
# -bk_x_mat,bk_mat: SSA solution applied to x and epsilon
# -x_mat data
# -m: Selection of m-th series (useful in univariate design)
# -L_short: is smaller than L and allows to compute the acausal filter closer to the series boundaries
# -L filter length
# -M_tilde,I_tilde: system matrices
# -var_target: variance of acausal target
# -ht_ssa_vec: HTs imposed in constraints
# -epsilon_mat: model residuals (currently not used)

# Outputs: filtered series and expected as well as sample performances
filter_perf_func<-function(hp_classic_concurrent,hp_classic_concurrent_eps,gammak_mse,gammak_x_mse,gamma_target,gamma_target_long,bk_x_mat,bk_mat,x_mat,m,L_short,L,M_tilde,I_tilde,var_target,ht_ssa_vec,Sigma,epsilon_mat=NULL)
{  
#x_mat<-x_mat[,m]  hp_classic_concurrent<-hp_classic_concurrent_x[,m]  hp_classic_concurrent_eps<-hp_classic_concurrent_eps[,m]
# gammak_mse<-gammaeps_mat[,m]   gammak_x_mse<-gammax_mat[,m]  gamma_target<-gamma_target_mat[,m]  
# gamma_target_long<-gamma_target_long_mat[,m]   bk_x_mat<-bx_mat[,m]  bk_mat<-beps_mat[,m] var_target<-var_target_mat[m]

# Sanity checks: works for uni- and multivariate designs
  if (!is.matrix(x_mat))
  {
    xnames<-names(x_mat)
    x_mat<-matrix(x_mat,ncol=1)
    rownames(x_mat)<-xnames
  }
  if (is.vector(hp_classic_concurrent))
    hp_classic_concurrent<-matrix(hp_classic_concurrent,ncol=1)
  if (is.vector(hp_classic_concurrent_eps))
    hp_classic_concurrent_eps<-matrix(hp_classic_concurrent_eps,ncol=1)
  if (is.vector(epsilon_mat))
    epsilon_mat<-matrix(epsilon_mat,ncol=1)
  if (is.vector(gammak_mse))
    gammak_mse<-matrix(gammak_mse,ncol=1)
  if (is.vector(gammak_x_mse))
    gammak_x_mse<-matrix(gammak_x_mse,ncol=1)
  if (is.vector(gamma_target))
    gamma_target<-matrix(gamma_target,ncol=1)
  if (is.vector(gamma_target_long))
    gamma_target_long<-matrix(gamma_target_long,ncol=1)
  if (is.vector(bk_x_mat))
    bk_x_mat<-matrix(bk_x_mat,ncol=1)
  if (is.vector(bk_mat))
    bk_mat<-matrix(bk_mat,ncol=1)
  n<-ncol(bk_x_mat)
  len<-nrow(x_mat)
# Transpose if necessary  
  if (dim(gamma_target)[1]<dim(gamma_target)[2])
    gamma_target<-t(gamma_target)
  if (is.vector(var_target))
    var_target<-matrix(var_target)
# Univariate design  
  if (n==1)
    m<-1
# Check that series are centered i.e. mean zero: otherwise sample HTs are generally not properly defined
  if (max(apply(x_mat,2,mean)/apply(x_mat,2,sd))>0.001)
  {
    print("Warning: the time series are possibly not centered (not zero-mean). Therefore sample HTs are eventually not properly defined")
  }

# 0. Classic concurrent HP: apply filter to m-th column of x_mat: we use hp_classic_concurrent (not hp_classic_concurrent_eps)
  z_classic_HP<-rep(NA,len)  
  for (j in L:len)
    z_classic_HP[j]<-apply(hp_classic_concurrent*(x_mat[j:(j-L+1),m]),2,sum)
# 1. Generate yt in either of two equivalent ways:        
# 1.a Apply bk_x_mat to x_mat
  bk<-NULL
# Extract coefficients applied to m-th series    
  for (j in 1:n)#j<-2
    bk<-cbind(bk,bk_x_mat[((j-1)*L+1):(j*L),m])
  ts.plot(scale(bk))
  y<-rep(NA,len)
#  y_bi<-matrix(nrow=len,ncol=n)
  for (j in L:len)
  {
    y[j]<-sum(apply(bk*(x_mat[j:(j-L+1),]),2,sum))
#    y_bi[j,]<-apply(bk*(x_mat[j:(j-L+1),]),2,sum)
  }
#  ts.plot(cbind(y_bi,apply(y_bi,1,sum)),lty=1:3,ylim=c(-0.5,0.5))
  
# 1.b Apply bk_mat to epsilon_mat
# The M-SSA solution bk_mat is the convolution of bk_x_mat and MA-inversion: it is applied to WN epsilon
  if (!is.null(epsilon_mat))
  {
    bk<-NULL
# Extract coefficients applied to m-th series    
    for (j in 1:n)#j<-2
      bk<-cbind(bk,bk_mat[((j-1)*L+1):(j*L),m])
    ts.plot(scale(bk))
    y_eps<-rep(NA,len)
    for (j in L:len)
      y_eps[j]<-sum(apply(bk*(epsilon_mat[j:(j-L+1),]),2,sum))
# y and y_eps should be nearly identical (up to finite MA-inversion error)
  }
  
# 2. MSE-estimate 
# 2.a Apply gammak_x_mse to x_mat
  gammak<-NULL
  for (j in 1:n)#j<-2
    # gammak_mse is convolution of MSE with Wold-decomposition: must be applied to epsilont      
    gammak<-cbind(gammak,gammak_x_mse[((j-1)*L+1):(j*L),m])
  ts.plot(gammak,lty=1:2)
  z_mse<-rep(NA,len)
#  z_mse_bi<-matrix(rep(NA,n*len),ncol=n)
  for (j in L:len)
  {  
    z_mse[j]<-sum(apply(gammak*x_mat[j:(j-L+1),],2,sum))
#    z_mse_bi[j,]<-apply(gammak*x_mat[j:(j-L+1),],2,sum)
  }
#  ts.plot(cbind(z_mse_bi,apply(z_mse_bi,1,sum)),lty=1:3,ylim=c(-0.5,0.5))
# 2.b Apply gammak_mse to epsilon_mat
  if (!is.null(epsilon_mat))
  {
    bk<-NULL
    for (j in 1:n)#j<-2
      gammak<-cbind(gammakk,gammak_mse[((j-1)*L+1):(j*L),m])
    ts.plot(scale(gammak))
    z_mse_eps<-rep(NA,len)
    for (j in L:len)
      z_mse_eps[j]<-sum(apply(gammak*(epsilon_mat[j:(j-L+1),]),2,sum))
# z_mse and z_mse_eps should be nearly identical (up to finite MA-inversion error)
  }
  
# 3.Target zt+delta: based on xt 
  gammak<-NULL
  for (j in 1:n)#j<-1
    #  gamma_target is original Signal extraction filter as applied to xt : L_short allows for longer sample, closer to sample boundary
    gammak<-cbind(gammak,(gamma_target)[((j-1)*L+1):((j-1)*L+L_short),m])
  z<-rep(NA,len)
  # Truncated two-sided filter
  for (j in L:(len-L_short))
    z[j]<-sum(apply(gammak*x_mat[j:(j-L_short+1),],2,sum))+sum(apply(matrix(gammak[-1,]*x_mat[(j+1):(j+L_short-1),]),2,sum))
  # Shift z by delta    
  if (delta>0)
  {  
    zdelta<-c(z[(delta+1):len],rep(0,delta))
  } else
  {
    if (delta<0)
    {
      zdelta<-c(rep(0,delta),z[1:(len-abs(delta))])
    } else
    {
      zdelta<-z
    }
  }
  names(zdelta)<-names(y)<-names(z_mse)<-rownames(x_mat)
  hp_classic_concurrent<-as.vector(hp_classic_concurrent)
  
#-------------------------------------------------------
  # Performances: MSE, target correlations HT and SA for (M-)SSA and MSE (expected and sample)
  # We need filters applied to WN (not xt) to compute theoretical performances (all formula rely on convolution of filters with MA-inversion) 
  # 1. MSE: empirical and expected MSE
  # For the MSE between M-SSA and acausal target we need
  # a. gamma_target_long : acausal target convolved with MA inversion and shifted by delta
  # b. bk_mat: bk convolved with MA-inversion
  #   But we must extended the filter with zeroes on left tail (corresponding to future epsilons of acausal target)  
  # c. Compute the differences between both filters and compute variance
  # d. For that purpose we need I_tilde_long computed for long filters: they all have length 2*L-1
  
  # Let's first proceed to a.
  # Compute two-sided shifted by delta
  # gamma_target_long is two-sided convolved with xi but without shift by delta
  # we then need to shift it by delta  
  gamma_long<-c(gamma_target_long[(delta+1):(2*L-1),m],rep(0,delta))
  if (n>1)
  { 
    for (i in 2:n)#i<-2
      gamma_long<-c(gamma_long,c(gamma_target_long[(i-1)*(2*L-1)+(delta+1):(2*L-1),m],rep(0,delta)))
  }
  # THis is used for d.  
  M_obj_long<-M_func(2*L-1,Sigma)
  I_tilde_long<-M_obj_long$I_tilde
  dim(I_tilde_long)
  # 1.1 Classic HP concurrent  
  HP_long<-gamma_HP_classic<-NULL
  # Compute one-sided univariate HP-concurrent with zeroes to the left tail 
  # Univariate filter is applied to m-th series only (all other coefficients vanish)  
  for (i in 1:n)#i<-1
  {
    HP_long<-c(HP_long,c(rep(0,L-1),hp_classic_concurrent_eps[(i-1)*L+1:L,m]))
  }
  gamma_HP_classic<-hp_classic_concurrent_eps[,m]
  
  # Compute difference of classic HP and acausal target shifted by delta 
  filter_diff_long<-HP_long-gamma_long
  ts.plot(cbind(HP_long,gamma_long))
  # MSE: SSA vs acausal target  
  true_mse_HP_classic_ref_target<-as.double(filter_diff_long%*%I_tilde_long%*%filter_diff_long)
  sample_mse_HP_classic_ref_target<-mean(na.exclude((z_classic_HP-zdelta))^2)
  # 1.2 SSA  
  filter_diff_long<-NULL
  # Compute one-sided SSA with zeroes to the left tail  
  filter_diff_long<-c(filter_diff_long,c(rep(0,L-1),bk_mat[1:L,m]))
  if (n>1)
  { 
    for (i in 2:n)
      filter_diff_long<-c(filter_diff_long,c(rep(0,L-1),bk_mat[(i-1)*L+1:L,m]))
  }
  # Compute difference of SSA and shifted acausal target: both as applied to WN  
  filter_diff_long<-filter_diff_long-gamma_long
  #  filter_diff_long<-filter_diff_long-gamma_target_long[,m]
  # MSE: SSA vs acausal target  
  true_mse_SSA_ref_target<-as.double(filter_diff_long%*%I_tilde_long%*%filter_diff_long)
  sample_mse_SSA_ref_target<-mean(na.exclude((y-zdelta))^2)
  # 1.3 MSE
  # Causal MSE vs. acausal target  
  filter_diff_long<-NULL
  # Compute one-sided SSA with zeroes to the left tail  
  filter_diff_long<-c(filter_diff_long,c(rep(0,L-1),gammak_mse[1:L,m]))
  if (n>1)
  { 
    for (i in 2:n)
      filter_diff_long<-c(filter_diff_long,c(rep(0,L-1),gammak_mse[(i-1)*L+1:L,m]))
  }
  # Compute difference of SSA and acausal target  
  filter_diff_long<-filter_diff_long-gamma_long
  true_mse_mse_ref_target<-as.double(filter_diff_long%*%I_tilde_long%*%filter_diff_long)
  sample_mse_mse_ref_target<-mean(na.exclude((z_mse-zdelta))^2)
  
  # 1.4 MSE of SSA when target is one-sided MSE (instead of two-sided acausal filter)  
  true_mse_SSA_ref_mse<-as.double((bk_mat[,n]-gammak_mse[,n])%*%I_tilde%*%(bk_mat[,n]-gammak_mse[,n]))
  sample_mse_SSA_ref_mse<-mean(na.exclude((z_mse-y))^2)
  # 1.5 MSE referenced against MSE (error vanishes)
  true_mse_mse_ref_mse<-0
  sample_mse_mse_ref_mse<-0
  
  perf_mat_true<-c(true_mse_HP_classic_ref_target,true_mse_SSA_ref_target,true_mse_mse_ref_target,true_mse_SSA_ref_mse,true_mse_mse_ref_mse)
  perf_mat_sample<-c(sample_mse_HP_classic_ref_target,sample_mse_SSA_ref_target,sample_mse_mse_ref_target,sample_mse_SSA_ref_mse,sample_mse_mse_ref_mse)
  names(perf_mat_true)<-names(perf_mat_sample)<-c("MSE: HP vs. target","MSE: SSA vs. target","MSE: MSE vs. Target","MSE: SSA vs. MSE","MSE: MSE vs. MSE")
  if (n==1)
  {
    names(perf_mat_true)<-names(perf_mat_sample)<-c("MSE: HP vs. target","MSE: SSA vs. target","MSE: MSE vs. Target","MSE: SSA vs. MSE","MSE: MSE vs. MSE")
  } else
  {
    names(perf_mat_true)<-names(perf_mat_sample)<-c("MSE: HP vs. target","MSE: M-SSA vs. target","MSE: MSE vs. Target","MSE: SSA vs. MSE","MSE: MSE vs. MSE")
  }
  
#ts.plot(cbind(gamma_HP_classic,gammak_mse[,m]),lty=1:2)  
# 2.1 Classic HP  
  sample_crit_HP_classic_ref_target<-cor(na.exclude(cbind(zdelta,z_classic_HP)))[2]
  true_crit_HP_classic_ref_target<-as.double(gamma_HP_classic%*%I_tilde%*%gammak_mse[,m]/sqrt(as.double(gamma_HP_classic%*%I_tilde%*%gamma_HP_classic)*var_target[m,m]))
# 2.2 SSA as referenced against acausal target  
  sample_crit_SSA_ref_target<-cor(na.exclude(cbind(zdelta,y)))[2]
#  true_crit_SSA_ref_target is the same as SSA_obj$crit_rhoy_target[m]
  true_crit_SSA_ref_target<-as.double(bk_mat[,m]%*%I_tilde%*%gammak_mse[,m]/sqrt(as.double(bk_mat[,m]%*%I_tilde%*%bk_mat[,m])*var_target[m,m]))
# 2.3 MSE as referenced against acausal target  
  sample_crit_mse_ref_target<-cor(na.exclude(cbind(zdelta,z_mse)))[2]
  true_crit_mse_ref_target<-as.double(gammak_mse[,m]%*%I_tilde%*%gammak_mse[,m]/sqrt(as.double(gammak_mse[,m]%*%I_tilde%*%gammak_mse[,m])*var_target[m,m]))
# 2.4 SSA as referenced against MSE  
  sample_crit_SSA_ref_mse<-cor(na.exclude(cbind(z_mse,y)))[1,2]
#  true_crit_SSA_ref_mse is the same as SSA_obj$crit_rhoy[m]
  true_crit_SSA_ref_mse<-as.double(bk_mat[,m]%*%I_tilde%*%gammak_mse[,m]/sqrt(as.double(bk_mat[,m]%*%I_tilde%*%bk_mat[,m])*gammak_mse[,m]%*%I_tilde%*%gammak_mse[,m]))
# 2.5 MSE as referenced against itself
  true_crit_mse_ref_mse<-1
  sample_crit_mse_ref_mse<-1
  
  perf_mat_true<-c(perf_mat_true,true_crit_HP_classic_ref_target,true_crit_SSA_ref_target,true_crit_mse_ref_target,true_crit_SSA_ref_mse,true_crit_mse_ref_mse)
  perf_mat_sample<-c(perf_mat_sample,sample_crit_HP_classic_ref_target,sample_crit_SSA_ref_target,sample_crit_mse_ref_target,sample_crit_SSA_ref_mse,sample_crit_mse_ref_mse)
  if (n==1)
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-4):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-4):length(perf_mat_sample)]<-c("Cor. HP vs. target","Cor. SSA vs. target","Cor. MSE vs. target","Cor. SSA vs. MSE","Cor. MSE vs. MSE")
  } else
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-4):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-4):length(perf_mat_sample)]<-c("Cor. HP vs. target","Cor. SSA vs. target","Cor. MSE vs. target","Cor. SSA vs. MSE","Cor. MSE vs. MSE")
  } 
  
  
# 3. Empirical and theoretical HT
# 3.1 HP classic
# Without centering: x_mat is assumed to be standardized
  mplot<-(na.exclude(z_classic_HP))
  sample_ht_HP_classic<-length(mplot)/length(which(mplot[2:length(mplot)]*mplot[1:(length(mplot)-1)]<0))
# First compute lag-one ACF and then HT 
  acf1_HP_classic<-gamma_HP_classic%*%M_tilde%*%gamma_HP_classic/gamma_HP_classic%*%I_tilde%*%gamma_HP_classic
  true_ht_HP_classic<-compute_holding_time_from_rho_func(acf1_HP_classic)$ht
# 3.2 SSA  
# Centering not necessary since data is standardized  
  mplot<-(na.exclude(y))
  sample_ht_SSA<-length(mplot)/length(which(mplot[2:length(mplot)]*mplot[1:(length(mplot)-1)]<0))
# First compute lag-one ACF and then HT 
  acf1_mssa<-bk_mat[,m]%*%M_tilde%*%bk_mat[,m]/bk_mat[,m]%*%I_tilde%*%bk_mat[,m]
  ht_mssa<-compute_holding_time_from_rho_func(acf1_mssa)$ht
# Is the same as ht_ssa_vec[m] if optimization converged i.e. difference below should be negligible (they become smaller when increasing split_grid)
  ht_mssa-ht_ssa_vec[m]
  true_ht_SSA<-ht_mssa
# 3.3 MSE
# Sample HT
  mplot<-(na.exclude(z_mse))
  sample_ht_MSE<-length(mplot)/length(which(mplot[2:length(mplot)]*mplot[1:(length(mplot)-1)]<0))
  acf1_mse<-t(gammak_mse[,m])%*%M_tilde%*%gammak_mse[,m]/t(gammak_mse[,m])%*%I_tilde%*%gammak_mse[,m]
  true_ht_MSE<-pi/acos(acf1_mse)

  perf_mat_true<-c(perf_mat_true,true_ht_HP_classic,true_ht_SSA,true_ht_MSE)
  perf_mat_sample<-c(perf_mat_sample,sample_ht_HP_classic,sample_ht_SSA,sample_ht_MSE)
  if (n==1)
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-c("HT HP","HT SSA","HT MSE")
  } else
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-c("HT HP","HT M-SSA","HT MSE")
  }
  

# 4. Sign accuracies SA (probabilities of same sign, see proof of proposition 1 in paper for 'true' criterion with arcsin transformation) 
# 4.1 HP classic
# We remove NA's  
  mplot<-na.exclude(cbind(zdelta,z_classic_HP))
  sample_SA_crit_HP_classic_ref_target<-length(which(sign(mplot[,1])==sign(mplot[,2])))/nrow(mplot)
  cor_HP_classic_target<-gammak_mse[,m]%*%I_tilde%*%gamma_HP_classic/sqrt(gamma_HP_classic%*%I_tilde%*%gamma_HP_classic*var_target[m,m])
  true_SA_crit_HP_classic_ref_target<-0.5+asin(cor_HP_classic_target)/pi
# 4.2 SSA  
  mplot<-na.exclude(cbind(zdelta,y))
  ts.plot(mplot)
  sample_SA_crit_SSA_ref_target<-length(which(sign(mplot[,1])==sign(mplot[,2])))/nrow(mplot)
  cor_MSSA_target<-gammak_mse[,m]%*%I_tilde%*%bk_mat[,m]/sqrt(bk_mat[,m]%*%I_tilde%*%bk_mat[,m]*var_target[m,m])
# Is the same as SSA_obj$crit_rhoy_target[m] i.e. difference below vanishes 
  cor_MSSA_target-SSA_obj$crit_rhoy_target[m]
  true_SA_crit_SSA_ref_target<-0.5+asin(cor_MSSA_target)/pi
  mplot<-na.exclude(cbind(zdelta,z_mse))
  sample_SA_crit_mse_ref_target<-length(which(sign(mplot[,1])==sign(mplot[,2])))/nrow(mplot)
# 4.3 MSE  
# Correlation of causal MSE with acausal target  
  cor_mse_target<-gammak_mse[,m]%*%I_tilde%*%gammak_mse[,m]/sqrt(gammak_mse[,m]%*%I_tilde%*%gammak_mse[,m]*var_target[m,m])
  true_SA_crit_mse_ref_target<-0.5+asin(cor_mse_target)/pi
  
  perf_mat_true<-c(perf_mat_true,true_SA_crit_HP_classic_ref_target,true_SA_crit_SSA_ref_target,true_SA_crit_mse_ref_target)
  perf_mat_sample<-c(perf_mat_sample,sample_SA_crit_HP_classic_ref_target,sample_SA_crit_SSA_ref_target,sample_SA_crit_mse_ref_target)
  if (n==1)
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-c("SA: HP vs. target","SA: SSA vs. target","SA: MSE vs. target")
  } else
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-c("SA: HP vs. target","SA: SSA vs. target","SA: MSE vs. target")
  }
  
  return(list(perf_mat_true=perf_mat_true,perf_mat_sample=perf_mat_sample,zdelta=zdelta,y=y,z_mse=z_mse,z_classic_HP=z_classic_HP))
}







# Perform simulations for Forecasting application: check convergence of sample performances to expected (true) expressions
sample_series_performances_func<-function(A,Sigma,len,bk_mat,bk_x_mat,gammak_mse,L,setseed)
{
  n<-dim(Sigma)[1]
  set.seed(setseed)
  eps1iid<-rnorm(len)
  eps2iid<-rnorm(len)
  
  eps_mat<-matrix(ncol=dim(Sigma)[1],nrow=len)
  
  # Generate eps with cross-correlation corresponding to Sigma
  eigen_obj<-eigen(Sigma)
  # Square-root of diagonal
  D<-diag(sqrt(eigen_obj$values))
  U<-eigen_obj$vectors
  Sigma_sqrt<-(U)%*%D%*%t(U)
  # Check_ should vanish
  t(Sigma_sqrt)%*%Sigma_sqrt-Sigma
  # Generate eps_mat
  eps_mat<-t(Sigma_sqrt%*%rbind(eps1iid,eps2iid))
  # Check: empirical cov should match Sigma
  cov(eps_mat)
  Sigma
  # 1.2 VAR  
  x_mat<-eps_mat
  for (i in 2:len)
  {
    x_mat[i,]<-as.double(A%*%(x_mat[i-1,]))+eps_mat[i,] 
  }
  #---------------  
  # 2. Generate yt (M-SSA), zt (target) and z_mse (MSE predictor) from eps_mat
  perf_mat_SSA<-perf_mat_MSE<-NULL
  z_mse_mat<-zdelta_mat<-y_mat<-yx_mat<-NULL
  for (m in 1:n)#m<-1
  {
    # Plot coefficients  
    mplot<-cbind(bk_mat[1:L,m],bk_mat[(L+1):(2*L),m])
    colo<-c("blue","red","green")
    ts.plot(mplot,col=colo)
    # 2.1 yt: we can generate M-SSA based on WN or VAR
    # 2.1.1 First case A above: Specify multivariate filter for m-th series: we select bk_mat as applied to epsilont   
    bk<-NULL
    for (j in 1:dim(Sigma)[1])
      bk<-cbind(bk,bk_mat[((j-1)*L+1):(j*L),m])
    y<-rep(NA,len)
    # Filter WN    
    for (j in L:len)
      y[j]<-sum(apply(bk*eps_mat[j:(j-L+1),],2,sum))
    y_mat<-cbind(y_mat,y)
    # 2.1.2 Second case B above: bk_x_mat as applied to xt differs from bk_mat as applied to epsilont    
    if (!is.null(xi))
    {
      bkx<-NULL
      for (j in 1:dim(Sigma)[1])
        bkx<-cbind(bkx,bk_x_mat[((j-1)*L+1):(j*L),m])
      yx<-rep(NA,len)
      # Filter VAR
      for (j in L:len)
        yx[j]<-sum(apply(bkx*x_mat[j:(j-L+1),],2,sum))
      yx_mat<-cbind(yx_mat,yx)
      # Check: both M-SSA predictors should be nearly the same (up to finite MA-inversion error)    
      max(abs(yx-y),na.rm=T)/sd(yx,na.rm=T) 
    } else
    {
      yx<-y
      yx_mat<-y_mat
    }
    # 2.2 Target: zt 
    z<-x_mat[,m]
    # Shift z by delta    
    if (delta>0)
    {  
      zdelta<-c(z[(delta+1):len],rep(0,delta))
    } else
    {
      if (delta<0)
      {
        zdelta<-c(rep(0,delta),z[1:(len-abs(delta))])
      } else
      {
        zdelta<-z
      }
    }
    par(mfrow=c(2,1))
    ts.plot(gammak)
    ts.plot(bk)
    
    ts.plot(scale(cbind(zdelta,y),scale=T)[500:600,],lty=1:2)
    zdelta_mat<-cbind(zdelta_mat,zdelta)
    # 2.3 MSE (this is used in MSSA_func internally)   
    gammak_n<-cbind(gammak_mse[m,1:L],gammak_mse[m,L+1:L])
    z_mse<-rep(NA,len)
    for (j in L:len)
      z_mse[j]<-sum(apply(gammak_n*eps_mat[j:(j-L+1),],2,sum))
    z_mse_mat<-cbind(z_mse_mat,z_mse)    
    #-------------------------
    # Performances: empirical and theoretical criterion values and lag-one acfs: one use either y (based on MA inversion) or yx
    perf_mat_SSA<-rbind(perf_mat_SSA,c(cor(na.exclude(cbind(z_mse[1:len],yx[1:len])))[1,2], SSA_obj$crit_rhoyz[m],length(na.exclude(yx))/length(which(yx[(L+1):len]*yx[L:(len-1)]<0)), ht_vec[m]))
    
    perf_mat_MSE<-rbind(perf_mat_MSE,c(length(na.exclude(z_mse))/length(which(z_mse[(L+1):len]*z_mse[L:(len-1)]<0))))
    
    
    # Criterion value is with respect to MSE-target: 
    #   Optimized solutions are identical but criterion measures performances against MSE i.e. against 'benchmark'    
  } 
  colnames(perf_mat_SSA)<-c("Sample crit.","True crit.","Sample ht SSA","True ht SSA")
  perf_mat_SSA
  
  # Keep sample for plot of filter outputs  
  y_mat<-y_mat[1:min(1000,len),]
  zdelta_mat<-zdelta_mat[1:min(1000,len),]
  z_mse_mat<-z_mse_mat[1:min(1000,len),]
  
  #-------------------  
  # Additional checks
  # 1. Holding-times (lag-one acfs)  
  M_obj<-M_func(L,Sigma)
  
  M_tilde<-M_obj$M_tilde
  I_tilde<-M_obj$I_tilde
  rho_mse_1<-gammak_mse[1,]%*%M_tilde%*%gammak_mse[1,]/gammak_mse[1,]%*%I_tilde%*%gammak_mse[1,]
  rho_ssa_1<-bk_mat[,1]%*%M_tilde%*%bk_mat[,1]/bk_mat[,1]%*%I_tilde%*%bk_mat[,1]
  rho_mse_2<-gammak_mse[2,]%*%M_tilde%*%gammak_mse[2,]/gammak_mse[2,]%*%I_tilde%*%gammak_mse[2,]
  rho_ssa_2<-bk_mat[,2]%*%M_tilde%*%bk_mat[,2]/bk_mat[,2]%*%I_tilde%*%bk_mat[,2]
  
  # Check: best approximation on grid should be close to effective holding-time constraints  
  compute_holding_time_from_rho_func(rho_ssa_1)$ht
  compute_holding_time_from_rho_func(rho_ssa_2)$ht
  ht_vec
  # 2. Criteria: MSE is trivially one since correlation of MSE with itself is one (our target is MSE which leads to the same solution as using z_{t+\delta})
  #   The criteria computed here correspond to the values in perf_mat_SSA above  
  crit_mse_1<-gammak_mse[1,]%*%I_tilde%*%gammak_mse[1,]/gammak_mse[1,]%*%I_tilde%*%gammak_mse[1,]
  crit_ssa_1<-gammak_mse[1,]%*%I_tilde%*%bk_mat[,1]/(sqrt(bk_mat[,1]%*%I_tilde%*%bk_mat[,1])*sqrt(gammak_mse[1,]%*%I_tilde%*%gammak_mse[1,]))
  crit_mse_2<-gammak_mse[2,]%*%I_tilde%*%gammak_mse[2,]/gammak_mse[2,]%*%I_tilde%*%gammak_mse[2,]
  crit_ssa_2<-gammak_mse[2,]%*%I_tilde%*%bk_mat[,2]/(sqrt(bk_mat[,2]%*%I_tilde%*%bk_mat[,2])*sqrt(gammak_mse[2,]%*%I_tilde%*%gammak_mse[2,]))

  return(list(perf_mat_SSA=perf_mat_SSA,y_mat=y_mat,zdelta_mat=zdelta_mat,z_mse_mat=z_mse_mat,perf_mat_MSE=perf_mat_MSE,z_mse_mat=z_mse_mat))
}






# Perform simulations for Smoothing application: check convergence of sample performances to expected (true) expressions
sample_series_performances_smooth_func<-function(A,Sigma,len,bk_mat,bk_x_mat,gammak_mse,L,setseed,SSA_obj)
{
  
  eps1iid<-rnorm(len)
  eps2iid<-rnorm(len)
  eps3iid<-rnorm(len)
  
  eps_mat<-matrix(ncol=dim(Sigma)[1],nrow=len)
  
  # Generate eps with cross-correlation corresponding to Sigma
  eigen_obj<-eigen(Sigma)
  # Square-root of diagonal
  D<-diag(sqrt(eigen_obj$values))
  U<-eigen_obj$vectors
  Sigma_sqrt<-(U)%*%D%*%t(U)
  # Check_ should vanish
  t(Sigma_sqrt)%*%Sigma_sqrt-Sigma
  # Generate eps_mat
  eps_mat<-t(Sigma_sqrt%*%rbind(eps1iid,eps2iid,eps3iid))
  # Check: empirical cov should match Sigma
  cov(eps_mat)
  Sigma
  
  # Generate x either of two ways: 
  # 1. Wold decomposition
  if (!is.null(xi))
  {  
    x<-rep(0,n)
    x_mat<-matrix(nrow=len,ncol=n)
    for (m in 1:n)#j<-1
    {
      x<-rep(NA,len)
      xi_mat<-cbind(xi[m,1:L],xi[m,L+1:L],xi[m,2*L+1:L])
      for (i in L:len)#i<-L
      {
        x[i]<-sum(apply(xi_mat*eps_mat[i:(i-L+1),],2,sum))
      }
      x_mat[,m]<-x
    }
  } else
  {
    x_mat<-eps_mat
  }
  # Check
  if (F)
  {  
    # 2. VAR(1) equation: note that Xi_0=xi[,(0:(n-1))*L+1] is an identity (otherwise one could account for identity with Sigma)
    #   Therefore we use eps_mat[i,] in the VAR-equation below i.e. e_{it} appears only in the equation of x_{it}   xx_mat<-NULL
    x<-x_mat[L,]
    xx_mat<-matrix(nrow=L,ncol=n)
    for (i in (L+1):len)#i<-L+1
    {
      xx_mat<-rbind(xx_mat,matrix(A%*%x+eps_mat[i,]+B%*%eps_mat[i-1,],nrow=1))
      x<-xx_mat[nrow(xx_mat),]
      x-x_mat[i,]
    }
    
    # Check: both VAR-computations are the same up to negligible errors dur to finite-length of Wold-decomposition   
    ts.plot(cbind(xx_mat[,1],x_mat[,1])[(len-100):len,],lty=1:2)
  }
  
  # 3. Generate yt, zt and z_mse from eps_mat
  perf_mat_sample<-perf_mat_true<-NULL
  y_mat<-zdelta_mat<-z_mse_mat<-NULL
  
  for (m in 1:dim(Sigma)[1])#m<-2
  {
    
    # Generate yt in either of two equivalent ways:        
    # 3.1. convolution of SSA with white noise solution bk_mat (solution of theorem)   
    bk<-NULL
    for (j in 1:dim(Sigma)[1])#j<-1
      bk<-cbind(bk,bk_mat[((j-1)*L+1):(j*L),m])
    y<-rep(NA,len)
    for (j in L:len)#j<-L
      y[j]<-sum(apply(bk*eps_mat[j:(j-L+1),],2,sum))
    y_mat<-cbind(y_mat,y[1: min(len,1000)])
    # Check:    
    if (F)
    {
      # 3.2 use deconvoluted solution bk_x_mat applied to xt (obtained from theorem after inversion of convolution) 
      bk<-NULL
      for (j in 1:dim(Sigma)[1])
        bk<-cbind(bk,bk_x_mat[((j-1)*L+1):(j*L),m])
      yh<-rep(NA,len)
      for (j in L:len)
        yh[j]<-sum(apply(bk*(x_mat[j:(j-L+1),]),2,sum))
      
      ts.plot(scale(cbind(y,yh)),lty=1:2)
    }
    # 3.3 Target: MSE-estimate of zt    
    gammak<-NULL
    for (j in 1:dim(Sigma)[1])#j<-1
      # in this example gammak_mse is identical with Wold-decomposition (delta=0)
      #  z_mse is just the original data   
      gammak<-cbind(gammak,gammak_mse[m,((j-1)*L+1):(j*L)])
    z_mse<-rep(NA,len)
    for (j in L:len)
      z_mse[j]<-sum(apply(gammak*eps_mat[j:(j-L+1),],2,sum))
    z_mse_mat<-cbind(z_mse_mat,z_mse[1: min(len,1000)])
    # Check: if Gamma is the identity then MSE should match Wold decomposition xi  
    max(abs(gammak_mse-xi))
    # Check: MSE is just X (because delta=0: nowcast)  
    max(abs(na.exclude(z_mse-x_mat[,m])))
    # 3.4 Target: zt+delta    
    gammak<-NULL
    for (j in 1:dim(Sigma)[1])#j<-1
      #  gamma_target is original Signal extraction filter as applied to xt     
      gammak<-cbind(gammak,gamma_target[m,((j-1)*L+1):(j*L)])
    z<-rep(NA,len)
    # Truncated bi-infinite sum (acausal filter) 
    # gamma_target is applied to xt (in contrast to gammak_mse above which is convolution of mse with wold-decomposition)    
    for (j in L:(len-L))
      z[j]<-sum(apply(gammak*x_mat[j:(j-L+1),],2,sum))+sum(apply(gammak[-1,]*x_mat[(j+1):(j+L-1),],2,sum))
    # Shift z by delta1: delta1 is zero (nowcast)    
    if (delta1>0)
    {  
      zdelta<-c(z[(delta1+1):len],rep(0,delta1))
    } else
    {
      if (delta1<0)
      {
        zdelta<-c(rep(0,abs(delta1)),z[1:(len-abs(delta1))])
      } else
      {
        zdelta<-z
      }
    }
    zdelta_mat<-cbind(zdelta_mat,zdelta[1: min(len,1000)])
    #----------------
    # Series of checks    
    sample_var_target<-var(na.exclude(zdelta))
    sample_var_target
    # Compute theoretical variance of z_mse
    variance_mse<-t(gammak_mse[m,])%*%I_tilde%*%gammak_mse[m,]
    # Check sample and theoretical values    
    variance_mse
    var(na.exclude(z_mse))
    # Criterion value with MSE target    
    gammak_mse[1,]%*%I_tilde%*%bk_mat[,1]/(sqrt(bk_mat[,1]%*%I_tilde%*%bk_mat[,1])*sqrt(gammak_mse[1,]%*%I_tilde%*%gammak_mse[1,]))
    # Criterion value with target: covariance in nominator is the same as with MSE but variance of target changes in denominator: use sample variance of target zdelta    
    gammak_mse[1,]%*%I_tilde%*%bk_mat[,1]/(sqrt(bk_mat[,1]%*%I_tilde%*%bk_mat[,1])*sqrt(sample_var_target))
    # Check: This should fit sample criterion with respect to effective target zdelta (instead of MSE as above)    
    cor(na.exclude(cbind(zdelta,y)))[2]
    
    #------------
    # Performances: empirical and theoretical criterion values and lag-one acfs
    sample_crit_SSA_ref_target<-cor(na.exclude(cbind(zdelta,y)))[2]
    sample_crit_SSA_ref_mse<-cor(na.exclude(cbind(z_mse,y)))[1,2]
    true_crit_SSA_ref_mse<-SSA_obj$crit_rhoyz[m]
    # Since the criterion is not calculated with respect to target in our function we use a trick:
    #   One can replace variance of MSE by variance of target in denominator of correlation
    true_crit_SSA_ref_target<-true_crit_SSA_ref_mse*sqrt(variance_mse)/sqrt(sample_var_target)
    # Latest update: criterion is calculated with respect to effective target     
    true_crit_SSA_ref_target<-SSA_obj$crit_rhoy_target[m]
    sample_ht_SSA<-length(na.exclude(y))/length(which(y[(L+1):len]*y[L:(len-1)]<0))
    true_ht_SSA<-ht_vec[m]
    sample_ht_MSE<-length(na.exclude(z_mse))/length(which(z_mse[(L+1):len]*z_mse[L:(len-1)]<0))
    acf1_mse<-t(gammak_mse[m,])%*%M_tilde%*%gammak_mse[m,]/t(gammak_mse[m,])%*%I_tilde%*%gammak_mse[m,]
    true_ht_MSE<-pi/acos(acf1_mse)
    # Here we compute sign accuracies i.e. probabilities of same sign (see proof of proposition 1 in paper for 'true' criterion with arcsin transformation)  
    sample_SA_crit_SSA_ref_target<-length(which(sign(zdelta)==sign(y)))/(nrow(na.exclude(cbind(y,zdelta))))
    true_SA_crit_SSA_ref_target<-0.5+asin(true_crit_SSA_ref_target)/pi
    
    perf_mat_sample<-rbind(perf_mat_sample,c(sample_crit_SSA_ref_target,sample_crit_SSA_ref_mse,sample_SA_crit_SSA_ref_target,sample_ht_SSA,sample_ht_MSE))
    perf_mat_true<-rbind(perf_mat_true,c(true_crit_SSA_ref_target,true_crit_SSA_ref_mse,true_SA_crit_SSA_ref_target,true_ht_SSA,true_ht_MSE))
    # Criterion value is with respect to MSE-target: 
    #   Optimized solutions are identical but criterion measures performances against MSE i.e. against 'benchmark'    
  } 
  colnames(perf_mat_sample)<-colnames(perf_mat_true)<-c("Cor. with target","Cor. with MSE","Sign accuracy","ht","ht MSE")
  rownames(perf_mat_sample)<-rownames(perf_mat_true)<-paste("Series ",1:n,sep="")
  
  perf_mat_sample
  perf_mat_true
  
  
  
  
  perf_mat<-rbind(perf_mat_sample,perf_mat_true)
  cor(na.exclude(x_mat))
  # Need only first 1000 data points of x_mat for plots: the other data files have been already shortened above 
  x_mat<-x_mat[1:1000,]
  return(list(perf_mat_sample=perf_mat_sample,perf_mat_true=perf_mat_true,bk_mat=bk_mat,gammak_mse=gammak_mse,y_mat=y_mat,zdelta_mat=zdelta_mat,z_mse_mat=z_mse_mat,x_mat=x_mat))
}












#########################################################################################################
# Old code
# Outputs: filtered series and expected as well as sample performances
filter_perf_func_old<-function(hp_classic_concurrent,hp_classic_concurrent_eps,gammak_mse,gammak_x_mse,gamma_target,gamma_target_long,bk_x_mat,bk_mat,x_mat,m,L_short,L,M_tilde,I_tilde,var_target,ht_ssa_vec,Sigma,epsilon_mat=NULL)
{  
  #x_mat<-x_mat[,m]  hp_classic_concurrent<-hp_classic_concurrent_x[,m]  hp_classic_concurrent_eps<-hp_classic_concurrent_eps[,m]
  # gammak_mse<-gammaeps_mat[,m]   gammak_x_mse<-gammax_mat[,m]  gamma_target<-gamma_target_mat[,m]  
  # gamma_target_long<-gamma_target_long_mat[,m]   bk_x_mat<-bx_mat[,m]  bk_mat<-beps_mat[,m] var_target<-var_target_mat[m]
  
  # Sanity checks: works for uni- and multivariate designs
  if (!is.matrix(x_mat))
  {
    xnames<-names(x_mat)
    x_mat<-matrix(x_mat,ncol=1)
    rownames(x_mat)<-xnames
  }
  if (is.vector(hp_classic_concurrent))
    hp_classic_concurrent<-matrix(hp_classic_concurrent,ncol=1)
  if (is.vector(hp_classic_concurrent_eps))
    hp_classic_concurrent_eps<-matrix(hp_classic_concurrent_eps,ncol=1)
  if (is.vector(epsilon_mat))
    epsilon_mat<-matrix(epsilon_mat,ncol=1)
  if (is.vector(gammak_mse))
    gammak_mse<-matrix(gammak_mse,ncol=1)
  if (is.vector(gammak_x_mse))
    gammak_x_mse<-matrix(gammak_x_mse,ncol=1)
  if (is.vector(gamma_target))
    gamma_target<-matrix(gamma_target,ncol=1)
  if (is.vector(gamma_target_long))
    gamma_target_long<-matrix(gamma_target_long,ncol=1)
  if (is.vector(bk_x_mat))
    bk_x_mat<-matrix(bk_x_mat,ncol=1)
  if (is.vector(bk_mat))
    bk_mat<-matrix(bk_mat,ncol=1)
  n<-ncol(bk_x_mat)
  len<-nrow(x_mat)
  # Transpose if necessary  
  if (dim(gamma_target)[1]<dim(gamma_target)[2])
    gamma_target<-t(gamma_target)
  if (is.vector(var_target))
    var_target<-matrix(var_target)
  # Univariate design  
  if (n==1)
    m<-1
  # Check that series are centered: otherwise sample HTs are not properly defined
  if (max(apply(x_mat,2,mean)/apply(x_mat,2,sd))>0.001)
  {
    print("Warning: the time series are possibly not centered. Therefore sample HTs are eventually not properly defined")
  }
  
  # 0. Classic concurrent HP: apply filter to m-th column of x_mat: we use hp_classic_concurrent (not hp_classic_concurrent_eps)
  z_classic_HP<-rep(NA,len)  
  for (j in L:len)
    z_classic_HP[j]<-apply(hp_classic_concurrent*(x_mat[j:(j-L+1),m]),2,sum)
  # 1. Generate yt in either of two equivalent ways:        
  # 1.a Apply bk_x_mat to x_mat
  bk<-NULL
  # Extract coefficients applied to m-th series    
  for (j in 1:n)#j<-2
    bk<-cbind(bk,bk_x_mat[((j-1)*L+1):(j*L),m])
  ts.plot(scale(bk))
  y<-rep(NA,len)
  for (j in L:len)
    y[j]<-sum(apply(bk*(x_mat[j:(j-L+1),]),2,sum))
  # 1.b Apply bk_mat to epsilon_mat
  # The M-SSA solution bk_mat is the convolution of bk_x_mat and MA-inversion: it is applied to WN epsilon
  if (!is.null(epsilon_mat))
  {
    bk<-NULL
    # Extract coefficients applied to m-th series    
    for (j in 1:n)#j<-2
      bk<-cbind(bk,bk_mat[((j-1)*L+1):(j*L),m])
    ts.plot(scale(bk))
    y_eps<-rep(NA,len)
    for (j in L:len)
      y_eps[j]<-sum(apply(bk*(epsilon_mat[j:(j-L+1),]),2,sum))
    # y and y_eps should be nearly identical (up to finite MA-inversion error)
  }
  
  # 2. MSE-estimate 
  # 2.a Apply gammak_x_mse to x_mat
  gammak<-NULL
  for (j in 1:n)#j<-2
    # gammak_mse is convolution of MSE with Wold-decomposition: must be applied to epsilont      
    gammak<-cbind(gammak,gammak_x_mse[((j-1)*L+1):(j*L),m])
  z_mse<-rep(NA,len)
  for (j in L:len)
    z_mse[j]<-sum(apply(gammak*x_mat[j:(j-L+1),],2,sum))
  # 2.b Apply gammak_mse to epsilon_mat
  if (!is.null(epsilon_mat))
  {
    bk<-NULL
    for (j in 1:n)#j<-2
      gammak<-cbind(gammakk,gammak_mse[((j-1)*L+1):(j*L),m])
    ts.plot(scale(gammak))
    z_mse_eps<-rep(NA,len)
    for (j in L:len)
      z_mse_eps[j]<-sum(apply(gammak*(epsilon_mat[j:(j-L+1),]),2,sum))
    # z_mse and z_mse_eps should be nearly identical (up to finite MA-inversion error)
  }
  
  # 3.Target zt+delta: based on xt 
  gammak<-NULL
  for (j in 1:n)#j<-1
    #  gamma_target is original Signal extraction filter as applied to xt : L_short allows for longer sample, closer to sample boundary
    gammak<-cbind(gammak,(gamma_target)[((j-1)*L+1):((j-1)*L+L_short),m])
  z<-rep(NA,len)
  # Truncated two-sided filter
  for (j in L:(len-L_short))
    z[j]<-sum(apply(gammak*x_mat[j:(j-L_short+1),],2,sum))+sum(apply(matrix(gammak[-1,]*x_mat[(j+1):(j+L_short-1),]),2,sum))
  # Shift z by delta    
  if (delta>0)
  {  
    zdelta<-c(z[(delta+1):len],rep(0,delta))
  } else
  {
    if (delta<0)
    {
      zdelta<-c(rep(0,delta),z[1:(len-abs(delta))])
    } else
    {
      zdelta<-z
    }
  }
  names(zdelta)<-names(y)<-names(z_mse)<-rownames(x_mat)
  
  ts.plot(gamma_target_long)
  hp_classic_concurrent<-as.vector(hp_classic_concurrent)
  #-------------------------------------------------------
  # Performances: MSE, target correlations HT and SA for (M-)SSA and MSE (expected and sample)
  # We need filters applied to WN (not xt) to compute theoretical performances (all formula rely on convolution of filters with MA-inversion) 
  # 1. MSE: empirical and expected MSE
  # For the MSE between M-SSA and acausal target we need
  # a. gamma_target_long : acausal target convolved with MA inversion
  # b. bk_mat: bk convolved with MA-inversion
  #   But we must extended the filter with zeroes on left tail (corresponding to future epsilons of acausal target)  
  # c. Compute the differences between both filters and compute variance
  # d. For that purpose we need I_tilde_long computed for long filters: they all have length 2*L-1
  M_obj_long<-M_func(2*L-1,Sigma)
  I_tilde_long<-M_obj_long$I_tilde
  dim(I_tilde_long)
  # 1.1 Classic HP concurrent  
  HP_long<-gamma_HP_classic<-NULL
  # Compute one-sided univariate HP-concurrent with zeroes to the left tail 
  # Univariate filter is applied to m-th series only (all other coefficients vanish)  
  if (F)
  {
    for (i in 1:n)
    {
      if (i==m)
      {
        # HP_long is used for determining the MSE with respect to acausal target: it has zeroes on the left tail     
        # This is based on hp_classic_concurrent_eps which is the convolution of hp_classic_concurrent and xi: it is applied to epsilon     
        HP_long<-c(HP_long,c(rep(0,L-1),hp_classic_concurrent_eps[,m]))
        # gamma_HP_classic is used for determining correlations and ACFs: it has the `normal' length
        # This is based on hp_classic_concurrent_eps which is the convolution of hp_classic_concurrent and xi: it is applied to epsilon     
        gamma_HP_classic<-c(gamma_HP_classic,hp_classic_concurrent_eps[,m])
        
      } else
      {
        HP_long<-c(HP_long,rep(0,2*L-1))
        gamma_HP_classic<-c(gamma_HP_classic,rep(0,L))
      }
    }
  }
  for (i in 1:n)#i<-1
  {
    HP_long<-c(HP_long,c(rep(0,L-1),hp_classic_concurrent_eps[(i-1)*L+1:L,m]))
  }
  gamma_HP_classic<-hp_classic_concurrent_eps[,m]
  
  # Compute difference of classic HP and acausal target 
  filter_diff_long<-HP_long-gamma_target_long[,m]
  # MSE: SSA vs acausal target  
  true_mse_HP_classic_ref_target<-as.double(filter_diff_long%*%I_tilde_long%*%filter_diff_long)
  sample_mse_HP_classic_ref_target<-mean(na.exclude((z_classic_HP-zdelta))^2)
  # 1.2 SSA  
  filter_diff_long<-NULL
  # Compute one-sided SSA with zeroes to the left tail  
  filter_diff_long<-c(filter_diff_long,c(rep(0,L-1),bk_mat[1:L,m]))
  if (n>1)
  { 
    for (i in 2:n)
      filter_diff_long<-c(filter_diff_long,c(rep(0,L-1),bk_mat[(i-1)*L+1:L,m]))
  }
  # Compute difference of SSA and acausal target  
  filter_diff_long<-filter_diff_long-gamma_target_long[,m]
  # MSE: SSA vs acausal target  
  true_mse_SSA_ref_target<-as.double(filter_diff_long%*%I_tilde_long%*%filter_diff_long)
  sample_mse_SSA_ref_target<-mean(na.exclude((y-zdelta))^2)
  # 1.3 MSE
  # Causal MSE vs. acausal target  
  filter_diff_long<-NULL
  # Compute one-sided SSA with zeroes to the left tail  
  filter_diff_long<-c(filter_diff_long,c(rep(0,L-1),gammak_mse[1:L,m]))
  if (n>1)
  { 
    for (i in 2:n)
      filter_diff_long<-c(filter_diff_long,c(rep(0,L-1),gammak_mse[(i-1)*L+1:L,m]))
  }
  # Compute difference of SSA and acausal target  
  filter_diff_long<-filter_diff_long-gamma_target_long[,m]
  true_mse_mse_ref_target<-as.double(filter_diff_long%*%I_tilde_long%*%filter_diff_long)
  sample_mse_mse_ref_target<-mean(na.exclude((z_mse-zdelta))^2)
  # 1.4 MSE of SSA when target is one-sided MSE (instead of two-sided acausal filter)  
  true_mse_SSA_ref_mse<-as.double((bk_mat[,n]-gammak_mse[,n])%*%I_tilde%*%(bk_mat[,n]-gammak_mse[,n]))
  sample_mse_SSA_ref_mse<-mean(na.exclude((z_mse-y))^2)
  # 1.5 MSE referenced against MSE (error vanishes)
  true_mse_mse_ref_mse<-0
  sample_mse_mse_ref_mse<-0
  
  perf_mat_true<-c(true_mse_HP_classic_ref_target,true_mse_SSA_ref_target,true_mse_mse_ref_target,true_mse_SSA_ref_mse,true_mse_mse_ref_mse)
  perf_mat_sample<-c(sample_mse_HP_classic_ref_target,sample_mse_SSA_ref_target,sample_mse_mse_ref_target,sample_mse_SSA_ref_mse,sample_mse_mse_ref_mse)
  names(perf_mat_true)<-names(perf_mat_sample)<-c("MSE: HP vs. target","MSE: SSA vs. target","MSE: MSE vs. Target","MSE: SSA vs. MSE","MSE: MSE vs. MSE")
  if (n==1)
  {
    names(perf_mat_true)<-names(perf_mat_sample)<-c("MSE: HP vs. target","MSE: SSA vs. target","MSE: MSE vs. Target","MSE: SSA vs. MSE","MSE: MSE vs. MSE")
  } else
  {
    names(perf_mat_true)<-names(perf_mat_sample)<-c("MSE: HP vs. target","MSE: M-SSA vs. target","MSE: MSE vs. Target","MSE: SSA vs. MSE","MSE: MSE vs. MSE")
  }
  
  #ts.plot(gamma_HP_classic)  ts.plot(gammak_mse[,m])
  # 2. Target correlations
  # 2.1 Classic HP  
  sample_crit_HP_classic_ref_target<-cor(na.exclude(cbind(zdelta,z_classic_HP)))[2]
  true_crit_HP_classic_ref_target<-as.double(gamma_HP_classic%*%I_tilde%*%gammak_mse[,m]/sqrt(as.double(gamma_HP_classic%*%I_tilde%*%gamma_HP_classic)*var_target[m,m]))
  # 2.2 SSA  
  sample_crit_SSA_ref_target<-cor(na.exclude(cbind(zdelta,y)))[2]
  #  true_crit_SSA_ref_target is the same as SSA_obj$crit_rhoy_target[m]
  true_crit_SSA_ref_target<-as.double(bk_mat[,m]%*%I_tilde%*%gammak_mse[,m]/sqrt(as.double(bk_mat[,m]%*%I_tilde%*%bk_mat[,m])*var_target[m,m]))
  # 2.3 MSE  
  sample_crit_SSA_ref_mse<-cor(na.exclude(cbind(z_mse,y)))[1,2]
  #  true_crit_SSA_ref_mse is the same as SSA_obj$crit_rhoy[m]
  true_crit_SSA_ref_mse<-as.double(bk_mat[,m]%*%I_tilde%*%gammak_mse[,m]/sqrt(as.double(bk_mat[,m]%*%I_tilde%*%bk_mat[,m])*gammak_mse[,m]%*%I_tilde%*%gammak_mse[,m]))
  
  perf_mat_true<-c(perf_mat_true,true_crit_HP_classic_ref_target,true_crit_SSA_ref_target,true_crit_SSA_ref_mse)
  perf_mat_sample<-c(perf_mat_sample,sample_crit_HP_classic_ref_target,sample_crit_SSA_ref_target,sample_crit_SSA_ref_mse)
  if (n==1)
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-c("Cor. HP","Cor. SSA","Cor. MSE")
  } else
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-c("Cor. HP","Cor. M-SSA","Cor. MSE")
  }
  
  
  # 3. Empirical and theoretical HT
  # 3.1 HP classic
  # Without centering: x_mat is assumed to be standardized
  mplot<-(na.exclude(z_classic_HP))
  sample_ht_HP_classic<-length(mplot)/length(which(mplot[2:length(mplot)]*mplot[1:(length(mplot)-1)]<0))
  # First compute lag-one ACF and then HT 
  acf1_HP_classic<-gamma_HP_classic%*%M_tilde%*%gamma_HP_classic/gamma_HP_classic%*%I_tilde%*%gamma_HP_classic
  true_ht_HP_classic<-compute_holding_time_from_rho_func(acf1_HP_classic)$ht
  # 3.2 SSA  
  # Centering not necessary since data is standardized  
  mplot<-(na.exclude(y))
  sample_ht_SSA<-length(mplot)/length(which(mplot[2:length(mplot)]*mplot[1:(length(mplot)-1)]<0))
  # First compute lag-one ACF and then HT 
  acf1_mssa<-bk_mat[,m]%*%M_tilde%*%bk_mat[,m]/bk_mat[,m]%*%I_tilde%*%bk_mat[,m]
  ht_mssa<-compute_holding_time_from_rho_func(acf1_mssa)$ht
  # Is the same as ht_ssa_vec[m] if optimization converged i.e. difference below should be negligible (they become smaller when increasing split_grid)
  ht_mssa-ht_ssa_vec[m]
  true_ht_SSA<-ht_mssa
  # 3.3 MSE
  # Sample HT
  mplot<-(na.exclude(z_mse))
  sample_ht_MSE<-length(mplot)/length(which(mplot[2:length(mplot)]*mplot[1:(length(mplot)-1)]<0))
  acf1_mse<-t(gammak_mse[,m])%*%M_tilde%*%gammak_mse[,m]/t(gammak_mse[,m])%*%I_tilde%*%gammak_mse[,m]
  true_ht_MSE<-pi/acos(acf1_mse)
  
  perf_mat_true<-c(perf_mat_true,true_ht_HP_classic,true_ht_SSA,true_ht_MSE)
  perf_mat_sample<-c(perf_mat_sample,sample_ht_HP_classic,sample_ht_SSA,sample_ht_MSE)
  if (n==1)
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-c("HT HP","HT SSA","HT MSE")
  } else
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-c("HT HP","HT M-SSA","HT MSE")
  }
  
  
  # 4. Sign accuracies SA (probabilities of same sign, see proof of proposition 1 in paper for 'true' criterion with arcsin transformation) 
  # 4.1 HP classic
  # We remove NA's  
  mplot<-na.exclude(cbind(zdelta,z_classic_HP))
  sample_SA_crit_HP_classic_ref_target<-length(which(sign(mplot[,1])==sign(mplot[,2])))/nrow(mplot)
  cor_HP_classic_target<-gammak_mse[,m]%*%I_tilde%*%gamma_HP_classic/sqrt(gamma_HP_classic%*%I_tilde%*%gamma_HP_classic*var_target[m,m])
  true_SA_crit_HP_classic_ref_target<-0.5+asin(cor_HP_classic_target)/pi
  # 4.2 SSA  
  mplot<-na.exclude(cbind(zdelta,y))
  ts.plot(mplot)
  sample_SA_crit_SSA_ref_target<-length(which(sign(mplot[,1])==sign(mplot[,2])))/nrow(mplot)
  cor_MSSA_target<-gammak_mse[,m]%*%I_tilde%*%bk_mat[,m]/sqrt(bk_mat[,m]%*%I_tilde%*%bk_mat[,m]*var_target[m,m])
  # Is the same as SSA_obj$crit_rhoy_target[m] i.e. difference below vanishes 
  cor_MSSA_target-SSA_obj$crit_rhoy_target[m]
  true_SA_crit_SSA_ref_target<-0.5+asin(cor_MSSA_target)/pi
  mplot<-na.exclude(cbind(zdelta,z_mse))
  sample_SA_crit_mse_ref_target<-length(which(sign(mplot[,1])==sign(mplot[,2])))/nrow(mplot)
  # 4.3 MSE  
  # Correlation of causal MSE with acausal target  
  cor_mse_target<-gammak_mse[,m]%*%I_tilde%*%gammak_mse[,m]/sqrt(gammak_mse[,m]%*%I_tilde%*%gammak_mse[,m]*var_target[m,m])
  true_SA_crit_mse_ref_target<-0.5+asin(cor_mse_target)/pi
  
  perf_mat_true<-c(perf_mat_true,true_SA_crit_HP_classic_ref_target,true_SA_crit_SSA_ref_target,true_SA_crit_mse_ref_target)
  perf_mat_sample<-c(perf_mat_sample,sample_SA_crit_HP_classic_ref_target,sample_SA_crit_SSA_ref_target,sample_SA_crit_mse_ref_target)
  if (n==1)
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-c("SA HP","SA SSA","SA MSE")
  } else
  {
    names(perf_mat_sample)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-names(perf_mat_true)[(length(perf_mat_sample)-2):length(perf_mat_sample)]<-c("SA HP","SA M-SSA","SA MSE")
  }
  
  return(list(perf_mat_true=perf_mat_true,perf_mat_sample=perf_mat_sample,zdelta=zdelta,y=y,z_mse=z_mse,z_classic_HP=z_classic_HP))
}



