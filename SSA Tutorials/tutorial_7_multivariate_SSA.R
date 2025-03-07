# M-SSA: extension of univariate to multivariate SSA
# Work in progress

# -The first M-SSA tutorial, this one, is based on a simulation example derived from an application of M-SSA to 
#   predicting German GDP (or BIP)
# -The data generating process here relies on the VAR fitted to German data
# -We here show that M-SSA is optimal (if the data generating is the true process) and that the most relevant 
#   sample performances converge to their expected numbers for sufficiently long samples of (artificial) data

# Clean sheet
rm(list=ls())

# Let's start by loading the required R-libraries

# Standard filter package
library(mFilter)
# Multivariate time series: VARMA model for macro indicators: used here for simulation purposes only
library(MTS)


# Load the relevant M-SSA functionalities
# M-SSA functions
source(paste(getwd(),"/R/functions_MSSA.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

#-----------------------------------------------
# Specify data generating process (DGP): this model is obtained by fitting a VAR(1) to quarterly German macro data
# The VAR is based on data up to Jan 2007 to demonstrate out-of-sample prioperties of the M-SSA predictor on 
#   a long out-of-sample span including the financial crisis as well as all subsequent `never ending' sequence of crises
# It is a 5-dimensional design comprising BIP (i.e. GDP), industrial production, economic sentiment, spread and an ifo indicator
#   -All series are log-transformed (except spread) and differenced (no cointegration)
# Since the series are relatively short (introduction of EURO up to Jan-2007) the VAR is sparsely parametrized
n<-5
# AR order
p_mult<-1
# AR(1) coefficient
Phi<-matrix(c( 0.0000000,  0.00000000, 0.4481816,    0, 0.0000000,
               0.2387036, -0.33015450, 0.5487510,    0, 0.0000000,
               0.0000000,  0.00000000, 0.4546929,    0, 0.3371898,
               0.0000000,  0.07804158, 0.4470288,    0, 0.3276132,
               0.0000000,  0.00000000, 0.0000000,    0, 0.3583553),nrow=n)

# covariance matrix
Sigma<-matrix(c(0.755535544,  0.49500481, 0.11051024, 0.007546104, -0.16687913,
  0.495004806,  0.65832962, 0.07810020, 0.025101191, -0.25578971,
  0.110510236,  0.07810020, 0.66385111, 0.502140497,  0.08539719,
  0.007546104,  0.02510119, 0.50214050, 0.639843288,  0.05908741,
 -0.166879134, -0.25578971, 0.08539719, 0.059087406,  0.84463448),nrow=n)

# Simulate a series corresponding to 25 years of quarterly data    
len<-100
set.seed(31)
x_mat=VARMAsim(len,arlags=c(p_mult),phi=Phi,sigma=Sigma)$series

# Use the original column names: GDP (BIP), industrial production, ifo, sentiment and spread
colnames(x_mat)<-c("BIP", "ip", "ifo_c","ESI", "spr_10y_3m") 
# Have a look at the data
# We expect to see mutually (cross-section) correlated and (longitudinal) autocorrelated series
# Can you glimpse `crises`, i.e., phases with strong negative growth?
ts.plot(x_mat[(len-99):len,])

# Generate a very long series for our simulation experiment    
len<-100000
set.seed(87)
x_mat=VARMAsim(len,arlags=c(p_mult),phi=Phi,sigma=Sigma)$series

# Use the original column names: GDP (BIP), industrial production, ifo, sentiment and spread
colnames(x_mat)<-c("BIP", "ip", "ifo_c","ESI", "spr_10y_3m") 

#-------------------------------------------------------------------
# Specify target: we want to nowcast and to predict a two-sided Hodrick Prescott filter
# The sampling frequency is quarterly: we selected a smaller lambda=160 than the usual quarterly setting of 1600
#   The classic 1600-setting leads to trends which are too smooth for the purpose of the analysis (predicting GDP-growth)
lambda_HP<-160
# Filter length: roughly 4 years. The length should be an odd number in order to have a symmetric HP 
#   with a peak in the middle (for even numbers the peak is truncated)
L<-31
# HP filter
HP_obj<-HP_target_mse_modified_gap(L,lambda_HP)

hp_symmetric=HP_obj$target
hp_classic_concurrent=HP_obj$hp_trend
hp_one_sided<-HP_obj$hp_mse


ts.plot(hp_symmetric)
#-------------------------------------------------------------------
# MA inversion of VAR
# MA inversion is used because the M-SSA optimization criterion relies an white noise
#   For autocorrelated data, we thus require the MA-inversion of the DGP
xi_psi<-PSIwgt(Phi = Phi, Theta = NULL, lag = L, plot = F, output = F)
xi_p<-xi_psi$psi.weight
# Transform Xi_p into Xi: first L entries, from left to right, are weights of first WN, next L entries are weights of second WN 
xi<-matrix(nrow=n,ncol=n*L)
for (i in 1:n)
{
  for (j in 1:L)
    xi[,(i-1)*L+j]<-xi_p[,i+(j-1)*n]
}
# Plot MA inversions  ????check the model
par(mfrow=c(1,n))
for (i in 1:n)#i<-1
{
  mplot<-xi[i,1:min(10,L)]
  
  for (j in 2:n)
  {
    mplot<-cbind(mplot,xi[i,(j-1)*L+1:min(10,L)])
    
  }
  ts.plot(mplot,col=rainbow(ncol(mplot)),main=paste("MA inversion ",colnames(x_mat)[i],sep=""))
}

#-----------------------------------------------------------------
# Target for M-SSA: two-sided acausal HP 
# A target must be specified for each of the 5 series of the VAR
#   The target for each of the five series is the two-sided HP applied to this series and possibly shifted forward (prediction)
# M-SSA filters are structured/organized in vectors and matrices
#   -The filter matrix has dimension nx(n*L)
#   -The i-th row of the filter matrix collects the filters for the i-th target
#   -Each row consists of n filters: one filter assigned to each of the n explanatory variables
#   -Therefore the number of columns is n*L (each sub-filter has length(L))
# We now specify the target filter accordingly: dimension nx(n*L)
# We start by the target for the first series
gamma_target<-c(hp_one_sided,rep(0,(n-1)*L))
# For the first series, the target is HP applied to the first series and then zeroes
#   The series delimitations are marked by vertical lines in the plot
ts.plot(gamma_target)
abline(v=(1:n)*L)
# Note that the filter is one-sided yet, see below for further details
# We now proceed to specifying the targets of the remaining n-1 series
for (i in 2:n)
  gamma_target<-rbind(gamma_target,c(rep(0,(i-1)*L),hp_one_sided,rep(0,(n-i)*L)))
# The target of each series is just HP applied to this series (still one-sided yet)
par(mfrow=c(1,1))
ts.plot(t(gamma_target),col=rainbow(n))
# In the above plots, the target filters where one-sided
# We now tell M-SSA that it has to mirror the above filters at their center points to obtain two-sided targets
symmetric_target<-T
# When symmetric_target==T, M-SSA is advised to compute a symmetric target by mirroring 
#   the right tail to the left as follows
ts.plot(c(hp_one_sided[L:2],hp_one_sided[1:L]))
# Note that this procedure (mirroring) assumes L to be an odd number (see previous comments above)
# Further down we will see another equivalent target specification based on supplying the two-sided filter directly

#--------------------------------------------------------------------------------
# Forecast horizon
# delta=0 means a nowcast
# delta>0 means a forecast
# delta<0 means a backcast: clearly, this is not our main application case
#   Note that for backcasts the filters may look weird because the HT in the constraint is too small (M-SSA will unsmooth the data)
delta<-6

# Holding times
# We have to specify a HT for each series
# The following numbers are derived from the Macro-tool: 
#   We impose roughly between 1.5 and 2 years as mean duration between consecutive zero-crossings of the predictor
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
# Compute corresponding lag-one ACF in HT constraint: see previous tutorials on the link between HT and lag-one ACF  
rho0<-compute_rho_from_ht(ht_ssa_vec)$rho

# Some default settings for numerical optimization
# with_negative_lambda==T allows the extend the search to unsmoothing (generate more zero-crossings than benchmark): 
#   Default value is FALSE (smoothing only)
with_negative_lambda<-F
# Default setting for numerical optimization
lower_limit_nu<-"rhomax"
# Optimization with half-way triangulation: effective resolution is 2^split_grid. Much faster than brute-force grid-search.
# 20 is a good value: fast and strong convergence in most applications
split_grid<-20

# Now we can apply M-SSA
MSSA_obj<-MSSA_func(split_grid,L,delta,grid_size,gamma_target,rho0,with_negative_lambda,xi,lower_limit_nu,Sigma,symmetric_target)

# In principle we could retrieve filters, apply to data and check performances
# But M-SSA delivers a much richer output, containing different filters and useful evaluation metrics
# These will be analyzed further down
# So let's pick out the real-time filter
bk_x_mat<-MSSA_obj$bk_x_mat
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-bk_x_mat[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,bk_x_mat[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("MSSA applied to x ",colnames(x_mat)[i],sep=""),col=rainbow(n))
}


#----------------------------------------------
# Apply M-SSA to data and check most important performance measures
# Select any of the series: m=1,...,n
m<-3
bk<-NULL
# Extract coefficients applied to m-th series    
for (j in 1:n)#j<-2
  bk<-cbind(bk,bk_x_mat[((j-1)*L+1):(j*L),m])
ts.plot((bk))
y<-rep(NA,len)
for (j in L:len)#j<-L
{
  y[j]<-sum(apply(bk*(x_mat[j:(j-L+1),]),2,sum))
}

# Apply target to m-th-series
gammak<-hp_one_sided[1:L]
par(mfrow=c(1,1))
ts.plot(gammak)
z<-rep(NA,len)
# Here the filter is mirrored
for (j in L:(len-L))
  z[j]<-gammak%*%x_mat[j:(j-L+1),m]+gammak[-1]%*%x_mat[(j+1):(j+L-1),m]
# Shift z by delta    
if (delta>0)
{  
  zdelta<-c(z[(delta+1):len],rep(NA,delta))
} else
{
  if (delta<0)
  {
    zdelta<-c(rep(NA,abs(delta)),z[1:(len-abs(delta))])
  } else
  {
    zdelta<-z
  }
}
names(zdelta)<-names(y)<-rownames(x_mat)

# Mean-square error
mean((zdelta-y)^2,na.rm=T)

# Correlation between target and M-SSA: sample estimates converge to criterion value for increasing sample size len
# Correlation is (1,2) element of correlation matrix 
cor(na.exclude(cbind(zdelta,y)))[1,2]
# This is the criterion value of M-SSA: the target correlation is maximized under the HT constraint
# We can see that the theoretical criterion value (maximized `true' correlation) matches the sample statistic
MSSA_obj$crit_rhoy_target[m]

# M-SSA optimites target correlation under holding time constraint:
# Compare empirical and theoretical (imposed) HTs: sample HT converges to imposed HT for increasing sample size len
compute_empirical_ht_func(y)
ht_mssa_vec[m]

#####################################################################################################################################
# Summary: The above simulation confirms that M-SSA optimization maximizes the target correlation subject to the HT constraint
############################################################################################################################## 












###########################################################################################

#-----------------------
# 4.2 Validation checks: compute  theoretical performances and check HT constraint  
# Target correlation: with respect to causal MSE benchmark  
MSSA_obj$crit_rhoyz
# Target correlation: with respect to effective acausal target  
MSSA_obj$crit_rhoy_target
# Lag one ACF: should match HT constraint  
MSSA_obj$crit_rhoyy
# Optimized nu  
MSSA_obj$nu_opt
# SSA as applied to epsilont (convolution)
bk_mat<-MSSA_obj$bk_mat
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-bk_mat[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,bk_mat[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("SSA applied to epsilon ",colnames(data_fit)[i],sep=""),col=rainbow(ncol(data)))
}
# SSA as applied to xt (deconvolution)
bk_x_mat<-MSSA_obj$bk_x_mat
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-bk_x_mat[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,bk_x_mat[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("MSSA applied to x ",colnames(data_fit)[i],sep=""),col=rainbow(ncol(data)))
}

# Check: deconvolute xi from bk_mat gives bk_x_mat  
deconv_M<-t(M_deconvolute_func(t(bk_mat),xi)$deconv)
max(abs(deconv_M-bk_x_mat))
# MSE as applied to epsilont: this is the same as gamma_mse above
gammak_mse<-MSSA_obj$gammak_mse
# Check: should vanish  
max(abs(gammak_mse-t(gamma_mse)))
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-gammak_mse[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,gammak_mse[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("M-MSE applied to eps ",colnames(data_fit)[i],sep=""),col=rainbow(ncol(data)))
}
# MSE as applied to xt
gammak_x_mse<-MSSA_obj$gammak_x_mse
# Check: should vanish
max(abs(gammak_x_mse-gamma_mse_x))
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-gammak_x_mse[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,gammak_x_mse[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("M-MSE applied to x ",colnames(data_fit)[i],sep=""),col=rainbow(ncol(data)))
}


# Target: effective acausal filter convolved with xi
gamma_target_long<-MSSA_obj$gammak_target
dim(gamma_target_long)
ts.plot(gamma_target_long,col=rainbow(ncol(data)))
# Variance-covariance matrix of acausal target  
var_target<-MSSA_obj$var_target
# Can compare variances on diagonal of acausal target (var_target) and causal MSE (below): the latter are roughly 50% smaller (because MSE is missing future epsilons)
t(gammak_mse)%*%I_tilde%*%gammak_mse
t(bk_mat)%*%I_tilde%*%bk_mat

# 1. Lag-one ACFs and HTs  
# Need system matrices M_tilde and I_tilde to compute ACFs
M_obj<-M_func(L,Sigma)
M_tilde<-M_obj$M_tilde
I_tilde<-M_obj$I_tilde
# Lag one ACF of MSE nowcast 
rho_mse<-gammak_mse[,1]%*%M_tilde%*%gammak_mse[,1]/gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]
for (i in 2:n)
  rho_mse<-c(rho_mse,gammak_mse[,i]%*%M_tilde%*%gammak_mse[,i]/gammak_mse[,i]%*%I_tilde%*%gammak_mse[,i])
rho_ssa<-bk_mat[,1]%*%M_tilde%*%bk_mat[,1]/bk_mat[,1]%*%I_tilde%*%bk_mat[,1]
for (i in 2:n)
  rho_ssa<-c(rho_ssa,bk_mat[,i]%*%M_tilde%*%bk_mat[,i]/bk_mat[,i]%*%I_tilde%*%bk_mat[,i])
# Compute HTs based on ACFs
ht_comp<-apply(matrix(rho_ssa,nrow=1),1,compute_holding_time_from_rho_func)[[1]]$ht
ht_comp
# Check: optimization successful if the above HTs based on optimal nowcasts match the imposed HTs
# Increasing split_grid tightens the fit
ht_ssa_vec
# Compute HTs of MSE
apply(matrix(rho_mse,nrow=1),1,compute_holding_time_from_rho_func)[[1]]$ht
# Check: should be identical  
ht_mse_vec

# 2. Criteria: MSE is trivially one since correlation of MSE with itself is one (our target is MSE which leads to the same solution as using z_{t+\delta})
#   The SSA criteria computed here correspond to MSSA_obj$crit_rhoyz
crit_mse<-gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]/gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]
for (i in 2:n)
  crit_mse<-c(crit_mse,gammak_mse[,i]%*%I_tilde%*%gammak_mse[,i]/gammak_mse[,i]%*%I_tilde%*%gammak_mse[,i])
crit_ssa<-gammak_mse[,1]%*%I_tilde%*%bk_mat[,1]/(sqrt(bk_mat[,1]%*%I_tilde%*%bk_mat[,1])*sqrt(gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]))
for (i in 2:n)
  crit_ssa<-c(crit_ssa,gammak_mse[,i]%*%I_tilde%*%bk_mat[,i]/(sqrt(bk_mat[,i]%*%I_tilde%*%bk_mat[,i])*sqrt(gammak_mse[,i]%*%I_tilde%*%gammak_mse[,i])))

crit_mse_1<-gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]/gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]
crit_ssa_1<-gammak_mse[,1]%*%I_tilde%*%bk_mat[,1]/(sqrt(bk_mat[,1]%*%I_tilde%*%bk_mat[,1])*sqrt(gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]))
crit_mse_2<-gammak_mse[,2]%*%I_tilde%*%gammak_mse[,2]/gammak_mse[,2]%*%I_tilde%*%gammak_mse[,2]
crit_ssa_2<-gammak_mse[,2]%*%I_tilde%*%bk_mat[,2]/(sqrt(bk_mat[,2]%*%I_tilde%*%bk_mat[,2])*sqrt(gammak_mse[,2]%*%I_tilde%*%gammak_mse[,2]))

criterion_mat<-rbind(crit_mse,crit_ssa)
colnames(criterion_mat)<-c(paste("Series ",1:n,paste=""))
rownames(criterion_mat)<-c("MSE","SSA")
criterion_mat
# Compare second row with MSSA_obj$crit_rhoyz
MSSA_obj$crit_rhoyz







