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

# Nowcast
delta<-0
# One year ahead forecast for quarterly data
delta<-4


# Holding times
# We have to specify a HT for each series
# The following numbers are derived from the Macro-tool: 
#   We impose roughly between 1.5 and 2 years as mean duration between consecutive zero-crossings of the predictor
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
# Compute corresponding lag-one ACF in HT constraint: see previous tutorials on the link between HT and lag-one ACF  
rho0<-compute_rho_from_ht(ht_mssa_vec)$rho

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
m<-m_check<-3
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

# M-SSA optimizes target correlation under holding time constraint:
# Compare empirical and theoretical (imposed) HTs: sample HT converges to imposed HT for increasing sample size len
compute_empirical_ht_func(y)
ht_mssa_vec[m]

#####################################################################################################################################
# Summary: The above simulation confirms that M-SSA optimization maximizes the target correlation subject to the HT constraint
############################################################################################################################## 

#######################################################################################
# Advanced (for readers of the M-SSA paper deriving and explaining formal expressions for expected performance measures)
#   We can now look in more detail at some of the additional outputs generated by M-SSA

# Some validation checks: compute  theoretical expected performances and check HT constraint  
# Target correlation: with respect to classic MSE benchmark 
# M-SSA replicates the classic MSE estimate when HT in the constraint matches HT of MSE
#   In general the MSE benchmark is `noisy' and we want M-SSA to be smoother (stronger noise suppression)`
#   Therefore the correlation with the MSE benchmark is generally smaller one
#   But the correlation is always larger than the target correlation with the two-sided (acausal) HP 
MSSA_obj$crit_rhoyz
# Target correlation: with respect to effective acausal target  
MSSA_obj$crit_rhoy_target
# Lag one ACF: should match imposed lag-one ACF 
# This is a quick check to verify successful numerical optimization
# If the discrepancy is `too large', the number of iteration steps specified by split_grid can be increased
MSSA_obj$crit_rhoyy
rho0
# Equivalently, the HT of the optimized M-SSA should match the imposed HT
compute_holding_time_from_rho_func(MSSA_obj$crit_rhoyy)
compute_holding_time_from_rho_func(rho0)

# Optimized nu in discrete difference equation, see paper for interpretation  
MSSA_obj$nu_opt
# M-SSA also computes optimal filters when applied to residuals of the VAR
#   This can be useful for visualization of the smoothing property (impulse response, for expert users)
#   Applying these filters to VAR residuals replicates the ordinary M-SSA output (see time series y above)
#   The approximation improves with larger L
# M-SSA computes bk_mat; the final filter bk_x_mat (used in above simulation exercise) is obtained by deconvolution
bk_mat<-MSSA_obj$bk_mat
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-bk_mat[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,bk_mat[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("SSA applied to epsilon ",colnames(x_mat)[i],sep=""),col=rainbow(n))
}
# SSA as applied to original series x_mat, see above simulation exercise 
#   Formally, bk_x_mat is obtained by deconvolution of MA-inversion xi from bk_mat above, see below
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

# Check: deconvolute xi from bk_mat gives bk_x_mat  
deconv_M<-t(M_deconvolute_func(t(bk_mat),xi)$deconv)
# Both filters match perfectly: absolute error is zero
max(abs(deconv_M-bk_x_mat))
# To summarize
# -bk_x_mat is the relevant filter
# -bk_mat is originally computed by M-SSA and provides some useful visual clues (impulse response)

# M-SSA also computes the classic MSE filter: this is the design computed by classic signal extraction
# First we look at the MSE-filter as applied to the VAR residuals
gammak_mse<-MSSA_obj$gammak_mse
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-gammak_mse[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,gammak_mse[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("M-MSE applied to eps ",colnames(x_mat)[i],sep=""),col=rainbow(n))
}
# Second we look at the MSE filter as applied to the orginal data (x_mat)
#   -If the HT constraint of M-SSA matches the HT of this filter, then M-SSA replicates MSE
#   -M-SSA is a generalization of classic signal extraction algorithms, allowing for a control of the HT (smoothness or noise suppression)
gammak_x_mse<-MSSA_obj$gammak_x_mse
par(mfrow=c(1,n))
for (i in 1:n)# i<-1
{
  mplot<-gammak_x_mse[1:L,i]
  for (j in 2:n)
  {
    mplot<-cbind(mplot,gammak_x_mse[(j-1)*L+1:L,i])
  }
  ts.plot(mplot,main=paste("M-MSE applied to x ",colnames(x_mat)[i],sep=""),col=rainbow(n))
}


# Target: two-sided HP filter convolved with MA-inversion xi
#   This is the symmetric filter when applied to VAR residuals
gamma_target_long<-MSSA_obj$gammak_target
dim(gamma_target_long)
# We plot the filters: vertical lines indicate delimitations between explanatory series i=1,...,n
#   -Note that for a particular series (say the first, i=1), the convolution of original target and MA-inversion xi 
#       generally assigns weight to all residuals in the presence of cross-correlation (non-diagonal Sigma) 
# -In contrast, the original gamma_target (see plot above) assigns weight only to the target series (all other series receive weight 0)
ts.plot(gamma_target_long,col=rainbow(n))
abline(v=(1:n*(nrow(gamma_target_long)/n)))
# M-SSA also computes the true variance of the two-sided HP output
var_target<-MSSA_obj$var_target
# We cCan compare variances on diagonal of acausal target (var_target) and causal MSE (below): the latter are roughly 50% smaller (because MSE is missing future epsilons)
# Need system matrices M_tilde and I_tilde to compute expectations
M_obj<-M_func(L,Sigma)
M_tilde<-M_obj$M_tilde
I_tilde<-M_obj$I_tilde

# Variance of target
diag(var_target)
# Variance of classic MSE predictor
diag(t(gammak_mse)%*%I_tilde%*%gammak_mse)
# Variance of M-SSA
diag(t(bk_mat)%*%I_tilde%*%bk_mat)
# The m-th number of the above vector should match the sample variance of the M-SSA output of the m-th series, see simulation above
diag(t(bk_mat)%*%I_tilde%*%bk_mat)[m]
var(na.exclude(y))
# Similarly, the m-th variance of the target should match the sample variance of zdelta, see above simulation experiment
diag(var_target)[m]
var(na.exclude(zdelta))


# We can now compute the true lag one ACFs of the classic MSE nowcast (consulte the M-SSA paper for technical details)
rho_mse<-gammak_mse[,1]%*%M_tilde%*%gammak_mse[,1]/gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]
for (i in 2:n)
  rho_mse<-c(rho_mse,gammak_mse[,i]%*%M_tilde%*%gammak_mse[,i]/gammak_mse[,i]%*%I_tilde%*%gammak_mse[,i])
# Similarly, we can compute the true lag-one ACFs of M-SSA, see M-SSA paper
rho_ssa<-bk_mat[,1]%*%M_tilde%*%bk_mat[,1]/bk_mat[,1]%*%I_tilde%*%bk_mat[,1]
for (i in 2:n)
  rho_ssa<-c(rho_ssa,bk_mat[,i]%*%M_tilde%*%bk_mat[,i]/bk_mat[,i]%*%I_tilde%*%bk_mat[,i])
# We can now derive the HTs based on ACFs
ht_comp<-apply(matrix(rho_ssa,nrow=1),1,compute_holding_time_from_rho_func)[[1]]$ht
ht_comp
# We have verified in simulation experiment that sample HTs match these `true' numbers`
# We can once again check if optimization was successful 
#  If successful, then the above HTs based on optimal nowcasts must match the imposed HTs
#  Increasing the number of iterations specified by split_grid tightens the fit
ht_mssa_vec
# We can also compute HTs of MSE: 
#   In general M-SSA is designed to be smoother (stronger nois suppression), i.e., ht_mssa_vec is larger than the below HTs of MSE design
#   We can of course change the HT in the constraint as specified in the call to M-SSA
apply(matrix(rho_mse,nrow=1),1,compute_holding_time_from_rho_func)[[1]]$ht

# Next we can compute the target correlations
#   -More precisely we here compute the correlation of M-SSA with MSE benchmark
#   -If HT of M-SSA matches MSE-benchmark, then M-SSA replicates the latter and correlation is one
#   -If HT of M-SSA is larger than that of MSE-benchmark, then correlation is smaller one: but M-SSA maximizes this correlation subject to the HT constraint
crit_mse<-gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]/gammak_mse[,1]%*%I_tilde%*%gammak_mse[,1]
for (i in 2:n)
  crit_mse<-c(crit_mse,gammak_mse[,i]%*%I_tilde%*%gammak_mse[,i]/gammak_mse[,i]%*%I_tilde%*%gammak_mse[,i])
# The correlation of the MSE-benchmark with itself is trivially one
# M-SSA tries to maximize this correlation (achieve one-correlöation) subject to the HT constraint
crit_mse
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
# Compare second row with MSSA_obj$crit_rhoyz computed by M-SSA 
#  crit_rhoyz is the objective function of the optimization criterion and is maximized by M-SSA
MSSA_obj$crit_rhoyz



##########################################################################################
# Summary:
# -M-SSA has a rich output with additional filters (including the classic MSE signal extraction filter) and performance metrics
# -Theoretical expressions (expected values) of any performance number (see M-SSA paper for derivation) match sample estimates (for sufficiently long samples)
# -M-SSA optimization concept: maximize target correlation (equivalently: minimize mean-square forecast error) subject to HT constraint
# -M-SSA replicates classic MSE signal extraction filter by inserting the HT of the latter into the M-SSA optimization
# -M-SSA can address backcasting (delta<0), nowcasting (delta=0) and forecasting (delta>0)
# -The target specification is generic: in the above experiment we relied on the two-sided HP
#   -Classic h-step ahead forecasting can be obtained by replacing the HP-filter by the identity (see univariate SSA tutorials on the topic)
# -The data generating process (DGP) is assumed to be stationary (could be generalized); otherwise the specification is completely general
#   -M-SSA relies on the (reduced-form) MA-inversion of the DGP which is straightforward to obtain for VARMA processes (see above illustration)
# -A convergence of sample performances towards expected numbers assumes the model to be `true'
#   -However, we shall see that the above application to German macro data is remarkably robust 
#     -against singular Pandemic data (outliers)
#     -against in-sample span for VAR: pre-financial crisis M-SSA (data up Jan-2007) performs nearly as well as full sample M-SSA
#     -against VARMA specification (as long as heavy overfitting is avoided)   
###########################################################################################
########################################################################################


# Densify code: let's pack the above code into functions with distinct tasks
# 1. Target function
HP_target_sym_T<-function(n,lambda_HP,L)
{
  HP_obj<-HP_target_mse_modified_gap(L,lambda_HP)
  
  hp_symmetric=HP_obj$target
  hp_classic_concurrent=HP_obj$hp_trend
  hp_one_sided<-HP_obj$hp_mse
  # Target first series  
  gamma_target<-c(hp_one_sided,rep(0,(n-1)*L))
  # We now proceed to specifying the targets of the remaining n-1 series
  for (i in 2:n)
    gamma_target<-rbind(gamma_target,c(rep(0,(i-1)*L),hp_one_sided,rep(0,(n-i)*L)))
  # The above target filters are one-sided (right half of two-sided filter)
  # We now tell M-SSA that it has to mirror the above filters at their center points to obtain two-sided targets
  symmetric_target<-T
  return(list(gamma_target=gamma_target,symmetric_target=symmetric_target))
}  



# 2. MA-inversion as based on VAR model
MA_inv_VAR_func<-function(Phi,Theta,L,n,Plot=F)
{
  # MA inversion of VAR
  # MA inversion is used because the M-SSA optimization criterion relies an white noise
  #   For autocorrelated data, we thus require the MA-inversion of the DGP
  xi_psi<-PSIwgt(Phi = Phi, Theta = NULL, lag = L, plot = F, output = F)
  xi_p<-xi_psi$psi.weight
  # Transform Xi_p into Xi as structured/organized for M-SSA
  #   First L entries, from left to right, are weights of first explanatory series, next L entries are weights of second WN 
  xi<-matrix(nrow=n,ncol=n*L)
  for (i in 1:n)
  {
    for (j in 1:L)
      xi[,(i-1)*L+j]<-xi_p[,i+(j-1)*n]
  }
  if (Plot)
  {
    # Plot MA inversions  
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
  }
  return(list(xi=xi))
}

# M-SSA
MSSA_main_func<-function(delta,ht_vec,xi,symmetric_target,gamma_target,Plot=F)
{
  # Compute lag-one ACF corresponding to HT in M-SSA constraint: see previous tutorials on the link between HT and lag-one ACF  
  rho0<-compute_rho_from_ht(ht_vec)$rho
  
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
  if (Plot)
  {
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
  }
  # We return the M-SSA filter as well as the whole M-SSA object which hides additional useful objects  
  return(list(bk_x_mat=bk_x_mat,MSSA_obj=MSSA_obj))
}


# 4. Filter function: apply M-SSA filter to data

filter_func<-function(x_mat,bk_x_mat,gamma_target,symmetric_target,delta)
{
  len<-nrow(x_mat)
  n<-dim(bk_x_mat)[2]
  # Compute M-SSA filter output 
  mssa_mat<-target_mat<-NULL
  for (m in 1:n)
  {
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
    mssa_mat<-cbind(mssa_mat,y)
  }  
  # Apply target to m-th-series
  target_mat<-NULL
  for (m in 1:n)#
  {
    # In general, m-th target is based on j=1,...,n filters applied to explanatory variables j=1,...,n
    gammak<-NULL
    for (j in 1:n)
    {
      # Retrieve j-th filter for m-th target       
      gammak<-cbind(gammak,gamma_target[m,(j-1)*L+1:L])
    }
    z<-rep(NA,len)
    if (symmetric_target)
    {
      # Here the right half of the filter is mirrored to the left at its peak
      # Moreover, the data is shifted by delta
      for (j in (L-delta):(len-L-delta+1))#j<-L-delta
        z[j]<-sum(apply(gammak*x_mat[delta+j:(j-L+1),],2,sum))+sum(apply(gammak[-1,]*x_mat[delta+(j+1):(j+L-1),],2,sum))
    } else
    {
      # Data shifted by delta: we do not mirror filter weights      
      for (j in (L-delta):(len-delta))
      {
        z[j]<-sum(apply(gammak*(x_mat[delta+j:(j-L+1),]),2,sum))
      }
    }
    
    names(zdelta)<-names(y)<-rownames(x_mat)
    target_mat<-cbind(target_mat,z)
  } 
  colnames(mssa_mat)<-colnames(target_mat)<-colnames(x_mat)
  return(list(mssa_mat=mssa_mat,target_mat=target_mat))
}

#------------------------------------------------------------------------
# Let's apply the above functions

# 1. Target
lambda_HP<-160
# Filter length: roughly 4 years. The length should be an odd number in order to have a symmetric HP 
#   with a peak in the middle (for even numbers the peak is truncated)
L<-31

target_obj<-HP_target_sym_T(n,lambda_HP,L)

gamma_target=target_obj$gamma_target
symmetric_target=target_obj$symmetric_target 

# The targets are one-sided
par(mfrow=c(1,1))
ts.plot(t(gamma_target),col=rainbow(n))

# But we tell M-SSA to mirror the target filter at its peak value
symmetric_target

# 2. MA-inversion as based on VAR model

MA_inv_obj<-MA_inv_VAR_func(Phi,Theta,L,n,T)

xi<-MA_inv_obj$xi

# 3. M-SSA function

# Nowcast
delta<-0
# One year ahead forecast for quarterly data
delta<-4

MSSA_main_obj<-MSSA_main_func(delta,ht_mssa_vec,xi,symmetric_target,gamma_target,T)


MSSA_main_obj$bk_x_mat=bk_x_mat
MSSA_obj=MSSA_main_obj$MSSA_obj 


# 4. Filter function: apply M-SSA filter to data

filt_obj<-filter_func(x_mat,bk_x_mat,gamma_target,symmetric_target,delta)

mssa_mat=filt_obj$mssa_mat
target_mat=filt_obj$target_mat

#------------------------
# Checks: the obtained output should be identical to previous y and zdelta for series m_check: differences should vanish
max(abs(y-mssa_mat[,m_check]),na.rm=T)
max(abs(zdelta-target_mat[,m_check]),na.rm=T)

# Mean-square errors
apply(na.exclude((target_mat-mssa_mat)^2),2,mean)

# Correlations between target and M-SSA: sample estimates converge to criterion value for increasing sample size len
for (i in 1:n)
  print(cor(na.exclude(cbind(target_mat[,i],mssa_mat[,i])))[1,2])
# This is the criterion value of M-SSA: the target correlation is maximized under the HT constraint
# We can see that the theoretical criterion value (maximized `true' correlation) matches the sample correlations (assuming a sufficiently large sample size len)
MSSA_obj$crit_rhoy_target

# M-SSA optimizes target correlation under holding time constraint:
# Compare empirical and theoretical (imposed) HTs: sample HT converges to imposed HT for increasing sample size len
apply(mssa_mat,2,compute_empirical_ht_func)
ht_mssa_vec

# Operation confirmed
