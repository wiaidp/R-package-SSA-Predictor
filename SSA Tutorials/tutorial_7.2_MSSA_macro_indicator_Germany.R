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
# Utility functions for M-SSA, see tutorial 
source(paste(getwd(),"/R/M_SSA_utility_functions.r",sep=""))

# 

#------------------------------------------------------------------------
# Let's apply the above functions to the previous simulation experiment

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

# The above functions can also be sourced
source(paste(getwd(),"/R/M_SSA_utility_functions.r",sep=""))


