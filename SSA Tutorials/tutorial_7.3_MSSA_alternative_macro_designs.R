# Tutorial 7.3: analyzing various M-SSA BIP predictor designs
# The concept of M-SSA predictors for BIP was introduced in tutorial 7.2
# We packed this proceeding into a single function to be able to analyze various M-SSA BIP predictor designs (hyperparameters)


# Start with a clean sheet
rm(list=ls())

# Load the required R-libraries

# Standard filter package
library(mFilter)
# Multivariate time series: VARMA model for macro indicators: used here for simulation purposes only
library(MTS)
# HAC estimate of standard deviations in the presence of autocorrelation and heteroscedasticity
library(sandwich)


# Load the relevant M-SSA functionalities
# M-SSA functions
source(paste(getwd(),"/R/functions_MSSA.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))
# Utility functions for M-SSA, see tutorial 
source(paste(getwd(),"/R/M_SSA_utility_functions.r",sep=""))


#------------------------------------------------------------------------
# Load data and select indicators: see tutorial 7.2 for background
load(file="C:\\Users\\marca\\OneDrive\\2025\\R-package-SSA-Predictor\\Data\\macro")
# We assume a publication lag of two quarters for BIP (the effective lag is smaller but we'd like to stay on the safe side, in particular since BIP is subject to revisions)
lag_vec<-c(2,rep(0,ncol(data)-1))
# Plot the data
# The real-time BIP (red) is lagging the target by lag_vec[1] quarters (publication lag)
par(mfrow=c(1,1))
mplot<-data
colo<-c("black",rainbow(ncol(data)-1))
main_title<-paste("Quarterly design BIP: the target (black) assumes a publication lag of ",lag_vec[1]," Quarters",sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(data)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# Select macro indicators for M-SSA 
select_vec_multi<-c("BIP","ip","ifo_c","ESI","spr_10y_3m")
x_mat<-data[,select_vec_multi] 
rownames(x_mat)<-rownames(data)
n<-dim(x_mat)[2]
# Number of observations
len<-dim(x_mat)[1]
#------------------------------

# Here's the head of the function derived from tutorial 7.2
head(compute_mssa_BIP_predictors_func)

# We can supply various hyperparameters (designs) and the function returns corresponding
#   -M-SSA predictors
#   -Performance measures: $
#     -Correlations with shifted HP-BIP or BIP
#     -HAC-adjusted p-values of regressions of predictors on shifted HP-BIP and BIP: to assess statistical significance

# In order to use the function, we need to specify hyperparameters, see tutorial 7.2 for background
# We here first replicate tutorial 7.2

# Target filter: lambda_HP is the single most important hyperparameter, see tutorial 7.1 for a discussion
# Briefly: we avoid the classic quarterly setting lambda_HP=1600 because the resulting filter would be too smooth
# Too smooth means: the forecast horizon would have nearly no effect on the M-SSA predictor (no left-shift, no anticipation)
lambda_HP<-160
# Filter length: nearly 8 years is fine for the selected lambda_HP (filter weights decay sufficiently fast)
L<-31
# In-sample span for VAR, i.e., M-SSA (the proposed design is quite insensitive to this specification because the VAR is parsimoniously parameterized)
date_to_fit<-"2008"
# VARMA model orders: keep the model simple in particular for short/tight in-sample spans
p<-1
q<-0
# Holding-times (HT): controls smoothness of M-SSA (the following numbers are pasted from the original predictor)
# Increasing these numbers leads to predictors with less zero-crossings (smoother)
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
# Forecast horizons: M-SSA is optimized for each forecast horizon in h_vec 
h_vec<-c(0,1,2,4,6)
# Forecast excesses: see tutorial 7.1 for background
f_excess<-c(4,2)

# Run the function packing and implementing our previous findings (tutorial 7.2) 
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess)

# We replicate performances obtained in tutorial 7.2  
cor_mat_BIP<-mssa_indicator_obj$cor_mat_BIP
cor_mat_HP_BIP<-mssa_indicator_obj$cor_mat
p_value_HAC_mat_HP_BIP<-mssa_indicator_obj$p_value_HAC_mat
p_value_HAC_mat_BIP<-mssa_indicator_obj$p_value_HAC_mat_BIP
BIP_target_mat=mssa_indicator_obj$BIP_target_mat
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
indicator_mat<-mssa_indicator_obj$indicator_mat

# Correlations between M-SSA predictors and forward-shifted HP-BIP (including the publication lag)
#   -We see that for increasing forward-shift (from top to bottom) the predictors optimized for 
#     larger forecast horizons (from left to right) tend to perform better
cor_mat_HP_BIP

# Let's visualize these correlations by plotting target against predictor
# Select a forward-shift of target (the k-th entry in h_vec)
k<-4
if (k>length(h_vec))
{
  print(paste("k should be smaller equal ",length(h_vec),sep=""))
  k<-length(h_vec)
}  
# Forward shift of target in quarters
h_vec[k]
# Select a M-SSA predictor: optimized for forecast horizon h_vec[j]
j<-k
if (j>length(h_vec))
{
  print(paste("j should be smaller equal ",length(h_vec),sep=""))
  j<-length(h_vec)
}  
# Plot targets (forward-shifted BIP and HP-BIP) and predictor
par(mfrow=c(1,1))
# Scale the data for better visual interpretation of effect of excess forecast on M-SSA (red) vs. previous M-SSA (blue)
mplot<-scale(cbind(BIP_target_mat[,k],target_shifted_mat[,k],indicator_mat[,j]))
rownames(mplot)<-rownames(x_mat)
colnames(mplot)<-c(paste("BIP left-shifted by ",h_vec[k]," quarters",sep=""),paste("HP-BIP left-shifted by ",h_vec[k]," quarters",sep=""),paste("M-SSA predictor optimized for h=",h_vec[j],sep=""))
colo<-c("black","violet","blue")
main_title<-"Standardized forward-shifted BIP and HP-BIP vs. M-SSA predictor"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(x_mat)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# Sample correlation: this corresponds to cor_mat_BIP computed by our function
cor(na.exclude(mplot))[2,ncol(mplot)]
cor_mat_HP_BIP[k,j]
# The following two correlations should match exactly
# However, they differ because by removing NAs (due to inclusion of the two-sided target) we change the sample-size
cor(na.exclude(mplot))[1,3]
cor_mat_BIP[k,j]
# We can easily amend by removing the two-sided target
mplot_without_two_sided<-scale(cbind(BIP_target_mat[,k],indicator_mat[,j]))
# This number now matches cor_mat_BIP[k,j]
cor(na.exclude(mplot_without_two_sided))[1,2]

# Assume one selects k=j=4 (one-year ahead) in the above plot (you might want to have a look at k=4 but j=5, too):
# Then the (weak) positive correlation between M-SSA and shifted BIP might suggest a (weak) predictability one year ahead
#    (including the publication lag) 
# Is this (weak) effect statistically significant?
# Let's have a look at the HAC-adjusted p-values
p_value_HAC_mat_BIP[k,j]
# Instead of BIP we might have a look at targeting HP-BIP instead (also shifted one year ahead)
p_value_HAC_mat_HP_BIP[k,j]

# Finding: statistical significance is stronger for shifted HP-BIP (than for BIP)
#   -Is it because short-term components of BIP are unpredictable?
#   -Or is it because lambda_HP=160 is not sufficiently adaptive (still too smooth)?


###############################################################################################
# Let's now analyze a more adaptive design by selecting a smaller lambda_HP

lambda_HP<-16
# Everything else in the above design is kept fixed
# Notes: 
#   -Keeping the above settings fixed is probably a bad idea because the `faster` filters (less smoothing required 
#       for lambda_HP=16) most likely do not require additional `acceleration' by the forecast excesses 
#   -You might try smaller values for f_excess

# Run the M-SSA predictor function
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess)

cor_mat_BIP<-mssa_indicator_obj$cor_mat_BIP
cor_mat_HP_BIP<-mssa_indicator_obj$cor_mat
p_value_HAC_mat_HP_BIP<-mssa_indicator_obj$p_value_HAC_mat
p_value_HAC_mat_BIP<-mssa_indicator_obj$p_value_HAC_mat_BIP
BIP_target_mat=mssa_indicator_obj$BIP_target_mat
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
indicator_mat<-mssa_indicator_obj$indicator_mat

# Look at correlations between M-SSA predictors and forward-shifted BIP (including the publication lag)
#   -We see that for increasing forward-shift (from top to bottom) the predictors optimized for 
#     larger forecast horizons (from left to right) tend to perform better
# Note: in contrast to the previous lambda_HP=160 setting, we here emphasize BIP (not HP-BIP)
p_value_HAC_mat_BIP
cor_mat_BIP

# Finding: the more adaptive design based on lambda_HP=16 seems to be able to track future BIP better

# Let's visualize these correlations by plotting target against predictor
# Select a forward-shift of target (the k-th entry in h_vec)
k<-4
if (k>length(h_vec))
{
  print(paste("k should be smaller equal ",length(h_vec),sep=""))
  k<-length(h_vec)
}  
# Forward shift of target in quarters
h_vec[k]
# Select a M-SSA predictor: optimized for forecast horizon h_vec[j]
j<-k
if (j>length(h_vec))
{
  print(paste("j should be smaller equal ",length(h_vec),sep=""))
  j<-length(h_vec)
}  
# Plot targets (forward-shifted BIP and HP-BIP) and predictor
par(mfrow=c(1,1))
# Scale the data for better visual interpretation of effect of excess forecast on M-SSA (red) vs. previous M-SSA (blue)
mplot<-scale(cbind(BIP_target_mat[,k],target_shifted_mat[,k],indicator_mat[,j]))
rownames(mplot)<-rownames(x_mat)
colnames(mplot)<-c(paste("BIP left-shifted by ",h_vec[k]," quarters",sep=""),paste("HP-BIP left-shifted by ",h_vec[k]," quarters",sep=""),paste("M-SSA predictor optimized for h=",h_vec[j],sep=""))
colo<-c("black","violet","blue")
main_title<-"Standardized forward-shifted BIP and HP-BIP vs. M-SSA predictor"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(x_mat)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# Sample correlation: this corresponds to cor_mat_BIP computed by our function
cor(na.exclude(mplot))[2,ncol(mplot)]
cor_mat_HP_BIP[k,j]





# The following two correlations should match exactly
# However, they differ because by removing NAs (due to inclusion of the two-sided target) we change the sample-size
cor(na.exclude(mplot))[1,3]
cor_mat_BIP[k,j]
# We can easily amend by removing the two-sided target
mplot_without_two_sided<-scale(cbind(BIP_target_mat[,k],indicator_mat[,j]))
# This number now matches cor_mat_BIP[k,j]
cor(na.exclude(mplot_without_two_sided))[1,2]

# Assume one selects k=j=4 (one year ahead) in the above plot:
# Then the positive correlation between M-SSA and shifted BIP suggests that the predictor is informative 
#   for BIP one-year ahead (including the publication lag) 
# Is predictability statistically significant?
# Let's have a look at the HAC-adjusted p-values
p_value_HAC_mat_BIP[k,j]
# In contrast to previous lambda_HP=160 setting, the predictor is now statisticially significant 
#   for forward-shifted BIP 
# Let's check significance for forward-shifted HP-BIP
p_value_HAC_mat_HP_BIP[k,j]
# Almost significant

# We might ask why the t-test suggests weaker significance while the correlation is larger for HP-BIP
# Let's have a look at the HAC-adjustment for autocorrelation and heteroscedasticity of regression residuals
# Consider HP-BIP and M-SSA predictor
mplot<-scale(cbind(target_shifted_mat[,k],indicator_mat[,j]))
# Correlation: quite large (at least for a one-year ahead forecast)
cor(na.exclude(mplot))
# Regress M-SSA on HP-BIP  
lm_obj<-lm(mplot[,1]~mplot[,2])
# OLS statistics: strongly significant (in accordance with large correlation)
summary(lm_obj)
# We can replicate the OLS t-statistics as follows
sd<-sqrt(diag(vcov(lm_obj)))
lm_obj$coef/sd
# We can now compare to HAC adjustment
# This is the HAC adjusted standard error: it is nearly twice as large as the OLS estimate above  
sd_HAC<-sqrt(diag(vcovHAC(lm_obj)))
sd_HAC
# The HAC-adjusted t-statistics is then nearly one half in size (compared to OLS) 
t_HAC<-lm_obj$coef/sd_HAC
t_HAC
# Accordingly, the p-values are larger
p_value<-2*pt(t_HAC, len-length(select_vec_multi), lower=FALSE)
p_value 
# So the HAC-adjustment leads to weaker statistical significance despite stronger correlation when targeting HP-BIP 








# Summary: transitioning from lambda_HP=160 (mildly adaptive) to lambda_HP=16 (adaptive) reverts the 
#       ordering of significance at the one-year ahead forecast horizon:
#   -The more adaptive design is better at forecasting BIP
#   -The mildly adaptive design is better at forecasting HP-BIP
