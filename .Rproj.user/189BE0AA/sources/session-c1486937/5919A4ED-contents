# Tutorial 7.3: propose various M-SSA BIP (German GDP) predictor designs
# The concept of M-SSA predictors for BIP was introduced in tutorial 7.2
# We packed this proceeding into a single function to be able to analyze various M-SSA BIP predictor designs (hyperparameters)
# We here present various predictor designs and you might be able to find better hyperparameters than ours below

# Main purposes of this tutorial
# -Illustrate M-SSA as applied to real data (in contrast to tutorial 7.1, based on simulated data)
#   -The application considers nowcasting and forecasting of German GDP (BIP) based on a set of well-known indicators
# -For the sake of interest, we shall consider forecast horizons of up to 6 quarters (one and a half year)
#   -Performances of institutional forecasters (`big five' German forecast institutes) degrade steeply 
#     beyond a one quarter forecast horizon, see up-coming publication by Heinisch and Neufing (currently working paper)
#   -We here illustrate that BIP can possibly be predicted consistently beyond half a year ahead
#   -Our main emphasize in this tutorial is the mid-term predictability: 2-6 quarters ahead
#   -Institutional forecasters are very good at nowcasting GDP: indeed, much better than M-SSA presented here
#   -But M-SSA as proposed in this tutorial could possibly provide additional insights into the prospect of mid-term GDP/BIP forecasting
# -In addition to forecasting BIP we also consider nowcasting and forecasting of the trend-growth of BIP
#   -For this purpose we apply a HP-filter to the differenced (and log-transformed) BIP
#   -This trend-growth component is termed HP-BIP
# -Forecast performance measures:
#   -We shall consider forecast performances of M-SSA against forward-shifted HP-BIP and BIP based on
#     -target correlations: correlations of predictors with forward-shifted BIP or HP-BIP
#     -rRMSE: relative root mean-square error when benchmarked against classic direct predictors or mean(BIP) (simple benchmark)
#     -HAC-adjusted p-values of t-statistics of regressions of predictors on target (HAC adjustment can account for autocorrelation and heteroscedasticity of regression residuals)

# To do: provide additional Diebold-Mariano (DM) and Giacomini-White (GW) tests of unequal predictability (benchmarked against mean(BIP))


# Start with a clean sheet
rm(list=ls())

# Load the required R-libraries
# Standard filter package
library(mFilter)
# Multivariate time series: VARMA model for macro indicators: used here for simulation purposes only
library(MTS)
# HAC estimate of standard deviations in the presence of autocorrelation and heteroscedasticity
library(sandwich)
# Extended time series
library(xts)
# Library for Diebold-Mariano test of equal forecast performance
library(multDM)


# Load the relevant M-SSA functionalities
# M-SSA functions
source(paste(getwd(),"/R/functions_MSSA.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))
# Utility functions for M-SSA, see tutorial 
source(paste(getwd(),"/R/M_SSA_utility_functions.r",sep=""))
# Set of performance metrics and tests of unequal predictability
source(paste(getwd(),"/R/performance_statistics_functions.r",sep=""))


#------------------------------------------------------------------------
# Load data and select indicators: see tutorial 7.2 for background
load(file=paste(getwd(),"\\Data\\macro",sep=""))
tail(data)
lag_vec<-c(2,rep(0,ncol(data)-1))
# -We assume a publication lag of two quarters for BIP (the effective lag is smaller but we'd like to stay on the safe side, in particular since BIP is subject to revisions)
#     -Therefore the target column (first column) in the above data file is up-shifted by two quarters as compared to the second column (BIP)
# -In general, we shall apply a two-sided HP to the target column: this is called HP-BIP
# -The challenge (and purpose of M-SSA) then consists in nowcasting or forecasting HP-BIP based on the 
#     explanatory variables as listed in columns 2-8 (available data at each time point, ignoring revisions)
#   -The two-quarter publication lag is too large (in practice it is one quarter). 
#   -But GDP/BIP is subject to revisions
#   -Since we ignore data revisions, we allow for a larger publication lag: in principle we discard the first (noisiest) release of GDP.



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
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
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
#   -Performance measures: 
#     -Correlations with shifted HP-BIP or BIP
#     -HAC-adjusted p-values of regressions of predictors on shifted HP-BIP and BIP: to assess statistical significance

# In order to use the function, we need to specify hyperparameters, see tutorial 7.2 for background
# We here first replicate tutorial 7.2

# Target filter: lambda_HP is the single most important hyperparameter, see tutorial 7.1 for a discussion
# Briefly: we avoid the classic quarterly setting lambda_HP=1600 because the resulting filter would be too smooth
# Too smooth means: the forecast horizon would have nearly no effect on the M-SSA predictor (almost no left-shift, no anticipation)
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
h_vec<-0:6
# Forecast excesses: see tutorial 7.1 for background
f_excess<-c(4,2)

# Run the function packing and implementing our previous findings (tutorial 7.2) 
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess)


# Forward-shifted HP-BIP
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
# M-SSA indicators
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
# M-MSE
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat

# For the direct predictor we can specify the macro-indicators in the expanding-window regressions
#   -Note: too complex designs lead to overfitting and thus worse out-of-sample performances
select_direct_indicator<-c("ifo_c","ESI")

perf_obj<-compute_perf_func(x_mat,target_shifted_mat,predictor_mssa_mat,predictor_mmse_mat,date_to_fit,select_direct_indicator) 
  
p_value_HAC_HP_BIP_full=perf_obj$p_value_HAC_HP_BIP_full
t_HAC_HP_BIP_full=perf_obj$t_HAC_HP_BIP_full
cor_mat_HP_BIP_full=perf_obj$cor_mat_HP_BIP_full
p_value_HAC_HP_BIP_oos=perf_obj$p_value_HAC_HP_BIP_oos
t_HAC_HP_BIP_oos=perf_obj$t_HAC_HP_BIP_oos
cor_mat_HP_BIP_oos=perf_obj$cor_mat_HP_BIP_oos
p_value_HAC_BIP_full=perf_obj$p_value_HAC_BIP_full
t_HAC_BIP_full=perf_obj$t_HAC_BIP_full
cor_mat_BIP_full=perf_obj$cor_mat_BIP_full
p_value_HAC_BIP_oos=perf_obj$p_value_HAC_BIP_oos
t_HAC_BIP_oos=perf_obj$t_HAC_BIP_oos
cor_mat_BIP_oos=perf_obj$cor_mat_BIP_oos
rRMSE_MSSA_HP_BIP_direct=perf_obj$rRMSE_MSSA_HP_BIP_direct
rRMSE_MSSA_HP_BIP_mean=perf_obj$rRMSE_MSSA_HP_BIP_mean
rRMSE_MSSA_BIP_direct=perf_obj$rRMSE_MSSA_BIP_direct
rRMSE_MSSA_BIP_mean=perf_obj$rRMSE_MSSA_BIP_mean
target_BIP_mat=perf_obj$target_BIP_mat

# Correlations between M-SSA predictors and forward-shifted HP-BIP (including the publication lag)
#   -We see that for increasing forward-shift (from top to bottom) the predictors optimized for 
#     larger forecast horizons (from left to right) tend to perform better
# 1. Full sample
cor_mat_HP_BIP_full
# 2. Out-of-sample (period following estimation span for VAR-model of M-SSA)
cor_mat_HP_BIP_oos

# Let's visualize these correlations by plotting target against predictor
# Select a forward-shift of target (the k-th entry in h_vec)
k<-5
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
mplot<-scale(cbind(target_BIP_mat[,k],target_shifted_mat[,k],predictor_mssa_mat[,j]))
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
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()

# Sample correlation: 
cor(na.exclude(mplot))[2,ncol(mplot)]
# This number corresponds to element (k,j) of the matrix cor_mat_HP_BIP_full computed by our function
cor_mat_HP_BIP_full[k,j]

# We can also look at the correlation of the predictor with the forward-shifted BIP (instead of HP-BIP)
cor(na.exclude(mplot))[1,3]
# Note: the (k,j) entry of cor_mat_BIP generally differs from cor(na.exclude(mplot))[1,3]
cor_mat_BIP_full[k,j]
# The reason is simple: by removing NAs (due to inclusion of the two-sided target in mplot) we change the sample-size
# We can easily correct by removing the two-sided target in mplot
mplot_without_two_sided<-scale(cbind(target_BIP_mat[,k],predictor_mssa_mat[,j]))
# This number now matches cor_mat_BIP[k,j]
cor(na.exclude(mplot_without_two_sided))[1,2]

# Assume one selects k=j=5 (one-year ahead) in the above plot (you might want to have a look at k=5 and j=7, too):
# Then the (weak) positive correlation between M-SSA and shifted BIP might suggest a (weak) predictability one year ahead
#    (including the publication lag) 
# Is this (weak) effect statistically significant?
# Let's have a look at the HAC-adjusted p-values
# 1. Full sample 
p_value_HAC_BIP_full[k,j]
# 2. Out-of-sample
p_value_HAC_BIP_oos[k,j]

# Instead of BIP we might have a look at targeting HP-BIP instead (also shifted one year ahead)
p_value_HAC_HP_BIP_full[k,j]
p_value_HAC_HP_BIP_oos[k,j]


#--------------------------------
# Findings: 
#   -Statistical significance is stronger for shifted HP-BIP (than for BIP)
#   -Is it because mid- and short-term components of BIP are effectively unpredictable?
#   -Or is it because lambda_HP=160 is not sufficiently adaptive to track mid/short-term dynamics (still too smooth)?
# To find an answer we now propose a more adaptive design based on lambda_HP=16
#   -We then check whether BIP can be predicted more consistently


################################################################################################################
################################################################################################################
# Let's now analyze a more adaptive design by selecting a smaller lambda_HP: see above for motivation
# We then verify if the more flexible design is able to predict BIP more consistently 

lambda_HP<-16
# Everything else in the above design is kept fixed
# Notes: 
#   -Keeping the above settings fixed is probably a bad idea because the `faster` filters (less smoothing required 
#       for lambda_HP=16) most likely do not require additional `acceleration' by the forecast excesses 
#   -You might try smaller values for f_excess
f_excess_adaptive<-f_excess

# Run the M-SSA predictor function
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess_adaptive)

# Forward-shifted HP-BIP
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
# M-SSA indicators
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
# M-MSE
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat

# For the direct predictor we can specify the macro-indicators in the expanding-window regressions
#   -Note: too complex designs lead to overfitting and thus worse out-of-sample performances
select_direct_indicator<-c("ifo_c","ESI")

perf_obj<-compute_perf_func(x_mat,target_shifted_mat,predictor_mssa_mat,predictor_mmse_mat,date_to_fit,select_direct_indicator) 

p_value_HAC_HP_BIP_full=perf_obj$p_value_HAC_HP_BIP_full
t_HAC_HP_BIP_full=perf_obj$t_HAC_HP_BIP_full
cor_mat_HP_BIP_full=perf_obj$cor_mat_HP_BIP_full
p_value_HAC_HP_BIP_oos=perf_obj$p_value_HAC_HP_BIP_oos
t_HAC_HP_BIP_oos=perf_obj$t_HAC_HP_BIP_oos
cor_mat_HP_BIP_oos=perf_obj$cor_mat_HP_BIP_oos
p_value_HAC_BIP_full=perf_obj$p_value_HAC_BIP_full
t_HAC_BIP_full=perf_obj$t_HAC_BIP_full
cor_mat_BIP_full=perf_obj$cor_mat_BIP_full
p_value_HAC_BIP_oos=perf_obj$p_value_HAC_BIP_oos
t_HAC_BIP_oos=perf_obj$t_HAC_BIP_oos
cor_mat_BIP_oos=perf_obj$cor_mat_BIP_oos
rRMSE_MSSA_HP_BIP_direct=perf_obj$rRMSE_MSSA_HP_BIP_direct
rRMSE_MSSA_HP_BIP_mean=perf_obj$rRMSE_MSSA_HP_BIP_mean
rRMSE_MSSA_BIP_direct=perf_obj$rRMSE_MSSA_BIP_direct
rRMSE_MSSA_BIP_mean=perf_obj$rRMSE_MSSA_BIP_mean
target_BIP_mat=perf_obj$target_BIP_mat

# We now examine performances targeting BIP (not HP-BIP)
# 1. Full sample
cor_mat_BIP_full
# 2. Out-of-sample (period following estimation span for VAR-model of M-SSA)
cor_mat_BIP_oos



# Are the results significant?
p_value_HAC_BIP_full
p_value_HAC_BIP_oos



#---------------------------------------------
# Findings overall:

# A. Classic direct predictors:
#   -Classic direct predictors generally do not perform better (out-of-sample) than the simple mean benchmark at 
#     forward-shifts exceeding 2 quarters
#   -Classic direct predictors are more sensitive (than M-SSA) to episodes subject to unusual singular readings (e,g,, the Pandemic)

# B. M-SSA
#   -Classic business-cycle designs (lambda_HP=1600) smooth out recessions and hide  
#     dynamics potentially relevant in a short- to mid-term forecast exercise (1-6 quarters ahead)
#   -Fairly adaptive designs (lambda_HP=160) show a (logically and) statistically consistent forecast pattern, 
#       suggesting that M-SSA outperforms both the mean and the direct forecasts out-of-sample when targeting HP-BIP
#     -This result suggest that M-SSA is also informative about forward-shifted BIP, although corresponding 
#       performance statistics are less conclusive (due to noise)
#   -More adaptive designs (lambda_HP=16) seem to be able to track forward-shifted BIP (more) consistently, 
#     by allowing the (more) flexible trend-component to provide (more) overlap with relevant mid- and high-frequency 
#     components of BIP

