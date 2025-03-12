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
#     -rRMSE (on to do list): relative root mean-square error when benchmarked against classic direct predictors or mean(BIP) (simple benchmark)
#     -HAC-adjusted p-values of t-statistics of regressions of predictors on target (HAC adjustment can account for autocorrelation and heteroscedasticity of regression residuals)
#     -Diebold-Mariano (DM) and Giacomini-White (GW) tests (on to do list) of unequal predictability (benchmarked against mean(BIP))

# To do: analyze performance of average forecasts (see end of file)


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
h_vec<-c(0,1,2,3,4,6)
# Forecast excesses: see tutorial 7.1 for background
f_excess<-c(4,2)

# Run the function packing and implementing our previous findings (tutorial 7.2) 
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess)

# Sample performances: target correlations and HAC-adjusted p-values for forward-shifted BIP and HP-BIP targets
#   -We replicate performances obtained in tutorial 7.2 
# 1. Target correlations of M-SSA predictors with forward-shifted BIP
cor_mat_BIP<-mssa_indicator_obj$cor_mat_BIP
# 2. Target correlation with forward-shifted HP-BIP 
cor_mat_HP_BIP<-mssa_indicator_obj$cor_mat
# HAC-adjusted p-Values of regressions of M-SSA predictors on forward-shifted BIP
p_value_HAC_mat_BIP<-mssa_indicator_obj$p_value_HAC_mat_BIP
# HAC-adjusted p-Values of regressions of M-SSA predictors on forward-shifted HP-BIP
p_value_HAC_mat_HP_BIP<-mssa_indicator_obj$p_value_HAC_mat
# Forward-shifted BIP
BIP_target_mat=mssa_indicator_obj$BIP_target_mat
# Forward-shifted HP-BIP
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
# M-SSA indicators
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
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
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

# Assume one selects k=j=4 (one-year ahead) in the above plot (you might want to have a look at k=4 and j=5, too):
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
f_excess_adaptive<-f_excess


# Run the M-SSA predictor function
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess_adaptive)

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
cor_mat_BIP
# In contrast to the previous setting lambda_HP=160, the new adaptive design based on lambda_HP=16  also leads to 
#   statistically significant predictors (with respect to BIP, not HP-BIP)
p_value_HAC_mat_BIP
# Findings: the more adaptive design based on lambda_HP=16 seems to be able to track future BIP better


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

# Target correlations with respect to forward-shifted BIP are now improved
cor_mat_HP_BIP[k,j]
cor_mat_BIP[k,j]
# We find statistical significance when targeting forward-shifted BIP
p_value_HAC_mat_HP_BIP[k,j]
p_value_HAC_mat_BIP[k,j]



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


#---------------------
# The following code is not working properly for the direct predictors!!!
# Ignore all results related to the latter!!!
# More extensive evaluation metrics: 
#   -Full-sample and out-of-sample 
#   -rRMSE, HAC adjusted p-values (the same as above),  DM and (HAC-) GW statistics
#     -rRMSE: relative root means-square forecast error (relative to simple mean benchmar)
#     -Reported p-Values of one-sided DM and GW tests evaluate whether M-SSA outperforms the simple mean(BIP) benchmark or the direct forecasts
# The function computes benchmark `direct predictors' and evaluates performances of M-SSA against direct predictors as well as 
#       the simple mean (mean of BIP) benchmark
#   -Direct predictors are simple OLS regressions of a selection of macro indicators on forward-shifted BIP
#   -The selection is specified in the vector select_direct_indicator below
# Note: the following functionality is currently under construction/test
# To conclude we can use the more general compute_all_perf_func function to obtain additional performance measures:

# 1. Select the indicators that we wish to use for predicting BIP `directly':
select_direct_indicator<-c("ifo_c","ESI")
select_direct_indicator<-select_vec_multi
# 2. Select any of the computed  M-SSA predictors
#     -Take the last one, i=6, which is optimized for forecast horizon h_vec[i] quarters ahead
i<-1
# Forecast horizon
h<-h_vec[i]
h
# M-SSA predictor
indicator_cal<-indicator_mat[,i]

# Call the performance evaluation function  
perf_obj<-compute_all_perf_func(indicator_cal,data,lag_vec,h_vec,h,select_direct_indicator,L,lambda_HP,date_to_fit)

# Here we have the rRMSE of the M-SSA predictor (first column) and the HAC-adjusted p-values (second column)
# This is for the full sample: in-sample and out-of-sample aggregated
# We can see that M-SSA outperforms significantly at the intended forward-shift  
perf_obj$mat_all
# We can compare the p-values (second column above) to the previously obtained p-values: they match perfectly
matrix(p_value_HAC_mat_BIP[,i],ncol=1)
# Same as above but out-of-sample only: once again, the M-SSA predictor seems to be informative for BIP at the 
#   forecast horizon for which M-SSA has been optimized (while intuitively appealing, this outcome is far of trivial given the noisy target)
perf_obj$mat_out
# Direct predictor benchmarks:
# Notes:
#   Note 1: the direct predictors do not depend on the selected forecast horizon h
#     -The forecast horizon h means that M-SSA has been optimized accordingly
#     -However, the direct predictors are computed within the above function, specifically for each forward-shift of BIP
#     -M-SSA depends on h but is independent of the forward-shift; direct predictors do not depend on h but on the forward-shift
#   Note 2: the direct predictors seem to perform best at shift 2
#     -Intuitively the predictors should perform best at shift 0 (nowcast)
#     -This counter intuitive outcome is mainly due to the singular outliers during the Pandemic (affect results despite trimming)
#   Note 3: direct predictors are sensitive to the singular Pandemic readings even for trimmed data:
#     -Performances tend to be substantially better when removing the Pandemic
#     -Performances tend to be somehow random, depending on the predictors hitting randomly the Covid-outliers 
#         in-phase our out-of-phase (depending on the forward-shift)
#   -In comparison, M-SSA tends to be less sensitive to Pandemic outliers
# -We can obtain the rRMSE and p-values of the direct predictors based on the indicators selected by select_direct_indicator above
# The direct forecasts generally perform poorly for shifts larger than 2 quarters
# Full sample: in-sample and out-of-sample aggregated
perf_obj$mat_all_direct
# Out-of-sample only: Adding regressors leads to worse performances (larger rRMSE)
perf_obj$mat_out_direct
# We also report p-values of one-sided DM and GW tests of unequal predictive ability
# Full sample
perf_obj$gw_dm_all_mat
# Out-of-sample only
perf_obj$gw_dm_out_mat
# Explanation:
#   -The first two columns are DM and GW tests verifying whether M-SSA performs better than mean(BIP), 
#        when targeting forward-shifted BIP (not HP-BIP)
#     -The M-SSA predictor optimized for forecast horizon 6 has the smallest p-values at the intended forward-shift of 6
#   -Columns 3 and 4 are DM and GW testing whether M-SSA performs better than mean(BIP) when targeting 
#       forward-shifted HP-BIP (not BIP) 
#     -The predictor optimized for forecast horizon 6 has small p-values at small and larger shifts (the latter is desirable, the former is random) 
#   -Columns 5 and 6 verify if the new benchmarks (classic direct predictors) outperform mean(BIP) when targeting BIP
#     -The direct predictor generally have a hard-time at forward-shifts larger than 2 quarters (see also see up-coming publication by Heinisch and Neufing (currently working paper))
#   -Finally, columns 7 and 8 test whether M-SSA outperforms the direct predictors when targeting BIP
#     -p-values smaller than 0.5 indicate relative outperformance of M-SSA
# Full sample: in-sample and out-of-sample aggregated
# In summary: DM and GW statistics suggest predictability in the following terms:  
#     -Significant test statistics at the intended forward-shift of BIP, out-of-sample 
#     -Outperformance against mean(BIP), see the last row, first two columns
#     -Outperformance against the direct predictors, see the last row, last two columns
# Note:
# 1. These findings refer to M-SSA optimized for the largest forecast horizon of 6 quarters
#     -Similar results are obtained for shorter forecast horizons
# 2. HAC adjustment is used for GW statistics

  



##########################################################################################
# Summary: transitioning from lambda_HP=160 (mildly adaptive) to lambda_HP=16 (adaptive) reverts the 
#       ordering of significance at the one-year ahead forecast horizon:
#   -The more adaptive design is better at forecasting BIP
#   -The mildly adaptive design is better at forecasting HP-BIP
# But we might be tempted to look at an even more adaptive design
#################################################################################################
# very adaptive design
# We now select a (very) small lambda_HP

lambda_HP<-2
# Everything else in the above design is kept fixed except f_excess
#   -Imposing positive excess-forecasts allows for a more pronounced left-shift in the case of smooth 
#       (not excessively adaptive) targets
#   -However, for very adaptive targets, as we are looking at here, a positive forecast-excess may lead to phase reversion: too much anticipation
# Since the target here is very adaptive, we impose zero forecast-excesses (you might try fine-tuning this hyperparameter) 
f_excess_very_adaptive<-rep(0,ncol(x_mat))

# Run the M-SSA predictor function
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess_very_adaptive)

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

# Sample correlation HP-BIP: this corresponds to cor_mat_BIP computed by our function
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


###########################################################################################
# Summary: 
# -In comparison to the adaptive design (lambda_HP=16) the very adaptive one (lambda_HP=2) did not 
#       provide further improvements when targeting forward-shifted BIP
#   -Both designs hover at roughly 0.2 target correlation at a one-year ahead forecast horizon (noting the additional publication lag of two quarters)
# -There are natural limitations when predicting a `noisy' series since the high-frequency portion is essentially unpredictable
# -The above results hint at a possible separation of predictable and non-predictable portions of the spectrum of BIP
#   -We could look at pass- and stop-bands of the HP(16): the stopband is a first guess of the unpredictable part  
# -While results did not improve by considering more adaptive designs, the forecasts differ qualitatively
#   -Having equivalent (in terms of target correlations) but different forecasts might suggest averaging
# -To do: analyze performance of average forecasts obtained by equal-weighting of lambda_HP=16 and lambda_HP=2 M-SSA predictors
#################################################################################################


