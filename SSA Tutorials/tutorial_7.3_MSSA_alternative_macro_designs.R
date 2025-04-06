# Tutorial 7.3: we propose various M-SSA BIP (GDP Germany) predictor designs
# The concept of M-SSA predictors for BIP was introduced in tutorial 7.2
# We wrapped this proceeding into a single function to be able to analyze various M-SSA BIP predictor designs (hyperparameters)
# We here propose a `fairly adaptive' predictor in exercise 1 and a `more adaptive' one in exercise 3
#   -One might be able to find better hyperparameters by fine-tuning adaptivity further

# Main purposes of this tutorial
# -Illustrate M-SSA as applied to real data (in contrast to tutorial 7.1, based on simulated data)
#   -The application considers nowcasting and forecasting of German GDP (BIP) based on a set of well-known indicators
# -For the sake of interest, we shall consider forecast horizons of up to 6 quarters (one and a half year)
#   -Performances of institutional forecasters (`big five' German forecast institutes) degrade steeply 
#     beyond a one quarter forecast horizon, see up-coming publication by Heinisch and Neufing (currently working paper)
#   -We here illustrate that BIP can possibly be predicted consistently beyond half a year ahead
#     -We emphasize mid-term predictability: 2-6 quarters ahead
#   -Institutional forecasters are very good at nowcasting GDP: indeed, much better than M-SSA presented here
#     -For this purpose they rely on a rich cross-section (many series) and mixed-frequency approaches, linking monthly and quarterly data
#     -In contrast, M-SSA proposed in this tutorial considers few (important) indicators, within a purely quarterly scheme 
#   -Tutorial 7.3 could eventually provide additional insights into the important prospect of mid-term GDP/BIP forecasting
# -In addition to forecasting BIP, we also consider nowcasting and forecasting of the trend-growth of BIP
#   -For this purpose we apply a HP-filter to the differenced (and log-transformed) BIP
#     -This trend-growth component is termed HP-BIP
# -Forecast performance measures:
#   -We shall consider forecast performances of M-SSA against forward-shifted HP-BIP and BIP based on
#     -The target correlation: correlation of predictor with forward-shifted BIP or HP-BIP
#     -The rRMSE: relative root mean-square error when benchmarked against classic direct predictors or mean(BIP) (simple benchmark)
#     -HAC-adjusted p-values of (t-statistics of) regressions of predictors on targets (HAC adjustment can account for autocorrelation and heteroscedasticity of regression residuals)
# To do: provide additional Diebold-Mariano (DM) and Giacomini-White (GW) tests of unequal predictability (benchmarked against mean(BIP))

# The tutorial is structured into five exercises
# Exercise 1: apply a fairly adaptive design based on targeting a HP(160) filter by M-SSA
#   -HP(160) deviates from the standard HP(1600) specification typically recommended for quarterly data
#     -See a critic by Phillips and Jin (2021), suggesting that HP(1600) is `too smooth' (insufficiently flexible)
#     -See also the lengthy discussion in tutorial 7.2 
#     -See also exercise 4 below
#   -We shall see that M-SSA can predict HP-BIP (for which it is explicitly optimized) consistently 
#      multiple quarters ahead (statistical significance)
#   -It is more difficult to predict BIP, though: the noisy high-frequency components of BIP are unpredictable 
# Exercise 2: apply M-SSA to white noise data to verify that the proposed performance measures and tests confirm unpredictability
#   -We shall see that the HAC-adjustment cannot fully account for all data idiosyncrasies
#   -However, empirical significance levels do not appear to be strongly biased
# Exercise 3: the proposed M-SSA predictor emphasizes dynamic changes of the trend growth-rate of BIP. It is 
#   not designed explicitly for mean-square forecast performances when targeting BIP. Therefore, exercise 3 proposes
#   to rely on the (sub-) components of the M-SSA predictor when targeting BIP explicitly. We shall see that the 
#   corresponding predictor is able to outperform the classic mean-benchmark as well as the direct forecasts in 
#   terms of out-sample MSE performances up to multiple quarters ahead.
# Exercise 4: analyze a more adaptive M-SSA design based on targeting HP(16) by M-SSA
# Finally, exercise 5 briefly analyzes the classic HP(1600) as a target for M-SSA (negative example to corroborate earlier results)

#-------------------------------------
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
# GARCH model: for improving regression estimates
library(fGarch)



# Load the relevant M-SSA functionalities
# M-SSA functions
source(paste(getwd(),"/R/functions_MSSA.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))
# Utility functions for M-SSA, see tutorial 
source(paste(getwd(),"/R/M_SSA_utility_functions.r",sep=""))
# Set of performance metrics and tests of unequal predictability
#source(paste(getwd(),"/R/performance_statistics_functions.r",sep=""))


#------------------------------------------------------------------------
# Load data and select indicators: see tutorial 7.2 for background
load(file=paste(getwd(),"\\Data\\macro",sep=""))
tail(data)
lag_vec<-c(2,rep(0,ncol(data)-1))
# Note: we assume a publication lag of two quarters for BIP, see the discussion in tutorial 7.2

# Plot the data
# The real-time BIP (red) is lagging the target (black) by lag_vec[1] quarters (publication lag)
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

###############################################################################################
# Exercise 1: Compute forecasts for German GDP up to 6 quarters ahead, based on the above selection of 
#   macro-indicators
# -Rely on a `fairly adaptive' HP(160) as the target, based on lambda_HP=160, 
#   -The standard HP(1600) is too smooth (smooths out recessions)
#   -But BIP (German GDP) is too noisy to be forecasted `directly'
#   -Finding a target that emphasizes the relevant dynamics while damping the unpredictable 
#     high-frequency noise is challenging
#   -HP(160) is possibly a good compromise: it damps high-frequency noise sufficiently strongly 
#     to let the `signal` (i.e. the predictable business-cycle part of BIP) appear more clearly to 
#     the forecast fitting-tool: M-SSA
# -In exercise 3 we shall rely on an even more adaptive design based on HP(16)

# 1.1 Apply M-SSA
# Here's the wrapper proposed in tutorial 7.2: 
head(compute_mssa_BIP_predictors_func)

# The head of the function needs the following specifications:
# -x_mat: data 
# -lambda_HP: HP parameter
# -L: filter length
# -date_to_fit: in-sample span for the VAR
# -p,q: model orders of the VAR
# -ht_mssa_vec: HT constraints (larger means less zero-crossings)
# -h_vec: (vector of) forecast horizon(s) for M-SSA
# -f_excess: forecast excesses, see tutorial 7.2, exercise 2 
# -lag_vec: publication lag (target is forward shifted by forecast horizon plus publication lag)
# -select_vec_multi: names of selected indicators

# We can supply various hyperparameters (designs) and the function returns M-SSA predictors as 
#     specified in tutorial 7.2
# -The main hyperparameter is lambda_HP: a smaller lambda_HP means increased adaptivity
# -Below, we shall also consider various settings for the forecast horizon (Timeliness) and the HT (smoothness) 

# We now first replicate tutorial 7.2 with the above wrapper

# Target filter: lambda_HP is the single most important hyperparameter, see tutorial 7.1 for a discussion
lambda_HP<-160
# Filter length: nearly 8 years is fine for the selected lambda_HP (filter weights decay sufficiently fast)
#   The length should be an odd number (see tutorial 7.1)
L<-31
# In-sample span for VAR, i.e., M-SSA (the proposed design is quite insensitive to this specification because the VAR is parsimoniously parameterized
date_to_fit<-"2008"
# VARMA model orders: keep the model simple in particular for short/tight in-sample spans
p<-1
q<-0
# Holding-times (HT): controls smoothness of M-SSA (the following numbers are pasted from the original predictor)
# Increasing these numbers leads to predictors with less zero-crossings (smoother), see tutorial 7.1
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
# Forecast horizons: M-SSA is optimized for each forecast horizon in h_vec 
h_vec<-0:6
# Forecast excesses: see tutorial 7.2, exercise 2 for background
f_excess<-rep(4,length(select_vec_multi))

# Run the wrapper  
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi)

# Retrieve predictors and targets from the above function-call
# Forward-shifted HP-BIP
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
# M-SSA predictors: for each forecast horizon h( the columns of predictor_mssa_mat), the M-SSA predictor is 
#   obtained as equally-weighted average of all series outputs: BIP, ip,..., see exercise 3 of tutorial 7.2
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
# M-MSE predictors
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat
# Sub-series of M-SSA predictors: for each forecast horizon h, the M-SSA predictor is obtained as equally-weighted 
#   average of all series outputs: BIP, ip,...
# The components or sub-series of the aggregate predictor are contained in mssa_array 
# This is a 3-dim array with dimensions: sub-series (BIP, ip,...), time, forecast horizon h in h_vec
#   -We shall see below that the sub-series can be important when interpreting the M-SSA predictors 
mssa_array<-mssa_indicator_obj$mssa_array

# Plot M-SSA: the vertical line indicates the end of the in-sample span
mplot<-predictor_mssa_mat
colnames(mplot)<-colnames(predictor_mssa_mat)
par(mfrow=c(1,1))
colo<-c(rainbow(ncol(predictor_mssa_mat)))
main_title<-c(paste("Standardized M-SSA predictors for forecast horizons ",paste(h_vec,collapse=","),sep=""),"Vertical line delimites in-sample and out-of-sample spans")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
abline(v=which(rownames(mplot)>date_to_fit)[1]-1,lty=2)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# How can we relate the above plot to the AST forecast trilemma?
# -We can see that M-SSA predictors are increasingly left-shifted with increasing forecast horizon, 
#     both in- as well as out-of-sample
#   -Timeliness aspect of the AST trilemma
# -Also, the number of zero-crossings is controlled by the holding-time (HT constraint)
#   -Smoothness aspect of the AST trilemma
# -The remaining A (for Accuracy) is examined below 

#----------------------------------------------------------
# 1.2. Compute performances
# 1.2.1 Specify the benchmarks against which M-SSA will be compared
#   -We compare M-SSA against the mean (of BIP) and against a classic direct forecast
#   -The direct forecast is based on regressing a selection of macro indicators on forward-shifted BIP

# We can specify the selection of macro-indicators for the direct forecast 
select_direct_indicator<-c("ifo_c","ESI")
# Note: too complex designs (too many indicators) lead to overfitting and thus worse out-of-sample performances
# To illustrate the direct predictor consider the following example of a h-step ahead direct forecast:
h<-2
# Shift BIP forward by publication lag+forecast horizon
forward_shifted_BIP<-c(x_mat[(1+lag_vec[1]+h):nrow(x_mat),"BIP"],rep(NA,h+lag_vec[1]))
# Regress selected indicators on forward-shifted BIP
lm_obj<-lm(forward_shifted_BIP~x_mat[,select_direct_indicator])
# You will probably not find statistically significant regressors for h>2: BIP is a noisy series
summary(lm_obj)
# Technical note: 
# -Residuals are subject to heteroscedasticity (crises) and autocorrelation
# -Therefore classic OLS tests for statistical significance are biased
# -We shall rely on HAC-adjusted p-values further down (R-package sandwich)

# Compute the predictor: one can rely on the generic R-function predict or compute the predictor manually
direct_forecast<-lm_obj$coef[1]+x_mat[,select_direct_indicator]%*%lm_obj$coef[2:(length(select_direct_indicator)+1)]
# Note that this is a full-sample predictor (no out-of-sample span)

# We can now plot target and direct forecast: for h>2 the predictor comes close to a flat line centered at zero 
ts.plot(cbind(forward_shifted_BIP,direct_forecast),main=paste("BIP shifted forward by ",h," quarters (black) vs. direct forecast (red)"),col=c("black","red"))
abline(h=0)

#------------
# 1.2.2 Performances
# The following function computes direct forecasts and evaluates M-SSA against mean(BIP) and direct forecasts
# -Performance measures:
#   -Target correlations (correlations of predictors with targets)
#     -Correlations emphasize the dynamic aspects of forecasts: they ignore static level and scale adjustments
#   -Relative root mean-square errors: potential gains of M-SSA over mean(BIP) and direct forecasts
#   -HAC-adjusted p-values: 
#     -we regress the predictor on forward-shifted BIP and compute the t-statistic 
#       of the regression coefficient
#     -Small p-values indicate statistical significance of the predictor
#     -Since target and predictor are subject to heteroscedasticity (COVID-outliers...) and autocorrelation
#       we here rely on HAC-adjusted p-values (R-package sandwich)
#   -We compute full-sample and out-of-sample results: the out-of-sample span is data after date_to_fit specified above
#   -We consider two different targets
#     a. Foreward-shifted HP applied to BIP: HP_BIP. This is the target for which M-SSA has been optimized
#     b. Forward-shifted BIP, i.e., we include the (unpredictable) noisy high-frequency part of BIP
# Note:
#   -M-SSA does not target BIP directly; overfitting is not (less) of concern

perf_obj<-compute_perf_func(x_mat,target_shifted_mat,predictor_mssa_mat,predictor_mmse_mat,date_to_fit,select_direct_indicator,h_vec) 

# Retrieve all performance measures
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

#------------
# 1.2.3 Evaluation
# We first look at the target correlations between M-SSA predictors and forward-shifted HP-BIP (including the publication lag)
# a. Full sample
cor_mat_HP_BIP_full
# We can recognize a systematic pattern: 
#   -For increasing forward-shift (from top to bottom in the above matrix) M-SSA designs optimized for 
#     larger forecast horizons h (from left to right) tend to perform better (larger correlations), 
#     until h is too large
#   -For a given row (forward-shift of target) the maximum correlations stays at (or close to) the 
#     diagonal element in this row

# b. Out-of-sample (period following estimation span for VAR-model of M-SSA)
cor_mat_HP_BIP_oos
# Similar to full-sample

# We now look at the target correlation when the target is forward-shifted BIP (instead of HP-BIP) 
cor_mat_BIP_full
cor_mat_BIP_oos
# Target correlations tend to be smaller (due to noise) and the previous systematic pattern is still recognizable
#   but attenuated (cluttered by noise) 


# Note: 
# -Out-of-sample correlations suggest that M-SSA predictors optimized for forecast horizons h>4 (the last two columns) 
#     correlate positively with forward-shifted BIP up to shifts of 4 quarters 
#   -Recall that we imposed a publication lag of 2 quarters to BIP, suggesting that a forward-shift of 3 quarters is close to a one-year ahead horizon
# -Are these results suggesting a systematic predictability of BIP one year ahead (plus the publication lag lag_vec[1])?
# -For answering this question we may look at HAC-adjusted p-values of regressions of out-of-sample predictors on 
#   forward-shifted BIP (see further down for details)
p_value_HAC_BIP_oos
# The last predictor, optimized for h=6, is on the edge of statistical significance up to a forward-shift=3 
#   -Note that the out-of-sample span is quite short here 

# We now look at pairwise comparisons with established benchmarks in terms of relative root mean-square 
#   errors: rRMSE
# rRMSE is the ratio of the root mean-Square forecast error (RMSE) of M-SSA over the RMSE of 
#   a. the mean of BIP or
#   b. the direct predictor obtained by simple regressions of the data (indicators) on forward-shifted BIP
# Remarks:
#   1. Since M-SSA predictors are standardized (equal-weighting cross-sectional aggregation), we need to 
#         calibrate them by regression onto the target (to determine static level and scale parameters)
#   2. Background:
#       -The M-SSA objective function is the target correlation (not the mean-square error) 
#       -Therefore, M-SSA ignores static level and scale adjustments
#   3. Root mean-square errors are evaluated on the out-of-sample span only (specified by date_to_fit)
#   5. The benchmark direct predictors are full-sample estimates 
#       -Estimates based on short in-sample spans are unreliable (insignificant regression coefficients)
#   6. The benchmark mean-predictor used in our comparisons is based on the mean of the target in the 
#       out-of-sample span (it is looking ahead)
#   7. The main purpose of these comparisons is to evaluate the dynamic capability of the M-SSA predictor out-of-sample
#       -Static level and scale adjustments are deemed less relevant

# With these remarks in mind let's begin:
# The first rRMSE emphasizes M-SSA vs. the mean benchmark (of BIP), both targeting HP-BIP: 
#   -Numbers smaller one signify an outperformance of M-SSA against the mean-benchmark when targeting HP-BIP
# All metrics are out-of-sample
rRMSE_MSSA_HP_BIP_mean
# We next look at M-SSA vs. direct predictors based on indicators selected  in select_direct_indicator: targeting HP-BIP
rRMSE_MSSA_HP_BIP_direct
# Next: M-SSA vs. mean (of BIP) when targeting BIP
rRMSE_MSSA_BIP_mean
# Finally: M-SSA vs. direct predictors based on indicators selected  in select_direct_indicator: targeting BIP
rRMSE_MSSA_BIP_direct
# We see similar systematic patterns as for the previous target correlations
#   -The systematic patterns are strong when targeting forward-shifted HP-BIP (for which M-SSA is explicitly optimized)
#   -For forward-shifted BIP the results are weaker (cluttered by noise)
# But we shall see below that a more adaptive design can improve performances slightly, see exercise 3


# The above correlations and rRMSE do not test for statistical significance (of predictability)
# The following HAC-adjusted p-values provide a way to infer statistical significance 
# We look at HAC-adjusted p-values of regressions of M-SSA on forward-shifted targets
# Remarks:
#   -In some cases the HAC standard error (of the regression coefficient) seems `suspicious' 
#     -HAC estimate of standard error could be substantially smaller than the ordinary OLS/unadjusted estimate
#   -We therefore compute both types of standard errors and we rely on the maximum for a derivation of p-values
#   -In this sense our p-values may be claimed to be `conservative'
# We first consider forward-shifted HP-BIP and full sample p-values
p_value_HAC_HP_BIP_full
# Out-of-sample: 
p_value_HAC_HP_BIP_oos
# We infer that the systematic patterns observed in target correlations and rRMSEs above are statistically significant
#   -M-SSA seems capable of predicting forward-shifted HP-BIP, i.e., the low-frequency (trend-growth) part of BIP




# Same as above but now targeting BIP
# Full sample
p_value_HAC_BIP_full
# We still see a systematic pattern from top to bottom and left to right but the overall picture 
#   is cluttered by noise 

# Out-of-sample: 
p_value_HAC_BIP_oos
# Note that weaker significance can be imputed, at least in part, to the shorter out-of-sample span (less observations)

#-------------------------------------------
# 1.3 Visualize performances: link performance measures to plots of predictors against targets
# 1.3.1 M-SSA predictors (without sub-series)
# Select a forward-shift of target (the k-th entry in h_vec)
k<-5
# This is the forecast horizon 
h_vec[k]
# To obtain the effective forward-shift we have to add the publication lag
shift<-h_vec[k]+lag_vec[1]
# Effective forward shift of BIP
shift
if (k>length(h_vec))
{
  print(paste("k should be smaller equal ",length(h_vec),sep=""))
  k<-length(h_vec)
}  
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
colnames(mplot)<-c(paste("BIP left-shifted by ",h_vec[k]," quarters (plus publication lag)",sep=""),paste("HP-BIP left-shifted by ",h_vec[k]," quarters (plus publication lag)",sep=""),paste("M-SSA predictor optimized for h=",h_vec[j],sep=""))
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


#------------------
# 1.3.2 We now relate the above plot to the performance measures
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

# Assume one selects k=j=5 (h_vec[5]=4: one-year ahead forecast) in the above plot:
# Then the (weak) positive correlation between M-SSA and shifted BIP might suggest a (weak) predictability one year ahead
# Is this (weak) effect statistically significant?
# Let's have a look at the HAC-adjusted p-values
# 1. Full sample 
p_value_HAC_BIP_full[k,j]
# 2. Out-of-sample
p_value_HAC_BIP_oos[k,j]
# Not significant at a one-year ahead horizon when targeting BIP (which is a very noisy series...)


# Instead of BIP we might have a look at targeting HP-BIP (also shifted one year ahead)
p_value_HAC_HP_BIP_full[k,j]
p_value_HAC_HP_BIP_oos[k,j]


# Findings: 
#   -Statistical significance is stronger for shifted HP-BIP (than for BIP)
# Questions:
#   -Is it because mid- and short-term components of BIP are effectively unpredictable?
#   -Or is it because lambda_HP=160 is not sufficiently adaptive to track mid/short-term dynamics (still too smooth)?
# To find an answer, we shall consider a more adaptive design based on targeting HP(16), see exercise 3 below


#---------------
# 1.4 Changing the HT
# -In the above exercises we addressed timeliness by the forecast horizon h_vec and the forecast excess f_excess
# -Here we briefly address smoothness by increasing the HT in the HT-constraint
# -This exercise can be viewed as a further validity or robustness check for the M-SSA predictor

# We impose twice as large HTs
ht_mssa_vec_long<-2*ht_mssa_vec
names(ht_mssa_vec_long)<-colnames(x_mat)
# Stronger smoothing might require longer filters 
#   -We here ignore this effect and set L_long=L (advantage of shorter filters: faster numerical computation)
L_long<-L

# Run the wrapper  
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L_long,date_to_fit,p,q,ht_mssa_vec_long,h_vec,f_excess,lag_vec,select_vec_multi)

target_shifted_mat=mssa_indicator_obj$target_shifted_mat
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat
mssa_array<-mssa_indicator_obj$mssa_array

# Plot 
mplot<-predictor_mssa_mat
colnames(mplot)<-colnames(predictor_mssa_mat)
par(mfrow=c(1,1))
colo<-c(rainbow(ncol(predictor_mssa_mat)))
main_title<-c(paste("Standardized M-SSA predictors for forecast horizons ",paste(h_vec,collapse=","),sep=""),"Vertical line delimites in-sample and out-of-sample spans")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
abline(v=which(rownames(mplot)>date_to_fit)[1]-1,lty=2)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# -We can observe increased smoothness of the predictors, as expected 
#   -The expected duration between consecutive zero-crossings should double (when compared to the previous results) 
#     on sufficiently long spans, assuming the VAR to be `true', see tutorial 7.1 for background
# -Overall, the general pattern of the predictors towards the sample end confirms earlier findings (based on smaller HTs)
# -But nowcast and short-term forecasts are more prudent with respect to up-turn dynamics


################################################################################################################
# Exercise 2
# Let's check what happens when we apply the above battery of tests and performance measures to white noise
#   -Target correlations should be small, rRMSEs should be close to one and p-values should be above 5%
# Note: 
# -We're looking into a multiple-test problem, since we consider 7*7=49 tests (p-values)
#   -We do not account/adjust for this problem. 
#   -But one should expect to see randomly significant results in the simultaneous 49 tests 
#     even if the data is white noise

# 2.1
# Generate artificial white noise data
# One can try multiple set.seed
# This one will generate multiple significant results (p<5%) for the out-of-sample span, but none below 1% 
set.seed(1)
# This one will generate only one p-value below 5% in the full sample span and none below 1% 
set.seed(2)
# None below 1%
set.seed(3)
# None below 1%
set.seed(4)
# None below 1%
set.seed(5)
# None below 1%
set.seed(9)
# None below 1%
set.seed(6)
# One single p-value below 1%
set.seed(7)
# None below 1%
set.seed(8)
# None below 1%
set.seed(9)
# None below 1%
set.seed(10)

# The outcome suggests that HAC-adjustments are unable to correct fully for data-dependence
#   -We observe p-values below 5% more often than in 5% of all cases
#   -Therefore, some care is needed when evaluating results on the verge of statistical significance
# However, our results also suggest that p-values below 1% are `rare` 
#   -We found only one p-value below 1% for the above set.seeds, out of 10*7*7*2~1000 computed values

x_mat_white_noise<-NULL
for (i in 1:ncol(x_mat))
  x_mat_white_noise<-cbind(x_mat_white_noise,rnorm(nrow(x_mat)))

# Provide colnames and rownames from x_mat: necessary because the function relies on dates and column names of selected indicators
rownames(x_mat_white_noise)<-rownames(x_mat)
colnames(x_mat_white_noise)<-colnames(x_mat)
tail(x_mat_white_noise)

# Check ACF
acf(x_mat_white_noise)


# 2.2 Apply M-SSA
# Target filter: lambda_HP is the single most important hyperparameter, see tutorial 7.1 for a discussion
# Briefly: we avoid the classic quarterly setting lambda_HP=1600 because the resulting filter would be too smooth
# Too smooth means: the forecast horizon would have nearly no effect on the M-SSA predictor (almost no left-shift, no anticipation)
lambda_HP<-160
# Filter length: nearly 8 years is fine for the selected lambda_HP (filter weights decay sufficiently fast)
L<-31
# In-sample span for VAR, i.e., M-SSA (the proposed design is quite insensitive to this specification because the VAR is parsimoniously parameterized)
date_to_fit<-"2008"
# VARMA model orders: keep the model simple in particular for short/tight in-sample spans
p<-0
q<-0
# Holding-times (HT): controls smoothness of M-SSA (the following numbers are pasted from the original predictor)
# Increasing these numbers leads to predictors with less zero-crossings (smoother)
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
# Forecast horizons: M-SSA is optimized for each forecast horizon in h_vec 
h_vec<-0:6
# Forecast excesses: see tutorial 7.1 for background
f_excess<-rep(4,length(select_vec_multi))

# Run the function packing and implementing our previous findings (tutorial 7.2) 
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat_white_noise,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi)


# 3.3 Check performances: 
# Forward-shifted HP-BIP
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
# M-SSA indicators
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
# M-MSE
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat

# For the direct predictor we can specify the macro-indicators in the expanding-window regressions
#   -Note: too complex designs lead to overfitting and thus worse out-of-sample performances
select_direct_indicator<-c("ifo_c","ESI")

perf_obj<-compute_perf_func(x_mat_white_noise,target_shifted_mat,predictor_mssa_mat,predictor_mmse_mat,date_to_fit,select_direct_indicator,h_vec) 

p_value_HAC_BIP_full=perf_obj$p_value_HAC_BIP_full
p_value_HAC_BIP_oos=perf_obj$p_value_HAC_BIP_oos


# We need to check whether we can forecast the white noise data WN (not HP-WN)  based on the predictors
# Full sample
p_value_HAC_BIP_full
# Out-of-sample: 
p_value_HAC_BIP_oos

# Check number of occurrences below 1%
length(which(p_value_HAC_BIP_full<0.01))
length(which(p_value_HAC_BIP_oos<0.01))

# Findings:
# -We did not account for the multiple-test problem
# -The above simulation experiment suggests that HAC-adjustments are unable to correct fully for the dependence
#   -Therefore, some care is needed when evaluating results on the verge of statistical significance
# -However, strongly significant results, such as found for HP-BIP, seem convincing, in a statistical sense, 
#   since occurrences with p-values below 1% are `rare' (one out of 500 observations) in the above experiment

# Discussion
# -HP applied to white noise (HP-WN) leads to an autocorrelated series which can be predicted
# -However, a predictor of HP-WN is not helpful in predicting WN
# -In contrast, we expect HP-BIP in exercise 1 (and 3 below) to be informative about future BIP because the 
#     latter is not WN:
#   -The occurrence of recessions (business-cycle) contradicts the WN assumption
#   -The acf and the VAR-model contradict the WN assumption
#   -The systematic structure in the performance matrices (top-down/left-right) suggests predictability
#   -The recent negative BIP-readings are not `random` events: they are due in part to exogenous shock-waves and 
#     also to endogenous decisions whose underlying(s) did not realize as wished for (self-critical assessments by the former `Ampel' minister of economic affairs are telling)
#   -These deviations of BIP from WN concern lower frequency components of the spectral decomposition, as 
#     emphasized by HP-BIP

##################################################################################
# Exercise 3 Working with (M-SSA) BIP predictor (sub-)components
# -Background: the M-SSA predictor (the matrix predictor_mssa_mat) is constructed from components (the array mssa_array)
# -We here briefly illustrate the construction principle: see exercise 3.1. 
# -Subsequently, we suggest that predictor components support additional information that can be exploited.
# -Specifically, we address interpretability, see exercise 3.2, and MSE forecast performances, see exercise 3.3
#   -Recall that the M-SSA predictor is designed to address dynamic changes (up-/downturns, target correlation, smoothness)
#   -Therefore, MSE performances are deemed less relevant, in particular when targeting BIP (instead of HP-BIP)
#   -We shall see that BIP predictor components can be used to address more explicitly MSE performances

# To start, let us intialize all settings as in exercise 1 above
lambda_HP<-160
L<-31
date_to_fit<-"2008"
p<-1
q<-0
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
h_vec<-0:6
f_excess<-rep(4,length(select_vec_multi))

# Run the wrapper  
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi)

target_shifted_mat=mssa_indicator_obj$target_shifted_mat
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat
mssa_array<-mssa_indicator_obj$mssa_array




# 3.1 What are BIP predictor (sub-)components?
# -The M-SSA BIP predictor was introduced in tutorial 7.2, exercise 3
#   -The BIP predictor is obtained as the equally-weighted aggregate of standardized M-SSA outputs of all indicators (BIP, ip, ifo, ESI, spread)
#   -For illustration, we here briefly replicate the predictor, as based on its components. 
#   -For this purpose, we select the M-SSA nowcast
j_now<-1
# This is the forecast horizon (nowcast)
h_vec[j_now]
# For forecast horizon h_vec[j_now], the sub-series of the M-SSA predictor are:  
tail(t(mssa_array[,,j_now]))
# These sub-series correspond to the outputs of the multivariate M-SSA optimized for horizon h_vec[j_now]
#   -One output for each series of the multivariate design
# We can check that the M-SSA predictor is the cross-sectional mean of the standardized sub-series: 
agg_std_comp<-apply(scale(t(mssa_array[,,j_now])),1,mean)

mplot<-cbind(agg_std_comp, predictor_mssa_mat[,j_now])
rownames(mplot)<-rownames(x_mat)
colnames(mplot)<-c("Cross-sectional mean of standardized predictor components","M-SSA predictor")
colo<-c("blue",rainbow(length(select_vec_multi)))
main_title<-"Replication of M-SSA predictor, based on its components"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=2,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=2)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()

# Both series are identical
# Alternative check: the maximal error/deviation should be `small' (zero up to numerical precision)
max(abs(apply(scale(t(mssa_array[,,j_now])),1,mean)-predictor_mssa_mat[,j_now]),na.rm=T)



#---------------
# 3.2 We now exploit the components for a better interpretation of the M-SSA predictor.
#   -We can examine which sub-series is (are) more/less likely to trigger a dynamic change of the predictor/nowcast
#   -For illustration, we select the M-SSA nowcast

# Plot M-SSA nowcast and sub-series
par(mfrow=c(1,1))
# Scale the data 
mplot<-scale(cbind(predictor_mssa_mat[,j_now],scale(t(mssa_array[,,j_now]))))
rownames(mplot)<-rownames(x_mat)
colnames(mplot)<-c(paste("M-SSA predictor optimized for h=",h_vec[j_now],sep=""),
                   paste("Subseries ",select_vec_multi,sep=""))
colo<-c("blue",rainbow(length(select_vec_multi)))
main_title<-c(paste("M-SSA predictor for h=",h_vec[j_now]," (solid blue) and sub-series (dashed lines)",sep=""),"In-sample span up to black vertical (dashed) line")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=2,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=2)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()
# Discussion:
# -All sub-series date the trough of the growth rate of the German economy in late 2023 
# -Currently (Jan-2025), the strongest positive dynamics are supported by the (leading) spread sub-series (violet dashed line)
# Notes:
# -The trough (minimum) of the grow-rate in the previous figure anticipates the trough of BIP by up to several quarters
# -The timing of the BIP-trough is sandwiched between the trough and the next zero-crossing of the growth-rate (recall that the zero-line corresponds to average growth)
# -Given that the nowcast just passed the zero-line, we infer that the 
#   trough of BIP might be behind, already (based on Jan-2025 data)
# -However, not all sub-series would support this claim
#   -The zero-crossing of the nowcast (solid line in the above plot) is triggered by the (leading) spread, mainly
#   -ifo and ESI are barely above the zero-line 
#   -ip and BIP are `waiting' for further evidence and confirmation
# -Looking at the sub-series can help when interpreting the predictor (explainability) 
# -Faint/fragile signals are sensitive to announced and/or unexpected disorders (tariffs, geopolitical contentions)
#   which are not yet `priced-in' (as of Jan-2025).

#-------------------------------------
# Exercise 3.3 Addressing MSE-performances
# -As stated above, the M-SSA predictor emphasizes dynamic changes (recessions/expansions); MSE performances are deemed less relevant
#   -In particular, the predictor is standardized: neither its level nor its scale are calibrated to BIP
# -In order to track BIP more explicitly, we may rely on the predictor (sub-)components in the previous figure
# -For this purpose, we regress the components on forward-shifted BIP
# -This is the same proceeding as for the direct forecasts, as illustrated in exercise 1.2.1, except that we rely on 
#   the M-SSA components as regressors (instead of the original indicators)

# 3.3.1 Selection
# -We can select the components which are deemed relevant for MSE performances
#   -ESI, ifo and spread are mainly relevant in a dynamic context (for the M-SSA predictor)
#   -In contrast, BIP and ip sub-components are more relevant in a MSE perspective that we here emphasize
#   -Note that ESI, ifo and spread are important in the determination of these two (BIP- and ip-) M-SSA components
sel_vec_pred<-select_vec_multi[c(1,2)]
sel_vec_pred
# We can select the forward shift of BIP: for illustration we assume a 2 quarters ahead forward-shift
shift<-2
# We can select the forecast horizon of the M-SSA components: we select a 4 quarters ahead horizon 
#   (below, we shall look all all combinations of shift and forecast horizon)
k<-5
# Check: k=5 corresponds to h_vec[k]=4, a one-year ahead horizon
h_vec[k]
# Define the data matrix for the regression
dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_pred,,k]))
rownames(dat)<-rownames(x_mat)
colnames(dat)<-c(paste("BIP shifted forward by lag_vec+shift=",shift+lag_vec[1],sep=""),
                    paste("Predictor component ",colnames(t(mssa_array[sel_vec_pred,,k])),": h=",h_vec[k],sep=""))
tail(dat)
# Remove NAs
dat<-na.exclude(dat)

#-----------------
# 3.3.2 Regression
# We now regress forward-shifted BIP (first column) on the components
#   For illustration we use data up to Q1-2017
i_time<-which(rownames(dat)>2017)[1]
tail(dat[1:i_time,])
# Regression
lm_obj<-lm(dat[1:i_time,1]~dat[1:i_time,2:ncol(dat)])
summary(lm_obj)
# Compute out-of-sample prediction for time point i+shift
oos_pred<-(lm_obj$coef[1]+lm_obj$coef[2:ncol(dat)]%*%dat[i_time+shift,2:ncol(dat)]) 
# Compute out-of-sample MSE
oos_error<-dat[i_time+shift,1]-oos_pred
# This is the out-of-sample error we observe shift=2 quarters ahead
oos_error

#---------------
# 3.3.3 Better regression
# Given that BIP is subject to heteroscedasticity we may apply a GARCH(1,1) to obtain an estimate of its variance
y.garch_11<-garchFit(~garch(1,1),data=dat[1:i_time,1],include.mean=T,trace=F)
summary(y.garch_11)
sigmat<-y.garch_11@sigma.t

# Plot BIP and its vola
par(mfrow=c(1,1))
# Scale the data 
mplot<-cbind(sigmat,dat[1:i_time,1])
rownames(mplot)<-names(dat[1:i_time,1])
colnames(mplot)<-c("GARCH-vola","BIP")
colo<-c("blue",rainbow(length(select_vec_multi)))
main_title<-"BIP and GARCH(1,1)-vola"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=2,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 1:ncol(mplot))
{
  lines(mplot[,i],col=colo[i],lwd=1,lty=2)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
abline(v=which(rownames(mplot)<=date_to_fit)[length(which(rownames(mplot)<=date_to_fit))],lwd=2,lty=2)
axis(1,at=c(1,4*1:(nrow(mplot)/4)),labels=rownames(mplot)[c(1,4*1:(nrow(mplot)/4))])
axis(2)
box()

# We can now apply weighted least-squares instead of OLS, using the (inverse of the) GARCH-vola for the weights 
weight<-1/sigmat^2
# Apply WLS instead of OLS
lm_obj<-lm(dat[1:i_time,1]~dat[1:i_time,2:ncol(dat)],weight=weight)
summary(lm_obj)
# Compute out-of-sample prediction for time point i+shift
oos_pred_wls<-(lm_obj$coef[1]+lm_obj$coef[2:ncol(dat)]%*%dat[i_time+shift,2:ncol(dat)]) 
# Compute out-of-sample MSE
oos_error_wls<-dat[i_time+shift,1]-oos_pred_wls
# This is the out-of-sample error we observe in shift=2 quarters later
oos_error_wls
# Compare to out-of-sample error based on OLS: 
oos_error
# WLS outperforms OLS out-of-sample. A more extensive analysis would show that WLS is better on average, 
#   i.e., over multiple time points

#------------
# 3.3.4 Apply the above findings to all data points after 2007 (entire financial crisis is out-of-sample)

start_fit<-"2007"
# Use WLS based on GARCH model when regressing explanatory on forward-shifted BIP
use_garch<-T
# The following function compute_component_predictors_func applies the above WLS-regression for all time points
#     after start_fit, based on an expanding window
#   -For each time point, a GARCH is fitted and the WLS-regression is re-computed based on the new GARCH-weights
# Note: we can ignore warnings which are generated by the GARCH estimation routine
perf_obj<-compute_component_predictors_func(dat,start_fit,use_garch,shift)

# The function computes HAC-adjusted p-values of the regression of the out-of-sample predictor (oos_pred_wls obtained in exercise 3.3.3 above)
#   on forward.shifted BIP
perf_obj$p_value
# The same but without singular Pandemic readings
perf_obj$p_value_without_covid
# We also obtain the out-of-sample MSE (mean of oos_error_wls^2, where oos_error_wls were obtained in exercise 3.3.3 above)
perf_obj$MSE_oos
# The same but without Pandemic
perf_obj$MSE_oos_without_covid
# The function also compute the out-of-sample MSE of the simple mean benchmark predictor: we expect the latter to 
#   be larger than for the M-SSA components above (at least for smaller forward-shifts of BIP)
perf_obj$MSE_mean_oos
perf_obj$MSE_mean_oos_without_covid

#----------------
# 3.3.5 Compute performances of M-SSA components as predictors for forward-shifted BIP
# -Performance metrics:
#   -In contrast to the M-SSA predictor in exercise 1, we here explicitly emphasize (out-of-sample) MSE performances
#   -Specifically, we compute HAC-adjusted p-values and rRMSEs when comparing M-SSA component predictors with 
#       the simple expanding mean as well as with the direct forecasts
#     -Direct forecasts are based on the more effective WLS-regression (in contrast to OLS regression in exercise 1 above)
# -We compute all combinations of forward-shift and forecast horizon (7*7 matrix of performance metrics)
# -All regressions rely on WLS, as based on the (inverse of the squared) GARCH(1,1)-vola
# -Computations may take several minutes (regressions and GARCH-models are recomputed anew for each time point)

p_mat_mssa<-p_mat_mssa_components<-p_mat_mssa_components_without_covid<-p_mat_direct<-rRMSE_mSSA_comp_direct<-rRMSE_mSSA_comp_mean<-rRMSE_mSSA_comp_direct_without_covid<-rRMSE_mSSA_comp_mean_without_covid<-matrix(ncol=length(h_vec),nrow=length(h_vec))
for (shift in h_vec)#shift<-1
{
  print(shift)
  
  for (j in h_vec)#j<-1
  {
    
    k<-j+1
    
# M-SSA components
# For a single predictor (vector) one does not have to rely on the transposition t(mssa_array[sel_vec_pred,,k])   
    if (length(sel_vec_pred)>1)
    {
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_pred,,k]))
    } else
    {
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mssa_array[sel_vec_pred,,k]))
    }
    
    rownames(dat)<-rownames(x_mat)
    dat<-na.exclude(dat)
    
    perf_obj<-compute_component_predictors_func(dat,start_fit,use_garch,shift)
    
    p_mat_mssa_components[shift+1,k]<-perf_obj$p_value
    p_mat_mssa_components_without_covid[shift+1,k]<-perf_obj$p_value_without_covid
    MSE_oos_mssa_comp<-perf_obj$MSE_oos
    MSE_oos_mssa_comp_without_covid<-perf_obj$MSE_oos_without_covid
    MSE_mean_oos<-perf_obj$MSE_mean_oos
    MSE_mean_oos_without_covid<-perf_obj$MSE_mean_oos_without_covid
    
# Direct forecasts
    dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),x_mat[,sel_vec_pred])
    rownames(dat)<-rownames(x_mat)
    dat<-na.exclude(dat)
    
    perf_obj<-compute_component_predictors_func(dat,start_fit,use_garch,shift)
    
    p_mat_direct[shift+1,k]<-perf_obj$p_value 
    MSE_oos_direct<-perf_obj$MSE_oos
    MSE_oos_direct_without_covid<-perf_obj$MSE_oos_without_covid
    
    rRMSE_mSSA_comp_direct[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_oos_direct)
    rRMSE_mSSA_comp_mean[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_mean_oos)
    rRMSE_mSSA_comp_direct_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_oos_direct_without_covid)
    rRMSE_mSSA_comp_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_mean_oos_without_covid)
  }
}

# Note: warnings are issued by GARCH estimation routine and can be ignored
# Assign column and rownames
colnames(p_mat_mssa_components)<-colnames(p_mat_direct)<-colnames(p_mat_mssa_components_without_covid)<-
  colnames(rRMSE_mSSA_comp_direct)<-colnames(rRMSE_mSSA_comp_mean)<-
  colnames(rRMSE_mSSA_comp_direct_without_covid)<-colnames(rRMSE_mSSA_comp_mean_without_covid)<-paste("h=",h_vec,sep="")
rownames(p_mat_mssa_components)<-rownames(p_mat_direct)<-rownames(p_mat_mssa_components_without_covid)<-
  rownames(rRMSE_mSSA_comp_direct)<-rownames(rRMSE_mSSA_comp_mean)<-
  rownames(rRMSE_mSSA_comp_direct_without_covid)<-rownames(rRMSE_mSSA_comp_mean_without_covid)<-paste("Shift=",h_vec,sep="")

# HAC-adjusted p-values of out-of-sample M-SSA components starting in start_fit=2007
p_mat_mssa_components
# Same but without Pandemic
p_mat_mssa_components_without_covid
# HAC-adjusted p-values of out-of-sample direct forecasts starting in start_fit=2007
p_mat_direct
# rRMSE of M-SSA components when benchmarked against mean, out-of-sample, all time points from start_fit=2007 onwards
rRMSE_mSSA_comp_mean
# rRMSE of M-SSA components when benchmarked against direct forecast, out-of-sample, all time points from start_fit=2007 onwards
rRMSE_mSSA_comp_direct
# Same but without Pandemic
rRMSE_mSSA_comp_direct_without_covid
rRMSE_mSSA_comp_mean_without_covid

# Summary (main findings)
# -The original M-SSA predictor (exercise 1) does not emphasize MSE-performances explicitly
# -For that purpose we can rely on the M-SSA components, underlying the construction of the M-SSA predictor
#   -The link between the M-SSA predictor and the components is illustrated in exercise 3.1, see also tutorial 7.2 (exercise 3)
#   -The components can be used for assessing and interpreting the M-SSA predictor, see exercise 3.2
#   -Addressing out-of-sample MSE performances is proposed in exercise 3.3
#     -Technical note: our results suggest that WLS regression dominates OLS, out-of-sample
# -Outcome (MSE performance gains): 
#   -The M-SSA components predictor outperforms significantly the simple mean as well as the direct forecasts 
#      (the latter also based on WLS regression) in terms of MSE-performances at forward-shifts of up to one year ahead
#     -At shifts larger than 4 quarters, p-values and rRMSEs seem to take a hit (in particular when excluding the Pandemic)
#   -These results hold irrespective of the singular Pandemic readings
#     -The Pandemic weakens efficiency gains and statistical significance but the overall picture remains roughly the same
#   -M-SSA designs optimized for larger forecast horizons tend to perform better 
#     -Possible explanation: the selected BIP- and ip- (M-SSA) components are targeting (slightly) lagging series
#     -The lags of the targets can be compensated by larger forecast horizons


#######################################################################################
# Exercise 4: More adaptive design
# -Exercise 1 above confirms that we can predict HP-BIP several quarters ahead by the M-SSA predictor
# -But predicting BIP is more challenging: for this purpose exercise 3 proposed an analysis of M-SSA components
# -However, we cannot exclude (a priori) that the target specification, HP(160), as specified in the previous 
#     exercises, is `too smooth'
#   -Maybe (too much) relevant information has been suppressed?
# -To verify this conjecture, we now analyze a more adaptive design by selecting lambda_HP=16 `small`

# Question: does the more flexible design allow to predict BIP (not HP-BIP) more reliably?

# 4.1 Run more adaptive M-SSA design
lambda_HP<-16
# Notes: 
# -For adaptive designs, a pronounced left-shift might lead to phase-reversal which is undesirable
# -Therefore we use forecast horizons up to 4 quarters (instead of 6) and no forecast excess
#   -Phase-reversal would be fine (optimal) if the data were in agreement with the implicit assumptions 
#     underlying the HP filter (which is not the case, see tutorial 2.0)
f_excess_adaptive<-rep(0,length(select_vec_multi))
h_vec_adaptive<-0:4

# Run the M-SSA predictor function
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec_adaptive,f_excess_adaptive,lag_vec,select_vec_multi)

# Forward-shifted HP-BIP
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
# M-SSA indicators
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
# M-MSE
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat

# 4.2 Compute performances
# For the direct predictor we can specify the macro-indicators in the expanding-window regressions
#   -Note: too complex designs lead to overfitting and thus worse out-of-sample performances
select_direct_indicator<-c("ifo_c","ESI")

perf_obj<-compute_perf_func(x_mat,target_shifted_mat,predictor_mssa_mat,predictor_mmse_mat,date_to_fit,select_direct_indicator,h_vec_adaptive) 

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

# We now examine performances when targeting specifically BIP (we already know that HP-BIP can be predicted)
# 1. Full sample
cor_mat_BIP_full
# 2. Out-of-sample (specified by date_to_fit above)
cor_mat_BIP_oos
# These numbers are larger than in exercise 1, suggesting that the more adaptive design here is better able to
#   track forward-shifted BIP

# Are the results significant?
p_value_HAC_BIP_full
p_value_HAC_BIP_oos

# We can see a slight improvement (smaller p-values) when using a more adaptive target HP(16) for M-SSA

# Notes: 
# -Slightly more adaptive designs (smaller lambda_HP) or more aggressive settings  (larger forecast excess) 
#   can improve performances further
# -Danger of data-snooping
# -Danger of phase-reversal 

##################################################################################
# Exercise 5
# As a counterpoint to exercise 4 above, we now briefly analyze the rather inflexible classic
#   quarterly HP(1600) design as a target for M-SSA, to back-up our previous discussion 
#   with empirical facts

lambda_HP<-1600
# Given a slower decay (stronger smoothing), we may consider longer filters
L_long<-2*L-1
# Or just keep it fixed (faster numerical optimization if filters are shorter)
L_long<-L
# We also use the larger HTs of exercise 1.5
ht_mssa_vec_long<-ht_mssa_vec_long
# Run the wrapper  
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L_long,date_to_fit,p,q,ht_mssa_vec_long,h_vec,f_excess,lag_vec,select_vec_multi)

target_shifted_mat=mssa_indicator_obj$target_shifted_mat
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat
mssa_array<-mssa_indicator_obj$mssa_array

# Plot 
mplot<-predictor_mssa_mat
colnames(mplot)<-colnames(predictor_mssa_mat)
par(mfrow=c(1,1))
colo<-c(rainbow(ncol(predictor_mssa_mat)))
main_title<-c(paste("Standardized M-SSA predictors for forecast horizons ",paste(h_vec,collapse=","),sep=""),"Vertical line delimites in-sample and out-of-sample spans")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
abline(v=which(rownames(mplot)>date_to_fit)[1]-1,lty=2)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# Discussion:
# -The stronger smoothing by HP(1600) emphasizes long-term dynamics which are less relevant in a one-year forecast perspective
#   -See a critic by Phillips and Jin (2021), suggesting that HP(1600) is `too smooth' (insufficiently flexible)
# -As a result, increasing the forecast horizon has only marginal effects on the M-SSA predictor
#   -The left-shift of the M-SSA predictors is less pronounced
# -Stated otherwise: increasing the forecast horizon has only a marginal effect on the `phase' of the filter
#   -the right tail of the HP(1600) corresponds to an AR(2) with a comparatively long periodicity
#   -advancing the filter by a full year has only small/marginal effects on the phase (when compared to HP(160) or HP(16)) 

# Further comments:
# -One can scroll back the plots generated in the plot-panel
# -Doing so would show that the two-sided filter (black-line in previous plots) behaves strangely after the 
#     financial crisis
#   -To a good extent, some of the anomalies are due to the singular Pandemic readings
#   -HP(1600) is more sensitive to the Pandemic than the previous (more adaptive) designs

#---------------------------------------------
# Findings overall:

# A. Classic direct predictors:
#   -Classic direct predictors often do not perform better (out-of-sample) than the simple mean benchmark at 
#     forward-shifts of 2 quarters or more
#   -Classic direct predictors are more sensitive (than M-SSA) to singular episodes (Pandemic)

# B. M-SSA
#   -Classic business-cycle designs (lambda_HP=1600) smooth out recessions and hide  
#     dynamics potentially relevant in a short- to mid-term forecast exercise (1-6 quarters ahead)
#   -Fairly adaptive designs (lambda_HP=160) show a (logically and) statistically consistent forecast pattern, 
#       suggesting that M-SSA outperforms both the mean and the direct forecasts out-of-sample when targeting HP-BIP
#     -This result suggests that M-SSA is informative about forward-shifted BIP, although corresponding 
#       performance statistics are less conclusive (cluttered by noise)
#   -More adaptive designs (lambda_HP=16) seem to be able to track forward-shifted BIP (more) consistently, 
#     by allowing the (more) flexible trend-component to provide (more) overlap with relevant mid- and high-frequency 
#     components of BIP
#   -Sub-components of the M-SSA predictor are potentially useful for interpretation purposes (see exercise 3.2) 
#     and for addressing MSE performances explicitly, see exercise 3.3

# C. Statistical significance
#   -HAC-adjustments (of test-statistics) seem unable to account fully for the observed data-idiosyncrasies
#     -Exercise 2 suggests that strongly significant results (HP-BIP) are convincing, especially so when 
#       considering the `logically consistent' pattern in the matrices (top/down and left/right))
#     -But results on the verge of significance (targeting BIP multiple quarters ahead) should be considered 
#       with caution
#   -However, the former stronger results (HP-BIP) provide additional evidence for the latter weaker (BIP) results  
#       -Predicting the low-frequency part of BIP is likely to tell something about future BIP, assuming 
#         the latter is not white noise 


# Final notes on the publication lag and data revisions
#   -All results relate to forward-shifts augmented by the publication lag
#   -According to feedback, our setting for the publication lag, i.e., lag_vec[1]=2, is too large (reflecting some prudence)
#     -The official/effective publication lag of BIP is one quarter
#     -But BIP is revised and we here ignore revisions
#     -On the other hand, the weight M-SSA assigns to BIP is rather weak (other, timely indicators are more important); 
#       moreover, data revisions affect mainly the so-called `direct forecasts'; 
#       finally, smoothing by HP mitigates the effect of data revisions
# In summary: we expect that performances at an indicated forward-shift of k quarters (in all the above evaluations) 
#   are likely to be representative of performances at effectively k+1 quarters ahead. 

