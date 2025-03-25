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
#     -The target correlations: correlations of predictors with forward-shifted BIP or HP-BIP
#     -The rRMSE: relative root mean-square error when benchmarked against classic direct predictors or mean(BIP) (simple benchmark)
#     -HAC-adjusted p-values of (t-statistics of) regressions of predictors on targets (HAC adjustment can account for autocorrelation and heteroscedasticity of regression residuals)
# To do: provide additional Diebold-Mariano (DM) and Giacomini-White (GW) tests of unequal predictability (benchmarked against mean(BIP))

# The tutorial is structured into 3 exercises
# Exercise 1: apply a fairly adaptive design based on targeting a HP(160) filter by M-SSA
#   -HP(160) deviates from the standard HP(1600) specification typically recommended for quarterly data
#     -See a critic by Phillips and Jin (2021), suggesting that HP(1600) is `too smooth' (insufficiently flexible)
#     -See also the lengthy discussion in tutorial 7.2
#   -We shall see that M-SSA can predict HP-BIP (for which it is explicitly optimized) consistently 
#      multiple quarters ahead (statistical significance)
#   -It is more difficult to predict BIP, though: the noisy high-frequency components of BIP are unpredictable 
# Exercise 2: apply M-SSA to white noise data to verify that the proposed performance measures and tests confirm unpredictability
# Exercise 3: analyze a more adaptive M-SSA design based on targeting HP(16) by M-SSA


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
# -In exercise 3 we shall rely on an even more adaptive design based on HP(16), for reference

# 1.1 Apply M-SSA
# Here's the wrapper summarizing findings derived in tutorial 7.2: 
head(compute_mssa_BIP_predictors_func)

# The head of the function needs the following specifications:
# -x_mat: data 
# -lambda_HP: HP parameter
# -L: filter length
# -date_to_fit: in-sample span for the VAR
# -p,q: model orders of the VAR
# -ht_mssa_vec: HT constraints (larger means less zero-crossings)
# -h_vec: (vector of) forecast horizon(s) for M-SSA
# -f_excess: forecast excesses, see exercises 2 and 3 above
# -lag_vec: publication lag (target is forward shifted by forecast horizon plus publication lag)

# We can supply various hyperparameters (designs) and the function returns M-SSA predictors as 
#     specified in tutorial 7.2
# -The main hyperparameter is lambda_HP: a smaller lambda_HP means increased adaptivity 

# We now first replicate tutorial 7.2 with the above wrapper

# Target filter: lambda_HP is the single most important hyperparameter, see tutorial 7.1 for a discussion
# Briefly: we avoid the classic quarterly setting lambda_HP=1600 because the resulting filter would be too smooth
# Too smooth means: the forecast horizon would have nearly no effect on the M-SSA predictor (almost no left-shift, no anticipation)
lambda_HP<-160
# Filter length: nearly 8 years is fine for the selected lambda_HP (filter weights decay sufficiently fast)
#   Should be an odd number (see tutorial 7.1)
L<-31
# In-sample span for VAR, i.e., M-SSA (the proposed design is quite insensitive to this specification because the VAR is parsimoniously parameterized)
#  -selecting date_to_fit<-"2008" means that the entire financial crisis is out-of-sample 
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
# Forecast excesses: see tutorial 7.2, exercise 2 for background
f_excess<-c(4,2)

# Run the wrapper  
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec)

# Retrieve predictors and targets from the above function-call
# Forward-shifted HP-BIP
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
# M-SSA predictors
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
# M-MSE predictors
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat

# Plot M-SSA: the vertical line indicates the end of the in-sample span
mplot<-predictor_mssa_mat
colnames(mplot)<-colnames(predictor_mssa_mat)
par(mfrow=c(1,1))
colo<-c(rainbow(ncol(predictor_mssa_mat)))
main_title<-c(paste("Standardized M-SSA predictors for forecast horizons ",paste(h_vec,collapse=","),sep=""),"Vertical line delimites in-sample and out-of-sample span")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=c(2,rep(1,ncol(data)-1)),ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
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

# We can specify the selection of macro-indicators 
select_direct_indicator<-c("ifo_c","ESI")
# Note: too complex designs (too many indicators) lead to overfitting and thus worse out-of-sample performances
# To illustrate the direct predictor consider the following example of a h-step ahead direct forecast:
h<-2
# Shift BIP forward by publication lag+forecast horizon
forward_shifted_BIP<-c(x_mat[(1+lag_vec[1]+h):nrow(x_mat),"BIP"],rep(NA,h+lag_vec[1]))
# Regress selected indicators on forward-shifted BIP
lm_obj<-lm(forward_shifted_BIP~x_mat[,select_direct_indicator])
# You will probably not find statistically significant regressors for h>2: BIP is a very noisy series
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
#   -M-SSA does not target BIP directly, unless the target-filter is the identity (instead of HP)

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

# Correlations between M-SSA predictors and forward-shifted HP-BIP (including the publication lag)
# 1. Full sample
cor_mat_HP_BIP_full
# We can recognize a systematic pattern: 
#   -For increasing forward-shift (from top to bottom in the above matrix) M-SSA designs optimized for 
#     larger forecast horizons h (from left to right) tend to perform better (larger correlations), 
#     until h is too large
#   -For a given row (forward-shift of target) the maximum correlations stays at (or close to) the 
#     diagonal element in this row

# 2. Out-of-sample (period following estimation span for VAR-model of M-SSA)
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
# -Are these results, suggesting a systematic predictability?
# -For answering this question we may look at HAC-adjusted p-values of regressions of out-of-sample predictors on 
#   forward-shifted BIP (see further down for details)
p_value_HAC_BIP_oos
# The last predictor, optimized for h=6, is on the edge of statistical significance up to a forward-shift=3 

# We now look at pairwise comparisons with established benchmarks in terms of relative root mean-square 
#   errors: rRMSE
# rRMSE is the ratio of the root mean-Square forecast error (RMSE) of M-SSA over the RMSE of 
#   a. the mean of BIP or
#   b. the direct predictor obtained by simple regressions of the data (indicators) on forward-shifted BIP
# Remarks:
#   1. Since M-SSA predictors are standardized (equal-weighting cross-sectional aggregation), we need to 
#         calibrate them by regression onto the target (to determine static level and scale parameters)
#   2. As a result, rRMSE of M-SSA against mean(BIP) and the above target correlations are redundant statistics: 
#         the information content is the same but it is presented in an alternative form 
#   3. Background:
#       -The M-SSA objective function is the target correlation (not the mean-square error) 
#       -Therefore, M-SSA ignores static level and scale adjustments
#   4. Root mean-square errors are evaluated on the out-of-sample span only (specified by date_to_fit)
#   5. The benchmark direct predictors are full-sample estimates 
#       -Estimates based on short in-sample spans are unreliable (insignificant regression coefficients)
#   6. The benchmark mean-predictor used in our comparisons is based on the mean of the target in the 
#       out-of-sample span (it is looking ahead)
#   7. The main purpose of these comparisons is to evaluate the dynamic capability of the M-SSA predictor
#       -The (ex post) static level and scale adjustments are deemed less relevant

# With these remarks in mind let's begin:
# The first rRMSE emphasizes M-SSA vs. the mean benchmark (of BIP), both targeting HP-BIP: 
#   -Numbers smaller one signify an outperformance of M-SSA against the mean-benchmark when targeting HP-BIP
rRMSE_MSSA_HP_BIP_mean
# We next look at M-SSA vs. direct predictors based on indicators selected  in select_direct_indicator: targeting HP-BIP
rRMSE_MSSA_HP_BIP_direct
# Next: M-SSA vs. mean (of BIP) when targeting BIP
rRMSE_MSSA_BIP_mean
# Finally: M-SSA vs. direct predictors based on indicators selected  in select_direct_indicator: targeting BIP
rRMSE_MSSA_BIP_direct
# We see similar systematic patterns as for the previous correlations
#   -The systematic patterns are strong when targeting forward-shifted HP-BIP (for which M-SSA is explicitly optimized)
#   -For forward-shifted BIP the results are weaker (clutterd by noise)
# But we shall see below that a more adaptive design can improve performances, see exercise 3


# The above performance metrics do not test for statistical significance (of predictability)
# The following HAC-adjusted p-values provide a way to infer statistical significance 
# We look at HAC-adjusted p-values of regressions of M-SSA on forward-shifted targets
# Remarks:
#   -In some cases the HAC standard error (of the regression coefficient) seems `suspicious' 
#     -HAC estimate of standard error could be substantially smaller than the ordinary OLS/unadjusted estimate
#   -We therefore compute both types of standard errors and we rely on the maximum for a derivation of p-values
#   -In this sense our p-values may be claimed to be `conservative'
# Looking at the matrix of p-values below, we can recognize a systematic pattern: 
#   -For increasing forward-shifts (from top to bottom), M-SSA designs optimized for larger forecast 
#       horizons h (from left to right) tend to perform better (smaller p-values), up to some point, 
#       where p-values decrease again for h too large
#   -In other words: when moving from left to right along a specific row, smaller p-values are on 
#       (or close to) the main diagonal element of that row

# We first consider forward-shifted HP-BIP and full sample p-values
p_value_HAC_HP_BIP_full
# Same but out-of-sample 
p_value_HAC_HP_BIP_oos
# We infer that the systematic patterns observed in correlations and rRMSE above are statistically significant
#   -M-SSA seems capable of predicting forward-shifted HP-BIP, i.e., the low-frequency (trend-growth) part of BIP


# Same as above but now targeting BIP
# Full sample
# We still see a systematic pattern from top to bottom and left to right but the overall picture 
#   is cluttered by noise (BIP is much noisier than HP-BIP) 
p_value_HAC_BIP_full
# Out-of-sample: weaker significance is due, in part, to the shorter length of the out-of-sample span
p_value_HAC_BIP_oos

#-------------------------------------------
# 1.3 Visualize performances: link performance measures to plots of predictors against targets
# Select a forward-shift of target (the k-th entry in h_vec)
k<-5
# This is the forward-shift (to which we add the publication lag of lag_vec[1])
h_vec[k]
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
colnames(mplot)<-c(paste("BIP left-shifted by ",h_vec[k]," quarters",sep=""),paste("HP-BIP left-shifted by ",h_vec[k]," quarters (plus publication lag)",sep=""),paste("M-SSA predictor optimized for h=",h_vec[j],sep=""))
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

# Assume one selects k=j=5 (h_vec[5]=4: one-year ahead forecast) in the above plot (you might want to have a look at k=5 and j=7, too):
# Then the (weak) positive correlation between M-SSA and shifted BIP might suggest a (weak) predictability one year ahead
#    (including the publication lag) 
# Is this (weak) effect statistically significant?
# Let's have a look at the HAC-adjusted p-values
# 1. Full sample 
p_value_HAC_BIP_full[k,j]
# 2. Out-of-sample
p_value_HAC_BIP_oos[k,j]
# Not significant at a one-year ahead horizon when targeting BIP (which is noisy series...)


# Instead of BIP we might have a look at targeting HP-BIP instead (also shifted one year ahead)
p_value_HAC_HP_BIP_full[k,j]
p_value_HAC_HP_BIP_oos[k,j]


#--------------------------------
# Findings: 
#   -Statistical significance is stronger for shifted HP-BIP (than for BIP)
#   -Is it because mid- and short-term components of BIP are effectively unpredictable?
#   -Or is it because lambda_HP=160 is not sufficiently adaptive to track mid/short-term dynamics (still too smooth)?
# To find an answer you might try a more adaptive design (such as for example based on lambda_HP=16)
#   -Then you can check whether BIP can be predicted more consistently (hint: it can)


################################################################################################################
# Exercise 2
# Let's check what happens when we apply the aobve battery of tests and performance measures to white noise
#   -Target correlations should be small, rRMSEs should be close to one and p-values shoule be above 5%


# Generate artificial white data
set.seed(345)

a1<-0.0
b1<-0.0

x_mat_white_noise<-NULL
for (i in 1:ncol(x_mat))
  x_mat_white_noise<-cbind(x_mat_white_noise,rnorm(nrow(x_mat)))

# Provide colnmaes and rwonames from x_dat: necessary because the function relies on true dates and column names
rownames(x_mat_white_noise)<-rownames(x_mat)
colnames(x_mat_white_noise)<-colnames(x_mat)
tail(x_mat_white_noise)




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
f_excess<-c(4,2)

# Run the function packing and implementing our previous findings (tutorial 7.2) 
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat_white_noise,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec)


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


# We just need to check whether we can forecast the white noise data based on the predictors
# Full sample
p_value_HAC_BIP_full
# Out-of-sample: note that weaker significance is due, in part, to the shorter span
p_value_HAC_BIP_oos
# Findings: p-values are above 5%: cannot predict original (white noise) data
# Note: multiple test problem

#######################################################################################
# Exercise 3: increasing adaptivity further
# -The previous exercise 1 suggests that we might be able to predict HP-BIP several quarters ahead
# -But predicting BIP is more challenging because the noisy high-frequency part of BIP is essentially unpredictable
# However, it could be, in principle, that the target specification, HP(160), as specified in exercise 1 is too smooth
#   -What if we suppressed relevant information for predicting BIP?
# To verify this conjecture, we now analyze a more adaptive design by selecting lambda_HP=16 `small`
# Question: does the more flexible design allow to predict BIP (not HP-BIP) more consistently, further ahead

lambda_HP<-16
# Notes: 
# -For adaptive designs, a pronounced left-shift might lead to phase-reversal which is undesirable
# -Therefore we use forecast horizons up to 4 quarters (instead of 6) and no forecast excess
#   -Phase-reversal would be fine (optimal) if the data were in agreement with the implicit assumptions 
#     underlying the HP filter (which is not the case, see tutorial 2.0)
f_excess_adaptive<-c(0,0)
h_vec_adaptive<-0:4

# Run the M-SSA predictor function
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec_adaptive,f_excess_adaptive,lag_vec)

# Forward-shifted HP-BIP
target_shifted_mat=mssa_indicator_obj$target_shifted_mat
# M-SSA indicators
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
# M-MSE
predictor_mmse_mat<-mssa_indicator_obj$predictor_mmse_mat

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

#---------------------------------------------
# Findings overall:

# A. Classic direct predictors:
#   -Classic direct predictors often do not perform better (out-of-sample) than the simple mean benchmark at 
#     forward-shifts exceeding 2 quarters
#   -Classic direct predictors are more sensitive (than M-SSA) to episodes subject to unusual singular readings (Pandemic)

# B. M-SSA
#   -Classic business-cycle designs (lambda_HP=1600) smooth out recessions and hide  
#     dynamics potentially relevant in a short- to mid-term forecast exercise (1-6 quarters ahead)
#   -Fairly adaptive designs (lambda_HP=160) show a (logically and) statistically consistent forecast pattern, 
#       suggesting that M-SSA outperforms both the mean and the direct forecasts out-of-sample when targeting HP-BIP
#     -This result suggest that M-SSA is informative about forward-shifted BIP, although corresponding 
#       performance statistics are less conclusive (cluttered by noise)
#   -More adaptive designs (lambda_HP=16) seem to be able to track forward-shifted BIP (more) consistently, 
#     by allowing the (more) flexible trend-component to provide (more) overlap with relevant mid- and high-frequency 
#     components of BIP

# Note about the publication lag
#   -All results relate to forward-shifts augmented by the publication lag
#   -According to feedback, our setting for the publication lag, i.e., lag_vec[1]=2, is too large (prudence)
#     -The effective publication lag of BIP is one quarter
#   -As a result, a forward-shift corresponding to 3 quarters in our evaluations is more likely to 
#     represent a full year ahead forecast horizon 

