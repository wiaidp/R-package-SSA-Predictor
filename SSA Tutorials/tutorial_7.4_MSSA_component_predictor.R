# Tutorial 7.4
# Main purposes:
# 1. Forecasting BIP multiple quarters ahead
#   -Derive a new M-SSA components predictor which addresses MSE forecast performances when tracking BIP
#     -The original M-SSA predictor (without the attribute `components') proposed in tutorial 7.3 is 
#         standardized: neither level nor scale were calibrated to track BIP 
#     -The original M-SSA predictor was designed to track dynamic changes in the trend growth-rate of BIP (HP-BIP)
#     -The original M-SSA predictor was based on aggregating equally the standardized M-SSA components 
#   -We here propose a more refined design, the `M-SSA components predictor', whereby equal weighting 
#       (of M-SSA components) is replaced by an additional (new) optimal weighting step.
# 2. Interpretability: 
#   -Exploit the M-SSA components to gauge forecasts by the proposed M-SSA predictor(s)
#   -How trustworthy is a change of the predictor outlook?
# 3. Explainability: 
#   -Justify the design and construction steps underlying the proposed M-SSA predictor(s)
#   -Determine which step(s) in the construction of the predictor(s) generate(s) forecast gains?

# Proceeding:
# -We rely on the design proposed in tutorial 7.3, exercise 1
# -We then add a new optimal weighting step in the construction of the final predictor

# The tutorial is organized into 6 exercises
# -Exercise 1
#   -M-SSA components: derive M-SSA components and replicate the original M-SSA predictor (tutorial 7.3)
#   -Interpretability: rely on M-SSA components to gauge forecasts (reliability/trustworthiness)
#   -Forecast BIP: new additional optimization step
#   -Out-of-sample performance evaluation
#     -Comparison vs. simple mean, direct forecast and original M-SSA predictor
# -Exercise 2
#   -Analysis of revisions of new (real-time out-of-sample) M-SSA components predictor
# -Exercise 3
#   -Application of the predictor to German BIP
# -Exercise 4
#   -Explainability: why does the M-SSA component predictor outperform specifically at multi-quarters 
#     ahead forecast horizons?
# -Exercise 5
#   -Specify and compute an `M-MSE component predictor' (same as M-SSA but without HT imposed: less smooth)
#   -Compare MSE forecast performances to the simple mean benchmark and the (new) M-SSA component predictor
# -Exercise 6
#   -Compute final M-SSA and M-MSE component predictors based on full data information and optimal 
#     WLS regression, discarding singular Pandemic data for estimation of parameters
#----------------------------------------------
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
# Load the data and select the relevant indicators: see tutorials 7.2 and 7.3 for background
load(file=paste(getwd(),"\\Data\\macro",sep=""))

# Publication lag: we assume a lag of two quarters for BIP 
#   -Effective publication lag is one
#   -But we here ignore data revisions
#   -The higher publication lag addresses (in part) the absence of revisions 
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
# The plot indicates that the publication lag of two quarters is too large 
#   -Peaks and dips of the target (black line) are left-shifted by one quarter at recessions
# The excessive publication lag is intended to compensate for data revisions (which are ignored in our design)
#   -We may claim prudence: results will be conservative (on the safe side)  

# Select macro indicators for M-SSA 
select_vec_multi<-c("BIP","ip","ifo_c","ESI","spr_10y_3m")
x_mat<-data[,select_vec_multi] 
rownames(x_mat)<-rownames(data)
n<-dim(x_mat)[2]
# Number of observations
len<-dim(x_mat)[1]


##################################################################################
# Exercise 1 Working with M-SSA (sub-)components
#   -We rely on the design proposed in tutorial 7.3, exercise 1

# Exercise 1.0: brief summary of original M-SSA predictor (tutorial 7.3)

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
#   -Increasing these numbers leads to predictors with less zero-crossings (smoother), see tutorial 7.1
ht_mssa_vec<-c(6.380160,  6.738270,   7.232453,   7.225927,   7.033768)
names(ht_mssa_vec)<-colnames(x_mat)
# Forecast horizons: M-SSA is optimized for each forecast horizon in h_vec 
h_vec<-0:6
# Forecast excesses: see tutorial 7.2, exercise 2 for background
f_excess<-rep(4,length(select_vec_multi))

# Run the wrapper, see tutorial 7.2
#   -The function computes M-SSA for each forecast horizon h in h_vec
mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi)

# Target series: output of two-sided HP applied to BIP: 
#   -This is the target for which the original M-SSA predictor (tutorial 7.3) has been designed
#   -The target is forward-shifted by the forecast horizon (plus publication lag)
target_shifted_mat<-mssa_indicator_obj$target_shifted_mat
# Original M-SSA predictor, see tutorial 7.3
#   -One predictor available for each forecast horizon in h_vec
predictor_mssa_mat<-mssa_indicator_obj$predictor_mssa_mat
# M-SSA components, see tutorial 7.2
#   -This is a three dimensional array
#   -For each forecast horizon and for each indicator we obtain the M-SSA predictor when targeting 
#     the two-sided HP applied to this indicator, see exercise 1.1 below
mssa_array<-mssa_indicator_obj$mssa_array
# M-MSE components
# -Same as mssa_array but without HT imposed, i.e., classic multivariate mean-square error signal extraction
# -Forecast performances, see exercise 5 below 
mmse_array<-mssa_indicator_obj$mmse_array

# Compute performances of original M-SSA predictor: these will be used as a benchmark when evaluating 
#   the new design
# Select start of out-of-sample span (entire financial crisis is out-of-sample)
in_out_separator<-"2007"
# We can specify the selection of macro-indicators for the direct forecast, see tutorial 7.3
# Note: these results will not be used in this tutorial (but we need to specify a selection anyway)
select_direct_indicator<-c("ifo_c","ESI")
perf_obj<-compute_perf_func(x_mat,target_shifted_mat,predictor_mssa_mat,predictor_mmse_mat,in_out_separator,select_direct_indicator,h_vec) 

# The above function generates a rich output, see tutorial 7.3
#   -But we need only one (important/representative) performance number for our comparisons further down
# HAC adjusted p-values of regression of original M-SSA predictor on forward-shifted BIP, see tutorial 7.3 (exercise 1.2.2)
p_value_HAC_BIP_oos=perf_obj$p_value_HAC_BIP_oos
p_value_HAC_BIP_oos
# The p-values are empirical significance levels of the regressions of the (out-of-sample) M-SSA predictors
#   (optimized for forecast horizon h in h_vec: the columns of the matrix) on BIP shifted forward by shift 
#   plus publication lag (the rows of the matrix)

# Example: check if the M-SSA predictor optimized for horizon h_vec[j] can significantly predict BIP shifted forward
#   by shift=i+1
# One-year shift
i<-5
# Forecast horizon 6
j<-7
p_value_HAC_BIP_oos[i,j]
# The original M-SSA predictor optimized for horizon h=6 does not significantly predict BIP 4 quarters ahead (plus publication lag)
#   -Below we shall illustrate that the new M-SSA component predictor will perform better (smaller p-value) in 
#     particular at larger forward-shifts (>=3 quarters)

# Technical note: 
# -The M-SSA predictor can track forward-shifted HP-BIP (the series in target_shifted_mat above) more tightly 
#   than forward-shifted BIP
perf_obj$p_value_HAC_HP_BIP_oos[i,j]
# The original M-SSA predictor optimized for horizon h=6 is strongly significant when predicting HP-BIP 
#   4 quarters ahead (plus publication lag)

# The original M-SSA predictor (tutorial 7.3) and the newly proposed M-SSA component predictor 
#   in this tutorial address different targets!  

#-----------
# Exercise 1.1 What are M-SSA (sub-)components?
# -The original M-SSA predictor is obtained as the equally-weighted aggregate of standardized M-SSA outputs of all indicators (BIP, ip, ifo, ESI, spread)
#   -For illustration, we here briefly replicate the predictor, as based on its (equally-weighted) components. 
# -For this purpose, we can select any forecast horizon in h_vec
#   -For illustration, we select the nowcast
j_now<-1
# This is the forecast horizon (nowcast)
h_vec[j_now]
# For forecast horizon h_vec[j_now], the sub-series of the M-SSA predictor are:  
tail(t(mssa_array[,,j_now]))

# Plot M-SSA components: these are the M-SSA outputs (predictors) tracking the two-sided HP(160) applied 
#   to the indicators. Specifically: HP-BIP, HP-ip, HP-ifo, HP-ESI and HP-spread.
# For each target, say HP-BIP, M-SSA can rely on all indicators for more effective tracking (than univariate filters)
mplot<-t(mssa_array[,,j_now])
colo<-c(rainbow(length(select_vec_multi)))
main_title<-"M-SSA components: outputs of M-SSA tracking two-sided HP"
par(mfrow=c(1,1))
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))),lty=2)
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

# These sub-series correspond to the M-SSA outputs optimized for horizon h_vec[j_now], see tutorial 7.2, exercise 1
#   -For each series of the multivariate design, the target is the two-sided HP applied to this series and 
#     shifted forward by the forecast horizon (plus the publication lag in case of BIP)
#   -For each of these targets, the explanatory variables are BIP, ip, ifo, ESI and spread
#   -The series in mssa_array[,,j_now] are the real-time (causal) predictors of the acausal targets

# We now check that the M-SSA predictor (tutorial 7.3, exercise 1) is the cross-sectional mean of these 
#   standardized sub-series: 
agg_std_comp<-apply(scale(t(mssa_array[,,j_now])),1,mean)
# Plot the equall-weighted aggregate and the M-SSA predictor
mplot<-cbind(agg_std_comp, predictor_mssa_mat[,j_now])
rownames(mplot)<-rownames(x_mat)
colnames(mplot)<-c("Cross-sectional mean of standardized predictor components","M-SSA predictor")
colo<-c("blue",rainbow(length(select_vec_multi)))
main_title<-"Replication of M-SSA predictor, based on its components"
par(mfrow=c(1,1))
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
# Both series are identical (overlap)
# Alternative check: the maximal error/deviation should be `small' (zero up to numerical precision)
max(abs(apply(scale(t(mssa_array[,,j_now])),1,mean)-predictor_mssa_mat[,j_now]),na.rm=T)

# Remarks:
# -Equal-weighting of the M-SSA components, as done above, indicates that we assume each M-SSA component to be equally
#   important for tracking dynamic changes of the BIP growth-rate by the resulting M-SSA predictor. This 
#   `naive' assumption might be questioned. But the rule (equal-weighting) is robust and simple.
# -Instead, we could think about a more sophisticated weighting scheme: for example, by regressing the components 
#     on forward-shifted BIP. 
#   -This way, we'd explicitly emphasize BIP-MSE forecast performances by the resulting (new) M-SSA `components` predictor
# -The corresponding `component predictor' will be derived and analyzed in exercise 1.3 below.
# -But first we consider an alternative usage of the components, namely interpretability (of the M-SSA predictor)  

#---------------
# 1.2 We now exploit the M-SSA components in view of a better interpretation (explanation/understanding) of the M-SSA predictor (tutorial 7.3, exercise 1).
#   -We can examine which sub-series is (are) more/less likely to trigger a dynamic change of the M-SSA predictor
#   -Thereby we can gauge the forecast (trustworthiness/reliability)
# -For illustration, we here select the M-SSA nowcast

# Plot M-SSA nowcast and components
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
# -All sub-series date the latest trough of the growth rate of the German economy in late 2023 
# -Currently (data up to Jan-2025), the strongest positive dynamics are supported by the (leading) spread sub-series (violet dashed line in plot)
# -Given that the nowcast (solid blue line) just passed the zero-line (long-term average-growth) in the 
#     above plot, we may infer that the trough of BIP (in levels) might be behind, based on Jan-2025 data.
# -However, not all components (sub-series) would support this claim:
#   -The strongest up-turn signal is supported by the (leading) spread (which has been subjected to critic as a leading indicator)
#   -ifo and ESI are barely above the zero-line 
#   -ip and BIP are `waiting' for further evidence and confirmation. 

# -We infer that we can gauge the M-SSA predictor by looking at its components; which component(s) trigger(s) a change in dynamics? 
# -Note that announced and/or unexpected disorders (tariffs, geopolitical contentions) are not yet `priced-in' (as of Jan-2025).

#-------------------------------------
# Exercise 1.3 Addressing BIP-MSE performances 
# -As stated above, the original M-SSA predictor emphasizes dynamic changes (recessions/expansions); 
#     MSE performances are deemed less relevant 
#   -In particular, the predictor is standardized: neither its level nor its scale are calibrated on BIP
# -In order to track (future) BIP explicitly, we may rely on the M-SSA components in the previous figure
# -For this purpose, we can regress the components on forward-shifted BIP (instead of equal-weighting)
# -This is the same proceeding as for the direct forecasts (see tutorial 7.3, exercise 1.2.1), except that 
#   we rely on the M-SSA components for the regressors (instead of the original un-filtered indicators)

# 1.3.1 Selection
# -We can select the M-SSA components which are deemed relevant for MSE performances when targeting BIP
#   -ESI, ifo and spread M-SSA components are mainly relevant in a dynamic context (for the original M-SSA predictor)
#   -In contrast, BIP and ip M-SSA components are natural candidates in a MSE-BIP perspective (which is emphasized here).
#   -Note, however, that the original indicators ESI, ifo and spread are important determinants 
#     of the selected (BIP- and ip-) M-SSA components, see tutorial 7.2, exercise 1.
# -In summary: one can try various component combinations, including a single M-SSA BIP component. Performances are 
#   roughly similar. The combination of BIP and ip M-SSA components is simple and intuitively appealing.
sel_vec_pred<-select_vec_multi[c(1,2)]
# Check the selected M-SSA components
sel_vec_pred
# We can select the forward shift of BIP: for illustration we here assume a 2 quarters ahead forward-shift
#   (plus publication lag). Below we shall analyze all shifts, from zero to five quarters ahead.
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
                 paste("M-SSA component ",colnames(t(mssa_array[sel_vec_pred,,k])),": h=",h_vec[k],sep=""))
tail(dat)
# We target BIP shifted forward by shift+publication lag (first column) based on M-SSA components BIP and ip
#   -As stated above, ifo, ESI and spread are important determinants of the two selected M-SSA components 
# For estimation purposes we can remove all NAs
dat<-na.exclude(dat)

#-----------------
# 1.3.2 Regression
# We now regress forward-shifted BIP (first column) on the two M-SSA components
#   -Specify an arbitrary in-sample span (below we shall use an expanding window starting in Q1-2007 and ending in Q4-2025)
i_time<-which(rownames(dat)>2010)[1]
# In-sample span: 
tail(dat[1:i_time,])
# Regression
lm_obj<-lm(dat[1:i_time,1]~dat[1:i_time,2:ncol(dat)])
summary(lm_obj)
# The M-SSA components are strongly significant (HAC-adjustments wouldn't contradict this statement)

# Compute an out-of-sample prediction for time point i_time+shift+lag_vec[1] 
#   -Due to the publication lag, the regression span cannot extend up to the sample end
#   -Therefore we shift the explanatory variables as well as the target forward by the additional publication lag 
#   -Note however that this effect (due to shifting explanatory and target by lag_vec[1]) is negligible, 
#     because the regression coefficients tend to converge to fixed values with increasing sample size, 
#     see exercise 2.2 below. 
oos_pred<-(lm_obj$coef[1]+lm_obj$coef[2:ncol(dat)]%*%dat[i_time+shift+lag_vec[1],2:ncol(dat)]) 
# Compute the out-of-sample forecast error
oos_error<-dat[i_time+shift+lag_vec[1],1]-oos_pred
# This is the out-of-sample error that will be observed shift (+publication lag) quarters ahead
oos_error

#---------------
# 1.3.3 Better/improved regression: we can improve the weighting of the M-SSA components further.
# -Given that BIP is subject to heteroscedasticity we may apply a GARCH(1,1) to obtain an estimate of its variance
y.garch_11<-garchFit(~garch(1,1),data=dat[1:i_time,1],include.mean=T,trace=F)
summary(y.garch_11)
# sigmat could be retrieved from GARCH-object
sigmat<-y.garch_11@sigma.t
# But this is lagged by one period
# Therefore we recompute the vola based on the estimated GARCH-parameters
eps<-y.garch_11@residuals
d<-y.garch_11@fit$matcoef["omega",1]
alpha<-y.garch_11@fit$matcoef["alpha1",1]
beta<-y.garch_11@fit$matcoef["beta1",1]
sigmat_own<-sigmat
for (i in 2:length(sigmat))#i<-2
  sigmat_own[i]<-sqrt(d+beta*sigmat_own[i-1]^2+alpha*eps[i]^2)
# This is now correct (not lagging anymore)
sigmat<-sigmat_own


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

# We can now apply weighted least-squares (WLS) instead of OLS, using the (inverse of the) GARCH-vola for the weights 
weight<-1/sigmat^2
# Apply WLS instead of OLS
lm_obj<-lm(dat[1:i_time,1]~dat[1:i_time,2:ncol(dat)],weight=weight)
summary(lm_obj)
# The M-SSA components are still strongly significant but the regression coefficients are slightly 
#   different (when compared to OLS above)

# Compute out-of-sample prediction for time point i+shift+lag_vec[1]: 
# Technical notes:
#   1. The GARCH is irrelevant when computing the predictor (the GARCH is us used for estimating regression coefficients, only)
#   2. Due to the publication lag, the regression span cannot extend up to the sample end
#     -Therefore we shift the explanatory variables as well as the target forward by the additional publication lag 
#     -Note however that this effect  (due to shifting explanatory and target by lag_vec[1]) is negligible, 
#       because the regression coefficients tend to converge to fixed values with increasing sample size, 
#       see exercise 2.2 below.
oos_pred_wls<-as.double(lm_obj$coef[1]+lm_obj$coef[2:ncol(dat)]%*%dat[i_time+shift+lag_vec[1],2:ncol(dat)]) 
# Compute out-of-sample forecast error
oos_error_wls<-dat[i_time+shift+lag_vec[1],1]-oos_pred_wls
# This is the out-of-sample WLS error that we observe in shift=2 quarters later
oos_error_wls
# Compare to out-of-sample error based on OLS: 
oos_error
# Depending on the selected time point, WLS performs better or worse than OLS.
# But on average, over many (out-of-sample) time points, WLS tends to outperform OLS (see exercise 1.3.4 below for confirmation). 
#   -Therefore we now apply WLS when deriving weights of M-SSA components
#   -For comparison and benchmarking, we also derive a better `direct forecast' based on WLS (in contrast to 
#     exercise 1.2.1 in tutorial 7.3 which is based on classic OLS)

# Finally, we can also compute the simple mean-benchmark 
#   -Due to the publication lag, the mean estimate cannot extend up to the sample end
#     -We shift the target forward by the additional publication lag 
#   -Note however that this effect (adding lag_vec[1] or not) is negligible, because the 
#       mean converges to a fixed value (long-term average growth) with increasing sample size (assuming stationarity...)
mean_bench<-mean(dat[1:i_time,1])
# Its out-of-sample forecast error is
oos_error_mean<-dat[i_time+shift+lag_vec[1],1]-mean_bench
# Compute the rRMSE of the WLS (M-SSA) component predictor referenced against the mean-benchmark when 
#   targeting BIP, shifted forward by shift (+publication lag):
rRMSE_mSSA_comp_mean<-sqrt(mean(oos_error_wls^2)/mean(oos_error_mean^2))
rRMSE_mSSA_comp_mean
# -Depending on the selected time point, the rRMSE is larger or smaller one
# -However, on average over a longer out-of-sample span, we expect the more sophisticated predictor(s) 
#   to outperform the simple mean benchmark, at least for `reasonably sized' forecast horizons

# We now apply the above proceeding to a longer out-of-sample span and compute average performances
#------------------
# 1.3.4 Average performances: apply the above proceeding to all data points after 2007, including the 
#   entire financial crisis for out-of-sample evaluations

# Start point for out-of-sample evaluation: 2007
in_out_separator<-in_out_separator
# Note that the in-sample span is rather short at the start (due in part to filter initialization)
#   -Therefore MSE forecast performances are likely to be worse than towards the sample end

# Use WLS based on GARCH(1,1) when regressing M-SSA components on forward-shifted BIP 
#   (setting the Boolean to F would amount to OLS, see below for details)
use_garch<-T
# The following function applies the above WLS-regression for all time points (expanding window)
#   -For each time point, a GARCH is fitted and the WLS-regression is computed based on the up-dated GARCH-weights
# Note: we can ignore `warnings' which are generated by the GARCH estimation routine
perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift,lag_vec)

# Here we have the out-of-sample forecast errors of the new M-SSA components predictor
# Note that the out-of-sample error series is shorter than the original data, due to publication lag and forward-shift
tail(perf_obj$epsilon_oos)
# We can see that the function replicates the proceeding in 1.3.3 above, i.e., oos_error_wls is obtained as one of the entries of the longer out-of-sample vector
which(perf_obj$epsilon_oos==oos_error_wls)
# We also obtain the out-of-sample forecast errors of the mean benchmark predictor
tail(perf_obj$epsilon_mean_oos)
# One again, the corresponding out-of-sample error (in 3.3.3) is replicated as an entry of the longer vector
which(perf_obj$epsilon_mean_oos==oos_error_mean)
# More importantly, the function computes HAC-adjusted p-values of the regression of the out-of-sample predictor (oos_pred_wls obtained in exercise 3.3.3 above)
#   on forward-shifted BIP
perf_obj$p_value
# The same but without singular Pandemic readings
perf_obj$p_value_without_covid
# We infer that there exists a statistically significant link, out-of-sample, between the new M-SSA component
#     predictor and BIP shifted by 2 quarters ahead 
#   -Recall that the classic direct forecast (tutorial 7.3) was insignificant at shifts larger than one quarter
# The singular pandemic data affects (negatively) the strength of this link

# In addition, we obtain the out-of-sample MSE of the M-SSA component predictor (the mean of oos_error_wls^2, where oos_error_wls was obtained in exercise 3.3.3 above)
perf_obj$MSE_oos
# The same but without Pandemic: we can s(e)ize the impact of the crisis on the MSE metric!
perf_obj$MSE_oos_without_covid
# The function also computes the out-of-sample MSE of the simple mean benchmark predictor (expanding window): 
#   -we expect the mean-benchmark to be slightly worse (larger MSE) than the M-SSA components, at least for 
#     smaller forward-shifts of BIP (in the long run, the mean is difficult to outperform)
perf_obj$MSE_mean_oos
perf_obj$MSE_mean_oos_without_covid

# We can also compute the rRMSE of the new M-SSA component predictor against the simple mean benchmark
sqrt(perf_obj$MSE_oos/perf_obj$MSE_mean_oos)
# Same but without Pandemic
sqrt(perf_obj$MSE_oos_without_covid/perf_obj$MSE_mean_oos_without_covid)

# For sake of comparison, we can compute a classic OLS regression and compare with WLS above. 
#   For this purpose we set use_garch<-F
use_garch<-F
perf_obj_OLS<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift,lag_vec)

perf_obj_OLS$MSE_oos
# Compare with WLS
perf_obj$MSE_oos
# The same but without Pandemic: we can s(e)ize the impact of the crisis on the MSE metric!
perf_obj_OLS$MSE_oos_without_covid
# Compare with WLS
perf_obj$MSE_oos_without_covid
# As claimed, WLS outperforms OLS out-of-sample, on average, Note also that OLS outperforms the mean benchmark

# In the next step, we compute the above performance metrics for all combinations of forward-shift (of BIP) 
#   and forecast horizons (of M-SSA components)

#----------------
# 1.3.5 Compute performances of new M-SSA component predictor for all combinations of forward-shift and 
#   forecast horizon (6*7 matrix of performance metrics)

# -Depending on the CPU, computations may last up to several minutes (regressions and GARCH-models are recomputed for each time point and for all combinations of shift and forecast horizon))
# -All results were previously computed and saved: 
#   recompute_results<-F loads these results without lengthy computations
#   recompute_results<-T recomputes everything
recompute_results<-F
shift_vec<-0:5
if (recompute_results)
{
# Initialize performance matrices
  MSE_oos_mssa_comp_without_covid_mat<-MSE_oos_mssa_comp_mat<-p_mat_mssa<-p_mat_mssa_components<-p_mat_mssa_components_without_covid<-p_mat_direct<-rRMSE_mSSA_comp_direct<-rRMSE_mSSA_comp_mean<-rRMSE_mSSA_comp_direct_without_covid<-rRMSE_mSSA_comp_mean_without_covid<-rRMSE_mSSA_direct_mean_without_covid<-rRMSE_mSSA_direct_mean<-p_mat_direct_without_covid<-matrix(ncol=length(h_vec),nrow=length(h_vec)-1)
# Use WLS
  use_garch<-T
# Set-up progress bar: indicates progress in R-console
  pb <- txtProgressBar(min=min(h_vec),max=max(h_vec)-1,style=3)
  
# The following double loop computes all combinations of forward-shifts (of BIP) and forecast horizons (of M-SSA)
  for (shift in shift_vec)#shift<-0
  {
# Progress bar: see R-console
    setTxtProgressBar(pb, shift)
    for (j in h_vec)#j<-5
    {
# Horizon j corresponds to k=j+1-th entry of array    
      k<-j+1
# A. M-SSA component predictor
# Specify data matrix for WLS regression
      if (length(sel_vec_pred)>1)
      {
        dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_pred,,k]))
      } else
      {
        dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mssa_array[sel_vec_pred,,k]))
      }
      rownames(dat)<-rownames(x_mat)
      colnames(dat)<-c(colnames(x_mat)[1],sel_vec_pred)
      dat<-na.exclude(dat)
# Apply the previous function: compute GARCH, WLS regression, out-of-sample MSEs and p-values    
      perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift,lag_vec)
# Retrieve out-of-sample performances 
# a. p-values with/without Pandemic    
      p_mat_mssa_components[shift+1,k]<-perf_obj$p_value
      p_mat_mssa_components_without_covid[shift+1,k]<-perf_obj$p_value_without_covid
# b. MSE forecast error out-of-sample
#   -M-SSA components with/without Pandemic    
      MSE_oos_mssa_comp_mat[shift+1,k]<-MSE_oos_mssa_comp<-perf_obj$MSE_oos
      MSE_oos_mssa_comp_without_covid_mat[shift+1,k]<-MSE_oos_mssa_comp_without_covid<-perf_obj$MSE_oos_without_covid
#   -mean-benchmark with/without Pandemic    
      MSE_mean_oos<-perf_obj$MSE_mean_oos
      MSE_mean_oos_without_covid<-perf_obj$MSE_mean_oos_without_covid
# Here we retrieve the final in-sample predictor (based on the full-sample WLS regression) as well as the 
#    real-time out-of-sample predictor (re-adjusted to new data at each time point)
# We can plot both predictors to illustrate revisions (due to WLS estimation at each time point), see below
# Note: the variables will be overwritten, i.e., we keep only the last run through the double loop, 
#   corresponding to maximal shift and maximal forecast horizon, see exercise 2.1 below 
      final_components_preditor<-perf_obj$final_in_sample_preditor
      oos_components_preditor<-perf_obj$cal_oos_pred
# We can also obtain the regression weights to track changes (systematic vs. noisy revisions) over time
# Note: the variable will be overwritten, i.e., we keep only the last run through the double loop, 
#   corresponding to maximal shift and maximal forecast horizon, see exercise 2.2 below 
      track_weights<-perf_obj$track_weights
      
# B. Direct forecasts
# -The main difference to M-SSA above is the specification of the explanatory variables in the data 
#     matrix dat: we here use x_mat instead of mssa_array. 
#   -We select all indicators (one could easily change this setting but results are only marginally effected as long as ifo and ESi are included)
#   -Note that the data matrix here does not depend on j, in contrast  to the M-SSA components above    
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),x_mat)
      rownames(dat)<-rownames(x_mat)
      dat<-na.exclude(dat)
      
      perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift,lag_vec)
# Retrieve out-of-sample performances: p-values and forecast MSE, with/without Pandemic 
      p_mat_direct[shift+1,k]<-perf_obj$p_value 
      p_mat_direct_without_covid[shift+1,k]<-perf_obj$p_value_without_covid 
      MSE_oos_direct<-perf_obj$MSE_oos
      MSE_oos_direct_without_covid<-perf_obj$MSE_oos_without_covid
      
# Compute rRMSEs
# a. M-SSA Components vs. direct forecast    
      rRMSE_mSSA_comp_direct[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_oos_direct)
# b. M-SSA Components vs. mean benchmark    
      rRMSE_mSSA_comp_mean[shift+1,k]<-sqrt(MSE_oos_mssa_comp/MSE_mean_oos)
# c. Direct forecast vs. mean benchmark    
      rRMSE_mSSA_direct_mean[shift+1,k]<-sqrt(MSE_oos_direct/MSE_mean_oos)
# Same as a, b, c but without Pandemic
      rRMSE_mSSA_comp_direct_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_oos_direct_without_covid)
      rRMSE_mSSA_comp_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid/MSE_mean_oos_without_covid)
      rRMSE_mSSA_direct_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_direct_without_covid/MSE_mean_oos_without_covid)
    }
  }
# Close progress bar
  close(pb)
# Note: possible warnings issued by the GARCH estimation routine during computations can be ignored
  
# Assign column and rownames
  colnames(p_mat_mssa_components)<-colnames(p_mat_direct)<-colnames(p_mat_mssa_components_without_covid)<-
    colnames(rRMSE_mSSA_comp_direct)<-colnames(rRMSE_mSSA_comp_mean)<-
    colnames(rRMSE_mSSA_comp_direct_without_covid)<-colnames(rRMSE_mSSA_comp_mean_without_covid)<-
    colnames(rRMSE_mSSA_direct_mean)<-colnames(rRMSE_mSSA_direct_mean_without_covid)<-
    colnames(p_mat_direct_without_covid)<-colnames(MSE_oos_mssa_comp_mat)<-
    colnames(MSE_oos_mssa_comp_without_covid_mat)<-paste("h=",h_vec,sep="")
  rownames(p_mat_mssa_components)<-rownames(p_mat_direct)<-rownames(p_mat_mssa_components_without_covid)<-
    rownames(rRMSE_mSSA_comp_direct)<-rownames(rRMSE_mSSA_comp_mean)<-
    rownames(rRMSE_mSSA_comp_direct_without_covid)<-rownames(rRMSE_mSSA_comp_mean_without_covid)<-
    rownames(rRMSE_mSSA_direct_mean)<-rownames(rRMSE_mSSA_direct_mean_without_covid)<-
    rownames(p_mat_direct_without_covid)<-rownames(MSE_oos_mssa_comp_mat)<-
    rownames(MSE_oos_mssa_comp_without_covid_mat)<-paste("Shift=",shift_vec,sep="")
# Define list for saving all matrices  
    list_perf<-list(p_mat_mssa_components=p_mat_mssa_components,p_mat_direct=p_mat_direct,
    p_mat_mssa_components_without_covid=p_mat_mssa_components_without_covid,rRMSE_mSSA_comp_direct=rRMSE_mSSA_comp_direct,
    rRMSE_mSSA_comp_mean=rRMSE_mSSA_comp_mean, rRMSE_mSSA_comp_direct_without_covid=rRMSE_mSSA_comp_direct_without_covid,
    rRMSE_mSSA_comp_mean_without_covid=rRMSE_mSSA_comp_mean_without_covid,rRMSE_mSSA_direct_mean=rRMSE_mSSA_direct_mean,
    rRMSE_mSSA_direct_mean_without_covid=rRMSE_mSSA_direct_mean_without_covid,p_mat_direct_without_covid=p_mat_direct_without_covid,
    final_components_preditor=final_components_preditor,oos_components_preditor=oos_components_preditor,
    track_weights=track_weights,MSE_oos_mssa_comp_mat=MSE_oos_mssa_comp_mat,
    MSE_oos_mssa_comp_without_covid_mat=MSE_oos_mssa_comp_without_covid_mat)
# The results can be saved (overwritten)    
    if (F)
    {
      save(list_perf,file=paste(getwd(),"/Results/list_perf",sep=""))
    }
} else
{
# Load all results  
  load(file=paste(getwd(),"/Results/list_perf",sep=""))
  p_mat_mssa_components=list_perf$p_mat_mssa_components
  p_mat_direct=list_perf$p_mat_direct
  p_mat_mssa_components_without_covid=list_perf$p_mat_mssa_components_without_covid
  rRMSE_mSSA_comp_direct=list_perf$rRMSE_mSSA_comp_direct
  rRMSE_mSSA_comp_mean=list_perf$rRMSE_mSSA_comp_mean
  rRMSE_mSSA_comp_direct_without_covid=list_perf$rRMSE_mSSA_comp_direct_without_covid
  rRMSE_mSSA_comp_mean_without_covid=list_perf$rRMSE_mSSA_comp_mean_without_covid
  rRMSE_mSSA_direct_mean=list_perf$rRMSE_mSSA_direct_mean
  rRMSE_mSSA_direct_mean_without_covid=list_perf$rRMSE_mSSA_direct_mean_without_covid
  p_mat_direct_without_covid=list_perf$p_mat_direct_without_covid  
  final_components_preditor=list_perf$final_components_preditor
  oos_components_preditor=list_perf$oos_components_preditor
  track_weights=list_perf$track_weights
  MSE_oos_mssa_comp_mat=list_perf$MSE_oos_mssa_comp_mat
  MSE_oos_mssa_comp_without_covid_mat=list_perf$MSE_oos_mssa_comp_without_covid_mat
}

# HAC-adjusted p-values of out-of-sample (M-SSA) components predictor when targeting forward-shifted BIP
#   -Evaluation based on out-of-sample span starting at in_out_separator and ending on Jan-2025
p_mat_mssa_components
# Same but without singular Pandemic
p_mat_mssa_components_without_covid
# The link between the new predictor and future BIP is statistically significant up to multiple quarters ahead 
#   -Designs optimized for larger forecast horizons (columns with h>=4) seem to perform significantly up to 
#     one year ahead

# We can compare new component and original M-SSA predictors (tutorial 7.3)
# a. New M-SSA components predictor
p_mat_mssa_components
# b. Original M-SSA predictor (based on naive equal-weighting of the components)
p_value_HAC_BIP_oos
# Findings: the new M-SSA component predictors optimized for h>=4 (last three columns) tend to track BIP 
#   better than the original M-SSA predictors at forward-shifts>=3 quarters ahead (last three rows) 

# -The above p-values are based on regressions (of out-of-sample predictors on BIP) thereby ignoring 
#   `static' level and scale parameters (which are automatically adjusted by the regressions)
# -The following rRMSEs account (also) for scale and level of the new predictor
#   -All results out-of-sample (starting just before the financial crisis)
# a. rRMSE of M-SSA components when benchmarked against mean, out-of-sample
rRMSE_mSSA_comp_mean
# b. rRMSE of M-SSA components when benchmarked against direct forecast, out-of-sample
rRMSE_mSSA_comp_direct
# c. rRMSE of direct forecasts when benchmarked against mean benchmark, out-of-sample
#   Note: the columns are all the same because for a given shift, the explanatory variables of the 
#     direct forecast do not depend on j in the above loop  
rRMSE_mSSA_direct_mean

# Same but without Pandemic
rRMSE_mSSA_comp_mean_without_covid
rRMSE_mSSA_comp_direct_without_covid
rRMSE_mSSA_direct_mean_without_covid

# Comments:
# -Systematic pattern: for larger forward-shifts (from top to bottom), designs optimized for larger 
#     forecasts horizons (from left to right) tend to perform better
# -Singular Pandemic data affects evaluation 
#   -p-values and rRMSEs increase; systematic patterns are cluttered by noise
#   -Direct forecasts barely outperform the simple mean benchmark, see rRMSE_mSSA_direct_mean
# -Analysis on data excluding the Pandemic:
#   -New predictor vs. mean: (see rRMSE_mSSA_comp_mean_without_covid)
#     -rRMSEs are below 90% for shifts up to 4 quarters and for M-SSA designs optimized for larger forecast horizons 
#   -New predictor vs. direct forecasts: (see rRMSE_mSSA_comp_direct_without_covid) 
#     -rRMSEs are below 90% for shifts 2<=shift<=4 and for M-SSA designs optimized for larger forecast horizons 
#     -Outperformance of M-SSA is less strong for shifts<=1: direct forecasts are informative at short forecast horizons
#   -Direct forecast vs. mean benchmark: (see rRMSE_mSSA_direct_mean_without_covid)
#     -rRMSEs are below  (or close to) 90% for small shifts (shift<=1)
#     -Confirmation: direct forecasts are informative at short forecast horizons



################################################################################################################
# Exercise 2 Analyze revisions of M-SSA components predictor
# -The new predictor relies on quarterly up-dating of the (WLS-) regression weights
#   -Note that M-SSA is not subject to revisions because the VAR is fixed (based on data up to 2008: no up-dating)
# -We here analyze the impact of the quarterly up-dating on the predictor as well as on the regression weights

# 2.1 Compare the final predictor (full sample regression) with the out-of-sample sequence of 
#   continuously re-calibrated predictors: ideally (in the absence of revisions), both series would overlap
# -Differences illustrate revisions due to re-estimating regression weights each quarter
par(mfrow=c(1,1))
mplot<-cbind(final_components_preditor,oos_components_preditor)
colnames(mplot)<-c("Final predictor","Real-time out-of-sample predictor")
colo<-c("blue",rainbow(length(select_vec_multi)))
main_title<-"Revisions: final vs. real-time predictor"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
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
# Notes: 
# 1. The above plot compares real-time and final predictors for maximal shift and maximal forecast horizon
#     -Last run in the double-loop of exercise 1.3.5 above 
#     -Similar plots could be obtained for all combinations of forward-shift and forecast horizon (with no systematic change/difference)
# 2. The last data point in the above plot does not correspond to Jan-2025 (because the target is forward-shifted and NAs are removed)
tail(mplot)
#     -The purpose of the above plot is to illustrate revisions 
#     -It does not show the current forecast at the series end (Jan-2025)

# Comments (revisions): 
# -To the left of the plot, the real-time predictor is volatile because the sample is short (revisions are large)
# -With increasing sample size (from left to right), the real-time predictor approaches the final estimate 
# -The vertical black line in the plot indicates the start of the evaluation period, relevant for 
#     out-of-sample MSEs and p-value statistics, see exercise 1.3.5 above
# -As time progresses, we expect better out-of-sample forecast performances because the part imputable to the 
#   above revision error will decrease

# 2.2 We now examine the effect of the revisions on the regression weights  
par(mfrow=c(1,1))
mplot<-track_weights
colnames(mplot)[2:3]<-paste("Weight of M-SSA component ",colnames(track_weights)[2:3])
colo<-c("black","blue","red")
main_title<-"Revisions: regression weights over time"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=1,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
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

# Comments (revisions)
# -Regression weights are quite volatile at the start
# -However, progressively over time the weights appear to converge to some fix-points (stationarity)
#   -The real-time predictor converges to the final predictor


################################################################################################################
# Exercise 3 Apply the new M-SSA components predictor
# -Rely on the `final' M-SSA component predictor (at the sample end) to assess the business-cycle 
#   (based on data up to Jan-2025)
# -For illustration we here use forecast horizon h=4 (M-SSA optimized for one year ahead forecast) and 
#     forward-shifts in shift_vec (of BIP target)
# Remarks: 
# -M-SSA here still relies on the in-sample span up to the financial crisis and is not up-dated yet
# -Moreover, for simplicity, we here rely on OLS regression (of M-SSA components on forward-shifted BIP)
# -The final `best' M-SSA component predictor is derived and discussed in exercise 6 below
#   -We then rely on the full data set for estimating the VAR
#   -We rely on WLS regression, using the GARCH(1,1) vola for weighting the data
#   -We remove the Pandemic to obtain better estimates for the VAR as well as for the WLS regression

k<-5
# Check: forecast horizon h=4:
h_vec[k]
# Forward-shifts of BIP (+publication lag)
shift_vec<-shift_vec
mssa_predictor_mat<-NULL
# We compute the final predictor, based on data up to the sample end
# Note: for simplicity we here compute an OLS regression (WLS looks nearly the same)
for (shift in shift_vec)
{
# Data matrix: forward-shifted BIP and M-SSA components  
  if (length(sel_vec_pred)>1)
  {
    dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_pred,,k]))
  } else
  {
    dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mssa_array[sel_vec_pred,,k]))
  }
  rownames(dat)<-rownames(x_mat)
# OLS regression  
  lm_obj<-lm(dat[,1]~dat[,2:ncol(dat)])
  optimal_weights<-lm_obj$coef
# Compute predictor for each forward-shift  
  mssa_predictor_mat<-cbind(mssa_predictor_mat,optimal_weights[1]+dat[,2:ncol(dat)]%*%optimal_weights[2:length(optimal_weights)])
}  

# Plot M-SSA components predictors (optimized for h=4) and shifts in shift_vec
par(mfrow=c(1,1))
# Standardize for easier visual inspection
mplot<-scale(cbind(dat[,1],mssa_predictor_mat))
colnames(mplot)<-c(paste("BIP forward-shifted by ",shift," quarters (plus publication lag)",sep=""),
                   paste("h=",h_vec[k],", shift=",shift_vec,sep=""))
colo<-c("black",rainbow(4*ncol(mssa_predictor_mat)))
main_title<-paste("Final predictors based on M-SSA-components ",paste(sel_vec_pred,collapse=","),": h=",h_vec[k],sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],lwd=2,ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
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

# Comments:
# -The standardization of the series in the above plot somehow defeats the purpose of the M-SSA component 
#     predictors which are designed with MSE performances in mind. 
#   -But standardization simplifies visual inspection.
# -As for the M-SSA predictor in tutorial 7.3 (exercise 1), we observe an increasing left-shift of the new 
#     M-SSA component predictor with increasing forward-shift of the target 
#   -This left-shift is much less pronounced when computing direct forecasts (substituting the original data 
#     to the M-SSA components as regressors) 
#   -Forecast MSE outperformance at shifts>=2 (of M-SSA components vs. mean-benchmark and/or direct forecasts) is directly 
#     related to this left-shift, see the column entitled h=4 in the corresponding matrices 
rRMSE_mSSA_comp_mean_without_covid[,"h=4"]
rRMSE_mSSA_comp_direct_without_covid[,"h=4"]
# -The M-SSA component predictors confirm the earlier assessment obtained by the original M-SSA predictor 
#   -Data up to Jan-2025 suggests evidence of a recovery over 2025/2026


#################################################################################################
# Exercise 4: Can we explain forecast gains of the new M-SSA components predictor at forecast horizons larger 
#   than a quarter?

# The M-SSA component forecast is a rather complex `stacked' design, involving multiple steps 
#     in the derivation of the BIP predictor
# Construction steps:
#   -Filtering: remove undesirable high frequency noise 
#     -Filter based on HP(160) target to emphasizes mid-term dynamics relevant in a mid-term (2-4 quarters) 
#       forecast perspective
#   -M-SSA optimization criterion: maximize the target correlation under a HT (holding time) constraint
#   -WLS regression of M-SSA components on forward-shifted BIP

# -In order to check pertinence and relevance of this stacked construction principle, we verified outperformance
#     of the new M-SSA component predictor over the simple mean (mean of BIP) and the direct forecasts 
#     (regressing un-filtered indicators on future BIP) on a longer out-of-sample span comprising the 
#     financial crisis as well as the Pandemic.
# -However, at this stage, we are still unable to assess the role and the importance of M-SSA as a contributing 
#   element to improved forecast performances. 

# -Questions: 
#   -why does the M-SSA components predictor perform better than classic forecast rules at 
#     forecast horizons>=2 quarters? 
#   -Which steps (in the the above construction) are relevant? 

# -To answer these questions we here propose to consider a simple intermediary step as an additional benchmark
#   -This new benchmark is a simplification (a special case) of the M-SSA components predictor
# -Specifically, we consider an extension of the direct forecast, called direct HP forecast, as follows:
#   -We apply the classic univariate HP concurrent filter (HP-C) to each indicator
#   -The `direct HP forecast' is obtained by regressing the HP-C filtered indicators on future BIP
#   -The direct HP forecast is a special case of the M-SSA component predictor, wherein the multivariate 
#     filter (M-SSA) is replaced by a univariate design (HP-C).
# -M-SSA component, direct forecast and direct HP forecast designs differ only with respect to the explanatory 
#     variables that are regressed on future BIP:
#   -M-SSA components rely on multivariate filters of the indicators 
#   -direct forecasts rely on original (un-filtered) indicators
#   -direct HP forecasts rely on univariate HP-C filters of the indicators 
# -A comparison of these predictors will hint at the cause(s) and origin(s) of efficiency gains by the 
#   M-SSA component predictor


# Exercise 4.1: Compute the new benchmark (direct HP forecast)
# 4.1.1 Classic one-sided HP filter: HP-C

lambda_HP<-lambda_HP
L<-L

HP_obj<-HP_target_mse_modified_gap(L,lambda_HP)
# Classic concurrent (one-sided) HP filter
hp_c<-HP_obj$hp_trend
ts.plot(hp_c,main=paste("One-sided (concurrent) HP(",lambda_HP,"): HP-C",sep=""),xlab="",ylab="")

# We can analyze the filter in the frequency-domain (but this topic will not be discussed further here)
if (F)
{
  # Analyze filter in frequency-domain (amplitude function)
  # Specify the number of equidistant frequency ordinates in [0,pi]
  K<-600
  # Compute transfer, amplitude and shift functions (shift=phase divided by frequency)
  amp_obj_hp_c<-amp_shift_func(K,hp_c,F)
  
  # Plot amplitude function
  par(mfrow=c(1,1))
  mplot<-matrix(amp_obj_hp_c$amp,ncol=1)
  colnames(mplot)<-paste("Concurrent HP, lambda=",lambda_HP,sep="")
  colo<-c("blue",rainbow(ncol(mplot)))
  plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP, lambda=",lambda_HP,sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
  if (ncol(mplot)>1)
  {
    lines(mplot[,2],col=colo[2])
    abline(v=which(mplot[,1]==max(mplot[,1])),col=colo[1])
    mtext(colnames(mplot)[1],line=-1,col=colo[1])
    
    for (i in 2:ncol(mplot))
    {
      lines(mplot[,i],col=colo[i])
      mtext(colnames(mplot)[i],col=colo[i],line=-i)
    }
  }
  axis(1,at=1+0:4*K/4,labels=expression(0, pi/4, 2*pi/4,3*pi/4,pi))
  axis(2)
  box()
}

# 4.1.2 Filter the indicators: apply HP-C
hp_c_mat<-NULL
for (i in 1:ncol(x_mat))
{
  hp_c_mat<-cbind(hp_c_mat,filter(x_mat[,i],hp_c,side=1))
}
colnames(hp_c_mat)<-colnames(x_mat)
rownames(hp_c_mat)<-rownames(x_mat)

# Plot
mplot<-hp_c_mat
par(mfrow=c(1,1))
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("Concurrent HP(",lambda_HP,") applied to indicators",sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# 4.1.3 Compute filter outputs for forecast horizons 0:6, assuming WN data
# Forecast horizons
h_vec<-0:6
hp_c_array<-array(dim=c(ncol(x_mat),nrow(x_mat),length(h_vec)))
for (j in 1:length(h_vec))
{
  for (i in 1:ncol(x_mat))
  {
# For forecast horizon h_vec[j], the first h_vec[j] filter coefficients are skipped (zeroes are appended at 
#     the end). 
#   -This simple rule is optimal if the data is (close to) WN (white noise).
#   -Log-returns of the indicators are fairly close to WN (one can inspect the ACFs) 
    hp_c_forecast<-c(hp_c[(h_vec[j]+1):L],rep(0,h_vec[j]))
    hp_c_array[i,,j]<-filter(x_mat[,i],hp_c_forecast,side=1)
  }
}
dimnames(hp_c_array)[[1]]<-colnames(x_mat)
dimnames(hp_c_array)[[2]]<-rownames(x_mat)
dimnames(hp_c_array)[[3]]<-paste("h=",h_vec,sep="")

# Plot
# Select an indicator: for illustration we here choose BIP
i<-1
colnames(x_mat)[i]
# Plot HP nowcast and forecasts for that indicator: we scale the series for better visual inspection
mplot<-scale(hp_c_array[i,,])
colnames(mplot)<-paste(colnames(x_mat)[i],": h=",h_vec,sep="")
par(mfrow=c(1,1))
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("HP(",lambda_HP,") now- and forecasts applied to ",colnames(x_mat)[i],sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()
# For increasing forecast horizon, the filter outputs tend to be left-shifted



#------------------
# 4.2: Compute performances of the new direct HP forecast
# Numerical computations may take a couple minutes: therefore we already computed results which can be loaded
#   -But one can run the double loop to check results 
recompute_results<-recompute_results
shift_vec<-shift_vec
if (recompute_results)
{
# Initialize performance matrices
  p_mat_HP_c<-p_mat_HP_c_without_covid<-rRMSE_mSSA_comp_HP_c<-rRMSE_mSSA_comp_HP_c_without_covid<-matrix(ncol=length(h_vec),nrow=length(h_vec)-1)
# Use WLS
  use_garch<-T
  
# The following double loop computes all combinations of forward-shifts (of BIP) and forecast horizons 
# -We compute HAC-adjusted p-values (significance of out-of-sample predictor) and 
#  out-of-sample rRMSEs (relative root mean-square forecast errors) for the direct HP forecast

# Set-up progress bar: indicates progress in R-console
  pb <- txtProgressBar(min=min(h_vec),max=max(h_vec)-1,style=3)
  
  for (shift in shift_vec)#shift<-2
  {
# Progress bar: see R-console
    setTxtProgressBar(pb, shift)
    for (j in h_vec)#j<-5
    {
# Horizon j corresponds to k=j+1-th entry of array    
      k<-j+1
  
# Direct HP forecast:
#   -The explanatory variables are based on hp_c_array[,,k] 
#   -We select all indicators (one could easily change this setting but results are only marginally effected as long as ifo and ESi are included)
      dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(hp_c_array[,,k]))
      rownames(dat)<-rownames(x_mat)
      dat<-na.exclude(dat)
  
      perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift,lag_vec)
# Retrieve out-of-sample performances: p-values and forecast MSE, with/without Pandemic 
      MSE_oos_HP_c<-perf_obj$MSE_oos
      MSE_oos_HP_c_without_covid<-perf_obj$MSE_oos_without_covid
      p_mat_HP_c[shift+1,k]<-perf_obj$p_value 
      p_mat_HP_c_without_covid[shift+1,k]<-perf_obj$p_value_without_covid 
# Note that MSEs of M-SSA predictor were computed in exercise 1.3.5
      rRMSE_mSSA_comp_HP_c[shift+1,k]<-sqrt(MSE_oos_mssa_comp_mat[shift+1,k]/MSE_oos_HP_c)
# Same but without Pandemic
      rRMSE_mSSA_comp_HP_c_without_covid[shift+1,k]<-sqrt(MSE_oos_mssa_comp_without_covid_mat[shift+1,k]/MSE_oos_HP_c_without_covid)
    }
  }
  close(pb)
# Note: possible warnings issued by the GARCH estimation routine during computations can be ignored
  
# Assign column and rownames
  colnames(p_mat_HP_c)<-colnames(p_mat_HP_c_without_covid)<-
    colnames(rRMSE_mSSA_comp_HP_c)<-colnames(rRMSE_mSSA_comp_HP_c_without_covid)<-paste("h=",h_vec,sep="")
  rownames(p_mat_HP_c)<-rownames(p_mat_HP_c_without_covid)<-rownames(rRMSE_mSSA_comp_HP_c)<-rownames(rRMSE_mSSA_comp_HP_c_without_covid)<-
    paste("Shift=",shift_vec,sep="")
# Save results
  list_2<-list(p_mat_HP_c=p_mat_HP_c,p_mat_HP_c_without_covid=p_mat_HP_c_without_covid,rRMSE_mSSA_comp_HP_c=rRMSE_mSSA_comp_HP_c,rRMSE_mSSA_comp_HP_c_without_covid=rRMSE_mSSA_comp_HP_c_without_covid)
  if (F)
  {
    save(list_2,file=paste(getwd(),"/Results/list_2",sep=""))
  }
} else
{
# Load results  
  load(file=paste(getwd(),"/Results/list_2",sep=""))
  p_mat_HP_c=list_2$p_mat_HP_c
  p_mat_HP_c_without_covid=list_2$p_mat_HP_c_without_covid
  rRMSE_mSSA_comp_HP_c=list_2$rRMSE_mSSA_comp_HP_c
  rRMSE_mSSA_comp_HP_c_without_covid=list_2$rRMSE_mSSA_comp_HP_c_without_covid  
}


# HAC-adjusted p-values out-of-sample 
# -We here compare the M-SSA components predictor with the direct forecast and the new direct HP forecast
p_mat_mssa_components
p_mat_direct
p_mat_HP_c

# Same but without Pandemic
p_mat_mssa_components_without_covid
p_mat_direct_without_covid
p_mat_HP_c_without_covid


# Findings:
# -Like the classic direct forecast, the new direct HP forecast is unable to forecast BIP at shifts larger 
#     than one quarter (plus publication lag)
# -In contrast, the M-SSA components predictor remains significant for shifts up to four quarters (plus publication lag)
 


# We can compare the M-SSA components predictor to the direct forecast and the direct HP forecast in terms 
#   of rRMSEs out-of-sample
rRMSE_mSSA_comp_direct_without_covid
rRMSE_mSSA_comp_HP_c_without_covid

# -The above results suggest that applying the classic (univariate) HP-C to the data does not improve performances
#   when compared to direct forecasts (based on un-filtered data)
# -Therefore, outperformance by the M-SSA components predictor at larger forward-shifts cannot be replicated 
#     or explained by a `simple` univariate (HP-C) filter 
# -Also, outperformance of the direct forecasts by M-SSA suggests that the multivariate  
#   aspect of the forecast problem cannot be handled by simple (WLS) regression (with or without application of HP-C)
# -We then infer that the BIP forecast problem requires a simultaneous treatment of longitudinal and 
#   cross-sectional aspects, such as provided by M-SSA in combination with (WLS-) regression 
# -But why? What is the added benefit of the `M-SSA step` in the construction of the predictor?

# In order to understand the contribution of the multivariate M-SSA framework (over HP-C) we here 
#   consider two different target series: BIP (lagging) and spread (leading)

# -When targeting HP-BIP, i.e., the two-sided HP applied to BIP, we expect the multivariate design to 
#     extract relevant information from the leading series
#   -The explanatory series ip, ESI, ifo and spread are all leading when referenced against BIP 
#       (accounting for the publication lag of BIP). 
#   -In this case, we expect M-SSA to outperform HP-C (because the latter cannot rely on leading data). 
# -On the other hand, when targeting HP-spread, the additional explanatory series (BIP,ip,ifo,ESI) are 
#     lagging, when referenced against spread
#   -Therefore we do not expect substantial gains of M-SSA over HP-C in this case.

# To verify the above conjectures we generate a main plot with 4 sub-panels: 
# -panel a (top-left): HP-C applied to BIP
# -panel b (top-right): M-SSA applied to BIP (in this case we expect M-SSA to outperform HP-C) 
# -panel c (bottom left): HP-C applied to spread
# -panel d (bottom right): M-SSA applied to spread (in this case we do not expect M-SSA to outperform HP-C) 

# a. HP-C applied to BIP
i<-1
colnames(x_mat)[i]
# Plot HP-C for that indicator: we scale the series for better visual inspection
mplot<-scale(hp_c_array[i,,])
colnames(mplot)<-paste(colnames(x_mat)[i],": h=",h_vec,sep="")
par(mfrow=c(2,2))
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("HP-C targeting HP-",colnames(x_mat)[i],sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()
# b. M-SSA applied to BIP 
mplot<-scale(mssa_array[i,,])
colnames(mplot)<-paste(colnames(x_mat)[i],": h=",h_vec,sep="")
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("M-SSA targeting HP-",colnames(x_mat)[i],sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()
# c. HP-C applied to spread
i<-5
colnames(x_mat)[i]
# Plot HP-C for that indicator: we scale the series for better visual inspection
mplot<-scale(hp_c_array[i,,])
colnames(mplot)<-paste(colnames(x_mat)[i],": h=",h_vec,sep="")
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("HP-C targeting HP-",colnames(x_mat)[i],sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()
# d. M-SSA applied to spread 
mplot<-scale(mssa_array[i,,])
colnames(mplot)<-paste(colnames(x_mat)[i],": h=",h_vec,sep="")
colo<-c(rainbow(ncol(mplot)))
main_title<-paste("M-SSA targeting HP-",colnames(x_mat)[i],sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (j in 1:ncol(mplot))
{
  lines(mplot[,j],col=colo[j],lwd=1,lty=1)
  mtext(colnames(mplot)[j],col=colo[j],line=-j)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# Analysis:
# -Let us first look at the top two panels: targeting HP-BIP 
#   -The main difference between the classic HP-C (top left) and M-SSA (top right) is the size and the 
#     `quality' of the left-shift of the respective predictors as a function of the forecast horizon:
#   -Size: the left-shift at the zero-crossings is stronger with M-SSA (top right)
#   -Quality: 
#     -In contrast to HP-C, M-SSA also shows evidence of a (more pronounced) left-shift of dips and peaks, 
#       in particular at recessions/crises
#     -The left-shift of M-SSA operates at all levels: not only at zero-crossings but also at local peaks 
#       and dips and more generally, at all levels of the data
# -Next, let us look at the bottom two panels: targeting HP-spread 
#   -In this case, M-SSA (bottom right) and HP-C (bottom left) are nearly identical 
#   -No outperformance (in terms of left-shift) by M-SSA

# -The explanation is pretty simple: 
#   -The multivariate M-SSA filter targeting HP-spread simplifies (`collapses') to a univariate design,  
#     see tutorial 7.1, exercise 1.5
#   -This is because all additional explanatory series are lagging (relative to spread): they are not 
#     informative when targeting HP-spread

# -From the above, we infer that M-SSA outperforms HP-C when targeting HP-BIP (a similar outcome would apply to 
#   HP-ip, not shown).
# -Moreover, M-SSA BIP and M-SSA ip components are the most important explanatory variables when targeting 
#   future BIP within the WLS regression (the remaining components are not significant) 
# -Therefore, M-SSA contributes to the forecast outperformance by generating `better' regressors for the 
#   WLS-regression, with a more effective left-shift than either HP-C (in the direct HP 
#   forecast) or the un-filtered indicators (in the direct forecasts)


# Findings:
# -Applying a classic one-sided HP-C does not improve forecast performances (over the classic direct 
#     forecast, i.e., un-filtered data).
# -The BIP forecast problem is more `demanding' (complex) and asks for more advanced `signal extraction' 
#   than afforded by regression, eventually in combination with univariate HP-C filtering.


# Let us summarize the key design-elements of the proposed M-SSA component predictor:
#   -Emphasis of the relevant mid-term components: HP(160) target (see exercises 3 and 4 of tutorial 7.3 
#       for a justification of this target)
#   -Efficient exploitation of longitudinal and cross-sectional information when targeting HP-BIP
#     -Exploit leading (in relative terms) indicators to (significantly) improve forecasts of HP-BIP, up to 
#       several quarters ahead
#     -Address more effectively the size as well as the quality of the left-shift of the 
#       predictor (as a function of the forecast horizon) by M-SSA
#   -Address and control smoothness (rate of zero-crossings) of the predictor: HT constraint, see exercise 5 below

# These findings justify the  `more complex' forecast design proposed in this tutorial, 
#   wherein the key property of the predictor, namely the pronounced left-shift as a function of the 
#   forecast horizon, is obtained within the M-SSA optimization framework to track forward-shifted BIP 
#   up to one year ahead.

#############################################################################################
# Exercise 5: M-MSE vs. M-SSA
# -We illustrated in exercise 4 that a multivariate filter approach outperforms univariate (HP-C) filtering 
#   (or no filtering at all) by generating a more systematic left-shift of the M-SSA component predictor
# -However, we have not compared M-SSA to M-MSE, the classic multivariate mean-square error filter 
#   -The M-MSE filter does not impose a HT constraint: it is less smooth (more zero-crossings) 
# -We here briefly derive forecast performances of the `M-MSE component predictor' and compare results to the 
#   simple mean as well as to the M-SSA component predictor
# -Construction of the M-MSE component predictor is straightforward: 
#   -Instead of M-SSA components (filter outputs) we regress M-MSE components (filter outputs) on 
#     forward-shifted BIP
#   -In the code snippet below we substitute mmse_array for mssa_array in the data-matrix dat

# Exercise 5.1: compute M-MSE-component predictor
recompute_results<-recompute_results
shift_vec<-shift_vec
if (recompute_results)
{
  # Initialize performance matrices
  rRMSE_mmse_comp_mean<-rRMSE_mmse_comp_mean_without_covid<-rRMSE_mmse_comp_mssa<-rRMSE_mmse_comp_mssa_without_covid<-matrix(ncol=length(h_vec),nrow=length(h_vec)-1)
  # Use WLS
  use_garch<-T
  
  # The following double loop computes all combinations of forward-shifts (of BIP) and forecast horizons 
  # -We compute HAC-adjusted p-values (significance of out-of-sample predictor) and 
  #  out-of-sample rRMSEs (relative root mean-square forecast errors) for the direct HP forecast
  
  # Set-up progress bar: indicates progress in R-console
  pb <- txtProgressBar(min=min(h_vec),max=max(h_vec)-1,style=3)
  
  for (shift in shift_vec)#shift<-2
  {
    # Progress bar: see R-console
    setTxtProgressBar(pb, shift)
    for (j in h_vec)#j<-5
    {
      # Horizon j corresponds to k=j+1-th entry of array    
      k<-j+1
      
# M-MSE component predictor
# Specify data matrix for WLS regression: we insert mmse_array instead of mssa_array for the explanatory variables
      if (length(sel_vec_pred)>1)
      {
        dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mmse_array[sel_vec_pred,,k]))
      } else
      {
        dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mmse_array[sel_vec_pred,,k]))
      }
      rownames(dat)<-rownames(x_mat)
      colnames(dat)<-c(colnames(x_mat)[1],sel_vec_pred)
      dat<-na.exclude(dat)
      
      perf_obj<-optimal_weight_predictor_func(dat,in_out_separator,use_garch,shift,lag_vec)
# Out-of-sample performances: p-values and forecast MSE, with/without Pandemic 
      MSE_oos_mmse<-perf_obj$MSE_oos
      MSE_oos_mmse_without_covid<-perf_obj$MSE_oos_without_covid
      MSE_oos_mean<-perf_obj$MSE_mean_oos
      MSE_oos_mean_without_covid<-perf_obj$MSE_mean_oos_without_covid
# Compute rRMSEs
# a. M-MSE vs- M-SSA
#   Note that MSEs of M-SSA predictor were computed in exercise 1.3.5
      rRMSE_mmse_comp_mssa[shift+1,k]<-sqrt(MSE_oos_mmse/MSE_oos_mssa_comp_mat[shift+1,k])
# Same but without Pandemic
      rRMSE_mmse_comp_mssa_without_covid[shift+1,k]<-sqrt(MSE_oos_mmse_without_covid/MSE_oos_mssa_comp_without_covid_mat[shift+1,k])
# b. M-MSE vs. mean
      rRMSE_mmse_comp_mean[shift+1,k]<-sqrt(MSE_oos_mmse/MSE_oos_mean)
# Same but without Pandemic
      rRMSE_mmse_comp_mean_without_covid[shift+1,k]<-sqrt(MSE_oos_mmse_without_covid/MSE_oos_mean_without_covid)
    }
  }
  close(pb)
  # Note: possible warnings issued by the GARCH estimation routine during computations can be ignored
  
  # Assign column and rownames
  colnames(rRMSE_mmse_comp_mssa)<-colnames(rRMSE_mmse_comp_mssa_without_covid)<-
    colnames(rRMSE_mmse_comp_mean)<-colnames(rRMSE_mmse_comp_mean_without_covid)<-paste("h=",h_vec,sep="")
  rownames(rRMSE_mmse_comp_mssa)<-rownames(rRMSE_mmse_comp_mssa_without_covid)<-
    rownames(rRMSE_mmse_comp_mean)<-rownames(rRMSE_mmse_comp_mean_without_covid)<-paste("Shift=",shift_vec,sep="")
  # Save results
  list_3<-list(rRMSE_mmse_comp_mssa=rRMSE_mmse_comp_mssa,rRMSE_mmse_comp_mssa_without_covid=rRMSE_mmse_comp_mssa_without_covid,
               rRMSE_mmse_comp_mean=rRMSE_mmse_comp_mean,rRMSE_mmse_comp_mean_without_covid=rRMSE_mmse_comp_mean_without_covid)
  if (F)
  {
    save(list_3,file=paste(getwd(),"/Results/list_3",sep=""))
  }
} else
{
  # Load results  
  load(file=paste(getwd(),"/Results/list_3",sep=""))
  rRMSE_mmse_comp_mssa=list_3$rRMSE_mmse_comp_mssa
  rRMSE_mmse_comp_mssa_without_covid=list_3$rRMSE_mmse_comp_mssa_without_covid
  rRMSE_mmse_comp_mean=list_3$rRMSE_mmse_comp_mean
  rRMSE_mmse_comp_mean_without_covid=list_3$rRMSE_mmse_comp_mean_without_covid
}

#-------------------
# Exercise 5.2: evaluate out-of-sample performances of M-MSE component predictor
#   -We here emphasize a four quarters ahead forecast (challenging forecast problem)

# 5.2.1 Compute Final M-MSE and M-SSA component predictors (whose regression relies on 
#   the full data sample)

# Select h and shift (should be smaller or equal 5)
h<-5
if (h>5)
  h=5
# Select forward-shift
shift<-h

# Compute the final M-MSE component predictor optimized for forecast horizon h
# Note: for simplicity we here compute an OLS regression (WLS looks nearly the same)
if (length(sel_vec_pred)>1)
{
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mmse_array[sel_vec_pred,,h+1]))
} else
{
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mmse_array[sel_vec_pred,,h+1]))
}
# OLS regression  
lm_obj<-lm(dat[,1]~dat[,2:ncol(dat)])
optimal_weights<-lm_obj$coef
# Compute predictor for each forward-shift  
mmse_predictor<-optimal_weights[1]+dat[,2:ncol(dat)]%*%optimal_weights[2:length(optimal_weights)]

# Compute the final M-SSA component predictor optimized for forecast horizon h
if (length(sel_vec_pred)>1)
{
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),t(mssa_array[sel_vec_pred,,h+1]))
} else
{
  dat<-cbind(c(x_mat[(shift+lag_vec[1]+1):nrow(x_mat),1],rep(NA,shift+lag_vec[1])),(mssa_array[sel_vec_pred,,h+1]))
}
# OLS regression  
lm_obj<-lm(dat[,1]~dat[,2:ncol(dat)])
optimal_weights<-lm_obj$coef
# Compute predictor for each forward-shift  
mssa_predictor<-optimal_weights[1]+dat[,2:ncol(dat)]%*%optimal_weights[2:length(optimal_weights)]


# 5.2.2 Holding times
# Let's measure smoothness in terms of empirical holding-times
# -M-SSA imposes a larger HT (than the `natural' HT of M-MSE) and therefore it should be smoother
# -Note: 
#   -We imposed a 50% larger expected (true) HT than M-MSE in the HT constraint of the optimization criterion
#   -Ideally, the empirical HT of M-SSA should be (roughly) 50% larger than M-MSE, too.
compute_empirical_ht_func(mssa_predictor)
compute_empirical_ht_func(mmse_predictor)
# M-SSA has approximately 33% less crossings (roughly 50% larger empirical HT)
#   -This feature of the predictor can be controlled by the HT hyperparameter
#   -The sample estimate is close to the expected (true) number 
#     -As shown in tutorial 7.1, the sample HT converges to the imposed (expected) HT for sufficiently 
#     long samples if the data generating process is known


# 5.2.3 MSE forecast performances
# M-MSE vs. mean
rRMSE_mmse_comp_mean_without_covid
# M-SSA components vs. mean
rRMSE_mSSA_comp_mean_without_covid
# M-MSE vs. M-SSA: rRMSE<1 signifies that M-MSE perform better
rRMSE_mmse_comp_mssa_without_covid

# Findings
# -M-SSA does not perform markedly worse than M-MSE in terms of forecast MSE, 
#     despite increased smoothness (larger HT, fewer zero-crossings). 
#   -This outcome is quite remarkable and pleads in favor of the M-SSA components predictor (as a 
#     potentially more interesting/relevant predictor than M-MSE).
# -Stronger outperformance of M-SSA (rMSE>1) at shifts=2,3 and forecast horizons h=2,3,4
# -At forecast horizon h=5 M-SSA does not outperform and at h=6 M-SSA is possibly slightly outperformed by M-MSE 


# Summary
# -M-SSA and M-MSE component predictors perform roughly similarly in terms of out-of-sample MSE forecast 
#     performances when targeting forward-shifted BIP
#   -Both predictors outperform the mean, the direct forecast, the direct HP forecast and the original 
#     M-SSA predictor (tutorial 7.3), specifically at larger forward shifts (shifts>=1 quarter).
# -The M-SSA component predictor is smoother (less noisy, fewer zero-crossings) 
#   -The smoothness of M-SSA can be controlled by the HT hyperparameter
# -Interestingly, increased smoothness of M-SSA does not impair (out-of-sample) MSE forecast performances or 
#     lag (retardation) when references against M-MSE
#   -Therefore we may prefer the M-SSA componnet predictor (over M-MSE) in this application.
# -The M-MSE component predictor can be replicated by the M-SSA component predictor by inserting the 
#     former's HTs into the constraint
#   -The M-SSA component predictor is more general 
#   -The optimization principle offers more control on important characteristics (`shape') of the predictor 

################################################################################################
# Exercise 6: Compute final M-SSA and M-MSE component predictors 
# -We here up-date the VAR-model for M-SSA and M-MSE
# -We apply the final WLS regression, based on all available data and the GARCH(1,1)-vola
# -We remove Pandemic to obtain better estimates for VAR and regression equations
# -We apply M-SSA and M-MSE component predictors to standardized data (exercise 6.2) as well as to 
#     original BIP growth (without standardization, see exercise 6.3). 
#   -Note that rRMSEs or p-value statistics are indifferent to standardization 

# 6.1 Up-date M-SSA and M-MSE
# In-sample span for VAR, use all data for VAR
date_to_fit<-"3000"
# Remove Pandemic
x_mat_wc<-x_mat[c(which(rownames(x_mat)<2020),which(rownames(x_mat)>2021)),]
par(mfrow=c(1,1))
ts.plot(x_mat_wc)

# Run the M-SSA wrapper, see tutorial 7.2
#   -The function computes M-SSA and M-MSE for each forecast horizon h in h_vec
final_mssa_indicator_obj<-compute_mssa_BIP_predictors_func(x_mat_wc,lambda_HP,L,date_to_fit,p,q,ht_mssa_vec,h_vec,f_excess,lag_vec,select_vec_multi)

# Final M-SSA components
final_mssa_array<-final_mssa_indicator_obj$mssa_array
# Final M-MSE components
final_mmse_array<-final_mssa_indicator_obj$mmse_array


#----------------------
# 6.2 Compute up-dated M-SSA and M-MSE component predictors based on WLS regression

# Select h and shift (should be smaller or equal 5 or 6)
# We select a one-year ahead forecast h=4, for its potential importance in applications
# Background (short resume):
# -The forecast horizon h conditions M-SSA to track HP-filtered targets h-steps ahead
# -The corresponding M-SSA or M-MSE components are then used as regressors in the subsequent WLS regressions 
#   on forward-shifted BIP 
# -The output of the WLS regressions are the M-SSA and M-MSE component predictors
#   -Two-stage (stacked) predictor design
h<-4
if (h>6)
  h=6
# Select forward-shift: we set shift=h 
#   -WLS regression targets BIP shifted forward by shift (+publication lag)
shift<-h

# 6.2.1 M-MSE component predictor optimized for forecast horizon h
# We here rely on GARCH(1,1) and WLS regression
if (length(sel_vec_pred)>1)
{
  dat<-cbind(c(x_mat_wc[(shift+lag_vec[1]+1):nrow(x_mat_wc),1],rep(NA,shift+lag_vec[1])),t(final_mmse_array[sel_vec_pred,,h+1]))
} else
{
  dat<-cbind(c(x_mat_wc[(shift+lag_vec[1]+1):nrow(x_mat_wc),1],rep(NA,shift+lag_vec[1])),(final_mmse_array[sel_vec_pred,,h+1]))
}
y.garch_11<-garchFit(~garch(1,1),data=na.exclude(dat[,1]),include.mean=T,trace=F)
# sigmat could be retrieved from GARCH-object
sigmat<-y.garch_11@sigma.t
# But this is lagged by one period
# Therefore we recompute the vola based on the estimated GARCH-parameters
eps<-y.garch_11@residuals
d<-y.garch_11@fit$matcoef["omega",1]
alpha<-y.garch_11@fit$matcoef["alpha1",1]
beta<-y.garch_11@fit$matcoef["beta1",1]
sigmat_own<-sigmat
for (i in 2:length(sigmat))#i<-2
  sigmat_own[i]<-sqrt(d+beta*sigmat_own[i-1]^2+alpha*eps[i]^2)
# This is now correct (not lagging anymore)
sigmat<-sigmat_own
# WLS 
weight_short<-1/sigmat^2
# OLS
weight_short<-rep(1,length(sigmat))
# Shift vola by shift+lag_vec[1] (see exercise 1.3.3) 
weight<-c(weight_short,rep(weight_short[1],shift+lag_vec[1]))
# Regression  
lm_obj<-lm(dat[,1]~dat[,2:ncol(dat)],weight=weight)
optimal_weights<-lm_obj$coef
# Compute predictor for each forward-shift  
final_mmse_predictor<-optimal_weights[1]+dat[,2:ncol(dat)]%*%optimal_weights[2:length(optimal_weights)]


# 6.2.2 M-SSA component predictor optimized for forecast horizon h
# We here rely on GARCH(1,1) and WLS regression
if (length(sel_vec_pred)>1)
{
  dat<-cbind(c(x_mat_wc[(shift+lag_vec[1]+1):nrow(x_mat_wc),1],rep(NA,shift+lag_vec[1])),t(final_mssa_array[sel_vec_pred,,h+1]))
} else
{
  dat<-cbind(c(x_mat_wc[(shift+lag_vec[1]+1):nrow(x_mat_wc),1],rep(NA,shift+lag_vec[1])),(final_mssa_array[sel_vec_pred,,h+1]))
}
# Regression  
lm_obj<-lm(dat[,1]~dat[,2:ncol(dat)],weight=weight)
optimal_weights<-lm_obj$coef
# Compute predictor for each forward-shift  
final_mssa_predictor<-optimal_weights[1]+dat[,2:ncol(dat)]%*%optimal_weights[2:length(optimal_weights)]



# 6.2.3 Plot final M-MSE and M-SSA predictors
par(mfrow=c(2,1))
mplot<-scale(cbind(c(x_mat_wc[(shift+lag_vec[1]+1):nrow(x_mat_wc),1],rep(NA,shift+lag_vec[1])),final_mssa_predictor,final_mmse_predictor))
colnames(mplot)<-c(paste("BIP shifted forward by ",shift," (plus publication lag)",sep=""),"M-SSA component predictor","M-MSE component predictor")
colo<-c("black","blue","green")
main_title<-paste("Forward-shifted BIP and Predictors: Pandemic episode removed",sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (jj in 1:ncol(mplot))
{
  lines(mplot[,jj],col=colo[jj],lwd=1,lty=1)
  mtext(colnames(mplot)[jj],col=colo[jj],line=-jj)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

mplot<-cbind(rep(0,nrow(final_mssa_predictor)),final_mssa_predictor,final_mmse_predictor)
colnames(mplot)<-c("","M-SSA component predictor","M-MSE component predictor")
main_title<-paste("Predictors: M-SSA component vs. M-MSE component, h=",h,", shift=",shift,sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (jj in 1:ncol(mplot))
{
  lines(mplot[,jj],col=colo[jj],lwd=1,lty=1)
  mtext(colnames(mplot)[jj],col=colo[jj],line=-jj)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# 6.2.4 Smoothness
# -The above plot suggests that the M-SSA component predictor is smoother than M-MSE without lagging the latter
# -Let us check the empüirical HTs for confirmation:

compute_empirical_ht_func(final_mssa_predictor)
compute_empirical_ht_func(final_mmse_predictor)

#---------------------------
# Exercise 6.3 Apply predictor to original BIP growth-rate (without standardization)
# -Purpose: obtain prediction of effective BIP growth 

# 6.3.1 Compute data set with original BIP (without standardization)
data_file_name<-c("Data_HWI_2025_02.csv","gdp_2025_02.csv")
# Quarterly data: BIP in the first data column
data_quarterly<-read.csv(paste(getwd(),"/Data/",data_file_name[2],sep=""))
BIP_original<-data_quarterly[,"BIP"]

# Plot BIP and diff-log BIP
par(mfrow=c(2,1))
mplot<-matrix(BIP_original)
rownames(mplot)<-data_quarterly[,"Date"]
colnames(mplot)<-"Original BIP"
colo<-c("black","blue","green")
main_title<-"Original BIP"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()
diff_log_BIP<-na.exclude(diff(log(BIP_original)))
names(diff_log_BIP)<-rownames(data)
# First differences of original log-BIP
mplot<-matrix(diff_log_BIP)
rownames(mplot)<-names(diff_log_BIP)
colnames(mplot)<-"diff-log BIP"
colo<-c("black","blue","green")
main_title<-"First differences of original log-BIP"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# Check: verify that our data, i.e., x_mat[,"BIP"], matches scaled diff-log BIP up to trimming
par(mfrow=c(1,1))
mplot<-cbind(scale(diff_log_BIP),x_mat[,"BIP"])
colnames(mplot)<-c("scaled diff-log BIP","trimmed and scaled diff-log BIP")
colo<-c("black","blue","green")
main_title<-"Trimmed and untrimmed scaled diff-log BIP"
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (jj in 1:ncol(mplot))
{
  lines(mplot[,jj],col=colo[jj],lwd=1,lty=1)
  mtext(colnames(mplot)[jj],col=colo[jj],line=-jj)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

# Define new data set with original diff-log BIP in first column
x_mat_original_BIP<-x_mat
x_mat_original_BIP[,"BIP"]<-diff_log_BIP
# Remove Pandemic
x_mat_original_BIP_wc<-x_mat_original_BIP[c(which(rownames(x_mat_original_BIP)<2020),which(rownames(x_mat_original_BIP)>2021)),]


# 6.3.2 M-MSE component predictor optimized for forecast horizon h
# We here rely on GARCH(1,1) and WLS regression
if (length(sel_vec_pred)>1)
{
  dat<-cbind(c(x_mat_original_BIP_wc[(shift+lag_vec[1]+1):nrow(x_mat_original_BIP_wc),1],rep(NA,shift+lag_vec[1])),t(final_mmse_array[sel_vec_pred,,h+1]))
} else
{
  dat<-cbind(c(x_mat_original_BIP_wc[(shift+lag_vec[1]+1):nrow(x_mat_original_BIP_wc),1],rep(NA,shift+lag_vec[1])),(final_mmse_array[sel_vec_pred,,h+1]))
}
y.garch_11<-garchFit(~garch(1,1),data=na.exclude(dat[,1]),include.mean=T,trace=F)
# sigmat could be retrieved from GARCH-object
sigmat<-y.garch_11@sigma.t
# But this is lagged by one period
# Therefore we recompute the vola based on the estimated GARCH-parameters
eps<-y.garch_11@residuals
d<-y.garch_11@fit$matcoef["omega",1]
alpha<-y.garch_11@fit$matcoef["alpha1",1]
beta<-y.garch_11@fit$matcoef["beta1",1]
sigmat_own<-sigmat
for (i in 2:length(sigmat))#i<-2
  sigmat_own[i]<-sqrt(d+beta*sigmat_own[i-1]^2+alpha*eps[i]^2)
# This is now correct (not lagging anymore)
sigmat<-sigmat_own
# WLS 
weight_short<-1/sigmat^2
# OLS
weight_short<-rep(1,length(sigmat))
# Shift vola by shift+lag_vec[1] (see exercise 1.3.3) 
weight<-c(weight_short,rep(weight_short[1],shift+lag_vec[1]))
# Regression  
lm_obj<-lm(dat[,1]~dat[,2:ncol(dat)],weight=weight)
optimal_weights<-lm_obj$coef
# Compute predictor for each forward-shift  
final_mmse_predictor<-optimal_weights[1]+dat[,2:ncol(dat)]%*%optimal_weights[2:length(optimal_weights)]


# 6.3.3 M-SSA component predictor optimized for forecast horizon h
# We here rely on GARCH(1,1) and WLS regression
if (length(sel_vec_pred)>1)
{
  dat<-cbind(c(x_mat_original_BIP_wc[(shift+lag_vec[1]+1):nrow(x_mat_original_BIP_wc),1],rep(NA,shift+lag_vec[1])),t(final_mssa_array[sel_vec_pred,,h+1]))
} else
{
  dat<-cbind(c(x_mat_original_BIP_wc[(shift+lag_vec[1]+1):nrow(x_mat_original_BIP_wc),1],rep(NA,shift+lag_vec[1])),(final_mssa_array[sel_vec_pred,,h+1]))
}
# Regression  
lm_obj<-lm(dat[,1]~dat[,2:ncol(dat)],weight=weight)
optimal_weights<-lm_obj$coef
# Compute predictor for each forward-shift  
final_mssa_predictor<-optimal_weights[1]+dat[,2:ncol(dat)]%*%optimal_weights[2:length(optimal_weights)]



# 6.3.4 Plot final M-MSE and M-SSA predictors
par(mfrow=c(2,1))
mplot<-(cbind(c(x_mat_original_BIP_wc[(shift+lag_vec[1]+1):nrow(x_mat_original_BIP_wc),1],rep(NA,shift+lag_vec[1])),final_mssa_predictor,final_mmse_predictor))
colnames(mplot)<-c(paste("BIP shifted forward by ",shift," (plus publication lag)",sep=""),"M-SSA component predictor","M-MSE component predictor")
colo<-c("black","blue","green")
main_title<-paste("Forward-shifted BIP and Predictors: Pandemic episode removed",sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (jj in 1:ncol(mplot))
{
  lines(mplot[,jj],col=colo[jj],lwd=1,lty=1)
  mtext(colnames(mplot)[jj],col=colo[jj],line=-jj)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()

mplot<-cbind(rep(0,nrow(final_mssa_predictor)),final_mssa_predictor,final_mmse_predictor)
colnames(mplot)<-c("","M-SSA component predictor","M-MSE component predictor")
main_title<-paste("Predictors: M-SSA component vs. M-MSE component, h=",h,", shift=",shift,sep="")
plot(mplot[,1],main=main_title,axes=F,type="l",xlab="",ylab="",col=colo[1],ylim=c(min(na.exclude(mplot)),max(na.exclude(mplot))))
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (jj in 1:ncol(mplot))
{
  lines(mplot[,jj],col=colo[jj],lwd=1,lty=1)
  mtext(colnames(mplot)[jj],col=colo[jj],line=-jj)
}
abline(h=0)
axis(1,at=c(1,12*1:(nrow(mplot)/12)),labels=rownames(mplot)[c(1,12*1:(nrow(mplot)/12))])
axis(2)
box()







#############################################################################################################


# Final remark: 
# -We assume a publication lag of two quarters for BIP which is systematically added to the 
#   forward-shift in our computations (the effective publication lag is only one quarter).  
# -This assumption is made to compensate for the omission of data revisions, which are ignored in our 
#     designs(s)
