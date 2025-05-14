# In this tutorial we propose an application of SSA to Hamilton's regression  filter (HF)
# For illustration, we consider quarterly GDP (example 1) and monthly non-farm payroll (examples 2 and 3) data  
# We analyze long sample periods (starting at WWII) as well as shorter spans (starting at the great moderation)

# Main outcomes: 
#   1.HF is a lowpass (transformed filter as applied to returns): it does not generate spurious cycles
#       In contrast to BK, CF or HP-gap bandpass designs, see tutorials 2 (example 7), 4 and 5
#   2.HF removes the (remaining) weak drift of the returns: it can address arbitrary integration orders by modifying the AR-order p of the regression
#   3.By its very definition, the HF filter depends on the data-sample (regression model)
#       -The out-of-sample cycle is non-stationary (regression parameters do not sum to one).
#       -Therefore, the filter-regression must be continuously up-dated, as new information becomes available.
#       -Up-dating the regression generates (undesirable) revisions.
#   4.HF has a rather small holding-time and a rather large phase-lag 
#       It is subject to noise leakage (`noisy' zero crossings) and retardation
#       Therefore, an application of SSA is not inopportune
#   5.We want the SSA-nowcast to be smoother than HF (~30% less crossings in the long run)
#   6.The SSA forecasts (6 months and 1-year) are smooth, too, and leading 


# Note: our intention is not to push a particular BCA-tool. Rather, we strive at illustrating that a particular 
#   predictor or BCA-filter (any one as long as it's linear in the data) can be replicated and modified by SSA 
#   in view of addressing 
# 1. smoothness (noise suppression) and 
# 2. timeliness (advancement)
# In this perspective, HF is considered as a basic platform and a vitrine for showcasing SSA
#   -We offer a number of compelling performance measures, confirming pertinence of a simple novel optimization principle  

#-----------------------------------------------------------------------
# Make a clean-sheet, load packages and functions

rm(list=ls())

library(xts)
# This package implements the `never hpfilter', i.e., the Hamilton filter:  
# It is used here for background information only (generate some plots): we deploy own code for the Hamilton filter 
# The Hamilton filter is a breeze to implement: simple linear regression.
#   But the filter can change substantially, depending on the sample window: see examples 2 and 3
library(neverhpfilter)
# Load data from FRED with library quantmod
library(quantmod)


# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))

# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))

#-------------------------------------------------
# Example 1
# Application of SSA to HF based on quarterly GDP data
# We briefly illustrate HF with the R-package neverhpfilter 
# Data is retrieved from neverhpfilter package, too
data(GDPC1)

gdp_trend <- yth_filter(100*log(GDPC1), h = 8, p = 4,output = c("x", "trend"))

plot.xts(gdp_trend, grid.col = "white", legend.loc = "topleft", main = "Log of GDP and trend")

gdp_cycle <- yth_filter(100*log(GDPC1), h = 8, p = 4,output = c("cycle", "random"))

# Often the cycle and the random components correlate strongly which contradicts the classic assumptions
# The cycle is centered at the zero-line: HF can remove p unit-roots, where p is the autoregressive order in the regression
plot.xts(gdp_cycle, grid.col = "white", legend.loc = "topleft", main = "Cycle and irregular")

#---------------------------------------------------------
# We now skip the above environment and R-package and implement the filter with own code. 
# This will allow to plug SSA on HF and to modify characteristics of the original design.

# Source data directly from FRED
getSymbols('GDPC1',src='FRED')


# Make double: xts objects are subject to lots of automatic/hidden assumptions which make an application of SSA 
#     more cumbersome, counter-intuitive, unpredictable and hazardous (try applying a filter to a xts-object...).
# We here skip the pandemic: outliers affect HF-regression.
# Effects of the pandemic are analyzed in our last example. 
y<-as.double(log(GDPC1["/2019"]))
len<-length(y)
#------------------
# 1.1 Hamilton filter
# Settings proposed by Hamilton for quarterly data
h<-2*4
p<-4

# Regression of y_{t+h} on y_t,y_{t-1},...,y_{t-p}
explanatory<-cbind(y[(p):(len-h)],y[(p-1):(len-h-1)],y[(p-2):(len-h-2)],y[(p-3):(len-h-3)])
target<-y[(h+p):len]
# The time window here is very long, reaching back to WWII
# We will be looking at a shorter subsample, starting in 1990, in example 4 further down
#   This problem does not affect relative performances of SSA (over HF)
lm_obj<-lm(y[(h+p):len]~explanatory)

# Only the first coefficient is significant, as is often the case in applications to non-stationary economic series.
# This is a potential problem because the sum of the parameters is not one and therefore 1-sum(coefficients)!=0.
# As a result the forecast residual y_{t+h}-\hat{y}_{t+h} is stationary in-sample (due to overfitting) but non-stationary out-of-sample. 
# Stated differently: the out-of-sample forecast and the future observation are not `cointegrated' (the MSE diverges asymptotically).
# One of the problems is that the growth-rate (drift) of the series is changing over time i.e. first differences are non-stationary.
# Selecting p=4 removes this second-order non-stationarity in-sample; but not out-of-sample.
# Therefore the model has to be up-dated continuously over time as new data becomes available which leads to revisions.
summary(lm_obj)

# Specify Hamilton filter
hamilton_filter<-c(1,rep(0,h-1),-lm_obj$coefficients[1+1:p])
ts.plot(hamilton_filter,main=paste("Hamilton filter: GDP from ",index(GDPC1["/2019"])[1]," to 2019",sep=""))
# We can include an intercept (though we'll skip mean-centering)
intercept<-lm_obj$coefficients[1]

# Replicate HF:
# Define Data matrix
#   We can fill any numbers for the columns corresponding to leads from t+1,...,t+h-1 since the hamilton filter coefficients vanish there: we here just repeat the target h-time
data_mat<-cbind(matrix(rep(target,h),ncol=h),explanatory)
# Apply the Hamilton filter to the data (log GDP)
residuals<-data_mat%*%hamilton_filter-intercept
# We just replicated the regression residuals (cycle)
ts.plot(cbind(residuals,lm_obj$residuals),main="Replication of Hamilton cycle: both series overlap")

# The regression coefficients nearly sum to one: this is because the series is trending and therefore the filter must remove 
#   the trend; otherwise the residuals wouldn't be stationary out-of-sample (cointegration)
sum(lm_obj$coefficients[1+1:p])
# The sum of the Hamilton filter nearly vanishes (should vanish exactly to cancel the unit root out-of-sample)
sum(hamilton_filter)
# The fact that the sum of the Hamilton filter coefficients doesn't vanish exactly is a slight 
#   drawback (problem is due to overfitting), when compared to HP (The coefficients of HP-gap add to zero but the gap-filter tends to generate spurious cycles, see tutorial 2, example 7)

# We now correct the filter such that the sum of the coefficients is zero: cointegration constraint
#   -We just distribute the difference (error) evenly across coefficients.
#   -Alternative corrections could be envisioned but the rounding-error (to achieve a zero-sum) is much smaller than 
#     the regression sampling error and therefore we can ignore such subtleties (you are welcome to experiment)
#   -In any case, these adjustments do not affect relative performances of SSA
hamilton_filter_adjusted<-hamilton_filter
hamilton_filter_adjusted[(h+1):(h+p)]<-hamilton_filter_adjusted[(h+1):(h+p)]-sum(hamilton_filter)/p
# Now the sum is zero
sum(hamilton_filter_adjusted)
# Compute corresponding adjusted residuals: forecast residuals are the cycle (note that we do not adjust for the intercept anymore)
residuals_adjusted<-data_mat%*%hamilton_filter_adjusted
# Center: not mandatory (we here rely on un-centered adjusted residuals or cycle estimates)
if (F)
  residuals_adjusted<-residuals_adjusted-mean(residuals_adjusted)

# Compare both cycles
# 1. Hamilton cycle (red), Hamilton cycle shifted by regression intercept (orange line), adjusted cycle (blue)
par(mfrow=c(1,1))
ts.plot(cbind(residuals,residuals+intercept,residuals_adjusted),col=c("red","orange","blue"),main="Cycles")
mtext("Unit-root adjusted un-centered cycle",col="blue",line=-1)
mtext("Hamilton Cycle",col="red",line=-2)
mtext("Hamilton cycle shifted by regression intercept",col="orange",line=-3)
abline(h=0)
# The series differ mainly in terms of level; the dynamics are indistinguishable

# 2. Hamilton original vs. adjusted
par(mfrow=c(2,2))
ts.plot(y,main="Log(GDPC1)")
ts.plot(cbind(residuals,residuals_adjusted),col=c("red","blue"),main="Cycles")
mtext("Hamilton Cycle",col="red",line=-1)
mtext("Unit-root adjusted un-centered cycle",col="blue",line=-2)
abline(h=0)
ts.plot(residuals-residuals_adjusted,main="Cycle difference")
# Both cycles (forecast residuals) series differ mainly in terms of levels (we did not center the adjusted residuals and 
#   the level of the adjusted residuals is slowly changing over time because the drift of GDP is changing).
# We here give a slight preference to the adjusted residuals which are to some extent closer to the definition of 'recessions' as (two consecutive) negative (quarters of) growth in the data.
#   -The Hamilton cycle is below zero 50% of the time: it would need to be up-lifted in order to track recessions. 
#   -The unit-root adjusted cycle is already `up-lifted' but its level is likely a bit high pre financial-crisis.
#   -Both designs would require some additional tweaking in order to track recessions (as declared by the NBER).
# The adjusted cycle lies further away from the classic cycle at the start, where the growth of log(GDP) is stronger.
# The adjusted cycles is closer to the classic cycle at the end, where the growth of log(GDP) is weaker.
# The cycle-difference in the bottom plot resembles a shrunken copy/version of the original data.
# Depending on the particular preference, one might consider classic or adjusted cycles
#   Since both cycles are essentially statistical artifacts, this choice mainly reflects subjective preferences
# For the purpose of engrafting SSA onto the Hamilton-filter, we here consider the adjusted and un-centered cycle 
#   But we also transform SSA back, to match the original cycle, see corresponding adjustments and plots below
#   For that purpose we need the difference between both cycles:
cycle_diffh<-residuals-residuals_adjusted
#---------------------------------------------------
# 1.2 Transformation: from levels to differences
# Next step: in order to plug SSA on HF we have to transform the empirical setting such that the data is stationary, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
#   -Recall that the concept of zero-crossings is not properly defined for non-stationary (integrated) series 
# Proceeding: we transform HF such that it can be applied to first differences (instead of levels)
# This proceeding can be applied to all bandpass-cycle designs working with data in levels: CF, BK (see tutorial 4), HP-gap (see tutorials 2 and 5)
# Ideas/background:
# -The filter-input (original or log-transformed) data is non-stationary
# -The filter-output (cycle) is a (nearly) stationary series
# -Therefore, the bandpass filter must remove the unit-root at frequency-zero (HF can remove more than one unit-root, depending on the auto-regressive order p)
# -We can always find a new filter, called ham_diff below, which, when applied to differences (instead of levels), replicates the output of the original HF (bandpass), as applied to levels
# See section 2.3 and proposition 4 in JBCY paper for background.

# Let's do so and construct ham_diff
# Select filter length of new filter ham_diff (which replicates output of Hamilton filter when applied to returns) 
L<-20
# Filter length: at least length of Hamilton filter
L<-max(length(hamilton_filter_adjusted),L)
if (L>length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L<-c(hamilton_filter_adjusted,rep(0,L-length(hamilton_filter_adjusted)))
# Convolution of Hamilton filter with summation filter (unit-root assumption)
ham_diff<-conv_with_unitroot_func(hamilton_filter_adjusted_L)$conv

# Plot and compare both HF designs
# Note that the coefficients of the new filter ham_diff vanish for lags larger than length of Hamilton_filter_adjusted and therefore we could set L=length(hamilton_filter_adjusted).
par(mfrow=c(2,1))
ts.plot(ham_diff,main="Hamilton filter as applied to first differences")
ts.plot(hamilton_filter_adjusted_L,main="Hamilton filter as applied to level")

# We now verify that the outputs of both filters are identical
# Difference data: this will be fed to ham_diff
x<-diff(y)
len_diff<-length(x)
# Compute new cycle based on new filter ham_diff applied to returns
residual_diff<-na.exclude(filter(x,ham_diff,side=1))

# Compare filter outputs
# Add NAs at the start of cycle_diff and Hamilton cycle (series are shorter due to applying Hamilton filter)
cycle_diff<-c(rep(NA,length(x)-length(cycle_diffh)),cycle_diffh)
original_hamilton_cycle<-c(rep(NA,length(x)-length(residuals)),residuals)
# Check: ham_diff as applied to first differences generates the same cycle as Hamilton filter applied to levels
par(mfrow=c(1,1))
ts.plot(residual_diff,col="blue",main="Outputs of HF applied to levels and to differences: both series overlap")
lines(residuals_adjusted[(L-p-h+2):length(residuals)],col="red")

# With this transformation in place we can engraft SSA onto HF
#   -The same transformation is required for BK-filter in tutorial 4 and HP-gap in tutorials 2 and 5: both filters are also bandpass applied to levels
#-------------------------------------------------------------------------------
# 1.3 Holding-times 
# We first compute the holding-time of the Hamilton-filter (as applied to noise i.e. differenced data)
ht_ham_diff_obj<-compute_holding_time_func(ham_diff)
# Approximately one crossing in 1.5 years
ht_ham_diff_obj$ht 
# We now compute the empirical ht
compute_empirical_ht_func(residuals_adjusted)
# The empirical holding time is much longer
# Explanation: look at the data
par(mfrow=c(1,1))
ts.plot(residuals_adjusted)
abline(h=0)
# The time series is un-centered (non-vanishing positive slowly drifting mean) 
# In principle, we can ignore this mismatch because a translation (shift) of the series does not affect SSA-computation
# But the expected holding time does not reflect the empirical holding-time anymore, see example 7 in tutorial 1
# This bias will be addressed when adjusting SSA to the original Hamilton cycle further down
#-----------------------------------------------------------------------
# 1.4 Autocorrelation
# Section 2 in the JBCY paper suggests a method for extending the SSA-framework to autocorrelated data
# The main tool for checking autocorrelation is the ACF:
acf(x,main="ACF")
# The ACF suggests weak dependency among the returns of GDPC1
# In such a case (weak dependency) we generally recommend to rely on the basic white noise assumption (section 1 in JBCY paper) 
#   -The latter assumption works well for many economic time series (Granger: typical spectral shape)
#   -The resulting SSA-design is not subject to revisions (fixed/robust)
#   -Th e simplifying white noise assumption is also used in section 4 of the JBCY paper, when applying HP and SSA to INDPRO, see tutorial 5
# xi are the weights of the Wold decomposition: xi=NULL means xt=epsilont (returns are white noise)
xi<-NULL
# Examples 3 and 4 further down show how to work with a fitted time series model in the case of autocorrelation

# We are now in a position to plug SSA on HF
#   -We can work with stationary series thanks to our transformation to ham_diff
#   -We can work with the Wold decomposition specified by xi: in our case just white noise
#   -But the data is non-stationary (slowly drifting) and un-centered and the holding-time is biased, recall tutorial 1, exercise 7

#--------------------------------------
# 1.4 Apply SSA
# The above holding-time of HF was approximately 1.5 years 
ht_ham_diff_obj$ht 
# This number is biased because the data is not white noise
# But our main purpose is to derive a smoother (SSA-) design: smoother relative to HF  
# Therefore, we can augment ht of the Hamilton filter by 50% 
# As a result, the SSA filter output will be automatically smoother
ht<-1.5*ht_ham_diff_obj$ht 
# We compute the corresponding rho1 for the holding time constraint: our SSA-function accepts rho1 as input (not ht)
rho1<-compute_rho_from_ht(ht)
# Specify the target: SSA should track the Hamilton-cycle 
# Since we are working with first differences, we supply ham_diff for the target
gammak_generic<-ham_diff
# Forecast horizon: nowcast (see forecasts below)
forecast_horizon<-0
# SSA of Hamilton-diff
# Note: if we do not supply xi then the SSA-function assumes white noise
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)
# SSA computes two filters: 
#   ssa_x: our main filter which is applied to the data xt
#   ssa_eps: the convolved filter which would be applied to epsilont (the innovation in the Wold decomposition of xt)
#     The latter is mainly used for checking the holding-time constraint (if OK then the optimization has converged to global optimum)
# In our case we assume xt=epsilont (white noise) and therefore both filters are identical
SSA_filt_ham_diff<-SSA_obj_ham_diff$ssa_x
# Check that both filters are identical: difference should vanish
if (F)
{
  SSA_obj_ham_diff$ssa_x-SSA_obj_ham_diff$ssa_eps
}
# Compare target and SSA: 
par(mfrow=c(1,1))
mplot<-cbind(ham_diff,SSA_filt_ham_diff)
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"),main="Target and SSA")
mtext("HF (as applied to differences)",col="black",line=-1)
mtext(paste("SSA: ht increased by ",100*(ht/ht_ham_diff_obj$ht-1),"%",sep=""), col="blue",line=-2)
abline(h=0)

# Check numerical optimization: the holding-time of the SSA-solution should match the imposed ht
# Compute effective holding-time of SSA
ht_obj<-compute_holding_time_func(SSA_filt_ham_diff)
ht_obj$ht 
# Compare with constraint: 
ht
# Both numbers match up to rounding errors, thus confirming that the optimization converged to the global maximum (this is invariably the case in applications because the optimization of 'non exotic' forecast problems is fairly easy)
#--------------------------------------------------
# 1.5 Filter series and compute performance measures
# Our version of HF does not remove the changing drift or slope: this can be desirable 
#   (we think it is...) or not, depending on priorities. 
# Below we present a simple adjustment for matching SSA to the original (zero centered) Hamilton cycle

# Apply SSA to data
SSA_out<-filter(x,SSA_filt_ham_diff,side=1)
# Compare empirical and theoretical (imposed) holding-times: they differ because xt is not centered
ht  
compute_empirical_ht_func(SSA_out)
# Apply HF to data and compare holding-times
ham_out<-filter(x,ham_diff,side=1)
ht_ham_diff_obj$ht
compute_empirical_ht_func(ham_out)
# The empirical holding-time of SSA is approximately 50% larger, as desired (the theoretical ht are biased, as explained above)

# Compare Hamilton-filter and SSA 
#   Both series are pretty close though the Hamilton filter is slightly noisier (more leakage)
#   Note that one could adjust for the non-vanishing level (see the plot thereafter)
mplot<-cbind(SSA_out,ham_out)
colo=c("blue","red")
par(mfrow=c(1,1))
ts.plot(mplot[,1],col=colo[1],main="SSA vs. Hamilton (not adjusted for level differences)")
mtext("SSA",col=colo[1],line=-1)
mtext("Hamilton",col=colo[2],line=-2)
lines(mplot[,2],col=colo[2])
abline(h=0)

# We can also compare the original Hamilton cycle with SSA shifted for the changing level (and scaled to original cycle)
# Proceeding for transformation of SSA
# 1. We must shift the level of SSA by cycle_diff: the difference between original Hamilton cycle and adjusted cycle 
# 2. We must account for different scalings of SSA_out, cycle_diff and original_hamilton_cycle
# These different scalings are linked to filter properties and could be computed exactly
# But the simplest proceeding consists in regressing SSA_out and cycle_diff on original_hamilton_cycle
lm_obj<-lm(original_hamilton_cycle~cycle_diff+SSA_out-1)
coef<-lm_obj$coef
# We now adjust SSA
scale_shifted_SSA<-coef[1]*cycle_diff+coef[2]*SSA_out
# Finally, we can plot adjusted SSA and original Hamilton cycle: data is missing at the start of scale_shifted_SSA shorter because it has been filtered
ts.plot(scale_shifted_SSA,col=colo[1],main="Shifted SSA cycle and original Hamilton cycle")
mtext("Shifted and scaled SSA",col=colo[1],line=-1)
mtext("Original Hamilton cycle",col="black",line=-2)
lines(original_hamilton_cycle)
abline(h=0)

# Compare  the empirical holding-times of both series: SSA is smoother: approximately 30% less crossings, as desired.
# Note that the empirical hts are now much closer to the expected hts since we corrected for some sources of `misspecification`, 
#   recall exercise 7 in tutorial 1
compute_empirical_ht_func(original_hamilton_cycle)
compute_empirical_ht_func(scale_shifted_SSA)
# We can also compute the mean shift at zero-crossings by the Tau-statistic, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# Idea
#   -We shift one series relative to the other until the absolute difference in timings of zero-crossings (of both series) is minimized
#   -The shift with the smallest timing difference of crossings (minimum of the plotted function) is a measure for the relative lead or lag
#   -In our example the minimum value is obtained at lead/lag 0: SSA and HF are synchronized
mat<-cbind(scale_shifted_SSA,original_hamilton_cycle)
lead_lag_obj<-compute_min_tau_func(mat)


#-----------------------------------------------
# 1.6 Forecasting
# Let us now address timeliness or lead/lags: the last plot suggested that SSA is synchronized with the target (and smoother)
# We here increase the forecast horizon to obtain a faster SSA filter 
# However, we do not give-up in terms of smoothness since we impose the same ht constraint to the forecast filter 

# One year forecast: everything else is unchanged 
forecast_horizon<-4
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_filt_ham_diff_forecast<-SSA_obj_ham_diff$ssa_x

# Filter series
SSA_out_forecast<-filter(x,SSA_filt_ham_diff_forecast,side=1)
# Compute and compare empirical holding times: SSA generates less crossings (30% less in the long run)
compute_empirical_ht_func(SSA_out_forecast)
compute_empirical_ht_func(ham_out)


# Compare all series
#   The SSA-forecast is now shifted to the left: leading. 
#   But it retains the same smoothness!
mplot<-cbind(SSA_out/sd(SSA_out,na.rm=T),SSA_out_forecast/sd(SSA_out_forecast,na.rm=T),ham_out/sd(ham_out,na.rm=T))
colo=c("blue","darkgreen","red")
par(mfrow=c(1,1))
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot,na.rm=T),max(mplot,na.rm=T)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon,sep=""),col=colo[2],line=-2)
mtext("Hamilton",col=colo[3],line=-3)
abline(h=0)


# We can adjust SSA-forecast to match the original Hamilton cycle, see previous plot 
# The adjusted SSA is smoother and shifted to the left
lm_obj<-lm(original_hamilton_cycle~cycle_diff+SSA_out_forecast-1)
coef<-lm_obj$coef
# We now adjust SSA
scale_shifted_SSA<-coef[1]*cycle_diff+coef[2]*SSA_out_forecast
# We can plot adjusted SSA and original Hamilton cycle: data is missing at the start of scale_shifted_SSA shorter because it has been filtered
ts.plot(scale_shifted_SSA,col=colo[2],main="Shifted SSA cycle and original Hamilton cycle")
mtext("Shifted and scaled SSA",col=colo[2],line=-1)
mtext("Original Hamilton cycle",col="black",line=-2)
lines(original_hamilton_cycle)
abline(h=0)

# Compare  the empirical holding-times of both series: SSA is smoother (approximately 30% less crossings in the long run)
compute_empirical_ht_func(original_hamilton_cycle)
compute_empirical_ht_func(scale_shifted_SSA)

# Compute the mean shift at zero-crossings: 
#   The minimum value is achieved at lead -1
#   SSA-forecast is indeed leading (which was already apparent from previous plot)
mat<-cbind(scale_shifted_SSA,original_hamilton_cycle)
lead_lag_obj<-compute_min_tau_func(mat)

# Finally, we can analyze smoothness and timeliness issues by looking at amplitude and phase-shift functions
# These results offer an alternative description, based on formal (data-independent) filter characteristics
# They confirm the above empirical results (obtained by comparing filtered series)

#----------------------------------------
# 1.7 Compute amplitude and phase-shift functions
# Amplitude at higher frequencies: closer to zero means less noise leakage
# Phase-shift in passband: a smaller shift means a relative lead

# Select the number of equidistant frequency ordinates (grid-size) in [0,pi] 
K<-600
# Compute amplitude and phase-shifts for SSA, SSA-forecast and ham-diff: all filters are applied to returns
amp_obj_SSA_now<-amp_shift_func(K,as.vector(SSA_filt_ham_diff),F)
amp_obj_SSA_for<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_forecast),F)
amp_obj_ham<-amp_shift_func(K,ham_diff,F)

# Plot
par(mfrow=c(2,1))
# 1. Plot amplitude functions
mplot<-cbind(amp_obj_SSA_now$amp,amp_obj_SSA_for$amp,amp_obj_ham$amp)
# Scale SSA such that all amplitudes are normalized to one at frequency zero (always possible for lowpass)
mplot[,1]<-mplot[,1]/mplot[1,1]
mplot[,2]<-mplot[,2]/mplot[1,2]
mplot[,3]<-mplot[,3]/mplot[1,3]
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"Hamilton")
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude Hamilton Filter",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# 2. Plot shifts
mplot<-cbind(amp_obj_SSA_now$shift,amp_obj_SSA_for$shift,amp_obj_ham$shift)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon,")",sep=""),"Hamilton")
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase-shift ",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()

# Discussion:
# The amplitude functions illustrate that
# 1. ham_diff (applied to returns) is a lowpass (the original HF, applied to levels, is a bandpass)
#     -This setting is similar to our application of HP-trend to returns of INDPRO in section 4 of JBCY paper, 
#       see tutorial 5
#     -The lowpass differs notably from classic bandpass such as HP-gap (see tutorials 2 and 5) or BK (see tutorial 4).
#     -HF does not generate spurious cycles; but neither does HP-trend applied to returns (see tutorial 5)
#     -In contrast, HP-gap and BK generate spurious cycles: filter-artifacts
# 2. The amplitude functions of both SSA are closer to zero (smaller than ham_diff) towards higher frequencies 
#     -Effect: stronger smoothing and less (noisy) zero-crossings
#     -This typical property of SSA-smoothing designs has coined the term 'noisy crossings' (of the benchmark filter)
#     -Typically, SSA damps undesirable high-frequency components (noise) more effectively than the benchmark

# Phase-shift, i.e., phase divided by frequency
#   -This is a measure for the lag (or lead) of the one-sided filters at all frequencies
#   -In general we look at the phase-shift for frequencies in the passband: where the amplitude is above 0.5
#   -Phase-shift of SSA-nowcast is marginally larger than HF (overall negligible): recall the plots of filter outputs
#   -Phase-shift of SSA-forecast is smallest (in passband), confirming the observed (and measured) relative lead
# The positive phase-shift or lag of HF is substantial
#   -It is larger than the classic HP-concurrent trend, applied to returns (the shift of HP-trend must vanish at frequency zero because of the implicit assumption of a second-order unit-root)
#   -The size of the lag is due to the forecast horizon (2 years) in the regression equation 
#   -It is not clear, at this point, "why you should never use the HP" (which HP?)
#     -Don't use HP-gap, applied to the original data: yes, see confirmation in tutorial 2.
#     -But HP-trend applied to returns performs well: its lag is smaller and the filter is smoother than HF, see tutorial 2.

########################################################################################################################
#################################################################################################################
# Example 2
# Application of SSA to HF based on monthly PAYEMS series (non-farm payroll) 

# The data starts pre WWII: series is non-stationary.
# Example 2 considers the whole data-sample, assuming log-returns to be white noise
# Example 3 fits an ARMA-model to the data
# Example 4 emphasizes post-1990 data only (great moderation) and it also analyzes the pandemic effect


# Let's briefly illustrate the Hamilton filter as applied to quarterly transformed PAYEMS: R-package neverhpfilter
library(neverhpfilter)


data(PAYEMS)
log_Employment <- 100*log(xts::to.quarterly(PAYEMS["1947/2016-6"], OHLC = FALSE))

employ_trend <- yth_filter(log_Employment, h = 8, p = 4, output = c("x", "trend"), family = gaussian)

plot.xts(employ_trend, grid.col = "white", legend.loc = "topleft", main = "Log of Employment and trend")

# Often the cycle and the random components correlate strongly which contradicts the classic assumptions
employ_cycle <- yth_filter(log_Employment, h = 8, p = 4, output = c("cycle", "random"), family = gaussian)
par(mfrow=c(1,1))
plot.xts(employ_cycle, grid.col = "white", legend.loc = "topright", main="Log of Employment cycle and random")
abline(h=0)

# Returns are non stationary: slowly drifting and series pre/post 1960 as well as pre/post 1990 differ  in term of 
#   dependence structure and variance (WWII, `great moderation') 
# In example 2, here, we ignore these issues: our main purpose is to illustrate SSA. 
# SSA is more or less insensitive to the misspecification: it will outperform the benchmark irrespective of `false` models
plot(diff(log(PAYEMS)))
abline(h=0)


#---------------------------------------------------------
# We now skip the above environment and R-package and implement the filter with own code, based on original monthly data
# This will allow to engraft SSA onto HF
getSymbols('PAYEMS',src='FRED')

# Discard Pandemic: extreme outliers affect regression of Hamilton filter
# Make double: xts objects are subject to lots of automatic/hidden assumptions which make an application of SSA 
# more cumbersome, unpredictable and hazardous
y<-as.double(log(PAYEMS["/2019"]))
len<-length(y)
#-----------------------------------------------------
# 2.1 Hamilton filter
# Settings proposed by Hamilton for quarterly data: two years ahead forecast and AR-order 4 to remove multiple unit-roots (if necessary)
h<-2*4
p<-4
# We here adapt to monthly PAYEMS: p remains the same (accounts for integration order, see Hamilton paper) 
h<-2*12
p<-4

explanatory<-cbind(y[(p):(len-h)],y[(p-1):(len-h-1)],y[(p-2):(len-h-2)],y[(p-3):(len-h-3)])
target<-y[(h+p):len]

lm_obj<-lm(y[(h+p):len]~explanatory)

# Only the first coefficient is significant, as is often the case in applications to non-stationary economic series.
# This is a potential problem because the sum of the parameters is not one and therefore 1-sum(coefficients)!=0.
# As a result the forecast residual y_{t+h}-\hat{y}_{t+h} is stationary in-sample (due to overfitting) but non-stationary out-of-sample. 
#   Stated differently: the forecast and the future observation are not cointegrated.
# The growth-rate (drift) of the series is changing over time i.e. first differences are non-stationary.
# Selecting p=4 removes this second-order non-stationarity in-sample; but not out-of-sample.
# Therefore the model has to be up-dated continuously over time as new data becomes available which leads to revisions.
summary(lm_obj)

# Plot cycle
ts.plot(lm_obj$residuals)

# Specify filter
hamilton_filter<-c(1,rep(0,h-1),-lm_obj$coefficients[1+1:p])
# Plot filter
ts.plot(hamilton_filter,main="HF filter coefficients")

# We could center the cycle by relying on the intercept
intercept<-lm_obj$coefficients[1]
# Apply HF to data
# We can fill any numbers for the leads from t+1,...,t+h-1 since the hamilton filter coefficients vanish there: we just repeat the target h-time
data_mat<-cbind(matrix(rep(target,h),ncol=h),explanatory)
residuals<-data_mat%*%hamilton_filter-intercept
# We just replicated the regression residuals (Hamilton filter)
ts.plot(cbind(residuals,lm_obj$residuals),main="Replication of regression residuals by HF (both series overlap)")

# The sum of the coefficients nearly vanishes: this is because the series is non-stationary and therefore the filter must 
#     remove the trend
sum(hamilton_filter)
# We now correct the filter such that the sum of the coefficients is zero:  cointegration constraint 
#   (we just distribute the difference evenly on coefficients)
hamilton_filter_adjusted<-hamilton_filter
hamilton_filter_adjusted[(h+1):(h+p)]<-hamilton_filter_adjusted[(h+1):(h+p)]-sum(hamilton_filter)/p
# Now the sum is zero
sum(hamilton_filter_adjusted)
# Compute corresponding adjusted residuals
residuals_adjusted<-data_mat%*%hamilton_filter_adjusted
# Center: not mandatory (we here rely on uncentered adjusted residuals or cycle estimates)
if (F)
  residuals_adjusted<-residuals_adjusted-mean(residuals_adjusted)

# See comments to cycles in example 1 above
par(mfrow=c(2,2))
ts.plot(y,main="Log(PAYEMS)")
ts.plot(cbind(residuals,residuals_adjusted),col=c("red","blue"),main="Cycles")
mtext("Classic cycle",col="red",line=-1)
mtext("Adjusted un-centered cycle",col="blue",line=-2)
ts.plot(residuals-residuals_adjusted,main="Cycle difference")

# For the purpose of engrafting SSA onto HF, we here consider the adjusted un-centered cycle 
#   But we also transform SSA back, to match the original (centered) cycle, see corresponding adjustments and plots below
#   For that purpose we need the difference between both cycles:
cycle_diffh<-residuals-residuals_adjusted


#---------------------------------------------------
# 2.2 Transformation: from levels to differences
# In order to plug SSA on HF we have to transform the empirical setting such that the data 
#   is stationary, see corresponding discussion in example 1 above
# -We can always find a new filter, called ham_diff, which, when applied to differences (instead of levels), replicates the output of the original bandpass, as applied to levels
# See section 2.3 and proposition 4 in JBCY paper for background.
L<-50
# Filter length: at least length of Hamilton filter
L<-max(length(hamilton_filter_adjusted),L)
if (L>length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L<-c(hamilton_filter_adjusted,rep(0,L-length(hamilton_filter_adjusted)))


# Convolution with summation filter (unit-root assumption)
ham_diff<-conv_with_unitroot_func(hamilton_filter_adjusted_L)$conv
# Compare both filters
par(mfrow=c(2,1))
ts.plot(ham_diff,main="Hamilton filter as applied to first differences")
ts.plot(hamilton_filter_adjusted_L,main="Hamilton filter as applied to level")

# Apply new filter ham_diff to returns and verify that its filter output is the same as hamilton_filter_adjusted_L
x<-diff(y)
len_diff<-length(x)
residual_diff<-na.exclude(filter(x,ham_diff,side=1))

# Compare outputs
# Add NAs at the start of cycle_diff and Hamilton cycle (series are shorter due to applying Hamilton filter)
cycle_diff<-c(rep(NA,length(x)-length(cycle_diffh)),cycle_diffh)
original_hamilton_cycle<-c(rep(NA,length(x)-length(residuals)),residuals)
# Check: ham_diff, as applied to first differences, generates the same cycle as original Hamilton filter applied to levels
par(mfrow=c(1,1))
ts.plot(residual_diff,col="blue",main="Replication of Hamilton filter: application to level vs. first differences")
lines(residuals_adjusted[(L-p-h+2):length(residuals)],col="red")

# With this transformation in place we are in a position to engraft SSA onto HF

#-------------------------------------------------------------------------------
# 2.3 Holding-times
# We first compute the holding-time of the Hamilton-filter ham_diff
ht_ham_diff_obj<-compute_holding_time_func(ham_diff)
# Quite short: more than two zero-crossings per year on average
ht_ham_diff_obj$ht
# This number will be used for reference in example 4 below
ht_ham_example2<-ht_ham_diff_obj$ht
# But the empirical holding time is much longer: see discussion in example 1 above and example 7 in tutorial 1
compute_empirical_ht_func(residuals_adjusted)

# Explanation:  xt (log-returns) are not white noise
par(mfrow=c(1,1))
ts.plot(x,ylim=c(-0.05,0.05),main="Log-returns PAYEMS")
abline(h=0)

#-----------------------------------------------------------------------
# 2.4 Autocorrelation
# Section 2 in the JBCY paper proposes an extension of SSA to autocorrelated data.
# The main tool for checking autocorrelation is the ACF:
acf(x,main="ACF")
# The ACF suggests dependency among the returns of PAYEMS
# We here (in example 2) deliberately assume a wrong model, namely white noise, to illustrate that SSA is quite robust against 
#   misspecification
xi<-NULL
# Examples 3 and 4 further down show how to work with a fitted time series model 

# We are now in a position to engraft SSA onto HF
#   -We can work with stationary series thanks to our transformation to ham_diff
#   -We can work with the Wold decomposition specified by xi: in our case white noise
#   -But the data is non-stationary (slowly drifting and variance is changing too) and un-centered and the holding-time is biased, recall tutorial 1, exercise 7


#--------------------------------------
# 2.5 Apply SSA
# The above holding-time was approximately half a year 
ht_ham_diff_obj$ht 
# This number is biased because the data is not white noise
# But our main purpose is to derive a smoother (SSA-) design: smoother relative to HF 
# Therefore, we can augment ht of the Hamilton filter by 50% 
# As a result, the SSA filter output will be automatically smoother: irrespective of biases or model misspecifications
ht<-1.5*ht_ham_diff_obj$ht 
# This is the corresponding rho1 for the holding time constraint: our SSA-function accepts rho1 as input (not ht)
rho1<-compute_rho_from_ht(ht)
# Specify the target: SSA should track Hamilton-cycle (beeing smoother)
gammak_generic<-ham_diff
# Forecast horizon: nowcast (see forecasts below)
forecast_horizon<-0
# SSA of Hamilton-diff
# Note: if we do not supply xi then the SSA-function assumes white noise
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1)
# SSA computes two filters: ssa_x and ssa_eps: in our case we assume xt=epsilont (white noise) and therefore both 
#   filters are identical
SSA_filt_ham_diff<-SSA_obj_ham_diff$ssa_x
# Check that both filters are identical: difference should vanish
if (F)
{
  SSA_obj_ham_diff$ssa_x-SSA_obj_ham_diff$ssa_eps
}

# Compare target and SSA
par(mfrow=c(1,1))
mplot<-cbind(ham_diff,SSA_filt_ham_diff)
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"))
mtext("HF (as applied to differences)",col="black",line=-1)
mtext(paste("SSA: ht increased by ",100*(ht/ht_ham_diff_obj$ht-1),"%",sep=""), col="blue",line=-2)

# Check holding time constraint
ht_obj<-compute_holding_time_func(SSA_filt_ham_diff)
ht_obj$ht 
ht
# Both numbers agree, confirming that the optimization converged to the global optimum

#--------------------------------------------------
# 2.6 Filter series and compute performance numbers
# 2.6.1. SSA
SSA_out<-filter(x,SSA_filt_ham_diff,side=1)
# Empirical ht much larger than targeted ht, see explanation above (series is non-stationary)
compute_empirical_ht_func(SSA_out)
ht  
# 2.6.2. Hamilton: SSA generates less crossings, as expected
ham_out<-filter(x,ham_diff,side=1)
compute_empirical_ht_func(ham_out)
ht_ham_diff_obj$ht 

# Although the series look similar, HF has more (noise-) leakage, see amplitude functions below, 
#   and therefore it generates  more `noisy' crossings (in the long run)
mplot<-na.exclude(cbind(SSA_out,ham_out))
colo=c("blue","red")
ts.plot(mplot[,1],col=colo[1])
lines(mplot[,2],col=colo[2])
abline(h=0)
mtext("Hamilton (as applied to differences)",col="red",line=-1)
mtext("SSA", col="blue",line=-2)

# 2.6.3 Empirical holding times: SSA approximately 50% larger than Hamilton in the long run (if the model is not misspecified)
#   Our model assumptions here are wrong: see example 4 below
#   One could increase ht in the function call in order to improve smoothness further, if desired
# We can clearly see the non-stationarity (changing shape) of the cycles.
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(ham_out)


# We can also compare the original Hamilton cycle with SSA shifted for the changing level (and scaled to original cycle)
# see example 1 above for details
lm_obj<-lm(original_hamilton_cycle~cycle_diff+SSA_out-1)
coef<-lm_obj$coef
# We now adjust SSA
scale_shifted_SSA<-coef[1]*cycle_diff+coef[2]*SSA_out
# We can plot adjusted SSA and original Hamilton cycle: data is missing at the start of scale_shifted_SSA shorter because it has been filtered
ts.plot(scale_shifted_SSA,col=colo[1],main="Shifted SSA cycle and original Hamilton cycle",ylim=c(min(original_hamilton_cycle,na.rm=T),max(original_hamilton_cycle,na.rm=T)))
mtext("Shifted and scaled SSA",col=colo[1],line=-1)
mtext("Original Hamilton cycle",col="black",line=-2)
lines(original_hamilton_cycle)
abline(h=0)

# Although both series look similar, SSA generates slightly less crossings (but not 30% less)
# We could either increase ht in SSA (making it smoother) or, better yet, split the series into 
#   shorter (more consistent or less stationary) time-frames: see example 4 below
compute_empirical_ht_func(scale_shifted_SSA)
compute_empirical_ht_func(original_hamilton_cycle)


#-----------------------------------------------
# 2.7 Forecasting
# Let us now address timeliness or lead/lags: 
# We augment the forecast horizon (delta in the JBCY paper) to obtain a faster (leading) SSA filter 
# We do not give-up in terms of smoothness or noise suppression since we do not change the ht-constraint

# We provide two different forecast horizons and compare with nowcast
# Compute half-a-year and full-year forecasts: we can supply a vector with the intended forecast horizons
#   SSA will return a matrix of filters: each column corresponds to the fixed forecast horizon 
forecast_horizon<-c(6,12)
# SSA of Hamilton-diff
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_filt_ham_diff_x_forecast<-SSA_obj_ham_diff$ssa_x
# This is now a matrix with two columns corresponding to the forecast horizons
head(SSA_filt_ham_diff_x_forecast)

# Plot and compare SSA filters
# Morphing: the forecast filters assign less weight to the remote past (faster/leading) but 
#   the particular filter patterns will ensure smoothness (ht unchanged)
par(mfrow=c(1,1))
ts.plot(SSA_filt_ham_diff_x_forecast,col=c("orange","darkgreen"),main="SSA forecasts vs nowcast",ylim=c(min(SSA_filt_ham_diff),max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff,col="blue")
mtext(paste("Forecast horizon ",forecast_horizon[1],sep=""),col="orange",line=-1)
mtext(paste("Forecast horizon ",forecast_horizon[2],sep=""),col="darkgreen",line=-2)
mtext("Nowcast",col="blue",line=-3)

#----------------------
# Filter series: 
# 6 months ahead
SSA_out_forecast_6<-filter(x,SSA_filt_ham_diff_x_forecast[,1],side=1)
# 12 months ahead
SSA_out_forecast_12<-filter(x,SSA_filt_ham_diff_x_forecast[,2],side=1)

# The SSA-forecast is now shifted to the left; smoothness is the same.
# The scale or variance of the forecast is smaller because the estimation problem is more difficult (zero-shrinkage)
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast_6,SSA_out_forecast_12,ham_out))
colo=c("blue","orange","darkgreen","red")
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
lines(mplot[,4],col=colo[4])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon[1],sep=""),col=colo[2],line=-2)
mtext(paste("SSA forecast: delta=",forecast_horizon[2],sep=""),col=colo[3],line=-3)
mtext("Hamilton",col=colo[4],line=-4)
abline(h=0)

# We here scale to unit variance in order to ease visual inspection
mplot<-scale(na.exclude(cbind(SSA_out,SSA_out_forecast_6,SSA_out_forecast_12,ham_out)),scale=T,center=F)
colo=c("blue","orange","darkgreen","red")
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
lines(mplot[,4],col=colo[4])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon[1],sep=""),col=colo[2],line=-2)
mtext(paste("SSA forecast: delta=",forecast_horizon[2],sep=""),col=colo[3],line=-3)
mtext("Hamilton",col=colo[4],line=-4)
abline(h=0)




# ht of SSA designs is approximately 50% larger in the long run (if the model is not misspecified)
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(SSA_out_forecast_6)
compute_empirical_ht_func(SSA_out_forecast_12)
compute_empirical_ht_func(ham_out)



# The nowcast SSA is synchronized with HF; the forecasts are left-shifted (leading) and smoother
max_lead=8
# SSA nowcast is synchronized with HF
shift_obj<-compute_min_tau_func(mplot[,c(1,4)],max_lead)
# SSA half-year forecast is leading HF by ~3 months 
shift_obj<-compute_min_tau_func(mplot[,c(2,4)],max_lead)
# SSA full-year forecast is leading HF by ~6 months
shift_obj<-compute_min_tau_func(mplot[,c(3,4)],max_lead)

# Finally, we can analyze smoothness and timeliness issues by looking at amplitude and phase-shift functions
# These results offer an alternative description, based on formal filter characteristics
# They confirm the above empirical results (obtained by comparing filtered series)


#----------------------------------------
# 2.8 Compute amplitude and phase-shift functions
# Amplitude at higher frequencies: closer to zero means less noise leakage
# Phase-shift in passband: a smaller shift means a relative lead

# Select the number of equidistant frequency ordinates (grid-size) in [0,pi] 
K<-600
# Compute amplitude and phase-shifts for SSA, SSA-forecast and ham-diff: all filters are applied to returns

colo=c("black","blue","darkgreen","red")
amp_obj_SSA_now<-amp_shift_func(K,as.vector(SSA_filt_ham_diff),F)
amp_obj_SSA_for_6<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,1]),F)
amp_obj_SSA_for_12<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,2]),F)
amp_obj_ham<-amp_shift_func(K,ham_diff,F)

par(mfrow=c(2,1))
mplot<-cbind(amp_obj_SSA_now$amp,amp_obj_SSA_for_6$amp,amp_obj_SSA_for_12$amp,amp_obj_ham$amp)
# Scale SSA such that all amplitudes are normalized to one at frequency zero
mplot[,1]<-mplot[,1]/mplot[1,1]
mplot[,2]<-mplot[,2]/mplot[1,2]
mplot[,3]<-mplot[,3]/mplot[1,3]
mplot[,4]<-mplot[,4]/mplot[1,4]
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")

# 1. Plot amplitude
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude Hamilton-Filter",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# 2. Plot phase-shift
mplot<-cbind(amp_obj_SSA_now$shift,amp_obj_SSA_for_6$shift,amp_obj_SSA_for_12$shift,amp_obj_ham$shift)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase-shift ",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()


# Discussion: see the discussion at the end of example 1 above 
# Note that the leads measured by the tau-statistic (in the time-domain and at zero-crossings) match 
#  the phase-shift differences in the passband closely
# Once again, the positive phase-shift or lag of HF is substantial
#   -It is larger than the classic HP-concurrent trend, applied to returns, whose shift must vanish at frequency zero
#   -The size of the lag is due to the forecast horizon (2 years) in the regression equation 
#   -It is not clear at this point "why you should never use the HP"
#     -Don't use HP-gap, applied to the original data: yes, see confirmation in tutorial 2.
#     -But HP-trend applied to returns performs well: it is quite smooth  and its lag is smaller than HF, see tutorial 2.



########################################################################################################################
#################################################################################################################
# Example 3: same as example 2 but we fit a model to the data
# Note: one has to run example 2 at least once before example 3 in order to initialize all settings 

# 3.1-3.3: We assume that steps 2.1 to 2.3 were done (otherwise run the corresponding code lines)
#--------------------------------------------  
# 3.4 Autocorrelation
# We here fit an ARMA-model to the data
# ACF suggests autocorrelation
acf(x,main="ACF")
ar_order<-1
ma_order<-1
estim_obj<-arima(x,order=c(ar_order,0,ma_order))
# We have the typical cancelling AR and MA-roots which can fit a weak but long lasting ACF 
estim_obj
# Diagnostics are OK for the purpose at hand
tsdiag(estim_obj)
# Compute the MA-inversion of the ARMA: Wold-decomposition
xi<-c(1,ARMAtoMA(ar=estim_obj$coef[1:ar_order],ma=estim_obj$coef[ar_order+1:ma_order],lag.max=L-1))
# Remark: L should be sufficiently large for xi to decay (converge) to zero: L=50 is fine
par(mfrow=c(1,1))
ts.plot(xi,main="Wold-decomposition: xi")
# Convolve xi and ham_diff: this is the filter that would be applied to the innovations epsilont in the Wold 
#   decomposition of xt
# If the above model were true, i.e. if epsilont were white noise, then the holding-time of this filter would match 
#   the observed empirical holding time of ham_diff
ham_conv<-conv_two_filt_func(xi,ham_diff)$conv
ts.plot(ham_conv,main="Convolved Hamilton filter")
# Compute ht
ht_ham_conv_obj<-compute_holding_time_func(ham_conv)
ht_ham_conv_obj$ht
# Compare with ht of ham_diff (based on white noise assumption)
#   ham_conv is much smoother because the ARMA-filter for the DGP is a lowpass  
ht_ham_diff_obj$ht
# Compare with empirical ht: residuals_adjusted is the output of ham_diff
compute_empirical_ht_func(residuals_adjusted)
# Part of the ht bias of ham_diff (assuming white noise) has been addressed and resolved by ham_conv by modelling the ACF-structure of the data
#   -ht of ham_conv is closer to the empirical ht. 
#   -But there is some bias left in the ht of ham_conv: the data is still un-centered and non-stationary 
# We use the holding time of ham_conv when specifying ht for SSA 


# We are now in a position to plug SSA on HF
#   -We can work with stationary series thanks to our transformation to ham_diff
#   -We can work with the Wold decomposition specified by xi 

#--------------------------------------
# 3.5 Apply SSA: 
# We here rely on ham_conv for the ht-constraint
ht_ham_conv_obj$ht
# We can lengthen by 50%: SSA will generate ~30% less crossings (if the model is not misspecified)
ht<-1.5*ht_ham_conv_obj$ht
# This is the corresponding rho1 for the holding time constraint
rho1<-compute_rho_from_ht(ht)
# Target: we want to `improve' upon ham_diff (which is the filter that is applied to xt)
# By supplying xi to SSA_func the algorithm `knows' that xt is not white noise
gammak_generic<-ham_diff
# Forecast horizon: nowcast
forecast_horizon<-0
# SSA of Hamilton-diff
# Note: this call to SSA is not correct!!!! We did not supply the Wold decomposition xi in the function call! 
#  Let's see what happens 
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1)

SSA_filt_ham_diff<-SSA_obj_ham_diff$ssa_eps

# Compare target and SSA: looks odd (SSA assumes the data to be white noise: in this case ht in the constraint 
#   is too large: the filter smooths excessively)
par(mfrow=c(1,1))
mplot<-cbind(ham_diff,SSA_filt_ham_diff)
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"))
mtext("Hamilton (as applied to differences)",col="black",line=-1)
mtext("Incorrect SSA (erroneously assuming white noise)", col="blue",line=-2)

# Check holding time constraint
ht_obj<-compute_holding_time_func(SSA_filt_ham_diff)
# It matches our constraint. But the constraint is too heavy (it is based on ham_conv which assumes autocorrelation)
ht_obj$ht 
ht
 
# Let's do it the right way now: add xi (Wold decomposition) in the function call
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)
# ssa_eps is the filter which is applied to epsilont: this is mainly used for verifying the ht-constraint (convergence of optimization to global maximum)
SSA_filt_ham_diff_eps<-SSA_obj_ham_diff$ssa_eps
# ssa_x is the filter which is applied to the data (xt) 
#   In the previous call both filters were identical because we forgot to supply xi (now both filters differ)
SSA_filt_ham_diff_x<-SSA_obj_ham_diff$ssa_x


# Compare target and SSA: we must compare ham_diff with ssa_x (both are applied to xt)
#   In contrast to the previous call and plot, SSA seems now to be `OK' 
mplot<-cbind(ham_diff,SSA_filt_ham_diff_x)
# Compare target and SSA: looks better than previous plot
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"),main="Filters as applied to first differences xt")
mtext("Hamilton ",col="black",line=-1)
mtext("SSA: assuming autocorrelation ", col="blue",line=-2)

# Alternatively, we could compare ham_conv and ssa_eps (both are applied to epsilont)
mplot<-cbind(ham_conv,SSA_obj_ham_diff$ssa_eps)
# Compare target and SSA: looks better than previous plot
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"),main="Convolved filters as applied to innovations epsilont")
mtext("Hamilton",col="black",line=-1)
mtext("SSA", col="blue",line=-2)

# Check holding time constraint: 
#   Both numbers are identical (up to rounding error) confirming convergence of the optimization to the global maximum
ht_obj<-compute_holding_time_func(SSA_filt_ham_diff_eps)
ht_obj$ht 
ht
# Note that the holding-time of ssa_x is much smaller because ssa_x is applied to xt which is smoother than epsilont
# This feature explains part of the ht-bias (mismatch of expected and empirical hts, see example 7 in tutorial 1)
ht_obj<-compute_holding_time_func(SSA_filt_ham_diff_x)
ht_obj$ht 



#--------------------------------------------------
# 3.6 Filter series and compute performance measures
# 1. SSA
SSA_out<-filter(x,SSA_filt_ham_diff_x,side=1)
# Empirical ht quite larger than targeted ht, see explanation above (series is non-stationary)
compute_empirical_ht_func(SSA_out)
ht  
# 2. Hamilton
ham_out<-filter(x,ham_diff,side=1)
compute_empirical_ht_func(ham_out)
# Have to compare with ht of ham_conv (not ham_diff)
ht_ham_conv_obj$ht 

# Although the series look quite similar, HF has more (noise-) leakage and therefore it generates 
#   more `noisy' crossings 
mplot<-na.exclude(cbind(SSA_out,ham_out))
colo=c("blue","red")
ts.plot(mplot[,1],col=colo[1])
lines(mplot[,2],col=colo[2])
abline(h=0)
mtext("Hamilton (as applied to differences)",col="red",line=-1)
mtext("SSA", col="blue",line=-2)

# Empirical holding times: ht of SSA is larger but not 50% larger  
# Problem: non-stationarity, cycle is changing (see example 4 below which emphasizes data post-1990)
# Hint: try ht<-2*ht_ham_conv_obj$ht instead of ht<-1.5*ht_ham_conv_obj$ht before SSA-call
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(ham_out)


# We can also compare the original Hamilton cycle with SSA shifted for the changing level (and scaled to original cycle)
# see example 1 above for details
lm_obj<-lm(original_hamilton_cycle~cycle_diff+SSA_out-1)
coef<-lm_obj$coef
# We now adjust SSA
scale_shifted_SSA<-coef[1]*cycle_diff+coef[2]*SSA_out
# We can plot adjusted SSA and original Hamilton cycle: data is missing at the start of scale_shifted_SSA shorter because it has been filtered
ts.plot(scale_shifted_SSA,col=colo[1],main="Shifted SSA cycle and original Hamilton cycle",ylim=c(min(original_hamilton_cycle,na.rm=T),max(original_hamilton_cycle,na.rm=T)))
mtext("Shifted and scaled SSA",col=colo[1],line=-1)
mtext("Original Hamilton cycle",col="black",line=-2)
lines(original_hamilton_cycle)
abline(h=0)

# Although both series look similar, SSA generates slightly less crossings (but not 30% less)
# We could either increase ht in SSA (making it smoother) or, better yet, split the series into 
#   shorter time-frames in order to mitigate misspecification (for example discard data prior 1990): see example 4 below
compute_empirical_ht_func(scale_shifted_SSA)
compute_empirical_ht_func(original_hamilton_cycle)

 


#-----------------------------------------------
# 3.7 Forecasting
# Let us now address timeliness or lead/lags: 
# We augment the forecast horizon (delta in the JBCY paper) to obtain a faster (leading) SSA filter 
# We do not give-up in terms of smoothness or noise suppression since we do not change the ht-constraint

# We compute two forecast horizons and compare with nowcast
# Compute half a year and full year forecasts: we can supply a vector with the intended forecast horizons
#   SSA will return a matrix of filters: each column corresponds to the intended forecast horizon 
forecast_horizon<-c(6,12)
# SSA of Hamilton-diff
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_filt_ham_diff_x_forecast<-SSA_obj_ham_diff$ssa_x

# Plot and compare SSA filter
# Morphing: the forecast filter assign less weight to remote past (faster/leading) but 
#   the particular filter pattern will ensure smoothness (ht unchanged)
# Note that these forecast filters differ from example 2 because we here assume an ARMA(1,1) model of the data
par(mfrow=c(1,1))
ts.plot(SSA_filt_ham_diff_x_forecast,col=c("orange","darkgreen"),main="SSA forecasts vs nowcast",ylim=c(min(SSA_filt_ham_diff_x),max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x,col="blue")
mtext(paste("Forecast horizon ",forecast_horizon[1],sep=""),col="orange",line=-1)
mtext(paste("Forecast horizon ",forecast_horizon[2],sep=""),col="darkgreen",line=-2)
mtext("Nowcast",col="blue",line=-3)

#----------------------
# Filter series: 
# 6 months ahead
SSA_out_forecast_6<-filter(x,SSA_filt_ham_diff_x_forecast[,1],side=1)
# 12 months ahead
SSA_out_forecast_12<-filter(x,SSA_filt_ham_diff_x_forecast[,2],side=1)

# The SSA-forecast is now shifted to the left; smoothness is the same.
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast_6,SSA_out_forecast_12,ham_out))
colo=c("blue","orange","darkgreen","red")
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
lines(mplot[,4],col=colo[4])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon[1],sep=""),col=colo[2],line=-2)
mtext(paste("SSA forecast: delta=",forecast_horizon[2],sep=""),col=colo[3],line=-3)
mtext("Hamilton",col=colo[4],line=-4)
abline(h=0)


# ht of SSA designs is approximately 50% larger: stronger noise suppression by SSA
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(SSA_out_forecast_6)
compute_empirical_ht_func(SSA_out_forecast_12)
compute_empirical_ht_func(ham_out)



# The forecast SSA filter has a lead at zero-crossings relative to HF (and less crossings)
max_lead=8
# SSA nowcast is synchronized with HF
shift_obj<-compute_min_tau_func(mplot[,c(1,4)],max_lead)
# SSA half-year forecast is leading HF by a quarter
shift_obj<-compute_min_tau_func(mplot[,c(2,4)],max_lead)
# SSA full-year forecast is leading HF by two quarters
shift_obj<-compute_min_tau_func(mplot[,c(3,4)],max_lead)

# Finally, we can analyze smoothness and timeliness issues by looking at amplitude and phase-shift functions
# These results offer an alternative description, based on formal filter characteristics
# They confirm the above empirical results (obtained by comparing filtered series)


#----------------------------------------
# 3.8 Compute amplitude and phase-shift functions
# Amplitude at higher frequencies: closer to zero means less noise leakage
# Phase-shift in passband: a smaller shift means a relative lead

# Select the number of equidistant frequency ordinates (grid-size) in [0,pi] 
K<-600
# Compute amplitude and phase-shifts for SSA, SSA-forecast and ham-diff: all filters are applied to returns

colo=c("black","blue","darkgreen","red")
amp_obj_SSA_now<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x),F)
amp_obj_SSA_for_6<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,1]),F)
amp_obj_SSA_for_12<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,2]),F)
amp_obj_ham<-amp_shift_func(K,ham_diff,F)

par(mfrow=c(2,1))
mplot<-cbind(amp_obj_SSA_now$amp,amp_obj_SSA_for_6$amp,amp_obj_SSA_for_12$amp,amp_obj_ham$amp)
# Scale SSA such that all amplitudes are normalized to one at frequency zero
mplot[,1]<-mplot[,1]/mplot[1,1]
mplot[,2]<-mplot[,2]/mplot[1,2]
mplot[,3]<-mplot[,3]/mplot[1,3]
mplot[,4]<-mplot[,4]/mplot[1,4]
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")

# 1. Plot amplitude
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude Hamilton-Filter",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# 2. Plot phase-shift
mplot<-cbind(amp_obj_SSA_now$shift,amp_obj_SSA_for_6$shift,amp_obj_SSA_for_12$shift,amp_obj_ham$shift)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase-shift ",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()


# Discussion: see the discussion at the end of example 1 above  
# General comments
# -SSA seems to be quite robust against misspecification of the dependence structure
#   -Examples 2 (white noise) and 3 (ARMA-model) lead to filters with similar/comparable performances
# -The data covers a very long history and is subject to non-stationarity which can lead to systematic biases of ht
#   -Biases could be addressed, to some extent, by adjusting SSA to the original Hamilton cycle  
#   -In all cases, SSA outperformed the target in terms of smoothness; in some cases the gain was less than projected, though
#     (the problem could be addressed by increasing ht in the SSA-constraint or by shortening the data, see example 4)
#   -In all cases, the forecast filters outperformed the target in terms of smoothness and lead (left shift)
# Once again, the positive phase-shift or lag of HF is substantial
#   -It is larger than the classic HP-concurrent trend, applied to returns, whose shift must vanish at frequency zero
#   -The size of the lag is due to the forecast horizon (2 years) in the regression equation 
#   -It is not clear at this point "why you should never use the HP" (which HP?)
#     -Don't use HP-gap, applied to the original data: yes, see confirmation in tutorial 2.
#     -But HP-trend applied to returns performs well: it is pretty smooth and its lag is smaller than HF, see tutorial 2.






##############################################################################################################
#############################################################################################################
# Example 4: same as example 3 but we use data past 1990 only.
# Our selection affects: 
#   1. The definition of the Hamilton cycle because the regression parameters will change 
#   2. The ARMA-model for deriving the Wold-decomposition xi

# Use data from 1990 up to 2019 (skip pandemic)
# Pandemic is analyzed in example 4.9, at the end of the tutorial
y<-as.double(log(PAYEMS["1990::2019"]))
ts.plot(y)
len<-length(y)
#--------------------------
# 4.1 Hamilton filter
# Settings proposed by Hamilton for quarterly data: two years ahead forecast and AR-order 4 to remove multiple unit-roots (if necessary)
h<-2*4
p<-4
# We here adapt to monthly PAYEMS: p remains the same (accounts for integration order, see Hamilton paper) 
h<-2*12
p<-4

explanatory<-cbind(y[(p):(len-h)],y[(p-1):(len-h-1)],y[(p-2):(len-h-2)],y[(p-3):(len-h-3)])
target<-y[(h+p):len]

lm_obj<-lm(y[(h+p):len]~explanatory)

# Substantially different regression parameters (as compared to example 3 above).
# In contrast to HP, which is fixed, Hamilton filter depends on data-fitting (data mining).
# This problem is sometimes overlooked in the literature (revision errors)
summary(lm_obj)
# Cycle appears  noisier (than in examples 2,3)
ts.plot(lm_obj$residuals)

# Specify filter
hamilton_filter<-c(1,rep(0,h-1),-lm_obj$coefficients[1+1:p])
intercept<-lm_obj$coefficients[1]
# We can fill any numbers for the leads from t+1,...,t+h-1 since the hamilton filter coefficients vanish there: we just repeat the target h-time
data_mat<-cbind(matrix(rep(target,h),ncol=h),explanatory)

residuals<-data_mat%*%hamilton_filter-intercept
# We just replicated the regression residuals (Hamilton filter)
par(mfrow=c(1,1))
ts.plot(cbind(residuals,lm_obj$residuals),main="Replication of regression residuals by hamilton_filter")

# The sum of filter coefficients nearly vanishes: this is because the series is non-stationary and therefore the filter must 
#   remove the trend 
# But there is no cointegration imposed: the out-of-sample cycle will be non-stationary
# Therefore HF must be regularly up-dated, as new data is available, which leads to revisions
sum(hamilton_filter)
# We now correct the filter such that the sum of the coefficients is zero: cointegration constraint 
#   (we just distribute the difference evenly on coefficients)
hamilton_filter_adjusted<-hamilton_filter
hamilton_filter_adjusted[(h+1):(h+p)]<-hamilton_filter_adjusted[(h+1):(h+p)]-sum(hamilton_filter)/p
# Now sum is zero
sum(hamilton_filter_adjusted)
# Compute corresponding adjusted residuals
residuals_adjusted<-data_mat%*%hamilton_filter_adjusted
# Center: not mandatory (we here rely on uncentered adjusted residuals or cycle estimates)
if (F)
  residuals_adjusted<-residuals_adjusted-mean(residuals_adjusted)

# Both cycles (regression residuals) differ in terms of `levels' , see discussions in previous examples
par(mfrow=c(2,2))
ts.plot(y,main="Log(PAYEMS)")
ts.plot(cbind(residuals,residuals_adjusted),col=c("red","blue"),main="Cycles")
mtext("Classic cycle",col="red",line=-1)
mtext("Adjusted un-centered cycle",col="blue",line=-2)
ts.plot(residuals-residuals_adjusted,main="Cycle difference")
# See discussion about cycle differences in example 2 above
# For the purpose of engrafting SSA onto HF, we here consider the adjusted and un-centered cycle 
#   But we also transform SSA back, to match the original cycle, see corresponding adjustments and plots below
#   For that purpose we need the difference between both cycles:
cycle_diffh<-residuals-residuals_adjusted


#---------------------------------------------------
# 4.2 Transformation: from levels to first differences, see previous examples
# We here select a larger filter-length L than in the previous examples because the weights of the Wold-decomposition 
#   decay more slowly: longer memory of the data (smoother pattern) after 1990 (great moderation)
# See section 2.3 and proposition 4 in JBCY paper for background.
L<-100
# Filter length: at least length of HF
L<-max(length(hamilton_filter_adjusted),L)
if (L>length(hamilton_filter_adjusted))
  hamilton_filter_adjusted_L<-c(hamilton_filter_adjusted,rep(0,L-length(hamilton_filter_adjusted)))

# Convolution with summation filter (unit-root assumption)
ham_diff<-conv_with_unitroot_func(hamilton_filter_adjusted_L)$conv
# Plot and compare both filters
par(mfrow=c(2,1))
ts.plot(ham_diff,main="HF as applied to first differences")
ts.plot(hamilton_filter_adjusted_L,main="HF as applied to level")

# Apply new filter ham_diff to returns and verify that it's filter output is the same as hamilton_filter_adjusted_L
x<-diff(y)
len_diff<-length(x)
residual_diff<-na.exclude(filter(x,ham_diff,side=1))

# Compare outputs
# Add NAs at the start of cycle_diff and Hamilton cycle (series are shorter due to applying HF)
cycle_diff<-c(rep(NA,length(x)-length(cycle_diffh)),cycle_diffh)
original_hamilton_cycle<-c(rep(NA,length(x)-length(residuals)),residuals)
# Check: ham_diff, as applied to first differences, generates the same cycle as original HF applied to levels
par(mfrow=c(1,1))
ts.plot(residual_diff,col="blue",main="Replication of HF: application to level vs. first differences")
lines(residuals_adjusted[(L-p-h+2):length(residuals)],col="red")

# With this transformation in place, we can now engraft SSA onto HF
#-------------------------------------------------------------------------------
# 4.3 Holding-times
# We first compute the holding-time of  ham_diff (as applied to differences)
ht_ham_diff_obj<-compute_holding_time_func(ham_diff)
# Much shorter holding time than in examples 2 (and 3): the regression equation now fits post-1990 data
ht_ham_diff_obj$ht 
ht_ham_example2

# But the empirical holding time is much longer: see discussion in examples 1-3 above
compute_empirical_ht_func(residuals_adjusted)

# Explanation:  xt (log-returns) are not white noise 
par(mfrow=c(1,1))
ts.plot(x,main="Log-returns PAYEMS")
abline(h=0)
#----------------------------------------------------------------------------
# 4.4 Autocorrelation
# One can split the data-sample into two-halves to assess out-of-sample performances
#   -SSA relies on ARMA-model to fit the dependence structure
# SSA is pretty insensitive as long as the ARMA-model is not severely overfitted
# Try either one (the comments refer to full-sample though both outcomes are nearly identical)
try_out_of_sample<-F
if (try_out_of_sample)
{
# Halve the data sample  
  in_sample_length<-length(x)/2
} else
{  
# Full sample  
  in_sample_length<-length(x)
}
# ACF suggests strong autocorrelation (`great moderation' effect?)
acf(x[1:in_sample_length],main="ACF: slowly decaying (longer memory)")
ar_order<-1
ma_order<-1
estim_obj<-arima(x[1:in_sample_length],order=c(ar_order,0,ma_order))
# We have the typical `cancelling' AR and MA-roots which can fit a weak but long lasting ACF 
estim_obj
tsdiag(estim_obj)
  
# Compute the MA-inversion of the ARMA: Wold-decomposition
xi<-c(1,ARMAtoMA(ar=estim_obj$coef[1:ar_order],ma=estim_obj$coef[ar_order+1:ma_order],lag.max=L-1))
# Remark: L should be sufficiently large for xi to decay (converge) to zero: this motivated the choice of L<-100
par(mfrow=c(1,1))
ts.plot(xi,main="Wold decomposition xi: slowly decaying (longer memory)")
# Convolve xi and ham_diff: filter applied to innovations in Wold decomposition (this filter is used for determining the holding-time only)
ham_conv<-conv_two_filt_func(xi,ham_diff)$conv
ht_ham_conv_obj<-compute_holding_time_func(ham_conv)


#--------------------------------------
# 4.5 Apply SSA
# The expected holding-time of ham_conv is slightly less than a year (ignoring the bias)
ht_ham_conv_obj$ht 
# We can lengthen by 50%: SSA will generate ~30% less crossings
ht<-1.5*ht_ham_conv_obj$ht
# This is the corresponding rho1 for the holding time constraint
rho1<-compute_rho_from_ht(ht)
# Target: we want to `improve' upon ham_diff: the filter applied to xt 
# By supplying xi to the SSA function-call, the algorithm `knows' that the data is not white noise`
gammak_generic<-ham_diff
# Forecast horizon: nowcast
forecast_horizon<-0

SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# ssa_eps is the filter which is applied to epsilont: it is mainly used to verify convergence of the optimization to the global maximum
SSA_filt_ham_diff_eps<-SSA_obj_ham_diff$ssa_eps
# ssa_x is the main filter of interest: it is applied to the data (xt)
SSA_filt_ham_diff_x<-SSA_obj_ham_diff$ssa_x

# Compare target and SSA: we must compare ham_diff with ssa_x (both are applied to xt)
mplot<-cbind(ham_diff,SSA_filt_ham_diff_x)
# Compare target and SSA: looks better than previous plot
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"),main="Filters as applied to first differences xt")
mtext("Hamilton ",col="black",line=-1)
mtext("SSA ", col="blue",line=-2)

# Alternatively, we could compare ham_conv and ssa_eps (both are applied to epsilont)
mplot<-cbind(ham_conv,SSA_obj_ham_diff$ssa_eps)
# Compare target and SSA: looks better than previous plot
par(mfrow=c(1,1))
ts.plot(mplot,ylim=c(min(mplot),max(mplot)),col=c("black","blue"),main="Convolved filters as applied to innovations epsilont")
mtext("Hamilton",col="black",line=-1)
mtext("SSA", col="blue",line=-2)


# Check optimization towards global maximum
# The holding-time of the solution should match the imposed number
ht_obj<-compute_holding_time_func(SSA_filt_ham_diff_eps)
# Matching numbers confirm that the optimization has converged 
ht_obj$ht 
ht


#--------------------------------------------------
# 4.6 Filter series and compute performance measures
# 4.6.1. SSA
SSA_out<-filter(x,SSA_filt_ham_diff_x,side=1)
# Empirical ht quite larger than targeted ht, see explanation above (series is non-stationary)
compute_empirical_ht_func(SSA_out)
ht  
# 4.6.2. Hamilton
ham_out<-filter(x,ham_diff,side=1)
compute_empirical_ht_func(ham_out)
ht_ham_conv_obj$ht 

# Although the series look quite similar, HF has more (noise-) leakage and therefore it generates 
#   more `noisy' crossings 
mplot<-na.exclude(cbind(SSA_out,ham_out))
colo=c("blue","red")
ts.plot(mplot[,1],col=colo[1])
lines(mplot[,2],col=colo[2])
abline(h=0)
mtext("Hamilton (as applied to differences)",col="red",line=-1)
mtext("SSA", col="blue",line=-2)

# Empirical holding times: SSA approximately 50% larger than HF (difference commensurate with sampling error)
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(ham_out)


# We can also compare the original Hamilton cycle with SSA shifted for the changing level (and scaled to original cycle)
# see example 1 above for details
lm_obj<-lm(original_hamilton_cycle~cycle_diff+SSA_out-1)
coef<-lm_obj$coef
# We now adjust SSA
scale_shifted_SSA<-coef[1]*cycle_diff+coef[2]*SSA_out
# We can plot adjusted SSA and original Hamilton cycle: data is missing at the start of scale_shifted_SSA shorter because it has been filtered
ts.plot(scale_shifted_SSA,col=colo[1],main="Shifted SSA cycle and original Hamilton cycle",ylim=c(min(original_hamilton_cycle,na.rm=T),max(original_hamilton_cycle,na.rm=T)))
mtext("Shifted and scaled SSA",col=colo[1],line=-1)
mtext("Original Hamilton cycle",col="black",line=-2)
lines(original_hamilton_cycle)
abline(h=0)

# Although both series look similar, SSA generates ~30% less crossings (even less than that) 
# Since model misspecification is smaller (than in previous examples), empirical and expected holding times are in better agreement 
compute_empirical_ht_func(scale_shifted_SSA)
compute_empirical_ht_func(original_hamilton_cycle)
# Note that scale_shifted_SSA is shorter than the original_hamilton_cycle (because of filtering)
# We here align time frames accordingly
# The correction does not affect our findings, though
compute_empirical_ht_func(original_hamilton_cycle[L:length(original_hamilton_cycle)])


#-----------------------------------------------
# 4.7 Forecasting
# Let us now address timeliness or lead/lags: 
# We augment the forecast horizon (delta in the JBCY paper) to obtain a faster (leading) SSA filter 
# We do not give-up in terms of smoothness or noise suppression since we do not change the ht-constraint

# We compute two forecast horizons and compare with nowcast
# Compute half a year and full year forecasts: we can supply a vector with the intended forecast horizons
#   SSA will return a matrix of filters: each column corresponds to the intended forecast horizon 
forecast_horizon<-c(6,12)
# SSA of Hamilton-diff
SSA_obj_ham_diff<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

SSA_filt_ham_diff_x_forecast<-SSA_obj_ham_diff$ssa_x

# Plot and compare SSA filter
# Morphing: the forecast filters assign less weight to remote past (faster/leading) but 
#   the particular filter patterns will ensure smoothness (ht unchanged)
par(mfrow=c(1,1))
ts.plot(SSA_filt_ham_diff_x_forecast,col=c("orange","darkgreen"),main="SSA forecasts vs nowcast",ylim=c(min(SSA_filt_ham_diff_x),max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x,col="blue")
mtext(paste("Forecast horizon ",forecast_horizon[1],sep=""),col="orange",line=-1)
mtext(paste("Forecast horizon ",forecast_horizon[2],sep=""),col="darkgreen",line=-2)
mtext("Nowcast",col="blue",line=-3)

# Check ht: matching numbers confirm convergence of the optimization to the global maximum
# Note that we supply ssa_eps: the filter which would be applied to epsilont (not xt)
apply(SSA_obj_ham_diff$ssa_eps,2,compute_holding_time_func)
ht

#----------------------
# Filter series: 
# 6 months ahead
SSA_out_forecast_6<-filter(x,SSA_filt_ham_diff_x_forecast[,1],side=1)
# 12 months ahead
SSA_out_forecast_12<-filter(x,SSA_filt_ham_diff_x_forecast[,2],side=1)

# The SSA-forecast is now shifted to the left; smoothness is the same.
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast_6,SSA_out_forecast_12,ham_out))
colo=c("blue","orange","darkgreen","red")
par(mfrow=c(1,1))
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot),max(mplot)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
lines(mplot[,4],col=colo[4])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon[1],sep=""),col=colo[2],line=-2)
mtext(paste("SSA forecast: delta=",forecast_horizon[2],sep=""),col=colo[3],line=-3)
mtext("Hamilton",col=colo[4],line=-4)
abline(h=0)

# ht of SSA designs is approximately 50% larger: stronger noise suppression by SSA
compute_empirical_ht_func(SSA_out)
compute_empirical_ht_func(SSA_out_forecast_6)
compute_empirical_ht_func(SSA_out_forecast_12)
compute_empirical_ht_func(ham_out)



# The forecast SSA filter has a lead at zero-crossings relative to HF (and less crossings)
max_lead=10
# SSA nowcast is synchronized with HF
shift_obj<-compute_min_tau_func(mplot[,c(1,4)],max_lead)
# SSA half-year forecast is leading Hamilton
shift_obj<-compute_min_tau_func(mplot[,c(2,4)],max_lead)
# SSA full-year forecast is leading Hamilton even more
shift_obj<-compute_min_tau_func(mplot[,c(3,4)],max_lead)

# Finally, we can analyze smoothness and timeliness issues by looking at amplitude and phase-shift functions
# These results offer an alternative description, based on formal filter characteristics
# They confirm the above empirical results (obtained by comparing filtered series)


#----------------------------------------
# 4.8 Compute amplitude and phase-shift functions
# Amplitude at higher frequencies: closer to zero means less noise leakage
# Phase-shift in passband: a smaller shift means a relative lead

# Select the number of equidistant frequency ordinates (grid-size) in [0,pi] 
K<-600
# Compute amplitude and phase-shifts for SSA, SSA-forecast and ham-diff: all filters are applied to returns

colo=c("black","blue","darkgreen","red")
amp_obj_SSA_now<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x),F)
amp_obj_SSA_for_6<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,1]),F)
amp_obj_SSA_for_12<-amp_shift_func(K,as.vector(SSA_filt_ham_diff_x_forecast[,2]),F)
amp_obj_ham<-amp_shift_func(K,ham_diff,F)

par(mfrow=c(2,1))
mplot<-cbind(amp_obj_SSA_now$amp,amp_obj_SSA_for_6$amp,amp_obj_SSA_for_12$amp,amp_obj_ham$amp)
# Scale SSA such that all amplitudes are normalized to one at frequency zero
mplot[,1]<-mplot[,1]/mplot[1,1]
mplot[,2]<-mplot[,2]/mplot[1,2]
mplot[,3]<-mplot[,3]/mplot[1,3]
mplot[,4]<-mplot[,4]/mplot[1,4]
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")

# 1. Plot amplitude: smaller amplitude at higher frequencies means stronger noise suppression (longer holding time)
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude Hamilton-Filter",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# 2. Plot phase-shift: smaller shift at lower frequencies means a lead
mplot<-cbind(amp_obj_SSA_now$shift,amp_obj_SSA_for_6$shift,amp_obj_SSA_for_12$shift,amp_obj_ham$shift)
colnames(mplot)<-c(paste("SSA(",round(ht,1),",",0,")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[1],")",sep=""),paste("SSA(",round(ht,1),",",forecast_horizon[2],")",sep=""),"Hamilton")
plot(mplot[,1],type="l",axes=F,xlab="Frequency",ylab="",main=paste("Phase-shift ",sep=""),ylim=c(min(mplot),max(mplot)),col=colo[1])
lines(mplot[,2],col=colo[2])
mtext(colnames(mplot)[1],line=-1,col=colo[1])
if (ncol(mplot)>1)
{
  for (i in 2:ncol(mplot))
  {
    lines(mplot[,i],col=colo[i])
    mtext(colnames(mplot)[i],col=colo[i],line=-i)
  }
}
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()


# Discussion: see the discussion at the end of example 1 above  
# Findings
# -by discarding the remote past (from WWII up to 1990), the shortened time series allows for a less inconsistent modelling of the 
#     data (non-stationarity is less pronounced from 1990-2023).
# -As a result, expected and empirical holding times match better than in the previous examples 2 and 3 (entire data set).
# -Also, empirical holding times of SSA match the intended 50% increase (over target) better than in examples 2 and 3.
#   -Sometimes a bit smaller: unadjusted cycles.
#   -sometimes quite a bit larger: adjusted cycles. 
#   -deviations are compatible with random sampling errors.
# The positive phase-shift or lag of HF is smaller than in the previous examples.
#   -HF depends on the selection of the data-window (for estimating parameters of the regression equation)
#   -The lag is still larger than the classic HP-concurrent trend, applied to returns, whose shift must vanish at frequency zero.
#   -The size of the lag is due to the forecast horizon (2 years) in the regression equation 
#   -It is not clear at this point "why you should never use the HP"
#     -Don't use HP-gap, applied to the original data: yes, see confirmation in tutorial 2.
#     -But HP-trend applied to returns performs well: it is pretty smooth and its lag is smaller, see tutorial 2.
#     -Moreover, HP-trend does not depend on the data-window or sample-size (there are pros and cons to this argument)


#----------------------------------------------------------------------------------------------
# 4.9 To conclude we apply the above filters unchanged (no re-estimation) out-of-sample, including the pandemic.
# With a potentially instructive outcome...

# Compute long sample
y<-as.double(log(PAYEMS["1990/"]))
x<-diff(y)
# The very strong pandemic outliers will act as `impulses`, triggering the impulse response of the filter, 
#   i.e., the proper (sign inverted) filter coefficients
par(mfrow=c(1,1))
ts.plot(x)

# Apply all filters unchanged (out of sample)
# 1. SSA nowcast
SSA_out<-filter(x,SSA_filt_ham_diff_x,side=1)
# 2. Hamilton
ham_out<-filter(x,ham_diff,side=1)
# 3. Both SSA forecasts
SSA_out_forecast_6<-filter(x,SSA_filt_ham_diff_x_forecast[,1],side=1)
# 12 months ahead
SSA_out_forecast_12<-filter(x,SSA_filt_ham_diff_x_forecast[,2],side=1)

# Interesting:
# The pandemic dip is mirrored by a later peak whose timing depends on the SSA-design
# Explanation: negative Impulse responses of the filters
mplot<-na.exclude(cbind(SSA_out,SSA_out_forecast_6,SSA_out_forecast_12,ham_out))
colo=c("blue","orange","darkgreen","red")
par(mfrow=c(1,1))
ts.plot(mplot[,1],col=colo[1],ylim=c(min(mplot,na.rm=T),max(mplot,na.rm=T)))
lines(mplot[,2],col=colo[2])
lines(mplot[,3],col=colo[3])
lines(mplot[,4],col=colo[4])
mtext("SSA nowcast",col=colo[1],line=-1)
mtext(paste("SSA forecast: delta=",forecast_horizon[1],sep=""),col=colo[2],line=-2)
mtext(paste("SSA forecast: delta=",forecast_horizon[2],sep=""),col=colo[3],line=-3)
mtext("Hamilton",col=colo[4],line=-4)
abline(h=0)

# In order to understand the observed pattern, it is instructive to have a look at filter coefficients, once again
# The (negative) Pandemic impulse replicates the (sign inverted) pattern of the corresponding filters 
par(mfrow=c(1,1))
ts.plot(SSA_filt_ham_diff_x_forecast,col=c("orange","darkgreen"),main="SSA forecasts vs nowcast",ylim=c(min(SSA_filt_ham_diff_x),max(SSA_filt_ham_diff_x_forecast)))
lines(SSA_filt_ham_diff_x,col="blue")
mtext(paste("Forecast horizon ",forecast_horizon[1],sep=""),col="orange",line=-1)
mtext(paste("Forecast horizon ",forecast_horizon[2],sep=""),col="darkgreen",line=-2)
mtext("Nowcast",col="blue",line=-3)

# We can now easily understand the secondary 'fake' peak of the filtered series as well as its different location on the time-axis
# Advantage of forecast filters
#   -a longer forecast horizon leads to a more rapid zero-decay of filter coefficients
#   -therefore, this type of filter `forgets' more rapidly extreme or singular observations (than nowcast or target)
#   -As an example, the one-year ahead forecast could be applied 1.5 years after the outliers occurred
#   -In contrast, the nowcast or HF would need another 10-11 months to `forget' the singularity
# It is not clear, at this point, "why you should never use the HP" (which HP?)
#   -The impulse response of the classic HP-trend decays faster than HF 
#   -Therefore the effects of gross outliers (pandemic) would be less pronounced (smeared)


###################################################################################################
###################################################################################################
# Summary
# -We proposed a variant of HF onto which SSA could be grafted
# -We also proposed a (simple) adjustment for SSA to match the original (Hamilton) cycle
# -SSA is quite robust against model-misspecification
#   -The very long sample in example 2 and 3, ranging from WWII up to 2023, is subject to changes in the time series dynamics (non-stationarity)
#     -White noise and ARMA specifications generated fairly similar results on the long time span 
#     -Empirical holding times are generally strongly biased. But correcting for misspecifications generally reduced this bias (see results for adjusted SSA vs. original Hamilton cycle)
#     -Imposing a 50% larger ht did not improve relative smoothness (of SSA against target) accordingly. However, 
#       one could simply increase ht in the SSA-constraint further (for example requiring 100% increase)
#     -These differences and deviations from target were mainly due to non-stationarity (model misspecification)
#   -The shorter sample, starting in 1990 (great moderation), led to more consistent results
# -SSA is also quite robust against overfitting, at least as long as the ARMA-model is not severely overfitted
#   -For typical applications, a simple ARMA(1,1) is often sufficiently flexible to extract the relevant features from the data   
# -SSA nowcasts generally improve upon the target in terms of smoothness (less noisy crossings). 
#   -Empirical improvements are commensurate with theoretical specifications, at least in the long run and assuming that the model is not severely misspecified  
# -SSA forecasts can retain the same smoothness while outperforming the target in terms of timeliness, too (lead/left-shift)
#   -One cannot improve speed and smoothness without loosing in terms of MSE-performances
#   -The tradeoff is a trilemma, see tutorial 0.1 on trilemma
#   -SSA allows to position the filter-design in the limits delineated by the trilemma: best MSE for given ht and shift
# -A more refined and effective handling of timeliness (left shift) is proposed in due time
