# We here 
# -briefly derive the classic mean-square error (MSE) predictor for a simple `toy' signal extraction problem
#   -Based on white noise: example 1
#   -Based on an autocorrelated process: example 2
# -Introduce SSA, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
#   -replicate MSE by SSA: example 3
#   -`play' with the input to SSA: example 4
# -In the subsequent tutorials we can view the MSE predictor both as a benchmark, for comparing performances, as well 
#   as a base-predictor, on which SSA can be plugged to alter performances: smoothness and/or timeliness 


rm(list=ls())

# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))
# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))


#----------------------------------------------------------------
# Example 1: xt=epsilont white noise
# Assume the following symmetric target filter (gamma does not have to be symmetric, see tutorials 2-5)
gamma<-c(0.25,0.5,0.75,1,0.75,0.5,0.25)

# Symmetric target filter
plot(gamma,axes=F,type="l",xlab="Lag-structure",ylab="filter-coefficients",main="Simple signal extraction (smoothing) filter")
axis(1,at=1:length(gamma),labels=(-(length(gamma)+1)/2)+1:length(gamma))
axis(2)
box()
# Note that the above plot indicates that gamma is meant as a two-sided (acausal) filter

# We can apply the filter to white noise: xt=epsilont
set.seed(231)
len<-120
# Scaling
sigma<-1
epsilon<-sigma*rnorm(len)
x<-epsilon
# No autocorrelation
acf(x)


# We can filter the data: either by assuming a two-sided acausal design (side=2) or a causal one-sided design (side=1)
y_sym<-filter(x,gamma,side=2)
y_one_sided<-filter(x,gamma,side=1)

tail(cbind(y_sym,y_one_sided))

# When the filter is two-sided (y_sym) the series is left-shifted and we do not observe the filter output 
# towards the sample end (NAs). In contrast, we observe the one-sided filter `till the sample end, but it is right-shifted (delayed)

ts.plot(cbind(y_sym,y_one_sided),col=c("black","black"),lty=1:2,main="One-sided vs. two-sided")

# -In applications one is often interested in obtaining estimates of y_sym towards the sample end
# -An estimate of y_sym at the sample end t=len is called a nowcast: in the above example, we have to compute 
#   forecasts for the future data x_{len+1}, x_{len+2},x_{len+3} missing in the symmetric filter. 
#   If we compute MSE-forecasts, then we obtain an MSE estimate of the target zt at t=len.
# -In our case (white noise) the MSE forecasts of the future data are zero and we obtain the one-sided truncated 
#   filter as MSE nowcast

b_MSE<-gamma[((length(gamma)+1)/2):length(gamma)]
plot(b_MSE,axes=F,type="l",xlab="Lag-structure",ylab="filter-coefficients",main="MSE-nowcast filter")
axis(1,at=1:((length(gamma)+1)/2),labels=-1+1:((length(gamma)+1)/2))
axis(2)
box()

# We can now filter xt with this filter to obtain yt and compare the estimate yt and the target zt
# The filter is one-sided: side=1
y_mse<-filter(x,b_MSE,side=1)

y_sym<-filter(x,gamma,side=2)

ts.plot(cbind(y_sym,y_mse),col=c("black","green"),lty=1:2,main="Target (black) vs MSE (green)")
abline(h=0)

# We can see that the MSE is slightly noisier. We can compute the empirical holding times of both filters
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)
# The MSE filter crosses the zero line more often than the target (excess of 'noisy' alarms)
# We can compare the empirical holding times to the expected ht, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
#   In the long run, empirical numbers converge to expected numbers (in the absence of model misspecification)
compute_holding_time_func(gamma)$ht
compute_holding_time_func(b_MSE)$ht
# We infer that MSE generates roughly 50% more crossings than the target: half of its crossings are 'false alarms'

# We see also that the MSE is right shifted relative to the target (lagging)
# Let's measure the shift at zero-crossings based on the tau statistic introduced in JBCY paper
# For that purpose we compare target and MSE
data_mat<-cbind(y_sym,y_mse)
# The function shifts the series in the second column against the series in the first column for various leads and lags
# For each shift it computes the distances between (closest) zero-crossings of both series and sums these distances, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# The shift at which the sum is smallest is a measure for the lead (left shift) or lag (right shift) of the series 
#   in the first column 
# The following plot suggest a lead of the target (a lag of the MSE) of one time-unit
max_lead<-4
compute_min_tau_func(data_mat,max_lead)

# Next, we can compute empirical and true or expected mean-square error
mean((y_sym-y_mse)^2,na.rm=T)
# The theoretical MSE can be obtained easily: since we replaced future epsilont by their forecast zero, the error between 
# the MSE predictor and the target corresponds to the missing (future) innovations in the target and the MSE is the sum 
# of the squared coefficients applied to these epsilont
# a. Weights assigned to future innovations by symmetric filter
gamma[1:3]
# b. MSE
sigma^2*sum(gamma[1:3]^2)

# We can also compute the percentage MSE which sets the error in relation to the signal or target
sum(gamma[1:3]^2)/sum(gamma^2)
# Outcome: the variance of the nowcast prediction error corresponds to 30% of the variance of the target

# To conclude, we can look at amplitude and phase-shift functions of the MSE-predictor
#   -The amplitude tells us something about noise suppression of the filter: a 'good' filter damps high-frequency components strongly
#   -The phase-shift informs about the lag (or lead) of the predictor: a good filter has a small shift
# Both functions can be computed easily with amp_shift_func
# For this we first specify the number of equidistant frequency ordinates in [0,pi] (we omit negative frequencies which are mirrored)
K<-600
amp_obj_mse<-amp_shift_func(K,as.vector(b_MSE),F)
amp_mse<-amp_obj_mse$amp
shift_mse<-amp_obj_mse$shift
par(mfrow=c(2,1))
plot(amp_mse,type="l",axes=F,xlab="Frequency",ylab="",main="Amplitude of MSE filter",col="green",ylim=c(0,max(amp_mse)))
abline(h=0)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
plot(shift_mse,type="l",axes=F,xlab="Frequency",ylab="",main="Shift MSE",col="green")
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# The amplitude function shows that the MSE-filter is a lowpass; the amplitude damps slightly higher frequencies 
#   but noise-leakage is quite important: high-frequency components can seep through the filter and generate
#   additional `noisy' crossings, as observed above 
# The shift in the passband of the filter is roughly one time unit, which confirms our previous findings
# An advantage of frequency-domain characteristics is that amplitude- and shift-functions do not depend on data: 
#   they are entirely determined by the filter
# Therefore, they offer an alternative `independent' ('data-free' and `model-free') assessment of vital characteristics
#   of the optimal estimate
# Tutorials 2-3 and 5 will confirm the above findings: 
#   -faster SSA (leading or left shifted) will have a smaller shift
#   -smoother SSA (larger ht) typically will have a smaller amplitude at higher-frequencies 
#     (tutorial 4 proposes an instructive counter-example) 
#  

# To conclude we derive the spectral density of the predictor y_mse
#   -Note that epsilont, the input of the MSE filter, has a flat spectral density
#   -The convolution theorem then implies that the spectral density of y_mse corresponds to the squared amplitude of b_MSE
par(mfrow=c(1,1))
plot(sigma^2*amp_mse^2,type="l",axes=F,xlab="Frequency",ylab="",main="Spectral density of MSE predictor",col="green",ylim=c(0,max(sigma^2*amp_mse^2)))
abline(h=0)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# The spectral density will be useful when assessing whether a particular BCA-tool (HP-filter, BK-filter, Hamilton-filter, BN-filter) 
#   generates spurious cycles or not. And some of them do, indeed!


# Discussion: the MSE solution above minimizes the mean-square filter error 
#   -No other competing design can outperform y_mse in terms of MSE performances 
# But y_mse is right-shifted (lagging) and y_mse is noisier (smaller ht)
# Can we improve somehow smoothness and/or timeliness? Both at once? What price do we have to pay in terms of MSE-performances?
# The formal SSA-framework introduced in Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5 addresses these questions

#######################################################################################################
######################################################################################################
# Example 2: xt=ARMA (not white noise anymore) 
# Same target as above
gamma<-c(0.25,0.5,0.75,1,0.75,0.5,0.25)
# New series
set.seed(76)
len<-1200
# Specify ARMA parameters for the DGP (data generating process)
a1<-0.4
b1<-0.3
sigma<-1
epsilon<-sigma*rnorm(len)
x<-epsilon
# Let's do it by hand (without arima.sim)
for (i in 2:len)
{
  x[i]<-a1*x[i-1]+epsilon[i]+b1*epsilon[i-1]
}

# The empirical autocorrelation function: no more white noise
acf(x)

# Set filter length: should be sufficiently large for filter weights to decay to zero
L<-50

# we can apply gamma to xt or epsilont: in the latter case we have to convolve gamma with the Wold-decomposition of xt
# For that purpose we have to compute the weights of the Wold-decomposition (MA-inversion) of the ARMA

xi<-c(1,ARMAtoMA(ar=a1,ma=b1,lag.max=L-1))
par(mfrow=c(1,1))
ts.plot(xi,main="Wold decomposition of ARMA(1,1)")

# Note that we can replicate xt by relying on the MA-inversion (Wold-decomposition) of the ARMA:
x_wold<-filter(epsilon,xi,side=1)
ts.plot(cbind(x,x_wold),main="ARMA vs. Wold: both series overlap")

# Let's now compute the convolution filter
gamma_conv<-conv_two_filt_func(xi,gamma)$conv
ts.plot(gamma_conv,main="Target filter as applied to epsilont")
# This filter is not symmetric anymore!

# Let's check that the output of both filters match
# We here rely on the one-sided filters (since otherwise the filter function would shift the outputs differently)
y_sym<-filter(x,gamma,side=1)
y_conv<-filter(epsilon,gamma_conv,side=1)  

# Perfect match
ts.plot(cbind(y_sym,y_conv),main="Target applied to xt overlaps convolved target applied to epsilont")

# Why do we transform the target filter by convolution? 
# Four main purposes: 
# 1. We can easily derive the MSE nowcast (or any forecast or backcast) by replacing future epsilont by their optimal MSE zero-forecast
# 2. we can apply the formula for ht in Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5 (because the expression assumes the data to be white noise)
# 3. We can easily derive the spectral density of the MSE predictor yt 
# 4. We can easily compute its expected MSE

# Let's first derive the MSE nowcast
# We set epsilon_{t+3}=epsilon_{t+2}=epsilon_{t+1}=0 and obtain the truncated filter
# Note that we append zeroes in order to keep the length fixed at L
gamma_conv_mse<-c(gamma_conv[4:L],rep(0,3))

ts.plot(gamma_conv_mse,main="MSE filter as applied to epsilont")

# In general we would like the MSE filter as applied to xt (not epsilont). Very easy: just deconvolve gamma_mse from gamma_conv_mse

gamma_mse<-deconvolute_func(gamma_conv_mse,xi)$dec_filt

ts.plot(gamma_mse,main="MSE filter as applied to xt")

# Check that filter outputs are identical

y_mse<-filter(x,gamma_mse,side=1)
y_conv_mse<-filter(epsilon,gamma_conv_mse,side=1)

# Same output
ts.plot(cbind(y_conv_mse,y_mse),main="Optimal filter applied to xt is the same as convolved filter applied to epsilont")

# We can now proceed to comparisons of MSE and target

y_sym<-filter(x,gamma,side=2)

ts.plot(cbind(y_sym,y_mse),col=c("black","green"),lty=1:2,main="Target (black) vs MSE (green)")
abline(h=0)

# We can see that the MSE is slightly noisier. Let's compute the empirical holding times of both filters
compute_empirical_ht_func(y_sym)
compute_empirical_ht_func(y_mse)
# The MSE filter crosses the zero line more often than the target (excess of 'noisy' alarms)

# We can compare the empirical holding times to the expected ht, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
#   In the long run, empirical numbers converge to expected numbers (in the absence of model misspecification)
# But these numbers do not seem to match 
#   Random sample deviations cannot explain the mismatch since the  series are sufficiently long
compute_holding_time_func(gamma)$ht
compute_holding_time_func(gamma_mse)$ht
# Solution: We need to provide the convolved filters, as applied to epsilont in order to obtain the correct 
#   expected ht (recall the second purpose of the convolved filters mentioned above)
compute_holding_time_func(gamma_conv)$ht
compute_holding_time_func(gamma_conv_mse)$ht
# We infer that MSE generates roughly 40% more crossings than the target ('false alarms') in the long run

# We see, also, that the MSE is right shifted relative to the target (lagging)
# Let's measure the shift at zero-crossings based on the tau statistic introduced in JBCY paper
# For that purpose we compare target and MSE
data_mat<-cbind(y_sym,y_mse)
# The following plot suggest a lead of the target (a lag of the MSE) of one time-unit
max_lead<-4
compute_min_tau_func(data_mat,max_lead)

# Next we can compute empirical and true expected mean-square errors
mean((y_sym-y_mse)^2,na.rm=T)
# True MSE (see example 1 for background)
sigma^2*sum(gamma_conv[1:3]^2)
# We can also compute the percentage MSE which sets the error in relation to the signal or target
sum(gamma_conv[1:3]^2)/sum(gamma_conv^2)
# The relative MSE, 17%, is smaller than in the previous example because xt is smoother than epsilont
# Therefeore, the MSE predictor has to solve an `easier' problem here
# SSA can address smoothness and timeliness issues (at cost of MSE), see tutorials 1-5



# To conclude we look at amplitude and phase-shift functions of the MSE-predictor
# In general we are interested in analyzing how the predictor-filter affects the data: how strong
#   is the noise suppression, how large a lag?
# Therefore we generally compute amplitude and shifts of gamma_mse (the filter applied to xt)
K<-600
amp_obj_mse<-amp_shift_func(K,as.vector(gamma_mse),F)
amp_mse<-amp_obj_mse$amp
shift_mse<-amp_obj_mse$shift
par(mfrow=c(2,1))
plot(amp_mse,type="l",axes=F,xlab="Frequency",ylab="",main="Amplitude of MSE filter",col="green",ylim=c(0,max(amp_mse)))
abline(h=0)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
plot(shift_mse,type="l",axes=F,xlab="Frequency",ylab="",main="Shift MSE",col="green")
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# We see that noise-leakage at higher frequencies is even more pronounced than in the previous example
#   This is because xt is smoother than epsilont and needs less smoothing by the predictor-filter
# As an effect, the lag (phase-shift in passband) is slightly smaller, though still close to one

# If we want to compute the spectral density of the predictor y_mse then we have to rely on gamma_conv_mse, the filter 
#   applied to epsilont, because xt does not have a flat spectrum (but epsilont does)
par(mfrow=c(1,1))
amp_obj_mse<-amp_shift_func(K,as.vector(gamma_conv_mse),F)
amp_conv_mse<-amp_obj_mse$amp
plot(sigma^2*amp_conv_mse^2,type="l",axes=F,xlab="Frequency",ylab="",main="Spectral density of predictor",col="green",ylim=c(0,max(sigma^2*amp_conv_mse^2)))
abline(h=0)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()
# The spectrum is rather weak towards higher frequencies: besides the squaring effect (we look at the squared amplitude function), 
#   this effect is also due to the fact that the ARMA-filter, linking xt to epsilont, is a lowpass in this example: 
#   xt is smoother than epsilont
amp_obj_xi<-amp_shift_func(K,as.vector(xi),F)
amp_arma<-amp_obj_xi$amp
par(mfrow=c(1,1))
plot(amp_arma,type="l",axes=F,xlab="Frequency",ylab="",main="Amplitude of ARMA filter",col="green",ylim=c(0,max(amp_arma)))
abline(h=0)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
axis(2)
box()

#####################################################################################################
# We provide context for understanding SSA, assuming the above example 2 for background 
# Let zt designate a target: zt=sum_{k=-\infty}^{\infty} gamma_k x_{t-k}
#   In the above examples, zt=y_sym is the output of the acausal two-sided filter 
# Let xt=sum_{j=0}^{\infty} xi_j epsilon_{t-k} be a stationary (or non-stationary integrated process)
# We want to estimate or predict z_{t+delta}, for integer delta

# The general proceeding in example 2 applies for SSA, too. Let's summarize the main steps
# 1. Find the Wold-decomposition: xi 
#   -Typically one fits a SARIMA-model to the data and inverts the model into an MA(\infty) using the above ARMAtoMA function
# 2. Convolve gamma_k with xi and replace future epsilon_t by zero: this leads to the MSE filter applied to epsilont
# 3. Deconvolute gamma from the MSE (applied to epsilont): this leads to the MSE filter applied to xt 

# These steps apply to SSA, too
# One specifies the target z_{t+delta} by providing:
#   1. gammak: forecasting or signal extraction (no restrictions imposed upon gammak other than being of finite length)
#   2. delta: forecast, nowcast or backcast
#   3. xi: if one does not supply xi, then SSA assumes by default the data to be white noise (xt=epsilont)
# Additionally to the MSE estimate we also specify 
#   3. ht or, more exactly, the lag-one ACF rho1 of the holding-time constraint
#   4. L: the filter length. Restriction: 3<=L<=len-1. A larger L automatically leads to a better predictor if xi (Wold decomposition) is correctly specified. 
#     For most applications the filter coefficients decay pretty fast towards zero: therefore truncation at smaller L 
#     generally does not affect performances.
# For given L, the SSA criterion computes the `best' or optimal finite-length filter subject to the holding-time constraint
#   Best or optimal means
#     1. The filter which best matches signs of the target z_{t+delta}: maximization of the sign-accuracy
#     2. The filter which correlates most strongly with the target z_{t+delta}: maximization of the target correlation
#   Both criteria are equivalent under Gaussianity (see JBCY paper) and the link is fairly robust against departures of Gaussianity (t-distribution up to nu=2 still works fine, equity data is OK, Macro data is fine too,...)

# The SSA function returns
# 1. The best SSA-filter as applied to epsilont: ssa_eps (compliant with holding-time constraint)
# 2. The best SSA-filter as applied to xt: ssa_x (compliant)
#     Typically, ssa_x is the requested filter: it can be applied to the original data (xt) and it is smooth (holding-time constraint) 
# SSA also computes:
# 3. The best MSE as applied to epsilont: mse_eps (generally not compliant with holding-time constraint)
# 4. The best MSE as applied to xt: mse_x (not compliant)
# Finally SSA computes theoretical criterion values and holding-times
# 5. crit_rhoyz: correlation with MSE predictor of target
# 6. crit_rhoy_target: correlation with two-sided-target
# 7. crit_rhoyy: lag one ACF of optimized solution: this number should match ht (if not, the optimization did not converge)

# If the model for xt (i.e. Wold decomposition xi) is correctly specified, then classic sample estimates will converge 
#   to crit_rhoyz, crit_rhoy_target and crit_rhoyy for sufficiently long samples of the data

# Examples 3 and 4 illustrate 
#   1. The function call: input and return
#   2. Convergence of sample estimates to returned criteria


#############################################################################################
#############################################################################################
# Example 3 Replicate MSE by SSA
# We here rely on the framework of example 2 above
# As stated in the introduction to these tutorials, SSA replicates MSE if ht in the holding-time constraint matches 
#   ht of the benchmark (here the MSE-predictor)

# Set hyperparameters
# Holding time: we want SSA to replicate the MSE predictor: therefore we set ht accordingly
# We need the convolved filter (applied to epsilont) for inferring ht
ht<-compute_holding_time_func(gamma_conv_mse)$ht
# Instead of ht, we provide the bijective `twin', namely the lag-one acf, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# We can transform ht in rho1 with the function compute_rho_from_ht 
rho1<-compute_rho_from_ht(ht)
# Alternatively we could set rho1 directly (the above function computes ht as well as the lag-one acf)
rho1<-compute_holding_time_func(gamma_conv_mse)$rho_ff1
# Filter length (should be at least twice the holding time)
L<-max(L,2*round(ht,0))
# Nowcast i.e. delta=0
delta<-0
# Recall that the (symmetric, two-sided) target gamma is right-sifted by (length(gamma)-1)/2=3 time units: it is a one-sided causal filter
# In order to obtain the two-sided acausal target we have to left-shift gamma by (length(gamma)-1)/2 lags or, 
#   stated otherwise, to forecast the causal gamma by (length(gamma)-1)/2+delta steps ahead 
forecast_horizon<-(length(gamma)-1)/2+delta
forecast_horizon
# We let SSA know that the data is an ARMA: recall that the ARMA-filter pre-whitens the noise (it's a lowpass) 
xi<-xi
# Target: symmetric (one-sided) filter: this will be shifted by forecast_horizon in SSA_func
# This is the filter applied to xt (not epsilont)
gammak_generic<-gamma

# Apply SSA
# Since gamma is applied to xt, we have to supply information about the link between xt and epsilont in terms of xi
# Warning messages inform that zeroes are appended to shorter filters and that SSA solution is very close to MSE after optimization
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

# Filter as applied to xt
ssa_x<-SSA_obj$ssa_x
# Convolved filter as applied to epsilont
ssa_eps<-SSA_obj$ssa_eps

mplot<-cbind(ssa_x,gamma_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to xt: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()


# Same for the convolved designs (filters applied to epsilont)
ssa_eps=SSA_obj$ssa_eps
mplot<-cbind(ssa_eps,gamma_conv_mse)
# Both filters overlap: SSA just replicated MSE up to arbitrary scaling
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to epsilont: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()


# SSA also computes the MSE-filters directly: as applied to xt and epsilont
mplot<-cbind(SSA_obj$mse_x,gamma_mse)
# Both filters overlap: SSA just replicated MSE up to arbitrary scaling
plot(mplot[,1],main="MSE by SSA and original MSE as applied to xt: both filters overlap",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()
# MSE as applied to epsilont
mplot<-cbind(SSA_obj$mse_eps,gamma_conv_mse)
# Both filters overlap: SSA just replicated MSE up to arbitrary scaling
plot(mplot[,1],main="MSE by SSA and original MSE as applied to epsilont: both filters overlap",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

# In applications one typically relies on ssa_x: but the package also returns all other filters (for diagnostics, benchmarling, validation,...)

#------------------------------
# Some checks of performances: convergence of sample estimates to theoretical criterion values and ht
# A. Criterion values: 
# 1. Maximize the correlation with the MSE solution (here gamma_conv_mse) or
# 2. Maximize the correlation with the effective target (here the two sided filter)
# Both criteria are equivalent, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# Let's compute the first one:
#   We can rely on the convolved filters (applied to epsilont) to compute correlations
#   Assuming white noise, the exact formula for the cross-correlation is
t(ssa_eps)%*%gamma_conv_mse/sqrt(t(ssa_eps)%*%ssa_eps*t(gamma_conv_mse)%*%gamma_conv_mse)
# Since SSA replicates MSE the cross-correlation is one
# SSA returns this criterion value
SSA_obj$crit_rhoyz
# The second criterion: maximize correlation with two-sided target
#   We just have to correct for the different squared lengths (or variances) of MSE and two-sided target, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
#   Since this ratio is independent of ssa_eps both criteria are equivalent (up to scaling)
length_ratio<-sqrt(sum(gamma_conv_mse^2)/sum(gamma_conv^2))
t(ssa_eps)%*%gamma_conv_mse/sqrt(t(ssa_eps)%*%ssa_eps*t(gamma_conv_mse)%*%gamma_conv_mse)*length_ratio
# SSA also returns the corresponding criterion value
SSA_obj$crit_rhoy_target
# The correlation with the effective target is smaller than one because the two-sided filter relies on future (unobserved) data
# In applications it is invariably the case that crit_rhoy_target <= crit_rhoyz 

# We can verify pertinence of the above criterion values by computing empirical correlations
y_ssa<-filter(x,ssa_x,side=1)
# First criterion
cor(na.exclude(cbind(y_mse,y_ssa)))[1,2]
# Second criterion
cor(na.exclude(cbind(y_sym,y_ssa)))[1,2]

# B. Besides optimization criteria we can and should also check holding-times or, equivalently, 
#   lag-one ACFs of the filters, see proposition 2 in JBCY paper
# We imposed 
ht
# This ht corresponds to the lag-one ACF
rho1
# After optimization, SSA achieves
SSA_obj$crit_rhoyy
# If both values match, then this is proof that the optimization reached the global maximum 
#   We could improve tightness of the approximation by increasing the number of iterations (of the numerical optimization)
#   The default value is 20: this is sufficient for typical (non-exotic) applications (in short: don't be concerned about this technical feature)

# Let's now check holding-times
# We can rely on the function compute_holding_time_func:
compute_holding_time_func(ssa_eps)$ht
# Note that we have to supply the convolved filter ssa_eps since computations of ht assume the data to be white noise

# Indeed, the filter as applied to xt has a substantially smaller ht 
compute_holding_time_func(ssa_x)$ht
# This is because the ARMA-filter has already a smoothing effect on epsilont
compute_holding_time_func(xi)$ht
# In fact, ssa_x lengthens the native ht of the ARMA-filter (which is a lowpass in our example) to match the 
#   imposed constraint. Note that ht of convolved filters do not combine additively (non-linear function of lag-one ACF, see proposition 2 in JBCY paper) 
# Let's have a look at the empirical lag-one ACF: it matches the imposed rho1, as desired
acf(na.exclude(y_ssa))$acf[2]
rho1
# Let's have a look at the empirical ht
compute_empirical_ht_func(y_ssa)
# It matches our constraint (error can be made arbitrarily small when increasing the sample size)
ht
# Empirical measures converge to expected numbers for longer time series because xi is the true model
# If empirical estimates differ substantially from returned theoretical or expected values, then the model (xi) is misspecified
# SSA generally performs well even if the model is misspecified: we provide ample empirical evidence in the following tutorials
# Note: one could re-interpret (divert) the above statistics as model-diagnostics for xi  (instead of Ljung-Box)

#########################################################################################
#########################################################################################
# Example 4: alternative target specifications
# We copy example 3 but we exchange the symmetric filter for the MSE-filter in the target specification, see Wildi, M. (2024) https://doi.org/10.1007/s41549-024-00097-5
# We can supply either gamma_mse (as applied to xt) or gamma_conv_mse (as applied to epsilont)
#   But we must be careful when doing so...
# A. First case: gamma_mse the MSE filter applied to xt
gammak_generic<-gamma_mse
# This target is causal and does not have to be shifted (in contrast to gamma in exercise 3). We set:
forecast_horizon<-delta
forecast_horizon
# Since gamma_mse is applied to xt, we have to supply information about the link between xt and epsilont in terms of xi
xi<-xi
# Don't forget xi in the function call: otherwise SSA assumes xt=epsilont white noise, by default
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi)

ssa_x=SSA_obj$ssa_x
mse_x<-SSA_obj$mse_x
mplot<-cbind(ssa_x,gamma_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to xt: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

mplot<-cbind(ssa_eps,gamma_conv_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to epsilont: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

# B. Second case: target is gamma_conv_mse which is applied to epsilont
gammak_generic<-gamma_conv_mse
# There is a subtle change in the function-call since we now omit xi
# By default (omission of xi), SSA assumes the data to be white noise
# This is correct since gamma_conv_mse is the convolved target and is applied to epsilont 
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1)

ssa_x=SSA_obj$ssa_x
ssa_eps=SSA_obj$ssa_eps
mplot<-cbind(ssa_eps,gamma_conv_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to xt: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

# Since we assume xt=epsilont (white noise) ssa_x and ssa_eps are identical in this case
mplot<-cbind(ssa_x,gamma_conv_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to epsilont: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

# We could obtain the same result as above by setting
xi_null<-NULL
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi_null)

ssa_x=SSA_obj$ssa_x
ssa_eps=SSA_obj$ssa_eps
mplot<-cbind(ssa_eps,gamma_conv_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to xt: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

# Or we could obtain the same result by specifying the identity
xi_id<-1
SSA_obj<-SSA_func(L,forecast_horizon,gammak_generic,rho1,xi_id)

ssa_x=SSA_obj$ssa_x
ssa_eps=SSA_obj$ssa_eps
# We do not have to re-scale the new filter
mplot<-cbind(ssa_eps,gamma_conv_mse)
plot(mplot[,1],main="Optimally scaled SSA and original MSE as applied to xt: SSA replicates MSE",axes=F,type="l",xlab="Lag-structure",ylab="filter-weights")
lines(mplot[,2])
axis(1,at=1:L,labels=-1+1:L)
axis(2)
box()

#----------------------------------------------------------------
# Summary
# -Splitting the estimation problem into Gamma (target filter applied to xt) and Xi (Wold decomposition of xt) is a convenience
# -By all means, our main filter of interest, from a theoretical point view, is always the convolved design, as applied to epsilont
#   -the convolved design is necessary for deriving optimal filters and performance numbers
# -Once computed, the optimal xt-filter can be obtained from simple deconvolution 
#   The xt-filter is convenient in the sense that the analyst can apply the filter to the original data xt (instead of epsilont)
# -SSA can replicate MSE
#   -The MSE solution can be substituted for the effective target without affecting the SSA-solution 
#   -One can provide either the MSE as applied to xt or the convolved MSE, as applied to epsilont, by accommodating 
#     the function call accordingly  
# -Theoretical criteria and ht match empirical estimates if the model (xi) is not misspecified

# Final note: one can modify rho1 in the above example: then SSA won't replicate MSE anymore
# -If rho1>rho(MSE) then SSA is smoother (typical application case)
# See the following tutorials for (more meaningful/interesting) applications
