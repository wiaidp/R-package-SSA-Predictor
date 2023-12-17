# In this tutorial we consider an application of SSA to the Hodrick-Prescott (HP) filter 

# We consider the HP-trend or lowpass filter, applied to stationary data
#   -This design is used in JBCY paper: HP-trend applied to log-returns of US INDPRO (does not generate spurious cycle)
#     -In contrast, the classic HP-gap applied to INDPRO (data in levels) tends to generate spurious cycles, see tutorial 5
#   -All examples in this tutorial rely on artificial (simulated) stationary series: knowing the true model allows for verification of theoretical results 
#     -Tutorial 5 applies SSA to US-INDPRO

# We apply SSA to different targets 
#   a.The one-sided MSE HP, assuming the data to be white noise, see example 1
#   b.The classic one-sided HP, assuming the data to be white noise, see example 2
#     (the classic HP-concurrent has some interesting properties and therefore it makes sense to consider the filter as applied to noise)
#   c.The classic one-sided HP assuming the true or estimated (stationary ARMA) model, see example 5
#   d.The one-sided MSE HP assuming the true or estimated (stationary ARMA) model, see example 5 (just change the target specification) 
#   e.The symmetric HP assuming the true or estimated (stationary ARMA) model, see example 6


# Main outcomes: 
#   1.The classic HP-gap (as applied to non-stationary data in levels) is not suited for BCA, see example 7 below: "never use HP-gap". 
#       -Therefore, we here emphasize the trend filter(s) only: as applied to stationary data (first differences of economic time series) 
#   2.The classic concurrent (one-sided) HP-trend can be derived from a particular (implicit) ARIMA(0,2,2) model for the data. 
#       -The implicit model assumes the data to be excessively smooth; economic time series are typically noisier than that (in levels and a fortiori in first differences).
#     Consequences: 
#       -In typical applications, HP-concurrent is not an optimal (MSE) nowcast of the symmetric two-sided HP, see example 6
#       -The holding-time of HP-concurrent is rather small i.e. the filter is subject to noise-leakage 
#         (noisy zero crossings).
#   3.In this scenario, SSA can be applied to control and to tame the number of noisy crossings of HP
#       -We typically impose 50% larger ht or, equivalently, 33% less (noisy) crossings
#   4.Besides nowcasts (delta=0) we also consider 12-steps ahead forecasts (one year for monthly data)
#       -SSA-forecasts adopt the same stringent holding-time constraint: 33% less noisy crossings than (one-sided) HP targets (in the long run)
#       -SSA-forecasts are left-shifted (relative advancement): they generally have a lead when referenced against the concurrent benchmarks
#       -SSA real-time (concurrent) designs can be smoother as well as leading, when compared to the concurrent benchmarks 
#   5. The forecast trilemma is visualized in example 8 for a SSA-design targeting HP-MSE

# Note: our intention is not to push a particular BCA-tool. Rather, we strive at illustrating that a particular 
#   predictor or BCA-filter (any one as long as it's linear in the data) can be replicated and modified by SSA 
#   in view of addressing 
# 1. smoothness (noise suppression) and 
# 2. timeliness (advancement)
# In this perspective, HP is considered as a basic platform and a vitrine for showcasing SSA
#   -We offer a number of compelling performance measures, confirming pertinence of a simple novel optimization principle  


#-----------------------------------------------------------------------
# Make a clean-sheet, load packages and functions
rm(list=ls())

library(xts)
# Load the library mFilter
# HP and BK filters
library(mFilter)
# Plot for heat map of Trilemma
library(ggplot2)
library("gplots")

# Load all relevant SSA-functions
source(paste(getwd(),"/R/simple_sign_accuracy.r",sep=""))
# Load tau-statistic: quantifies time-shift performances (lead/lag)
source(paste(getwd(),"/R/Tau_statistic.r",sep=""))

# Load signal extraction functions used for JBCY paper (relies on mFilter)
source(paste(getwd(),"/R/HP_JBCY_functions.r",sep=""))


# 1. Whittaker henderson
# ARIMA-model: Tucker's package for MA
# Simulations of ARIMA with MA corresponding to lambda
# HP-gap vs. HP-trend

# 2. Are economic series compatible with I(2)?
# Links in JBCY paper
# Simulations
# Which BCA-tool is compatible with I(1): HP-trend applied to returns/diffs
#   -explain why

# 3. Derivation of HP
# Amplitude and shift of: trend one and two-sided, gap one and two-sided
# Transform HP-trend in diffs to new gap in levels: compare with original gap


# 4. Conclusions: HP-trend applied to returns
#   -Compatible with I(1)
#   -No spurious cycle
#   -Peak amplitude OK with business-cycle spec.
#   -Shift vanishing

##########################################################################################################
##########################################################################################################
# Introduction
# a.Derivation of HP 
# b.Brief analysis of the classic one-sided HP concurrent trend filter
# c. Summary
#--------------------------
# a. Derivation
# We use the R-package mFilter for computing HP 
# Specify filter length: should be an odd number since otherwise the two-sided HP filter could not be adequately centered 
L<-201
# Should be an odd number
if (L/2==as.integer(L/2))
{
  print("Filter length should be an odd number")
  print("If L is even then HP cannot be adequately centered")
  L<-L+1
}  
# Specify lambda: monthly design
lambda_monthly<-14400
par(mfrow=c(1,1))
HP_obj<-HP_target_mse_modified_gap(L,lambda_monthly)
# Bi-infinite two-sided (symmetric) HP
hp_target<-HP_obj$target
ts.plot(hp_target)
# Concurrent gap: as applied to series in levels: this is a high pass filter
hp_gap=HP_obj$hp_gap
ts.plot(hp_gap)
# Concurrent HP assuming I(2)-process 
# This is the Classic concurrent or one-sided low pass HP, see e.g. McElroy (2006)
hp_trend=HP_obj$hp_trend
ts.plot(hp_trend)

# Compute lag one acf and holding time of HP concurrent
htrho_obj<-compute_holding_time_func(hp_trend)
rho_hp<-htrho_obj$rho_ff1
ht_hp<-htrho_obj$ht
ht_hp

# Compare hts of one- and two-sided filters
compute_holding_time_func(hp_target)$ht
ht_hp
# The large (atypical) discrepancy between holding-times of two- and one-sided filters is discussed in the JBCY paper

#------------------------------
# b. Classic concurrent HP trend
# Let us briefly have a look at the above discrepancy and some of its effects: for this purpose we compute and compare filter outputs
len<-L+1000
set.seed(67)
# Data: white noise (in the JBCY paper we apply the filter to log-returns of INDPRO which are close to noise)
a1<-0
x<-arima.sim(n = len, list(ar = a1))
# Compute filter output of SSA-HP filter
y_hp_concurrent<-filter(x,hp_trend,side=1)
y_hp_symmetric<-filter(x,hp_target,side=2)

ts.plot(y_hp_concurrent,main="HP: two-sided vs one-sided filter",col="red")
lines(y_hp_symmetric)
abline(h=0)
mtext("Two-sided HP",col="black",line=-1)
mtext("One-sided HP",col="red",line=-2)

# Let us compute empirical holding times of both filters:
compute_empirical_ht_func(y_hp_concurrent)
compute_empirical_ht_func(y_hp_symmetric)
# The difference of empirical hts is large, as expected (these numbers would converge to the above `true' hts for very long samples)


# -The plot of the time series suggests that the two-sided filter can stay away from the zero-line over long time episodes.
# -This characteristic of the two-sided HP can lead to over-smoothing, whereby critical recession dips are 
#   washed-out by the filter: the resulting overdamped cycle underestimates the felt impact of crises, see Phillips and Jin (2021) for background.
# -Interestingly, the one-sided HP can track recession dips better, due to its weaker noise-suppression (smaller ht) 
#   -We can gather more information by looking at frequency domain characteristics

# Let us compute and plot amplitude and time-shifts of the classic concurrent HP
K<-600
amp_obj<-amp_shift_func(K,as.vector(hp_trend),F)
par(mfrow=c(1,2))
# Amplitude
plot(amp_obj$amp,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Amplitude HP",sep=""))
mtext("Amplitude classic concurrent HP trend",line=-1)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()
# Shift
plot(amp_obj$shift,type="l",axes=F,xlab="Frequency",ylab="",main=paste("Time-shift HP",sep=""))
mtext("Shift  classic concurrent HP trend",line=-1)
axis(1,at=1+0:6*K/6,labels=expression(0, pi/6, 2*pi/6,3*pi/6,4*pi/6,5*pi/6,pi))
#axis(1,at=1+0:6*K/6,labels=(c("0","pi/6","2pi/6","3pi/6","4pi/6","5pi/6","pi")))
axis(2)
box()

# The peak of the amplitude function corresponds roughly to a periodicity of 7 years, which is in accordance with the concept of a `business-cycle'
# The time-shift is small, in particular towards frequency zero
#   -Recall that the filter output must track the level of an I(2) series: a non-vanishing shift would generate a non-stationary (integrated) error with respect to the two-sided target output

#-----------------------
# c. Summary
# -The two-sided HP is not necessarily a worthwhile target for BCA: it is possibly `too smooth' 
#   -The classic values of lambda, proposed in the literature, are eventually too large, see Phillips and Jin (2021) for background 
# -The classic one-sided HP, on the other hand, is less smooth: therefore it can better track short but severe recession dips
#   -The peak amplitude matches business-cycle frequencies
#   -The vanishing time-shift means that the filter is a tough benchmark
#     -The filter is typically faster than Hamilton's regression filter in real-time applications
# -We therefore propose to target two-sided  (examples 4 and 6)  as well as one-sided designs by SSA (examples 1,2,3,5 and 8) 
# -Example 1 addresses specifically the one-sided hp_mse, assuming the data to be white noise
#   -Example 4 illustrates that this particular target is equivalent to the two-sided HP when the data is white noise, thus confirming proposition 5 in the JBCY paper
