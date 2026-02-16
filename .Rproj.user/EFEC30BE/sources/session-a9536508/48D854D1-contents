# This series of tutorials is about optimal prediction.

# For background: see Wildi, M. (2024) Business Cycle Analysis and Zero-Crossings of Time Series: a Generalized Forecast Approach: https://doi.org/10.1007/s41549-024-00097-5.

# The structure of a prediction problem  can put forward different aspects of a predictor
#   -One-step ahead forecasting: short-term performances, track higher frequency components
#   -Multi-step ahead forecasting: fit and extrapolate short- and mid-term components
#   -Cycle or trend extraction: emphasize mid and/or long-term components, suppress short-term `noise' 
#   -Level performances vs. sign-changes vs. lead/lag (advancement, retardation)

# Subjective preferences, priorities, risk aversion of the analyst can also put forward different aspects of a predictor

# Accordingly, forecast performances can address multiple characteristics of a predictor such as:
#   -Mean-square error performances (MSE) or `closeness to target': 
#   -Smoothness or noise suppression: controlling the `wiggleness' of a predictor or the rate of false noisy alarms. 
#   -Timeliness or lead/lag: controlling the systematic advancement or retardation (until an alarm is set)


# SSA is an acronym for Simple Sign Accuracy or Smooth Sign Accuracy. 
# SSA: addresses prediction 
#   -one step ahead, multi-step ahead, backcasting, nowcasting or forecasting 
# SSA emphasizes characteristics of a predictor which are related to 
#   -MSE, 
#   -noise suppression (wiggleness, smoothness, rate of zero-crossings) 
#   -timeliness (lead, left-shift, reduced phase-lag)
# SSA can be configurated such that it replicates the classic MSE predictor, see tutorial 0.3

# Forecast trilemma, see tutorial 0.1
#   -MSE performances, smoothness and timeliness constitute a forecast trilemma
#   -the MSE predictor does not directly relate to smoothness or timeliness
# Therefore  
#   -the predictor is often subject to noise-leakage (unsystematic random dynamics, wiggleness), see tutorial 0.3
#   -the predictor is generally lagging behind the target (right-shift or lag), see tutorial 0.3
 
    
# SSA optimization principle, see tutorial 0.2
#   Maximize sign-accuracy under a holding-time constraint
#     -Sign-accuracy: probability of matching the correct sign of the target by the predictor
#     -Holding-time: expected duration between consecutive zero-crossings of the predictor
# Criterion as implemented in R-code:
#   -Maximize correlation of predictor and target (the same as sign-accuracy)
#   -subject to a constraint of the lag-one ACF of the predictor (the same as expected duration between crossings)
# See section 2 in JBCY paper for background

# Motivation
# -In applications, zero-crossings (sign changes) can be markers of important or relevant events 
#     asking for possible interventions of decision-makers or market players:
# -Examples:
#   -a. Automatic trading algorithms often rely on zero-crossings of suitably filtered (financial) time series (e.g. MA-cross filters)
#     to identify investment opportunities into bullish or de-investment from bearish markets 
#   -b. Recession indicators often rely on zero-crossings of a suitably filtered macro-series (or aggregate of series)
#   -c. Business-cycle analysis (BCA) assumes the existence of a more less regular and recurrent pattern (cycle) of the 
#     economy, which can be divided into expansion (growth) and contraction (recession) phases.
#       -`Anti-cyclical' policies are derived from the state of the cycle
#       -the state of the cycle changes at its zero-crossings
#   -d. Control problems (for example monitoring of industrial processes) can often be formulated in terms of filters exceeding 
#     some thresholds (which could be transformed into zero-crossings straightforwardly)
# -The classic MSE predictor is often noisy and lagging, see tutorial 0.3
# -SSA can improve upon a benchmark in terms of 
#   -noise suppression (less noisy alarms)  
#   -timeliness (lead or left-shift). 
#   -SSA can be both `smoother' and `faster' than the benchmark, see tutorials 2-5

# Background
#   -SSA can set different priorities within the limits spanned by the cited forecast trilemma
#   -Control (navigation) is provided by a set of two hyperparameters: the holding-time (or lag-one ACF) and the forecast horizon
#   -It is possible to assign full-weight to MSE and to replicate the classic predictor, see tutorial 0.3
#     -In this sense SSA generalizes MSE (see the remark after theorem 1 in the JBCY paper for background)
#   -In principle, any linear forecast rule can be replicated by SSA 
#     -In this sense, SSA generalizes the class of linear predictors
#   -All our examples emphasize univariate linear predictors: a multivariate SSA extension is on the way

# Plug-on
# -SSA can be considered as a self-contained and original forecast algorithm  
# -But SSA can also be engrafted onto an existing benchmark in view of controlling its smoothness and/or timeliness characteristics 
#   in a foreseeable way. 
# -In this series of tutorials we propose `plug-on' applications to
#   -MSE: see tutorials 0-5
#   -Hodrick-Prescott (HP) filter: see tutorials 2 and 5
#   -Hamilton filter (HF): see tutorial 3
#   -Baxter and King (BK) filter: see tutorial 4
# -Since SSA tracks the benchmark optimally, we argue that SSA retains interpretability (the original meaning or economic content) 
#     of the latter 
# -In our tutorials we typically ask SSA 
#   -to increase ht by up to 50% (reduction by up to 33% of unwanted `noisy' alarms)
#   -to lead the benchmark, typically between 1 and up to 6 time units
#   -Asking for more or less is possible; but always within the confines of the trilemma, i.e. at costs of MSE
# -Our R-package allows to evaluate the trilemma tradeoff by computing explicitly smoothness, MSE or timeliness contributions


# Assumptions
# -We assume a stationary zero-mean target or predictor
#   -Otherwise zero-crossings would not be properly defined anymore
# -Formally, we assume Gaussianity
#   -In that case, sign accuracy and holding-time can be addressed by correlation and lag-one ACF exactly, see section 2 of JBCY paper
#   -SSA is fairly robust against departures of Gaussianity (vola-clustering, heavy-tails) due to a central limit theorem.
#     -This means that `typical' departures from Gaussianity, such as found in many economic series, do not affect performances unduly, in general. 
# Non-Zero crossings 
#   -if crossings are measured and counted on a non-zero level, then ht is biased in absolute terms but relative 
#     performances, against a benchmark, are unaffected, see tutorials 1-5 



