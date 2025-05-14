
# The structure of a prediction problem  can put forward different aspects of a predictor
#   -One-step ahead forecasting: short-term performances, track higher frequency components
#   -Multi-step ahead forecasting: fit and extrapolate short- and mid-term components
#   -Cycle or trend extraction: emphasize mid and/or long-term components, suppress short-term `noise' 
#   -Level performances vs. sign-changes vs. lead/lag

# Subjective preferences, priorities, risk aversion of the analyst can also put forward different aspects of a predictor

# Accordingly, forecast performances can address multiple characteristics of a predictor such as, e.g.:
#   -MSE or closeness to target: 
#     -this measure typically emphasizes performances further away from the zero-line (or center-line)
#     -The squaring magnifies peaks and troughs more than zero-crossing phases
#   -Smoothness or noise suppression: controlling the rate of false noisy alarms. 
#     -This measure emphasizes zero-crossings or sign-changes exclusively
#   -Timeliness or lead/lag: controlling the systematic waiting-time until an alarm is set
#     -In principle, this measure addresses zero-crossings too. But a lead at zero-crossings generally extends uniformly, at all levels


# The classic MSE paradigm does not account explicitly for smoothness and/or timeliness
# SSA extends the classic MSE approach
#   -It optimizes MSE performances subject to smoothness and timeliness requirements: trilemma
#   -MSE addresses only a fixed/single point of the trilemma
# SSA 
#   -can replicate that particular fixed/single point; 
#   -the additional hyperparameters (delta,ht) allow for operationalization of alternative priorities (than MSE only): 
#     -Positioning within the confines of the trilemma

