# ======================================================================================
# Forecast Problem Structure:
# ======================================================================================
# Different prediction problems emphasize different aspects of forecast performance:
# - One-step-ahead forecasting:   captures short-term dynamics and high-frequency components
# - Multi-step-ahead forecasting: fits and extrapolates short- to medium-term components
# - Cycle or trend extraction:    emphasizes medium- to long-term components while
#                                 suppressing short-term noise
# - Additional dimensions:        level accuracy vs. sign-change detection vs. lead/lag behavior

# ======================================================================================
# User Priorities:
# ======================================================================================
# Subjective priorities, risk preferences, and analytical objectives shape
# which aspects of a predictor are considered most relevant

# ======================================================================================
# Matching Optimization Principle, Problem Structure, and User Priorities:
# ======================================================================================
# Forecast performance can be assessed along multiple complementary dimensions:
# - MSE / closeness to target:
#   * Emphasizes deviations far from the mean (peaks and troughs)
#   * Squaring amplifies large deviations more than near-mean phases
# - Smoothness / noise suppression:
#   * Controls the rate of false (noise-driven) alarms
#   * Focuses on mean-crossings and sign-changes (above/below the mean)
# - Timeliness / lead-lag:
#   * Controls the systematic delay before an alarm is triggered
#   * Also linked to mean-crossings; a lead at the crossing typically
#     extends uniformly across all amplitude levels
# - Risk aversion: find the optimal balance in the Timeliness-Smoothness tradeoff

# ======================================================================================
# Classic MSE vs. (M-)SSA:
# ======================================================================================
# The classical MSE paradigm does not explicitly account for smoothness or timeliness:
# - It corresponds to a single fixed point on the efficient Accuracy-Smoothness frontier 
#   traced-out by M-SSA
# (M-)SSA extends the classical MSE framework:
# - Optimizes MSE performance subject to an explicit smoothness constraint
#   (controlling the frequency of sign-changes / mean-crossings)
# - Nests the classical MSE solution as a special case
# - Additional hyperparameters (delta, ht) allow the analyst to navigate the trilemma
#   and operationalize priorities beyond pure MSE minimization

# ======================================================================================
# Forecasting Noisy Data:
# ======================================================================================
# Attenuating noise can improve multi-step-ahead forecasting of economic time series:
# - The additional smoothness enforced by M-SSA reduces spurious (noise-driven) sign-changes,
#   which can enhance discriminatory power (AUC) in a ROC analysis
















# Forecast Problem Strucure:
# Different prediction problems emphasize different aspects of forecast performance:
# - One-step-ahead forecasting:   captures short-term dynamics and high-frequency components
# - Multi-step-ahead forecasting: fits and extrapolates short- to medium-term components
# - Cycle or trend extraction:    emphasizes medium- to long-term components while
#                                 suppressing short-term noise
# - Additional dimensions:        level accuracy vs. sign-change detection vs. lead/lag behavior

# User Priorities
# Subjective priorities, risk preferences, and analytical objectives further shape
# which aspects of a predictor are considered most important

# Matching Forecast Optimization Principle, Poblem structure and User Priorities: 
# Accordingly, forecast performance can be assessed along multiple dimensions:
# - MSE / closeness to target:
#   * Emphasizes deviations far from the mean-line (peaks and troughs)
#   * Squaring amplifies large deviations more than near-zero phases
# - Smoothness / noise suppression:
#   * Controls the rate of false (noisy) alarms
#   * Focuses exclusively on mean-crossings and sign-changes (above/below the mean)
# - Timeliness / lead-lag:
#   * Controls the systematic delay before an alarm is triggered
#   * Also addresses zero-crossings; a lead at the zero-crossing
#     generally extends across all amplitude levels

# Classic MSE vs. M-SSA
# The classical MSE paradigm does not explicitly account for smoothness or timeliness:
# - It corresponds to a single fixed point in the trilemma (MSE, smoothness, timeliness)
# (M-)SSA extends the classical MSE framework:
# - Optimizes MSE performance subject to a novel smoothness constraints (frequency of sign changes, mean-crossings)
# - Can replicate the classical MSE solution as a special case
# - Additional hyperparameter (h,ht) allow the analyst to navigate the trilemma
#   and operationalize alternative priorities beyond pure MSE minimization
#
# Forecasting Noisy Data
# Attenuating noise can improve multi-step-ahead prediction of (noisy) economic time series:
# - The additional smoothness of M-SSA reduces spurious (noise-driven) zero-crossings,
#   which can enhance the AUC in a ROC analysis
