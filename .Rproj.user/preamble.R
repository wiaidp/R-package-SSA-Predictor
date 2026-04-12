# ══════════════════════════════════════════════════════════════════════════════
# PREAMBLE TO TUTORIAL 8
# ══════════════════════════════════════════════════════════════════════════════
# A. Contextual Mapping to Macroeconomic and Financial Data
# ──────────────────────────────────────────────────────────────────────────────
# Many economically relevant time series are non-stationary, exhibiting a
# slowly evolving level (trend). First-differencing such series typically yields
# an approximately stationary, near-white-noise process — the so-called
# "typical spectral shape" of economic time series.
#
# First differences emphasise high-frequency variation, making smoothing
# (i.e., noise removal or attenuation) particularly valuable for differenced
# data. This is especially true because first differences represent growth
# rates — a quantity of primary interest to practitioners, policymakers, and
# statistical agencies. Noise in these series obscures the underlying growth
# dynamics that are central to decision-making by economic actors, investors,
# and institutions alike.
#
# Tutorial 6, which introduced I-SSA, addressed smoothness in first differences
# while simultaneously tracking the non-stationary HP-trend optimally via
# I-SSA ("double-stroke"). The present Tutorial 8 also emphasises smoothness
# in stationary (differenced) data, but shifts focus to tracking growth
# dynamics in the (stationary) differenced series directly, rather than 
# recovering the non-stationary HP-trend.
#
# Accordingly, the exercises below use simulated white noise as the input
# series — a reasonable approximation to the first differences of many
# key economic indicators. All conclusions extend straightforwardly to
# arbitrary stationary processes (of differenced data), since M-SSA retains
# its optimality properties when the Wold decomposition of the differenced
# series is supplied to the design via ξ (the variable "xi" is an argument of 
# the SSA function call).
#
# ──────────────────────────────────────────────────────────────────────────────
# B. Turning Points, Inflection Points
# ──────────────────────────────────────────────────────────────────────────────
#
# Turning point:
# ──────────────
#   A point at which a series changes direction — from increasing to decreasing,
#   or vice versa — corresponding to a local maximum or minimum. A series
#   evolves monotonically between any two consecutive turning points.
#
# Inflection point:
# ─────────────────
#   A point at which the curvature (concavity) of a series changes sign —
#   from bending downward to bending upward, or vice versa. An inflection point
#   corresponds to a local maximum or minimum of the growth (slope), but
#   does not necessarily involve a change in direction of the series itself
#   (though it may anticipate one).
#
# Differenced Series x_t vs. Integrated Level I_t
# ───────────────────────────────────────────────
# Let I_t be a non-stationary series and x_t := I_t − I_{t−1} its stationary
# first differences. The following correspondence holds:
#
#   • Zero-crossings of x_t  ↔  Turning points of I_t
#   • Turning points of x_t  ↔  Inflection points of I_t
#
# M-SSA Applied to Noisy Data: Latent Trend Specification
# ───────────────────────────────────────────────────────
# When x_t, and hence I_t, are noisy, M-SSA specifies the growth of a latent
# trend T_t of I_t, characterised by two properties:
#
#   • Smoothness  : the mean duration between consecutive turning points of T_t
#                   equals the user-specified HT.
#   • Fidelity    : the drift of T_t tracks x_t — the noisy growth of I_t — as 
#                   closely as possible.
#
# M-SSA Smoothing Trade-Off: 
# ────────────────────────
# The M-SSA criterion jointly optimises two competing objectives,
# trading off smoothness against fidelity to the underlying growth dynamics:
#
#   • Smoothness : controlling the HT in first differences x_t governs the
#                  rate of turning points in the non-stationary level I_t.
#
#   • Fidelity   : maximising target correlation (or sign accuracy) of x_{t+δ}
#                  ensures optimal estimation of the growth rate at lag δ.
#
# The HT parameter provides direct, interpretable control over this trade-off:
# a larger HT enforces greater smoothness (fewer turning points) at the cost
# of reduced fidelity to the raw growth series, and vice versa.
#
# ══════════════════════════════════════════════════════════════════════════════
# SMOOTHING IN TUTORIAL 8
# ══════════════════════════════════════════════════════════════════════════════





















# ──────────────────────────────────────────────────────────────────────────────
# Smoothing vs. Smoother: Curvature vs. Backcasting
# ──────────────────────────────────────────────────────────────────────────────
#   • "Smoothing" (classical, WH/HP): tracks a target series subject to a
#     regularisation term that penalises roughness. Smoothness is defined
#     implicitly through the choice of penalty (e.g., curvature in HP).
#
#       → Exercise 1 compares SSA's HT-based smoothness criterion against
#         WH/HP's curvature-based criterion in the context of two-sided
#         (symmetric) filters.
#
#   • "Smoother" (backcasting sense): refers to the retrospective refinement of
#     a historical estimate at time T + δ (δ < 0) for a smooth latent component
#     (e.g., trend or cycle), by incorporating all observations up to the current
#     time T. The additional observations at T+δ+1, …, T — unavailable in
#     real time at T+δ but accessible at the sample end T — allow the smoother
#     to suppress spurious noise more effectively. As a result, historical
#     estimates at T+δ are generally smoother than their real-time counterparts 
#     (δ = 0, nowcast).
#
#     The maximally smooth estimate is typically obtained at the middle of the 
#     sample, δ = −(T−1)/2, at which point the smoother is fully
#     symmetric (two-sided), exploiting an equal number of leads and lags.
#     This symmetric design is the focus of Exercise 1.
#
#       → Exercise 2 explores SSA within this backcasting framework, examining
#         how the filter evolves as the lag δ on x_{t+δ} increases from
#         δ = −(L−1)/2 (fully symmetric, two-sided smoother) to δ = 0
#         (nowcast, fully one-sided filter). Along this continuum:
#           – Fewer future observations are available to the filter.
#           – The filter transitions progressively from two-sided to one-sided.
#           – The classical HP filter loses smoothness as δ → 0, because the
#             acausal (forward-looking) component that suppresses noise is
#             gradually removed.
#
#       → In contrast to classical smoothing, SSA in exercise 2 is asked to 
#         maintain a fixed HT across all lags δ — anchored to the HT of the 
#         acausal two-sided HP filter — thereby guaranteeing constant smoothness 
#         (in terms of mean-crossings) regardless of δ. This is a demanding 
#         constraint, since the fully symmetric HP filter is very smooth (i.e., 
#         it has a very large HT).
#
# ──────────────────────────────────────────────────────────────────────────────
# (M-)SSA vs. WH/HP 
# ──────────────────────────────────────────────────────────────────────────────
#   • Core contrast: (M-)SSA penalises turning points in I_t (via HT);
#     WH/HP penalises inflection points in I_t (via curvature).
#   • Classical smoothing (WH/HP) does not appeal to an explicit statistical
#     model of the data, although optimality can often be derived under implicit
#     model assumptions.
#       → The HP filter is the optimal trend estimate under the smooth trend-plus-
#         noise model (ARIMA(0,2,2)); see Tutorial 2.0.
#
#   • M-SSA, by contrast, incorporates an explicit parametric model for the data,
#     formulated through ξ (via the Wold decomposition or finite MA inversion).
#       → By default (i.e., when xi is ignored, or set to xi = NULL or xi = 1),
#         M-SSA assumes white noise as the input process.
#
#   • M-SSA can therefore be interpreted in two ways:
#       (i)  As a model-free smoother (analogous to WH/HP), by ignoring ξ; or
#       (ii) As a model-based smoother (analogous to the Kalman smoother),
#            by explicitly specifying ξ to reflect the assumed data-generating
#            process.



















# ─────────────────────────────────────────────────────────────────────────────
# NOTE ON SMOOTHNESS CRITERIA
# ─────────────────────────────────────────────────────────────────────────────
#
# In the WH framework, smoothness is enforced by penalising "unsmooth"
# behaviour through a regularisation term — typically squared differences
# of order d. HP is a special case of WH with d = 2, penalising curvature
# (squared second-order differences).
#
# The key question is therefore which notion of smoothness is most
# appropriate for a given application — curvature-based (WH/HP) or
# mean-crossing-based ((M-)SSA). 
#
# In particular, the discussion is not about the pertinence of applying HP to white noise
#
#
# ─────────────────────────────────────────────────────────────────────────────
# THE PRIMAL PERSPECTIVE: FIX HT, MAXIMISE GROWTH TRACKING
# ─────────────────────────────────────────────────────────────────────────────
# The series x_t (first differences) is an natural unbiased but noisy estimate of
# growth in the underlying levels series. For a prescribed mean distance
# between consecutive turning points in levels (i.e., a fixed HT in differences), SSA
# maximises tracking of x_t — the natural, unbiased growth signal.
#
# This constitutes a compelling and operationally meaningful criterion:
# it simultaneously controls the frequency of sign changes (via HT) and
# minimises noise in the growth estimate (via target correlation / MSE).
#
# Example — Business-cycle analysis:
#   Business cycles are conventionally defined over durations of 2–8 years,
#   with a typical mean cycle length of approximately 5 years. Imposing an
#   HT of 5 years in the SSA design would yield the closest possible
#   tracking of the unbiased (but noisy) growth estimate x_t, while
#   ensuring that the smoothed output generates turning-point signals at
#   the prescribed frequency.
# ─────────────────────────────────────────────────────────────────────────────

#
#
# The dual perspective: fixing MSE, maximising HT
# ─────────────────────────────────────────────────
# The argument can be reversed: suppose the tracking ability (MSE, target
# correlation, or sign accuracy) of x_t is fixed a priori. A natural
# complementary smoothness objective is then to maximise the HT subject to
# this tracking constraint — directly controlling the distance between
# consecutive turning points in levels. This combines:
#
#   • An MSE criterion on growth (first differences), and
#   • An explicit turning-point control on levels.
#
# Exercise 1.2 below explores exactly this dual formulation.
#
# Why M-SSA smoothness differs from HP smoothness
# ─────────────────────────────────────────────────
# Once the filtered series is clearly away from zero, noisy ripples
# (turning points in first differences, i.e., inflection points in levels)
# are operationally harmless — they do not generate false turning-point
# signals. The visual roughness of M-SSA at non-zero levels is therefore
# inconsequential in sign-based (turning points in levels) applications.
#
# What matters is the behaviour near zero: spurious zero-crossings at this
# boundary generate noisy turning-point signals. M-SSA controls precisely
# this rate — the frequency of zero-crossings — via the HT constraint.
#
# HP, by contrast, controls curvature (turning-point rate) uniformly across
# all levels, including regions far from zero where such control is
# operationally unnecessary. This makes HP's smoothness criterion less
# targeted in sign-based (turning points in levels) decision-making applications.
# ─────────────────────────────────────────────────────────────────────────────
















# The smaller curvature of HP (violet) suggests a visually smoother output
# than M-SSA. However, both smoothers exhibit the same rate of mean-crossings
# (identical empirical HT), so they are equally smooth in the sense that
# matters most for sign-based decision-making: the mean duration between
# consecutive turning points in the level series I_t (here, a random walk).
#
# The apparent visual roughness of the M-SSA output — reflected in more
# frequent sign changes of the slope (shorter monotonicity intervals in the
# smoothed series) — is not a deficiency but a necessary consequence of the
# HT constraint. It is precisely this additional variation that allows M-SSA
# to track the target x_{t+delta) more closely than HP,
# as measured by target correlation, sign accuracy, and MSE.
#
# In summary: for a given HT constraint — equivalently, a fixed rate of
# zero-crossings in x_t or of turning points in I_t — M-SSA achieves
# superior tracking of x_{t+delta} (growth) relative to HP. The trade-off is
# greater curvature (more inflection points in I_t), but inflection points
# are operationally less relevant than turning points, whose rate is held
# fixed by construction.
#
# Stated differently: the curvature-based smoothness criterion enforced by
# WH/HP does not target the rate of turning points in I_t — the quantity
# of primary interest in sign-based applications. Consequently, for equal
# HT, M-SSA outperforms HP in tracking accuracy. The cost of unconstrained
# curvature in M-SSA (i.e., increased inflection points) is a minor concern
# in settings where turning-point control is the primary objective.
#
# From a visual standpoint, HP appears smoother — an impression driven by
# its lower curvature and more regular (oscillatory) ACF structure (see below).
# This visual impression should not, however, distract from the operational
# objective: when the rate of turning points is the primary control criterion,
# the HT constraint is the more relevant measure of smoothness, and M-SSA is
# the more efficient smoother for tracking x_{t+delta}, i.e., growth (I_{t+delta}-I_{t+delta-1}).


























# HP has less turning points (local maxima and minima)



# Suppose we interpret zero-crossings of the smoother of x_t as turning points on levels I_t, where it is understood that 
# x_t=I_t-I_{t-1}.
# For the same frequency of TPs, SSA and HP in this exercise date TPs differently.
# Question: given this difference, which TP dating in I_t (zero-crossings in differences) is 
# potentially more informative/interesting?
# -HP exhibits smaller curvature than SSA: is smaller curvature in differences (i.e. third differences of I_t) relevant?
# -SSA exhibits optimal tracking of x_{t+delta}=I_{t+delta}-I_{t+delta-1}, i.e., growth.

# We contend that defining TP's on levels based on optimal growth-tracking is an intuitively appealing 
# approach by linking TP's (i.e. zero crossings in diffs) to optimal estimation in diffs.

# Defining TP's in levels based on minimal curvature (in differences), on the other hand, links TPs to inflection points.
# It is not clear why having fewest inflection points should determine implicitly TPs: what 
# is the rationale?



