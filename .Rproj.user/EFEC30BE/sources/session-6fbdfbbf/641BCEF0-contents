# ─────────────────────────────────────────────────────────────────
# M-SSA PREDICTOR: A TUTORIAL SERIES
# ─────────────────────────────────────────────────────────────────

# The M-SSA provides a unified framework for solving general prediction problems while simultaneously 
# accommodating specific research priorities and objectives.

# This series of tutorials introduces M-SSA,
# with a focus on controlling key characteristics of the predictor.

# ── BACKGROUND REFERENCES ────────────────────────────────────────
# The following papers provide the theoretical foundations:
#
#   Wildi, M. (2024) Business Cycle Analysis and Zero-Crossings of
#     Time Series: a Generalized Forecast Approach. Published in Journal of Business-Cycle Research
#     https://doi.org/10.1007/s41549-024-00097-5
#
#   Wildi, M. (2026a) Sign Accuracy, Mean-Squared Error and the Rate
#     of Zero Crossings: a Generalized Forecast Approach.
#     https://doi.org/10.48550/arXiv.2601.06547 (published on arXiv)
#
#   Wildi, M. (2026b). The Accuracy-Smoothness Dilemma in Prediction:
#   A Novel Multivariate M-SSA Forecast Approach.
#   Journal of Time Series Analysis, http://doi.org/10.1111/jtsa.70058 
#   arXiv: https://doi.org/10.48550/arXiv.2602.13722
#
#   Heinisch, K., Van Norden, S., and Wildi, M. (2026).
#   Smooth and Persistent Forecasts of German GDP:
#   Balancing Accuracy and Stability.
#   IWH Discussion Papers, 1/2026.
#   Halle Institute for Economic Research.
#   https://doi.org/10.18717/dp99kr-7336
#
# ============================================================

#
# Note: Working paper versions are available in the 'Papers' folder
# of this GitHub repository. Working papers contain full proofs and
# detailed technical results, whereas published versions are more
# streamlined, occasionally moving proofs to online appendices.

# ── PREDICTION OBJECTIVES ────────────────────────────────────────
# The structure of a prediction problem shapes which predictor
# properties matter most. Key distinctions include:
#
#   • One-step ahead forecasting
#       → Prioritizes short-term accuracy; captures rapidly evolving
#         (higher-frequency) dynamics
#
#   • Multi-step ahead forecasting
#       → Fits and extrapolates short- to medium-term components;
#         high-frequency variation is less critical
#
#   • Cycle or trend extraction
#       → Emphasizes medium- and long-term components;
#         short-term noise is suppressed
#
#   • Performance criteria
#       → Level accuracy vs. sign-change detection
#         vs. lead/lag behavior (advancement or retardation)
# ─────────────────────────────────────────────────────────────────

# ── ANALYST PREFERENCES AND PREDICTOR PRIORITIES ─────────────────
# Beyond problem structure, subjective preferences, priorities, and
# risk aversion of the analyst further shape which predictor
# properties are emphasized.
#
# Forecast performance can accordingly be assessed along multiple
# dimensions:
#
#   • Mean-Squared Error (MSE) / Closeness to target
#       → Measures how closely the predictor tracks the target signal
#
#   • Smoothness / Noise suppression
#       → Controls the "wiggliness" of the predictor and reduces
#         the rate of spurious, noise-driven alarms
#
#   • Timeliness / Lead-lag behavior
#       → Controls the systematic advancement or retardation of the
#         predictor relative to the target (i.e., how early or late
#         an alarm is triggered)

# ── SSA: SIMPLE (or SMOOTH) SIGN ACCURACY ─────────────────────────────
# SSA is the original and simpler univariate version of M-SSA. 
# It is a flexible prediction framework applicable to:
#   • One-step ahead, multi-step ahead, backcasting,
#     nowcasting, and forecasting settings
#
# SSA explicitly targets key predictor characteristics:
#   • MSE performance (Mean-Squared Error) and sign accuracy
#   • Noise suppression  (smoothness, wiggliness,
#                         rate of zero-crossings)
#   • It can also address Timeliness  `indirectly'   (lead, left-shift, reduced phase-lag)
#
# Note: SSA can be configured to replicate the classical MSE
# predictor as a special case — see Tutorial 0.3.

# ── M-SSA: MULTIVARIATE EXTENSION ────────────────────────────────
# M-SSA generalizes the SSA framework to a multivariate setting,
# allowing joint control of predictor characteristics across
# multiple time series.

# ── THE FORECAST TRILEMMA ─────────────────────────────────────────
# MSE performance, smoothness, and timeliness constitute a
# forecast trilemma — see Tutorial 0.1.
# These three objectives are inherently in tension:
#
#   • The classical MSE predictor does not directly optimize
#     for smoothness or timeliness.
#
# As a consequence, the MSE predictor typically exhibits:
#
#   • Noise leakage
#       → Unsystematic random dynamics ("wiggliness") contaminate
#         the predictor — see Tutorial 0.3
#
#   • Lag bias
#       → The predictor systematically trails behind the target
#         (right-shift or phase-lag) — see Tutorial 0.3
# ─────────────────────────────────────────────────────────────────
    
# ── SSA OPTIMIZATION PRINCIPLE ────────────────────────────────────
# See Tutorial 0.2 for implementation details.
#
# Core objective:
#   Maximize sign-accuracy subject to a holding-time constraint
#
#   • Sign-accuracy
#       → Probability that the (mean-zero) predictor correctly
#         matches the sign of the (mean-zero) target
#
#   • Holding-time
#       → Expected duration between consecutive zero-crossings
#         (sign changes) of the predictor
#
# As implemented in the R code, the criterion is reformulated as:
#
#   • Maximize the correlation between predictor and target
#       → Equivalent to maximizing sign-accuracy under Gaussianity;
#         near-equivalent for non-Gaussian processes in practice
#
#   • Subject to a constraint on the first-order ACF of the predictor
#       → Equivalent to controlling holding-time under Gaussianity;
#         near-equivalent for non-Gaussian processes in practice
#
# Theoretical background:
#   → Section 2 of the cited JBCY, SSA and M-SSA papers 

# ── MOTIVATION: WHY ZERO-CROSSINGS MATTER ─────────────────────────
# In many applications, zero-crossings (sign changes) serve as
# markers of significant events, triggering decisions or
# interventions by analysts, decision-makers, or market participants.
#
# Illustrative examples:
#
#   a. Algorithmic trading
#       → Automated strategies frequently rely on zero-crossings of
#         filtered financial time series (e.g., MA-crossover filters)
#         to signal entry into bullish markets or exit from bearish ones
#
#   b. Recession indicators
#       → Turning-point detection often hinges on zero-crossings of a
#         filtered macroeconomic series or composite aggregate. 
#
#   c. Business cycle analysis (BCA)
#       → BCA posits a broadly regular, recurrent economic cycle
#         alternating between expansion and contraction phases
#         → Anti-cyclical policy responses are derived from the
#           current phase of the cycle
#         → Phase transitions occur precisely at zero-crossings
#
#   d. Industrial process control
#       → Monitoring problems are often framed as a filter exceeding
#         a threshold — readily recast as a zero-crossing problem
#
# Limitations of the classical MSE predictor:
#   → Tends to be noisy (spurious alarms) and lagging
#     behind the target — see Tutorial 0.3
#
# SSA improves upon the MSE benchmark by offering:
#   • Noise suppression  → fewer spurious alarms
#   • Timeliness         → lead or left-shift relative to target
#
# Notably, SSA can be simultaneously smoother and faster than the
# benchmark — see Tutorials 2–5.

# ── SSA: BACKGROUND AND SCOPE ─────────────────────────────────────
# SSA navigates the forecast trilemma through two hyperparameters:
#   • Holding-time (or equivalently, first-order ACF)
#   • Forecast horizon
#
# These controls allow the analyst to set priorities freely within
# the space defined by the trilemma.
#
# Key properties:
#
#   • Generalization of MSE
#       → Assigning full weight to MSE replicates the classical
#         predictor exactly — see Tutorial 0.3 and the remark
#         after Theorem 1 in the JBCY paper
#
#   • Generalization of linear predictors
#       → In principle, any linear forecast rule can be replicated
#         within the SSA framework (see tutorials 1-5)
#
#   • Customization of benchmarks
#       → Once a classical linear forecast is replicated, SSA can
#         be used to make it faster, smoother, or both (see tutorials 1-5)
#
#   • Scope
#       → SSA addresses univariate linear predictors;
#         M-SSA extends the framework to multivariate designs

# ─────────────────────────────────────────────────────────────────
# ── SSA AS A PLUG-ON ──────────────────────────────────────────────
# SSA can be used in two distinct modes:
#
#   • Standalone
#       → SSA operates as a self-contained, original forecast
#         algorithm in its own right
#
#   • Plug-on
#       → SSA is grafted onto an existing benchmark predictor,
#         enhancing its smoothness and/or timeliness in a
#         controlled and predictable manner
#
# In this tutorial series, plug-on applications are demonstrated
# for the following benchmarks:
#
#   • MSE predictor              → Tutorials 0–5
#   • Hodrick-Prescott (HP) filter  → Tutorials 2 and 5
#   • Hamilton filter (HF)          → Tutorial 3
#   • Baxter-King (BK) filter        → Tutorial 4
#   • Beveridge-Nelson (refined)     → Tutorial 5


# ── TYPICAL PLUG-ON CONFIGURATIONS IN THIS TUTORIAL SERIES ──────────────
# In the tutorials, SSA plug-on applications are configured to:
#
#   • Increase holding-time (ht)
#       → Mean duration between consecutive zero-crossings
#         extended by up to 50%, reducing spurious noisy alarms
#         by up to 33%
#
#   • Advance the benchmark (lead / left-shift)
#       → Typically between 1 and 6 time units ahead of the target
#
# More aggressive settings (faster and/or smoother) are possible,
# but always within the constraints of the forecast trilemma —
# i.e., at the cost of increased MSE.

# ── TRILEMMA TRADEOFF EVALUATION ──────────────────────────────────
# The accompanying R package provides explicit decomposition of the
# trilemma tradeoff, computing separate contributions from:
#   • Smoothness
#   • MSE
#   • Timeliness

# ── INTERPRETABILITY ──────────────────────────────────────────────
# Since SSA tracks the benchmark optimally, it inherits and
# preserves the interpretability of the latter — including its
# original economic meaning and content.

# ─────────────────────────────────────────────────────────────────
# =============================================================================
# THEORETICAL FOUNDATION: THE SSA EFFICIENCY FRONTIER
# =============================================================================
# Wildi (2026a, 2026b) establish a fundamental DUALITY RESULT for SSA and
# its multivariate extension M-SSA:
#
# DEFINITION — Accuracy-Smoothness (AS) Frontier:
#   SSA and M-SSA trace out an EFFICIENT FRONTIER in the two-dimensional
#   space of filter performance:
#
#     ACCURACY  : correlation between filter output and the target signal
#                 (equivalently: sign accuracy of the predictor)
#     SMOOTHNESS: holding time of the filter output
#                 (mean duration between consecutive sign changes)
#
# DUALITY THEOREM:
#   The AS frontier is characterized by two equivalent optimization problems:
#
#   Primal problem:
#     Maximize ACCURACY (target correlation / sign accuracy)
#     subject to a SMOOTHNESS constraint (holding time >= h*)
#
#   Dual problem:
#     Maximize SMOOTHNESS (holding time)
#     subject to a ACCURACY constraint (target correlation >= rho*)
#
#   Both problems yield the SAME efficient frontier — SSA solves both
#   simultaneously by varying the constraint level (h* or rho*).
#
# EFFICIENCY PROPERTY:
#   No other linear predictor or filter can:
#     (i)  achieve higher accuracy     for a given level of smoothness, OR
#     (ii) achieve greater smoothness  for a given level of accuracy
#
#   => SSA / M-SSA are PARETO OPTIMAL in the accuracy-smoothness space.
#
# PRACTICAL IMPLICATION:
#   The constraint and objective are fully interchangeable:
#     - Fix smoothness (holding time h*)    => maximize accuracy
#     - Fix accuracy  (target correlation rho*) => maximize smoothness
#   The resulting filter is the same efficient design in both cases.
#   This flexibility allows practitioners to parameterize SSA according
#   to whichever performance dimension is most relevant for their application.
# =============================================================================


# ── ASSUMPTIONS ───────────────────────────────────────────────────
#
# ── Stationarity and zero mean ────────────────────────────────────
# For simplicity, the target and predictor are assumed to be
# stationary and zero-mean.
#   • For processes with a non-zero mean, zero-crossings should be
#     replaced by mean-crossings throughout
#   • An extension to non-stationary processes is given in Wildi (2026a) 
#     → Max-monotone and min-curvature SSA predictors
#

# ── Gaussianity ───────────────────────────────────────────────────
# The formal theoretical framework assumes Gaussian processes.
#   • Under Gaussianity, sign-accuracy and holding-time map exactly
#     onto correlation and first-order ACF, respectively
#     → See Section 2 of the JBCY, SSA, and M-SSA papers
#
#   • SSA is robust to departures from Gaussianity
#     (e.g., volatility clustering, heavy tails)
#     → This robustness follows from a central limit theorem argument
#     → Typical deviations from Gaussianity, as commonly encountered
#       in economic time series, do not materially affect performance
#     → See applications in Wildi (2024) and simulation studies in Wildi (2026 a,b)


# ── Crossings at non-mean thresholds ──────────────────────────────
# When zero-crossings are measured at a threshold above or below
# the mean (rather than at the mean itself):
#
#   • The holding-time (ht) statistic becomes biased at the
#     absolute level
#
#   • However, relative performance against a benchmark is
#     generally preserved:
#     → SSA remains smoother and produces fewer crossings than
#       the benchmark even at off-mean thresholds
#     → See Tutorials 1–5 for empirical illustration
#
# ─────────────────────────────────────────────────────────────────
# ─────────────────────────────────────────────────────────────────
# ── M-SSA, MDFA AND DFP/PCS PREDICTORS: A COMPARATIVE OVERVIEW ─────────────
#
# Historical context:
#   • DFA/MDFA   → origins in 2002 research and culminates in new MDFA book coauthored with Tucker McElroy (MDFA tutorials repository on github)
#   • M-SSA      → developed from early 2020 (M-SSA tutorials repository on github)
#   • DFP/PCS    → developed from mid 2020 (not yet on github)
#
# Common ground:
#   All three prediction frameworks are organized around the forecast trilemma,
#   jointly addressing Accuracy, Timeliness, and Smoothness —
#   albeit with practically relevant differences in formulation
#   and interpretation.
#
# Key distinctions:
#
#   • Domain
#       → MDFA operates in the frequency domain: https://github.com/wiaidp/MDFA-tutorial
#       → M-SSA and DFP/PCS are formulated in the time domain
#
#   • Trilemma decomposition in MDFA
#       → MSE is decomposed into amplitude and phase contributions,
#         which define the smoothness and timeliness terms
#         respectively — see cited literature for details and https://github.com/wiaidp/MDFA-tutorial
#
#   • Accuracy and Smoothness in M-SSA
#       → Accuracy measured as the predictor's sign accuracy
#       → Smoothness measured as the mean duration between consecutive
#         sign changes of the predictor (holding-time)
#       → Yields more direct and intuitive interpretations
#         than the MDFA amplitude-based formulation
#
#   • Timeliness in DFP/PCS
#       → Quantified via the effective time-shift of the predictor
#         (rather than phase in the frequency domain)
#       → Yields more direct and intuitive interpretation
#         than the MDFA phase-based formulation
#       → DFP/PCS maximize the lead among linear predictors:
#         We derive a universal upper bound on lead over classical MSE for any linear predictor 
#         under a consistency constraint and prove that DFP/PCS hit this ceiling.
#     