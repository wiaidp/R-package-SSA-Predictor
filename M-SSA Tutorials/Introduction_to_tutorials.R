# ─────────────────────────────────────────────────────────────────
# M-SSA PREDICTOR: A TUTORIAL SERIES
# ─────────────────────────────────────────────────────────────────

# The M-SSA provides a unified framework for solving general prediction problems 
# while simultaneously accommodating specific research priorities and objectives.

# This series of tutorials introduces M-SSA, with a focus on controlling key 
# characteristics of the predictor.

# Optimization criteria are simple, effective and interpretable. Optimization 
# problems have unique solutions and the optimization algorithms converge 
# rapidly to the global optimum, due to convexity (except in pathological cases).


# ── BACKGROUND REFERENCES ────────────────────────────────────────
# The following papers provide the theoretical foundations:
#
#   Wildi, M. (2024) Business Cycle Analysis and Zero-Crossings of
#     Time Series: a Generalized Forecast Approach. Journal of 
#     Business-Cycle Research,
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
# Note: Working paper versions are available in the 'Papers' folder
# of this GitHub repository. Working papers contain full proofs and
# detailed technical results, whereas published versions are more
# streamlined, occasionally moving proofs to online appendices.


###################################################################


# ────────────── GUIDED TOUR ACROSS (M-/I-) SSA/TUTORIAL ─────────────

# ── A. PREDICTION OBJECTIVES ────────────────────────────────────────
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
#         vs. lead/lag behavior (advancement or retardation) vs. 
#         noise suppression/smoothness/regularity

# ── B. ANALYST PREFERENCES AND PREDICTOR PRIORITIES ─────────────────
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


# ── C. SSA: SMOOTH SIGN ACCURACY ─────────────────────────────
#
# SSA is a flexible univariate prediction framework applicable to
# stationary time series:
#   • One-step ahead or multi-step ahead forecasting: tutorial 1
#   • Backcasting, nowcasting, and forecasting of Signals (Trends/cycles)
#     tutorials 2-7
#   • Smoothing: tutorials 8-10
#
# SSA explicitly targets key predictor characteristics:
#   • sign accuracy, target correlation and  MSE (Mean-Squared Error), see, 
#     e.g., tutorial 0.2.
#   • Noise suppression  (smoothness, wiggliness, rate of zero-crossings), see, 
#     e.g., tutorial 0.2.
#   • It can also address Timeliness  `indirectly'   (lead, left-shift, 
#     reduced phase-lag), see, e.g., tutorial 0.1 and tutorials 2-7.
#

# ── D. I-SSA: NON-STATIONARY PROCESSES ──────────────────────────────────
#
# I-SSA generalises the univariate SSA framework to handle integrated 
# (non-stationary) processes. 
#   - SIMULTANEOUS LEVEL AND GROWTH TRACKING:
#     I-SSA jointly addresses two complementary objectives in a single design:
#       (i)  Target tracking in levels: minimising the mean-squared error
#            between the filter output and the target signal on the
#            non-stationary level series.
#       (ii) Growth-sign tracking in first differences: ensuring that the
#            direction of change (sign of the first difference) in the
#            filter output correctly reflects the direction of change in
#            the underlying target, a property that is critical for
#            real-time turning-point detection.
# The theoretical foundations, algorithmic details, and empirical applications
# of I-SSA are developed in Tutorial 6.


# ── E. M-SSA: MULTIVARIATE EXTENSION ────────────────────────────────
#
# M-SSA generalises the univariate SSA framework to a multivariate setting,
# enabling the simultaneous extraction of cycle or trend signals from multiple
# time series under a common set of smoothness constraints. 
#
# The theoretical foundations, algorithmic implementation, and empirical
# applications of M-SSA are developed in Tutorial 7.


# ── F. SSA / I-SSA / M-SSA SMOOTHING ───────────────────────────────────────────
#
# Beyond classic target tracking and customization, SSA and its extensions 
# (I-SSA, M-SSA) can be directed at the original data series itself rather 
# than at a pre-specified benchmark filter or target. In this mode, the
# smoothness constraint is applied directly to the raw input, giving rise to
# two conceptually novel constructs:
#
#   - A NEW SMOOTHING PARADIGM:
#       Rather than approximating a fixed target filter, SSA minimises the
#       distance to the observed data subject to the holding-time constraint.
#       The result is a data-adaptive smoother whose degree of smoothness is
#       directly controlled by the analyst through the holding-time parameter,
#       without reference to any external benchmark, see tutorials 8 and 10.
#
#   - A NEW I-SSA TREND DEFINITION:
#       The smoothed output can be interpreted as a principled, operationally
#       well-defined trend: the closest approximation to the (non-stationary) 
#       data subject to a pre-specified (imposed) mean distance between 
#       consecutive turning points. This trend definition is
#       grounded in the SSA optimality framework: it is interpretable, 
#       logically consistent and statistically efficient. See tutorial 9.
#
# The theoretical foundations, algorithmic details, and empirical applications
# of SSA/I-SSA/M-SSA smoothing are developed in Tutorials 8–10.

# ── G. SSA OPTIMIZATION PRINCIPLE ───────────────────────────────────
#
# Core objective:
#   Maximize sign-accuracy (or target correlation) subject to a 
#   holding-time constraint
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
# Details and empirical applications are developed in Tutorial 0.2 and then 
# applied all along the tutorial series.
#
# Theoretical background:
#   → Section 2 Wildi (2024), Wildi (2026b); section 4 Wildi (2026a) 


# ── H. MOTIVATION: WHY ZERO-CROSSINGS MATTER ─────────────────────────
#
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
#     behind the target — see Tutorial 0.3 (+tutorials 2-7)
#
# SSA improves upon the MSE benchmark by offering:
#   • Noise suppression  → fewer spurious alarms
#   • Timeliness         → lead or left-shift relative to target
#
# Notably, SSA can be simultaneously smoother and faster than the
# benchmark — see Tutorials 2–5.


# ── I. FORECAST TRILEMMA ─────────────────────────────────────────
#
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
#   • Lag/retardation 
#       → The predictor systematically trails behind the target
#         (right-shift or phase-lag) — see Tutorial 0.3


# ── J. SSA (M-SSA/I-SSA): SCOPE ─────────────────────────────────────
#
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
#       → Imposing the HT of MSE to SSA replicates the classical
#         predictor exactly — see Tutorial 0.3 and the remark
#         after Theorem 1 in Wildi 2024, 2026a
#
#   • Generalization of linear predictors
#       → In principle, any linear forecast rule can be replicated
#         within the SSA framework (see tutorials 1-5)
#
#   • Customization of benchmarks
#       → Once a classical linear forecast is replicated, SSA can
#         be used to make it faster, smoother, or both (see tutorials 1-5)
#

# ── K. SSA AS A PLUG-ON ──────────────────────────────────────────────
#
# SSA can be used in two distinct modes:
#
#   • Standalone
#       → SSA operates as a self-contained, original forecast
#         algorithm in its own right
#
#   • Plug-on
#       → SSA is grafted onto an existing benchmark predictor,
#         enhancing its smoothness and/or timeliness in a
#         controlled and predictable manner, see section 3.2 in Wildi 2026b
#
# In this tutorial series, plug-on applications are demonstrated
# for the following benchmarks:
#
#   • MSE predictor              → Tutorials 0–5
#   • Hodrick-Prescott (HP) filter  → Tutorials 2, 6 and 7
#   • Hamilton filter (HF)          → Tutorial 3
#   • Baxter-King (BK) filter       → Tutorial 4
#   • Beveridge-Nelson (refined)    → Tutorial 5


# ── L. INTERPRETABILITY ──────────────────────────────────────────────
#
# OPTIMIZATION CRITERION:
#   The objective (sign accuracy, target correlation, or MSE) and the
#   HT constraint are interpretable and intuitively appealing.

# OPTIMIZATION:
#   Numerical computations are fast and the search space is
#   convex (except in singular cases requiring extreme smoothing),
#   guaranteeing convergence to the unique global optimum.

# CUSTOMIZATION:
#   When operated in plug-on mode, SSA grafts onto a benchmark filter (e.g., 
#   Hodrick Prescott, Hamilton, Baxter King, Beveridge Nelson, ARMA, VARMA), 
#   seeking the optimal causal approximation to the benchmark subject to the 
#   imposed smoothness (holding-time) constraint. This design choice has 
#   three important consequences:
#
#   - INHERITED INTERPRETABILITY:
#       Because SSA tracks the benchmark as closely as possible within the
#       imposed constraints, the resulting filter inherits the economic
#       interpretation of the benchmark. The SSA output can be read and
#       communicated in exactly the same terms as the original filter.
#
#   - PRESERVED ECONOMIC CONTENT:
#       The original economic meaning embedded in the benchmark — whether it
#       captures the business cycle, the output gap, or employment deviations
#       from trend — is carried through to the SSA filter by construction.
#       SSA refines the benchmark's real-time properties without redefining
#       what is being measured.
#
#   - TRANSPARENT IMPROVEMENTS:
#       Any gains in smoothness or timeliness relative to the benchmark are
#       directly attributable to the SSA constraints, making the source of
#       improvement explicit and quantifiable rather than an opaque artefact
#       of a black-box procedure.

# =============================================================================
# THEORETICAL FOUNDATION: THE SSA EFFICIENT FRONTIER
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
#   simultaneously by varying the constraint level (ht* or rho*).
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
#     - Fix smoothness (holding time ht*)    => maximize accuracy
#     - Fix accuracy  (target correlation rho*) => maximize smoothness
#   The resulting filter is the same efficient design in both cases.
#   This flexibility allows practitioners to parameterize SSA according
#   to whichever performance dimension is most relevant for their application.
# Currently, SSA, M-SSA and I-SSA are implemented in the primal form only: 
#   -Maximize tracking accuracy subject to a HT constraint.
# =============================================================================



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