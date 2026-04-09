
# ══════════════════════════════════════════════════════════════════════════════
# Tutorial 10: M-SSA SMOOTHING
# ══════════════════════════════════════════════════════════════════════════════

# This tutorial extends the SSA smoothing framework of tutorial 8 to 
# multivariate M-SSA smoothing. For a comprehensive discussion of the
# distinction between smoothing and prediction, the reader is referred
# to Tutorial 8.

# ══════════════════════════════════════════════════════════════════════════════

# Tutorial 7 emphasized M-SSA applications to prediction.
# The sole modification relative to Tutorial 7 is the choice of target: here the
# target is the raw observed series x_{t+delta} rather than an acausal
# filter applied to the data.

# ══════════════════════════════════════════════════════════════════════════════
# REFERENCES
# ──────────────────────
#
# Wildi, M. (2024). Business Cycle Analysis and Zero-Crossings of Time Series:
#    A Generalized Forecast Approach.  Journal of Business Cycle Research,
#   https://doi.org/10.1007/s41549-024-00097-5

# Wildi, M. (2026a). Sign Accuracy, Mean-Squared Error and the Rate of 
# Zero Crossings:
#     a Generalized Forecast Approach, https://doi.org/10.48550/arXiv.2601.06547
#
# Wildi, M. (2026b). The Accuracy-Smoothness Dilemma in Prediction:
#   A Novel Multivariate M-SSA Forecast Approach.
#   Journal of Time Series Analysis, http://doi.org/10.1111/jtsa.70058 
#   arXiv: https://doi.org/10.48550/arXiv.2602.13722
#
# Parts in this  tutorial are  based on Wildi (2026b), Section 4.2. Additional 
#   applications are given in Wildi (2024). Extensions to non-stationary series 
#   (I-SSA) are presented in Wildi (2026a).

# ══════════════════════════════════════════════════════════════════════════════


# Clear the workspace to ensure a clean environment
rm(list = ls())

# ══════════════════════════════════════════════════════════════════════════════
# LOAD REQUIRED LIBRARIES
# ══════════════════════════════════════════════════════════════════════════════

# HP filter and other standard time-series filters
library(mFilter)
# ROC curve calculation
library(pROC)
# NBER recession datings for the US
library(tis)


# ══════════════════════════════════════════════════════════════════════════════
# LOAD CUSTOM M-SSA FUNCTION LIBRARIES
# ══════════════════════════════════════════════════════════════════════════════

# Core M-SSA filter construction and optimisation routines
source(paste(getwd(), "/R/functions_MSSA.r", sep = ""))

# HP-filter utilities used in the JBCY paper (depends on mFilter)
source(paste(getwd(), "/R/HP_JBCY_functions.r", sep = ""))

# M-SSA utility functions: data preparation, plotting helpers, and wrappers
source(paste(getwd(), "/R/M_SSA_utility_functions.r", sep = ""))

# ROC plot
source(paste(getwd(), "/R/ROCplots.r", sep = ""))

#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# There are potential problems when loading SSA together with MSSA.
# Some function names are the same but the underlying functions are different.
# Advice: M-SSA generalizes SSA, therefore there is no need to load the SSA 
# functions in addition to M-SSA. All relevant function for M-SSA are packed 
# in functions_MSSA.r. DO NOT SOURCE simple_sign_accuracy.r when working with 
# M-SSA
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!




# ══════════════════════════════════════════════════════════════════════════════
# Exercise 1: M-SSA Smoothing
# Nowcasting with Moderate Smoothing
# ══════════════════════════════════════════════════════════════════════════════
# This example is drawn from Wildi (2026b), Section 4.2. It is based on a
# three-dimensional VAR(1) process.

# ─────────────────────────────────────────────────────────────────────────────
# 1.1. Three-Dimensional VAR(1)
# ─────────────────────────────────────────────────────────────────────────────
n <- 3
# Specify a random innovation covariance matrix Sigma and an AR(1) coefficient
# matrix A.
set.seed(1)
Sigma_sqrt <- matrix(rnorm(n * n), ncol = n)
Sigma <- Sigma_sqrt %*% t(Sigma_sqrt)
Sigma
# Sigma is full rank: all eigenvalues are strictly positive.
# If Sigma is rank-deficient, the problem simplifies; see Wildi (2026b).
eigen(Sigma)$values
A <- rbind(c(0.7, 0.4, -0.2), c(-0.6, 0.9, 0.3), c(0.5, 0.2, -0.3))
A
# Check stationarity: all eigenvalues of A must be strictly less than one
# in absolute value.
abs(eigen(A)$values)
# Optional: replace the VAR(1) with a VARMA(1,1) specification.
if (F)
{
  A <- diag(rep(0, n))
  B <- 0.05 * rbind(c(0.5, 0.4, -0.4), c(0.6, -0.9, 0.3), c(0.8, -0.7, -0.8))
}

# ─────────────────────────────────────────────────────────────────────────────
# 1.2. M-SSA Settings
# ─────────────────────────────────────────────────────────────────────────────
L <- 51
# Nowcasting: the target coincides with the current observation (delta = 0).
# A symmetric multivariate backcast will be explored below.
delta <- delta1 <- 0
# Target filter matrix: Gamma is the n x nL identity mapping, where each
# series targets its own contemporaneous value.
gamma_target <- matrix(rep(0, n^2 * L), nrow = n)
gamma_target[1, 1] <- gamma_target[2, 1 + L] <- gamma_target[3, 1 + 2 * L] <- 1
# Structure of gamma_target (n x nL matrix):
#   - Row i specifies the filter coefficients for the i-th target series.
#   - Columns (j-1)*L+1 : j*L of row i hold the coefficients applied to the
#     j-th input series when estimating target i.
#   - In the nowcasting case, all entries are zero except at position
#     (i, (i-1)*L+1), which equals one (Kronecker delta structure).
gamma_target
# Verify the Kronecker delta structure for each series:
i <- 1
gamma_target[i, (i - 1) * L + 1]
i <- 2
gamma_target[i, (i - 1) * L + 1]
i <- 3
gamma_target[i, (i - 1) * L + 1]
# Confirm identity structure along the block diagonal.
cbind(gamma_target[, 1], gamma_target[, L + 1], gamma_target[, 2 * L + 1])
# Note: in Tutorial 7 (M-SSA), gamma_target[i, ((i-1)*L)+1:L] was populated
# with HP filter coefficients, making HP the smoothing target. Here, the
# target is the raw series X_{t+delta} with delta = 0, i.e., a 3-dimensional
# VAR(1) observed contemporaneously.

# ─────────────────────────────────────────────────────────────────────────────
# 1.3. Wold Decomposition
# ─────────────────────────────────────────────────────────────────────────────
# M-SSA requires the Wold decomposition of the process into orthogonal
# innovations epsilon_t. Below, the MA coefficient matrices are computed
# for the VAR(1) specification.

# Method a: Analytic recursion for VAR(1)
B <- diag(rep(0, n))
if (F)
{
  A <- diag(rep(0, n))
  B <- 0.05 * rbind(c(0.5, 0.4, -0.4), c(0.6, -0.9, 0.3), c(0.8, -0.7, -0.8))
}
det(A)
Ak   <- A
Akm1 <- diag(rep(1, n))
xi   <- matrix(nrow = dim(A)[1], ncol = L * dim(A)[1])
xi[, L * (0:(dim(xi)[1] - 1)) + 1] <- diag(rep(1, n))
for (i in 2:L)
{
  # Row k of xi holds the MA weights for the k-th target series.
  # Columns 1:L correspond to the weights on the first noise series
  # eps_{1,t}, eps_{1,t-1}, ...; columns L+1:2L to the second noise
  # series eps_{2,t}, eps_{2,t-1}, ...; and so on.
  # This parametrisation follows the IJFOR paper convention.
  for (j in 1:n)
    xi[, i + (j - 1) * L] <- Ak[, j] + (Akm1 %*% B)[, j]
  Ak   <- Ak %*% A
  Akm1 <- Akm1 %*% A
}

# Method b: General MA inversion for arbitrary VARMA processes.
B    <- NULL
# Use non-orthogonalised innovations (consistent with Sigma).
orth <- F
irf  <- M_MA_inv(A, B, Sigma, L - 1, orth)$irf
# Populate Xi (capital X) from the impulse response function.
Xi <- 0*xi
for (i in 1:n)
{
  Xi[, (i - 1) * L + 1:L] <- irf[(i - 1) * n + 1:n, 1:L]
}
# Verification: both MA inversions should agree to numerical precision.
max(abs(xi - Xi))

# ─────────────────────────────────────────────────────────────────────────────
# 1.4. Holding-Time and M-SSA Default Settings
# ─────────────────────────────────────────────────────────────────────────────
# Specify the desired holding times (HTs) for each of the three series, see 
# Wildi 2026b.
ht_vec <- matrix(c(min(8,  L / 2), min(6, L / 2), min(10, L / 2)), nrow = 1)
# Convert the specified HTs to first-order autocorrelations (rho), as M-SSA
# is parameterised in terms of rho rather than HT directly.
rho0 <- apply(ht_vec, 1, compute_rho_from_ht)[[1]]$rho

# Hyperparameters for controlling numerical computation.
# Setting with_negative_lambda = TRUE extends the search to the unsmoothing
# regime (generating more zero-crossings than the benchmark). 
with_negative_lambda <- T
# Numerical optimisation settings for M-SSA.
# The setting chosen here allows for extreme smoothing (see example below)
lower_limit_nu <- "rhomax"
# The target is asymmetric (Gamma is the identity), so this flag has no
# effect on estimation; either TRUE or FALSE is admissible here.
symmetric_target <- F
# Use bisection-based optimisation with 2^split_grid effective resolution.
# This is substantially faster than brute-force grid search and exploits
# the monotonicity of the first-order ACF when |nu| > 2 * rho_max(L).
# For |nu| < 2 * rho_max(L), grid search should be used instead, as it
# enumerates all solutions to the HT constraint (at higher computational cost).
split_grid <- 20

# ─────────────────────────────────────────────────────────────────────────────
# 1.5. M-SSA Estimation
# ─────────────────────────────────────────────────────────────────────────────
MSSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho0,
                      with_negative_lambda, xi, lower_limit_nu, Sigma,
                      symmetric_target)

# Target correlations of M-SSA with the causal M-MSE smoother, for each
# series i = 1, ..., n. If the HT of M-SSA matches that of M-MSE exactly,
# these correlations equal one (M-SSA replicates M-MSE). Smaller correlations
# indicate stronger smoothing.
MSSA_obj$crit_rhoyz
# Target correlations of M-SSA with the acausal target. In a smoothing
# context the target is always causal, so these values coincide with
# crit_rhoyz above.
MSSA_obj$crit_rhoy_target
# First-order ACFs implied by the optimised smoother. If numerical
# optimisation has converged, these should match rho0 closely.
MSSA_obj$crit_rhoyy
# Verify convergence: the two sets of values should agree up to negligible
# errors.
rho0
# Optimal nu for each series. Values of nu > 2
# indicate active smoothing (fewer zero-crossings than M-MSE).
MSSA_obj$nu_opt

# ─────────────────────────────────────────────────────────────────────────────
# Smoother Coefficients
# ─────────────────────────────────────────────────────────────────────────────
# 1. M-SSA filter applied to innovations epsilon_t (Wold decomposition).
#    Primarily used for diagnostic purposes; also relevant when the process
#    is non-invertible (see Exercise 2.7 in Tutorial 8).
bk_mat <- MSSA_obj$bk_mat
# 2. M-SSA filter applied directly to the observed series x_t (deconvolution).
#    This is the operationally relevant smoother in nearly all applications.
bk_x_mat <- MSSA_obj$bk_x_mat

# ─────────────────────────────────────────────────────────────────────────────
# M-MSE Reference Smoother
# ─────────────────────────────────────────────────────────────────────────────
# M-SSA optimisation principle: for a given HT (encoded via rho0), M-SSA
# seeks the smoother that is as close as possible to the classical M-MSE
# smoother. See Wildi (2026b) for the formal statement.
#
# 1. M-MSE filter applied to innovations epsilon_t from the Wold decomposition.
gammak_mse <- MSSA_obj$gammak_mse
# Verification: in the nowcasting case (delta = 0, identity target), M-MSE
# reduces to the Wold decomposition itself.
max(abs(t(gammak_mse) - xi))
# 2. M-MSE filter applied to the observed series x_t.
#    Since the target is x_t and delta = 0, the optimal MSE solution is the
#    identity: the best causal approximation of x_t is x_t itself.
gammak_x_mse <- MSSA_obj$gammak_x_mse
# Verify the identity
cbind(gammak_x_mse[ 1,], gammak_x_mse[ L + 1,], gammak_x_mse[ 2 * L + 1,])


# ─────────────────────────────────────────────────────────────────────────────
# 1.6. Performance: Theoretical (Expected) Values
# ─────────────────────────────────────────────────────────────────────────────

# 1.6.1. Compute True First-Order ACFs
# Compute the system matrices required for theoretical performance evaluation.
# Notation follows Wildi (2026b).
M_obj    <- M_func(L, Sigma)
M_tilde  <- M_obj$M_tilde
I_tilde  <- M_obj$I_tilde

# Theoretical lag-one ACF for each series, derived from the filter coefficients.
# The lag-one ACF is in bijective correspondence with the holding time (HT);
# see Wildi (2026b), Section 2.
# Series 1:
rho_mse_1 <- (gammak_mse[, 1]) %*% M_tilde %*% gammak_mse[, 1] /
  gammak_mse[, 1]  %*% I_tilde  %*% gammak_mse[, 1]
rho_ssa_1 <- bk_mat[, 1] %*% M_tilde %*% bk_mat[, 1] /
  bk_mat[, 1] %*% I_tilde  %*% bk_mat[, 1]
# Expected: rho_ssa_1 > rho_mse_1, confirming that M-SSA is smoother than M-MSE.
rho_mse_1
rho_ssa_1
# Series 2:
rho_mse_2 <- gammak_mse[, 2] %*% M_tilde %*% gammak_mse[, 2] /
  gammak_mse[, 2] %*% I_tilde  %*% gammak_mse[, 2]
rho_ssa_2 <- bk_mat[, 2] %*% M_tilde %*% bk_mat[, 2] /
  bk_mat[, 2] %*% I_tilde  %*% bk_mat[, 2]
# Series 3:
rho_mse_3 <- gammak_mse[, 3] %*% M_tilde %*% gammak_mse[, 3] /
  gammak_mse[, 3] %*% I_tilde  %*% gammak_mse[, 3]
rho_ssa_3 <- bk_mat[, 3] %*% M_tilde %*% bk_mat[, 3] /
  bk_mat[, 3] %*% I_tilde  %*% bk_mat[, 3]

# 1.6.2. Convert ACFs to Holding Times (Equation 4 in Wildi 2026b)
# Verify that the optimised M-SSA HTs match the imposed targets in ht_vec.
# Increasing split_grid in the MSSA_func call improves the fit.
compute_holding_time_from_rho_func(rho_ssa_1)$ht
compute_holding_time_from_rho_func(rho_ssa_2)$ht
compute_holding_time_from_rho_func(rho_ssa_3)$ht
# Target HTs for reference: both numbers should match after successful 
# optimization.
ht_vec
# M-MSE HTs (all smaller than the corresponding M-SSA HTs, confirming that
# M-SSA imposes greater smoothness):
compute_holding_time_from_rho_func(rho_mse_1)$ht
compute_holding_time_from_rho_func(rho_mse_2)$ht
compute_holding_time_from_rho_func(rho_mse_3)$ht

# 1.6.3. Objective Functions (Target Correlations); see Wildi (2026b), Section 2
# M-MSE trivially achieves a target correlation of one with itself, since the
# target is defined as the M-MSE output (equivalently, x_{t+delta} with delta=0).
# M-SSA maximises the target correlation in the second row subject to the HT
# constraint. The values computed here should match MSSA_obj$crit_rhoyz.
crit_mse_1 <- gammak_mse[, 1] %*% I_tilde %*% gammak_mse[, 1] /
  gammak_mse[, 1] %*% I_tilde  %*% gammak_mse[, 1]
crit_ssa_1 <- gammak_mse[, 1] %*% I_tilde %*% bk_mat[, 1] /
  (sqrt(bk_mat[, 1]    %*% I_tilde %*% bk_mat[, 1]) *
     sqrt(gammak_mse[, 1] %*% I_tilde %*% gammak_mse[, 1]))

crit_mse_2 <- gammak_mse[, 2] %*% I_tilde %*% gammak_mse[, 2] /
  gammak_mse[, 2] %*% I_tilde  %*% gammak_mse[, 2]
crit_ssa_2 <- gammak_mse[, 2] %*% I_tilde %*% bk_mat[, 2] /
  (sqrt(bk_mat[, 2]    %*% I_tilde %*% bk_mat[, 2]) *
     sqrt(gammak_mse[, 2] %*% I_tilde %*% gammak_mse[, 2]))

crit_mse_3 <- gammak_mse[, 3] %*% I_tilde %*% gammak_mse[, 3] /
  gammak_mse[, 3] %*% I_tilde  %*% gammak_mse[, 3]
crit_ssa_3 <- gammak_mse[, 3] %*% I_tilde %*% bk_mat[, 3] /
  (sqrt(bk_mat[, 3]    %*% I_tilde %*% bk_mat[, 3]) *
     sqrt(gammak_mse[, 3] %*% I_tilde %*% gammak_mse[, 3]))

# Verify: M-SSA target correlations should match MSSA_obj$crit_rhoyz.
c(crit_ssa_1, crit_ssa_2, crit_ssa_3)
MSSA_obj$crit_rhoyz

# Summary table of target correlations:
# Row 1 (M-MSE): all entries are one in this smoothing exercise.
# Row 2 (M-SSA): maximised values subject to the HT constraint.
criterion_mat <- rbind(c(crit_mse_1, crit_mse_2, crit_mse_3),
                       c(crit_ssa_1, crit_ssa_2, crit_ssa_3))
colnames(criterion_mat) <- c(paste("Series ", 1:n, paste = ""))
rownames(criterion_mat) <- c("MSE smoother", "SSA smoother")
criterion_mat

# We now compute the sample performance measures. Convergence toward the
# expected values reported above provides empirical support for the optimality
# of M-SSA as a smoother: in that case, M-SSA also maximizes the sample
# correlation between the estimate and the target, subject to the empirical
# smoothness constraint.
# ─────────────────────────────────────────────────────────────────────────────
# 1.7. Sample Performances
# ─────────────────────────────────────────────────────────────────────────────
# Simulate a long realisation of the VAR(1) process and compute sample
# performance measures. Convergence of sample values to their theoretical
# counterparts validates the M-SSA framework as a pertinent multivariate
# extension of univariate I-SSA smoothing.
len <- 10000

# Note: depending on len simulation may take some time to go through.
# Source the simulation utilities used in Wildi (2026b).
source(paste(getwd(), "/R/M_SSA_paper_functions.r", sep = ""))
# Simulate the VAR(1) process and evaluate filter performance.
setseed <- 16
sample_obj     <- sample_series_performances_smooth_func(
  A, Sigma, len, bk_mat, bk_x_mat,
  t(gammak_mse), L, setseed, MSSA_obj)
perf_mat_sample <- sample_obj$perf_mat_sample
perf_mat_true   <- sample_obj$perf_mat_true
bk_mat          <- sample_obj$bk_mat
gammak_mse      <- sample_obj$gammak_mse
y_mat           <- sample_obj$y_mat
zdelta_mat      <- sample_obj$zdelta_mat
z_mse_mat       <- sample_obj$z_mse_mat
x_mat           <- sample_obj$x_mat

# Display theoretical values alongside sample estimates (in parentheses).
# Convergence of sample to theoretical values confirms the validity of the
# M-SSA approach. Increasing len improves agreement at the cost of
# longer computation time.
perf_mat <- matrix(paste(round(perf_mat_true[,c("Sign accuracy", 
            "Cor. with MSE", "ht","ht MSE")], 5), " (",
            round(perf_mat_sample[, c("Sign accuracy", "Cor. with MSE", "ht", 
            "ht MSE")], 5), ")", sep = ""),ncol = 4)
colnames(perf_mat) <- c("Sign accuracy", "Cor. with data", "HT M-SSA", 
                        "HT of data")
perf_mat
# Outcome: 
# Sample performances (in parentheses) converge to expected values 
# with increasing sample length.

# ─────────────────────────────────────────────────────────────────────────────
# 1.8. Plots
# ─────────────────────────────────────────────────────────────────────────────

# Plot 1: M-SSA filter weights (bk_x_mat) for each of the three targets.
# Each panel shows the lag structure of the weights applied to all three
# input series when constructing one smoothed output series.

# Panel 1: Smoothers for first target x_{1t}
mplot<-cbind(bk_x_mat[1:L,1],bk_x_mat[L+1:L,1],bk_x_mat[2*L+1:L,1])
colnames(mplot)<-c("Series 1","Series 2","Series 3")
colo<-c("blue","red","darkgreen","violet","black")
par(mfrow=c(1,3))
plot(mplot[,1],main="SSA: first target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()
# Panel 2: Smoothers for second target x_{2t}
mplot<-cbind(bk_x_mat[1:L,2],bk_x_mat[L+1:L,2],bk_x_mat[2*L+1:L,2])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: second target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()
# Panel 3: Smoothers for third and last target x_{3t}
mplot<-cbind(bk_x_mat[1:L,3],bk_x_mat[L+1:L,3],bk_x_mat[2*L+1:L,3])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: third target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

#--------------
# Plot 2: Cross-correlation functions (CCFs) against Series 2 (left panel)
# and a short realisation of the VAR(1) process (right panel).
# See Wildi (2026b), Section 4.2, for a discussion of this example.
# Hint: The smoother weights in the above plot is related to the CCF and sample 
# realizations in the below plot (explainability).

par(mfrow=c(1,2))

colo<-c("blue","red","darkgreen")

lag_max<-20
j<-1
k<-2
pc<-peak_cor_func(x_mat,j,k,lag_max)
j<-2
k<-2
pc<-cbind(pc,peak_cor_func(x_mat,j,k,lag_max))
j<-3
k<-2
pc<-cbind(pc,peak_cor_func(x_mat,j,k,lag_max))
plot(pc[,1],lty=1,main=paste("CCF against series ",k,sep=""),axes=F,type="l",
     ylab="",xlab="Lag-structure",ylim=c(min(pc),max(pc)))
for (i in 1:ncol(pc))
  lines(pc[,i],col=colo[i],lwd=1,lty=1)
abline(h=0)
abline(v=lag_max)
axis(1,at=1:nrow(pc),labels=-lag_max+1:nrow(pc))
axis(2)
box()
# Panel 2: short subsample
ts.plot(x_mat[850:875,],col=colo,main="VAR(1)")
for (i in 1:3)
  mtext(paste("Series ",i,sep=""),col=colo[i],line=-i,xlab="time",ylab="")


#--------------
# Plot 3: Observed data (black), M-SSA smoother (cyan), and M-MSE smoother
# (violet) over a short window, with vertical dashed lines marking
# zero-crossings of the M-SSA output.
# In the nowcasting case (delta = 0, identity target), M-MSE reduces to
# the identity: z_mse = x_t (bloth lines overlap). The plot therefore 
# illustrates how M-SSA tracks the data subject to the imposed HTe constraint.

# Select a short sub-sample of the very longth simulation path
anf<-10*L
enf<-11*L
y_mat[anf:enf,]
z_mse_mat[anf:enf,]
zdelta_mat[anf:enf,]
colo<-c("cyan","violet","black")
par(mfrow=c(2,2))
# MSE, target and x are identical: gammak is an identity and delta=0
# We here demonstrate the multivariate SSA trend specification
#   yt matches xt conditional on holding-time constraint
for (i in 1:n)
{
  mplot<-cbind(y_mat[anf:enf,i],z_mse_mat[anf:enf,i],zdelta_mat[anf:enf,i])
  
  ts.plot(scale(mplot,center=F,scale=T),col=colo,
          main=paste("Series ",i,sep=""),lwd=2)
  abline(h=0)
  abline(v=1+which(mplot[2:nrow(mplot),1]*mplot[1:(nrow(mplot)-1),1]<0),
         col=colo[1],lty=2)
  lines(scale(mplot,center=F,scale=T)[1],col=colo[1])
}

# Explanation:
# - The first plot (smoother weights) shows that the second series (red) is a
#   major source of information, as it receives relatively large weights across
#   all three dimensions.
#
# - More generally, each series i is also an important determinant of its own
#   target i.
#
# - The importance of the second series is also evident from the second plot:
#   - In the left panel, the peaks of the CCF indicate that the second series
#     is leading, since the CCF peaks for the first (blue) and third (green)
#     series are shifted to the right.
#   - This lead-lag pattern is confirmed in the subsample shown in the right
#     panel, where series 2 (red) is systematically shifted to the left,
#     consistent with a leading role.
#
# - The third plot (top-right panel) further suggests that series 2 is the
#   smoothest of the three.
#
# - The prominent role of series 2 in the smoother weights from the nowcast
#   experiment above can therefore be explained by the fact that series 2 is
#   both leading and relatively smooth.
#
# - This experiment underscores the importance of leading indicators in a 
#   multivariate framework.
#
# To conclude, Plot 3 illustrates the varying difficulty of the smoothing task:
# - Series 2 (top-right panel) is the easiest to smooth, as the estimated
#   smoother lies very close to the original data.
# - For series 1 (top-left panel), the task is somewhat more demanding, since
#   the smoother displays noticeably fewer zero-crossings than the original
#   series.
# - Series 3 (bottom panel) is the most challenging case, because the data are
#   highly noisy while the imposed HT is relatively large.


# ══════════════════════════════════════════════════════════════════════════════
# Exercise 2: M-SSA Smoothing — Three Special Cases
#   2.1  Symmetric multivariate backcast smoother
#   2.2  Replicating M-MSE (identity nowcast)
#   2.3  Extreme smoothing
# ══════════════════════════════════════════════════════════════════════════════

# ─────────────────────────────────────────────────────────────────────────────
# 2.1. Symmetric M-SSA Backcast Smoother
# ─────────────────────────────────────────────────────────────────────────────
# Extend Exercise 1 to the symmetric backcasting setting by shifting the
# target to the centre of the filter window. Only delta needs to change;
# all other settings remain identical to Exercise 1.
delta <- delta2 <- -(L - 1) / 2

# Retain the same holding-time targets as in Exercise 1.
ht_vec <- matrix(c(min(8, L / 2), min(6, L / 2), min(10, L / 2)), nrow = 1)
# Convert HTs to first-order ACFs for input to M-SSA.
rho0 <- apply(ht_vec, 1, compute_rho_from_ht)[[1]]$rho

MSSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho0,
                      with_negative_lambda, xi, lower_limit_nu, Sigma,
                      symmetric_target)

# Extract key performance summaries.
crit_sym     <- MSSA_obj$crit_rhoyz   # Target correlations with M-MSE.
rho_sym       <- MSSA_obj$crit_rhoyy   # Achieved first-order ACFs.
# Check: these differences should all be small if the optimization converged
rho_sym-rho0

# Extract filter coefficient matrices.
bk_mat_sym   <- MSSA_obj$bk_mat       # M-SSA applied to innovations epsilon_t.
bk_x_mat_sym <- MSSA_obj$bk_x_mat    # M-SSA applied to observed series x_t.
gammak_mse_sym <- MSSA_obj$gammak_mse # M-MSE applied to innovations epsilon_t.

# Plot filter weights for each target series.
mplot<-cbind(bk_x_mat_sym[1:L,1],bk_x_mat_sym[L+1:L,1],bk_x_mat_sym[2*L+1:L,1])
colnames(mplot)<-c("Series 1","Series 2","Series 3")
colo<-c("blue","red","green","violet","black")
par(mfrow=c(1,3))
plot(mplot[,1],main="SSA: first target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()


mplot<-cbind(bk_x_mat_sym[1:L,2],bk_x_mat_sym[L+1:L,2],bk_x_mat_sym[2*L+1:L,2])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: second target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

mplot<-cbind(bk_x_mat_sym[1:L,3],bk_x_mat_sym[L+1:L,3],bk_x_mat_sym[2*L+1:L,3])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: third target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Outcome: The symmetric M-SSA backcast smoother is virtually
# indistinguishable from a univariate SSA backcast smoother applied separately 
# to each series (see Tutorial 8, Exercise 1).
#
# The intuition is straightforward: when future observations are available, as
# in the backcasting case, the additional series provide little incremental
# information for constructing the optimal smoother. In this setting, the
# multivariate structure therefore offers no material advantage over the
# corresponding univariate solution.



# ─────────────────────────────────────────────────────────────────────────────
# 2.2. Replicating M-MSE (Identity Nowcast)
# ─────────────────────────────────────────────────────────────────────────────
# Repeat the nowcasting exercise from Exercise 1, but now calibrate the HT
# targets to match the empirical HTs of the data. When the imposed HT equals
# the HT of the data, M-SSA reduces to M-MSE (the identity in this case).

# Nowcast (this differs from exercise 2.1)
delta <- 0

# Estimate the empirical HT of each series from the simulated sample.
# Note: a longer sample (larger len) yields more accurate HT estimates.
ht_vec <- matrix(c(compute_empirical_ht_func(x_mat[, 1])$empirical_ht,
                   compute_empirical_ht_func(x_mat[, 2])$empirical_ht,
                   compute_empirical_ht_func(x_mat[, 3])$empirical_ht),
                 nrow = 1)
colnames(ht_vec) <- paste("Series ", 1:3, sep = "")
# This differs from exercises 1 and 2.1
ht_vec

# Convert empirical HTs to first-order ACFs for M-SSA input.
rho0 <- apply(ht_vec, 1, compute_rho_from_ht)[[1]]$rho

MSSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho0,
                      with_negative_lambda, xi, lower_limit_nu, Sigma,
                      symmetric_target)

# Expected outcome: target correlations are (approximately) equal to one,
# since the imposed HT matches the HT of the data and M-SSA replicates M-MSE.
MSSA_obj$crit_rhoyz

# Extract filter coefficient matrices.
bk_mat_mse     <- MSSA_obj$bk_mat       # M-SSA applied to innovations.
bk_x_mat_mse   <- MSSA_obj$bk_x_mat    # M-SSA applied to observed series.
gammak_mse_mse <- MSSA_obj$gammak_mse  # M-MSE applied to innovations.
gammak_x_mse   <- MSSA_obj$gammak_x_mse # M-MSE applied to observed series (identity).



# Plot filter weights for each target series.
# Panel 1: series 1
mplot<-cbind(bk_x_mat_mse[1:L,1],bk_x_mat_mse[L+1:L,1],bk_x_mat_mse[2*L+1:L,1])
colnames(mplot)<-c("Series 1","Series 2","Series 3")
colo<-c("blue","red","green","violet","black")
par(mfrow=c(1,3))
plot(mplot[,1],main="SSA: first target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Panel 2: series 2
mplot<-cbind(bk_x_mat_mse[1:L,2],bk_x_mat_mse[L+1:L,2],bk_x_mat_mse[2*L+1:L,2])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: second target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Panel 3: series 3
mplot<-cbind(bk_x_mat_mse[1:L,3],bk_x_mat_mse[L+1:L,3],bk_x_mat_mse[2*L+1:L,3])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: third target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Outcome: the M-SSA nowcast filter weights are approximately those
# of the identity, confirming that M-SSA replicates M-MSE when the imposed
# HT matches the empirical HT of the data. Increasing the sample length (len)
# yields more accurate empirical HT estimates and brings M-SSA closer to the
# exact identity solution.


# ─────────────────────────────────────────────────────────────────────────────
# 2.3. Extreme Smoothing
# ─────────────────────────────────────────────────────────────────────────────
# Impose nearly the maximum admissible degree of smoothing by setting all HTs 
# equal to the filter length L. At this boundary, the M-SSA smoother is heavily 
# constrained by the extremely demanding smoothness requirement. In particular, 
# the coefficient decay is very slow. 
#
# Practical note: extreme smoothing is not recommended in applications, as
# it leaves very few effective degrees of freedom. If such strong smoothing
# is required, the filter length L should be increased so that the target
# HT satisfies HT < L/2.
ht_vec <- matrix(c(L, L, L), nrow = 1)
# Very large HTs: HT/L=1
ht_vec

# Convert the extreme HTs to first-order ACFs for M-SSA input.
rho0 <- apply(ht_vec, 1, compute_rho_from_ht)[[1]]$rho

MSSA_obj <- MSSA_func(split_grid, L, delta, grid_size, gamma_target, rho0,
                      with_negative_lambda, xi, lower_limit_nu, Sigma,
                      symmetric_target)

# Expected outcome: target correlations are close to zero, reflecting the
# severe trade-off between smoothness and tracking accuracy at the HT boundary.
MSSA_obj$crit_rhoyz

# Extract filter coefficient matrices.
bk_mat_es     <- MSSA_obj$bk_mat       # M-SSA applied to innovations.
bk_x_mat_es   <- MSSA_obj$bk_x_mat    # M-SSA applied to observed series.
gammak_mse_es <- MSSA_obj$gammak_mse  # M-MSE applied to innovations.



# Plot filter weights for each target series.

# Panel 1: series 1
mplot<-cbind(bk_x_mat_es[1:L,1],bk_x_mat_es[L+1:L,1],bk_x_mat_es[2*L+1:L,1])
colnames(mplot)<-c("Series 1","Series 2","Series 3")
colo<-c("blue","red","green","violet","black")
par(mfrow=c(1,3))
plot(mplot[,1],main="SSA: first target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Panel 2: series 2
mplot<-cbind(bk_x_mat_es[1:L,2],bk_x_mat_es[L+1:L,2],bk_x_mat_es[2*L+1:L,2])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: second target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# Panel 3: series 3
mplot<-cbind(bk_x_mat_es[1:L,3],bk_x_mat_es[L+1:L,3],bk_x_mat_es[2*L+1:L,3])
colnames(mplot)<-c("Series 1","Series 2","Series 3")

plot(mplot[,1],main="SSA: third target",axes=F,type="l",xlab="Lag-structure",
     ylab="filter-weights",ylim=c(min(mplot),max(na.exclude(mplot))),
     col=colo[1],lwd=1)
mtext(colnames(mplot)[1],col=colo[1],line=-1)
for (i in 2:ncol(mplot))
{
  #  lines(mplot[,i]-ifelse(i==ncol(mplot),0.3,0),col=colo[i],lwd=1)
  lines(mplot[,i],col=colo[i],lwd=1)
  mtext(colnames(mplot)[i],col=colo[i],line=-i)
}
abline(h=0)
axis(1,at=1:nrow(mplot),labels=-1+1:nrow(mplot))
axis(2)
box()

# ─────────────────────────────────────────────────────────────────────────────
# Technical Note: Smoothers at the HT Boundary
# ─────────────────────────────────────────────────────────────────────────────
# When the imposed HT approaches the upper limit (L+1), rho0 approaches the 
# largest eigenvalue of the
# system matrix M. In this limiting regime, the optimal smoother converges to
# the eigenvector associated with the maximum eigenvalue of M; see Wildi
# (2026b), Section 3.
M_obj   <- M_func(L, Sigma)

M<-M_obj$M
eigen_obj <- eigen(M)
# Largest eigenvalue of M (upper bound on achievable rho0):
max(eigen_obj$values)
# Imposed rho0 in the above example (close to the maximum eigenvalue):
rho0

# Plot the eigenvector corresponding to the maximum eigenvalue of M.
# This vector defines the theoretically smoothest possible filter of length L,
# and the extreme-smoothing M-SSA coefficients should approximate it in the 
# above plot.
par(mfrow = c(1, 1))
ts.plot(eigen_obj$vectors[, 1],
        main = paste("Smoothest admissible filter of length ", L,
                     " (eigenvector of max eigenvalue of M)", sep = ""),
        xlab = "Lag", ylab = "")



# ─────────────────────────────────────────────────────────────────────────────
# Main Take-Aways
# ─────────────────────────────────────────────────────────────────────────────
# M-SSA smoothing generalizes SSA-smoothing (tutorial 8) to multivariate 
# frameworks






