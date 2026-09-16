# ============================================================
# Boosted HP filter coefficients (two-sided and one-sided)
# ============================================================
#
# Standard HP smoother matrix (for a series of length n):
#   S_lambda = [I + lambda * D'D]^{-1}
# where D is the second-difference operator.
#
# L-boosted HP (m iterations):
#   S_lambda^(m) = I - (I - S_lambda)^m
#
# Two-sided coefficients: extracted from a row in the middle
#   of the matrix (mid-sample, symmetric filter, truncated to
#   length L on each side).
#
# One-sided coefficients: extracted from the LAST row of the
#   matrix (real-time / end-of-sample filter), truncated to
#   length L (i.e., using only the most recent L observations).
# ============================================================

boosted_hp_filter <- function(lambda, m, L, n = NULL, buffer = NULL) {
  # lambda : HP smoothing parameter
  # m      : number of boosting iterations
  # L      : desired truncation length for the filter coefficients
  # n      : size of the underlying matrix used for computation
  #          (must be >> L to avoid boundary effects for two-sided
  #          coefficients, and >= L+1 for one-sided)
  # buffer : extra padding added to n to reduce edge effects;
  #          defaults to a generous multiple of L
  
  if (is.null(buffer)) buffer <- max(50, 5 * L)
  if (is.null(n))      n <- 2 * L + 1 + 2 * buffer
  
  # ---- Build second-difference operator D (n-2) x n ----
  D <- matrix(0, n - 2, n)
  for (i in 1:(n - 2)) {
    D[i, i]     <- 1
    D[i, i + 1] <- -2
    D[i, i + 2] <- 1
  }
  
  I_n <- diag(n)
  
  # ---- Standard HP smoother matrix ----
  S_lambda <- solve(I_n + lambda * t(D) %*% D)
  
  # ---- Boosted HP smoother matrix: S^(m) = I - (I - S)^m ----
  IminusS <- I_n - S_lambda
  
  # (I - S_lambda)^m via repeated matrix multiplication
  IminusS_m <- diag(n)
  if (m > 0) {
    for (j in 1:m) {
      IminusS_m <- IminusS_m %*% IminusS
    }
  }
  S_boosted <- I_n - IminusS_m
  
  # ---- Extract two-sided (mid-sample) coefficients ----
  mid <- ceiling(n / 2)
  two_sided_full <- S_boosted[mid, ]
  # Coefficients relative to the center, from -L to +L
  idx_range <- (mid - L):(mid + L)
  two_sided <- two_sided_full[idx_range]
  names(two_sided) <- -L:L
  
  # ---- Extract one-sided (end-of-sample) coefficients ----
  # Row n = filter applied at the last available time point,
  # using only current and past data (causal / real-time filter)
  one_sided_full <- S_boosted[n, ]
  # Keep only the most recent L+1 coefficients (lag 0 to L)
  one_sided <- rev(one_sided_full[(n - L):n])
  names(one_sided) <- 0:L
  
  list(
    two_sided = two_sided,
    one_sided = one_sided,
    S_boosted = S_boosted   # returned in case full matrix is useful
  )
}

# Faster version for large m using eigendecomposition
boosted_hp_filter_fast <- function(lambda, m, L, n = NULL, buffer = NULL) {
  if (is.null(buffer)) buffer <- max(50, 5 * L)
  if (is.null(n))      n <- 2 * L + 1 + 2 * buffer
  
  D <- matrix(0, n - 2, n)
  for (i in 1:(n - 2)) {
    D[i, i] <- 1; D[i, i+1] <- -2; D[i, i+2] <- 1
  }
  I_n <- diag(n)
  S_lambda <- solve(I_n + lambda * t(D) %*% D)
  IminusS <- I_n - S_lambda
  
  eig <- eigen(IminusS, symmetric = TRUE)
  IminusS_m <- eig$vectors %*% diag(eig$values^m) %*% t(eig$vectors)
  S_boosted <- I_n - IminusS_m
  
  mid <- ceiling(n / 2)
  two_sided <- S_boosted[mid, (mid - L):(mid + L)]
  names(two_sided) <- -L:L
  
  one_sided <- rev(S_boosted[n, (n - L):n])
  names(one_sided) <- 0:L
  
  list(two_sided = two_sided, one_sided = one_sided)
}


# Notes on the implementation:
  
#  Matrix-based, exact approach. Rather than deriving the residue/partial-fraction expansion symbolically, 
#  this computes $S_\lambda^{(m)} = I-(I-S_\lambda)^m$ directly as a matrix operation. This is exact 
#  (up to floating-point precision) and avoids the tedious algebra of repeated poles discussed earlier, 
#  at the cost of being $O(n^3)$ for the matrix inversion — fine for moderate $n$ (a few hundred to ~1000).



# Truncation/edge-effect control via buffer. Since we need a finite matrix to represent an infinite (or very long) filter, 
# coefficients extracted from rows too close to the top/bottom of the matrix will be contaminated by boundary effects. 
# The buffer argument pads the matrix so that:
  
  
# The row used for the two-sided filter is far from both ends (mimicking the doubly-infinite/mid-sample filter), and
# the row used for the one-sided filter is at the very last row of a sufficiently long matrix on its left side (mimicking a long history), but note this one-sided filter still implicitly reflects an infinite/very long causal history — if you want the "genuine" end-of-sample filter for a short series of length $n_0$, just set n = n0 directly and extract row n0 without extra buffering (see below).


# Two variants of "one-sided" filter:
  
# As coded above: approximates the steady-state one-sided filter (as if the sample were very long), which is what's 
# typically meant by "the" one-sided boosted HP filter, comparable to the Kalman-filter steady-state gain.

# If you instead want the one-sided filter for a specific finite sample length $n_0$ (i.e., exact end-of-sample behavior for a 
# short series), simply call:

#     res_finite <- boosted_hp_filter(lambda, m, L = n0 - 1, n = n0, buffer = 0)
#     one_sided_finite <- res_finite$one_sided
     
# This directly uses row $n_0$ of the $n_0\times n_0$ boosted-smoother matrix, which is the *exact* filter applied at 
# the end of a sample of length $n_0$ (no steady-state approximation).


# Truncation length $L$. Since boosted HP filters typically decay geometrically (with possibly polynomial-in-$k$ prefactors, 
# as derived analytically earlier), truncation at moderate $L$ (e.g., 20–50) should capture the vast majority of the filter's 
# mass unless $\lambda$ is very large or $m$ is large (both of which slow decay).



# Performance for large $m$: the loop computing $(I-S_\lambda)^m$ via repeated multiplication is $O(m \cdot n^3)$. 
# For large $m$, consider diagonalizing $I-S_\lambda$ once (eigen()) and computing powers via eigenvalues raised to the 
# $m$-th power, which is much faster.
  
  