# HamiltonFilter_Restricted(x, p, h, Diff = FALSE)
# This is the main function for estimating Hamilton filters on a single series.
#
# SYNTAX:
#   x     a T x 1 object containing the single series to filter.
#         Ideally an xts object, but should work with any numeric type.
#   p     Integer > 0   Number of lags in the AR forecasting model.
#   h     Integer > 0   Number of periods ahead to forecast
#   Diff  FALSE (default) runs the original Hamilton filter.
#         TRUE  imposes the restriction that the sum of the AR coefficients = 1.
#               This ensures a stationary cycle when x is I(1).
#
# VALUE: a list with elements
#   cycle_xts     (T-p-h) x 1  the estimated cycle
#   trend_xts     (T-p-h) x 1  the estimated trend (x - cycle-xts)
#   coefficients  the p+1 estimated AR coefficients
#   filter_kernel the p+h implied filter weights
#   verification  text giving the sum of the AR coefficients.

HamiltonFilter_Restricted <- function(x, p, h, Diff = FALSE) {
  
  # --- Step 1: Handle Input Type and Extract Core Data ---
  is_xts <- inherits(x, "xts")
  if (is_xts) {
    original_index <- index(x)
    x_numeric <- as.vector(coredata(x))
  } else {
    x_numeric <- x
    original_index <- seq_along(x)
  }
  
  # --- Input Validation ---
  if (!is.numeric(x_numeric) || !is.vector(x_numeric)) stop("x must be a numeric vector.")
  if (!is.numeric(p) || p < 1 || p != as.integer(p)) stop("p must be a positive integer.")
  if (!is.numeric(h) || h < 1 || h != as.integer(h)) stop("h must be a positive integer.")
  if (length(x_numeric) < p + h) stop(paste0("Length of x must be at least ", p + h, "."))
  
  len <- length(x_numeric)
  
  # --- Part 2: Standard Data Setup (Levels) ---
  
  y <- x_numeric[(p + h):len]
  
  X_embed <- embed(x_numeric, p)
  Xt_lags_raw <- X_embed[1:(nrow(X_embed) - h), , drop = FALSE]
  Xt_full <- cbind(Intercept = 1, Xt_lags_raw)
  
  output_index <- original_index[(p + h):len]
  
  # --- Part 3: Estimation (Conditional Logic) ---
  
  if (Diff == FALSE) {
    # Scenario A: Unrestricted OLS
    lm_obj <- lm.fit(x = Xt_full, y = y)
    final_coeffs <- coef(lm_obj)
    
  } else {
    # Scenario B: Restricted Least Squares (Sum of lags = 1.0)
    
    # 1. Transform Dependent Variable: y* = y - x_{t-p+1}
    last_lag_col_idx <- p + 1
    last_lag_vec <- Xt_full[, last_lag_col_idx]
    y_restricted <- y - last_lag_vec
    
    # 2. Transform Design Matrix
    # CRITICAL CHECK FOR p=1
    # If p=1, Xt_restricted is JUST the intercept.
    # If p>1, Xt_restricted is Intercept + (Lags - LastLag)
    
    Xt_restricted <- Xt_full[, 1:p, drop = FALSE]
    
    if (p > 1) {
      # This loop creates the transformed columns: x_{t-j+1} - x_{t-p+1}
      # It ONLY runs if there are actually lags to transform (p >= 2)
      for (j in 2:p) {
        Xt_restricted[, j] <- Xt_full[, j] - last_lag_vec
      }
    }
    # If p=1, the loop is skipped. Xt_restricted remains a single column of 1s.
    # The matrix is Non-Singular (Rank 1).
    
    # 3. Perform Regression
    lm_obj_restr <- lm.fit(x = Xt_restricted, y = y_restricted)
    
    # 4. Reconstruct Coefficients
    coeffs_est <- coef(lm_obj_restr)
    beta_0 <- coeffs_est[1]
    
    if (p > 1) {
      beta_1_to_p_minus_1 <- coeffs_est[2:p]
    } else {
      # If p=1, there are no estimated betas (besides intercept)
      beta_1_to_p_minus_1 <- numeric(0)
    }
    
    # beta_p absorbs the residual weight to ensure sum is 1.0
    beta_p <- 1 - sum(beta_1_to_p_minus_1)
    
    final_coeffs <- c(beta_0, beta_1_to_p_minus_1, beta_p)
  }
  
  # --- Part 4: Calculation of Trend and Cycle ---
  
  lag_names <- paste0("lag_", 1:p)
  names(final_coeffs) <- c("Intercept", lag_names)
  
  # Trend = Xt * Beta (Restricted or Unrestricted)
  trend_vals <- as.vector(Xt_full %*% final_coeffs)
  cycle_vals <- y - trend_vals
  
  # --- Part 5: Filter Kernel Construction ---
  
  p_params <- final_coeffs[2:(p + 1)]
  implied_cycle_filter <- c(1, rep(0, h - 1), -p_params)
  
  # --- Step 6: Format Output ---
  
  if (is_xts) {
    cycle_xts <- xts(cycle_vals, order.by = output_index)
    trend_xts <- xts(trend_vals, order.by = output_index)
  } else {
    cycle_xts <- cycle_vals
    trend_xts <- trend_vals
    names(cycle_xts) <- as.character(output_index)
    names(trend_xts) <- as.character(output_index)
  }
  
  sum_lags <- sum(p_params)
  verif_msg <- paste0("Sum of lag coefficients: ", round(sum_lags, 10))
  
  return(list(
    cycle_xts = cycle_xts,
    trend_xts = trend_xts,
    coefficients = final_coeffs,
    filter_kernel = implied_cycle_filter,
    verification = verif_msg
  ))
}


