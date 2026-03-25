# ROCplots : a function for comparing AUCs and ROC plots
#   by Simon van Norden, HEC Montreal
#   19-05-2025
#
# SYNTAX
#   
#   auc <- ROCplots(df, showROC = TRUE, smoothROC = FALSE, showLegend = TRUE)
#
# where
#   df  is a data.frame of series to analyse. See below for format.
#   showROC       [FALSE] show the plot comparing ROC curves?
#     main        [NULL]  title for plot
#     lbls                X and Y axis labels. See below for options.
#     smoothROC   [FALSE] smooth the plotted ROC curves? (Changes AUCs!)
#     colours             user-supplied colours for curves
#     lwd         [2]     line width for curves
#     showLegend  [TRUE]  include a legend in the ROC plots?
#       lg_cex    [1]     additional scaling factor for legends
#       lg_ncol   [1]     number of columns in legend
#
# VALUE
#
#   auc   a 1-column data.frame containing the AUC for each predictor in df.
#
# NOTES
#
# The data.frame df should contain
# - the target series in column 1, consisting only of 0s and 1s
# - an arbitrary number of additional numeric series in the remaining columns
# All series should be the same length and contain no NAs
#
# There are various options for axis labels
#   lbls = NA         no X- or Y-axis labels
#   lbls = "Hit"      Y-axis is "Hit Rate", X-axis is "False Alarm Rate"
#                     This does not change the shape of the curves. However,
#                     the X-axis will run from 0 to 1 instead of 1 to 0, and 
#                     the x-data plotted will be 1-ROC$specificities instead of
#                     ROC$specificities.
#       All other values use the default label and x-axis settings



ROCplots <- function(df, showROC = TRUE, main = NULL, lbls = "roc",
                     smoothROC = FALSE, colours = NULL, lwd = 2,
                     showLegend = TRUE, lg_cex = 1, lg_ncol = 1) {
  df_names <- names(df) # series names
  k <- length(df_names) # 1 + number of predictors to compare
  print(paste("Target variable is", df_names[1]))
  # Create table to hold AUCs
  aurocs <- as.data.frame(matrix(NA, nrow = k-1))
  row.names(aurocs) <- df_names[-1]
  names(aurocs) <- "AUC"
  
  # make the first plot and store the first AUC
  # ROC_OBJ <- roc_(df, response = df_names[1],
  #                 predictor = df_names[2],
  #                 quiet = T, plot = showROC,
  #                 smooth = smoothROC)
  # aurocs[1,1] <- ROC_OBJ$auc
  
  ROC_OBJ <- roc_(df, response = df_names[1],
                  predictor = df_names[2],
                  quiet = T, plot = F,
                  smooth = smoothROC)
  # Store the AUC
  aurocs[1,1] <- ROC_OBJ$auc
  
  if (showROC) {
    # preliminaries for all plots
    xlim = c(1,0)
    a_line = 1
    b_line = -1
    xlbl = "Specificity"
    ylbl = "Sensitivity"
    # User-defined colours
    if (is.null(colours)) colours = 1:(k-1)
    
    # reset axes and their labels if needed
    if (lbls == 'Hit') {
      xlbl <- 'False Alarm Rate'
      ylbl <- "Hit Rate"
      xlim = c(0,1)
      a_line = 0
      b_line = 1
    } else if (lbls == ""){
      xlbl <- ""
      ylbl <- ""
    }

    # plot the 1st ROC
    # plotting specificity or False Alarms?
    specifty <- ROC_OBJ$specificities
    if(lbls == "Hit") specifty <- 1 - specifty
    
    plot(specifty, 
         ROC_OBJ$sensitivities,
         xlim = xlim, ylim = c(0,1), 
         type = "l", lwd = lwd, 
         col = colours[1],
         xlab = xlbl, ylab = ylbl,
         main = main, cex = lg_cex)
    abline(a = a_line, b = b_line, lty = 3)
  }
  if (k>2) {
  # Calculate and plot for the rest of the series
    for (j in 3:k) {
      ROC_OBJ <- roc_(df, response = df_names[1], 
                      predictor = df_names[j], 
                      quiet = T, plot = F,
                      smooth = smoothROC)
      aurocs[j-1,1] <- ROC_OBJ$auc
      # Plot the ROC?
      if (showROC) {
        # plotting specificity or False Alarms?
        specifty <- ROC_OBJ$specificities
        if(lbls == "Hit") specifty <- 1 - specifty
        
        lines(specifty,
              ROC_OBJ$sensitivities,
              col = colours[j - 1],
              lwd = lwd)
      }
    }
  }
  
  if (showROC & showLegend) legend("bottomright", 
                                   legend = df_names[-1], 
                                   col = colours, 
                                   lwd = lwd,
                                   cex = lg_cex,
                                   ncol = lg_ncol)
  
  return(aurocs)
}
