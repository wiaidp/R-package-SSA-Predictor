M-SSA Tutorial — GitHub Project



Overview:

M-SSA (Multivariate Smooth Sign Accuracy) provides a unified framework for solving general prediction problems while simultaneously accommodating specific, practically relevant research priorities and objectives.



The M-SSA Tutorial is an R-based project comprising a collection of exercises and case studies designed to introduce users to — and provide hands-on experience with — the M-SSA framework.



\####

Author: Marc Wildi — https://marcwildi.com



Repository: https://github.com/wiaidp/R-package-SSA-Predictor



Background (references \& links): https://github.com/wiaidp/R-package-SSA-Predictor/about



\####

Project Structure:



The project directory is organized into sub-folders:



1.Data

2.M-SSA Tutorials

3.Papers

4.R (repository of SSA, I-SSA and M-SSA optimization)

5.R utility functions 

6.Results



\####



Getting started: Open the R project by clicking the project icon located in the main repository. This will launch the project in RStudio. From there, load any tutorial file from the `M-SSA Tutorials' sub-folder and run the code. Tutorials are arranged in order of increasing complexity.





\################################################################################################################

Background:

\################################################################################################################



M-SSA is built to address the prediction ATS Trilemma. Forecasting inherently involves three partly competing goals:



I. Accuracy — correctly predicting future levels or signs



II. Timeliness — avoiding undue delays or premature signals



III. Smoothness — suppressing spurious noise and erratic fluctuations



Together, these constitute the ATS Trilemma.





\####

Optimization Principles:

M-SSA addresses each dimension of the trilemma as follows:



Accuracy: optimized via sign accuracy and MSE (mean-squared error).



Smoothness: enforced by imposing a constraint on the mean duration between consecutive sign changes in the predictor (less sign changes means smoother).



Timeliness: can be incorporated as an additional consideration through the choice of forecast horizon, though it remains secondary to the core optimization framework.



M-SSA is Specialized to the AS-Dilemma (with possible extension to the ATS Trilemma, discussed in the tutorials).



\####

Why Zero-Crossings (Sign Changes) Matter:

In many applications, zero-crossings serve as markers of significant events, triggering decisions or interventions by analysts, decision-makers, or market participants.



a. Algorithmic trading

Automated strategies frequently rely on zero-crossings of filtered financial time series to trigger market orders (buy low, sell high)



b. Recession indicators

Turning-point detection often hinges on zero-crossings of a filtered macroeconomic series or composite aggregate, marking the onset or end of a recessionary episode.



c. Business cycle analysis (BCA)

BCA identifies a broadly regular, recurrent economic cycle alternating between expansion and contraction phases. Phase transitions — and thus the timing of anti-cyclical policy responses — occur precisely at zero-crossings of the cycle.



d. Industrial process control

Monitoring problems are often framed as a filter exceeding a threshold and readily recast as a zero-crossing problem.



Note:

In contrast to methods that depend exclusively on the sign of observations (e.g., logit), M-SSA utilizes fully observed interval-scaled data to address signs, resulting in greater efficiency (see a corresponding tutorial).



\####

Efficient Frontier and Pareto Optimality:

In an M-SSA-optimized predictor, any gain in sign accuracy inevitably incurs a higher rate of zero-crossings — and vice versa. There is no free lunch.



This trade-off is a direct consequence of M-SSA residing on the efficient frontier of the Accuracy-Smoothness (AS) Dilemma (Pareto optimality):



Classical Max-Likelihood predictors represent a single point on this frontier.



M-SSA extends the solution space to the full frontier, offering a richer and more flexible set of forecasting solutions.



\####

What Makes M-SSA Distinctive:



A. Generality: classical linear forecasting methods emerge as special cases, which can then be refined within M-SSA to reflect specific research priorities and objectives (customization). The tutorial proposes customization of Hodrick-Prescott, Hamilton, Christiano-Fitzgerald, refined Beveridge-Nelson filter designs as well as of ARMA/VARMA forecasting.



B. Interpretability: optimization criteria are grounded in clear, fundamental principles, yielding solutions that are uniquely determined and straightforward to communicate.



C. Transparency: unlike black-box methods, M-SSA provides a direct window into the forecasting mechanism. Optimization is fast, numerically stable, and leads to unique solutions.



These qualities make M-SSA especially well-suited for settings where opacity is either prohibited — such as compliance-driven or regulatory environments — or simply undesirable, such as when a deeper understanding of the underlying forecasting logic is required.





\####

Alternative Prediction Packages:

The author proposes the following complementary R-based prediction frameworks (https://marcwildi.com):



1. MDFA: https://github.com/wiaidp/MDFA-tutorial. MDFA is a generic prediction approach addressing the full ATS trilemma in the frequency domain. Accuracy, smoothness, and timeliness are derived from and defined on the amplitude and phase characteristics of the predictor filter.



2\. Look-Ahead DFP/PCS: tutorial in preparation. DFP/PCS specializes in the Accuracy-Timeliness trade-off, targeting applications where the cost of delay is particularly significant.



\####

Positioning of the Approaches:



\-MDFA is the most general, all-round framework — operating in the frequency domain and addressing all three ATS dimensions simultaneously.



\-M-SSA and DFP/PCS are formulated in the time domain and specialize in targeted aspects of the prediction problem, making them more focused than MDFA.



\-Their domain-specific optimization principles may offer greater intuitive appeal than frequency-domain statistics for certain users and applications.



\-Current research aims at unifying M-SSA and DFP/PCS to combine their respective strengths into a single, more comprehensive time-domain prediction framework.

