# R-package_Simple_Sign_Accuracy_SSA_Predictor
 
SSA-predictor (Simple Signa Accuracy): this is a generalization of the classic mean-square error (MSE) forecast paradigm 
·	It can replicate classic one-step ahead, multi-step ahead forecasting; 
·	It can address signal extraction: backcasting, nowcasting, forecasting of `signals’ (cycles, trends, …)
·	It generalizes the classic MSE forecast approach by providing explicit control on smoothness and (to some extent) timeliness issues
o	Smoothness: controlling the rate of false alarms (for example fewer noisy alarms than a selected benchmark)
o	Timeliness: advancement/lead or retardation/lag of filter output relative to a target/benchmark
The R-package contains tutorials on the cited applications. In particular, SSA is applied to Hodrick-Prescott, Baxter-King and Hamilton filters for extracting a cyclical component from data: US GDP, Non-Farm Payroll, Industrial Production. 
·	Take control of the zero-crossings of the cycle (less noisy alarms)
·	Take control of the lag or retardation of one-sided causal concurrent real-time nowcasts

There should be three folders in this package: 
·	Data 
·	R: functions for computing the SSA criterion 
·	SSA Tutorials: sample R-code on various topics (introductory examples, forecasting, signal extraction with HP, BK and HF filters)
There’s also an R-project file called SSA_package
·	Open the R-project SSA_package (in Rstudio)
o	Doing so sets the paths to the folders
·	Once the project is opened (in Rstudio): load any of the tutorials from the folder `SSA Tutorials’
o	Depending on the tutorial one has to add additional packages (so called libraries) such as xts, mFilter or packages for loading/updating the data: 
o	When all packages/libraries are loaded one can work through the tutorials
o	Start with lowest numbers (of tutorials)
Experience


