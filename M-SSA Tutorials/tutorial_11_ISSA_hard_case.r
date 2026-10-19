
# This tutorial is under construction

# Tutorial 11: I-SSA and the limiting hard-case of the GTRS problem structure 

# We illustrate and analyze the subtle cancellation of singularities when the 
# Lagrangian parameter approaches or hits the spectral boundary of the Hessian: 
# the lower bound of the region of strict positivity of H=A'A+lambda Q.

# See Wildi 2026c



# ---------------------------------------------------------
# Illustrative Example: Objective vs Autocorrelation Trade-off
# ---------------------------------------------------------

# Set up the problem

# 1. Setup dimensions and matrices
L <- 10

# Summation operator Sigma (lower triangular with 1s)
Sigma <- matrix(0, L, L)
Sigma[lower.tri(Sigma, diag = TRUE)] <- 1

t(Sigma)%*%Sigma

# Stationary MA(1) operator Xi (Toeplitz)
theta <- 0.5
Xi <- matrix(0, L, L)
diag(Xi) <- 1
for(i in 2:L) Xi[i, i-1] <- theta

# Matrix M for lag-1 autocorrelation
M <- matrix(0, L, L)
for(i in 1:(L-1)) {
  M[i, i+1] <- 0.5
  M[i+1, i] <- 0.5
}
VM<-eigen(M)$vectors

# This leads to eigenvector 9
rho_1<-0.8
# This leads to eigenvector 10
rho_1<-0.9
# This leads to eigenvector 10
rho_1<-0.95
rho_1<-eigen(M)$values[1]-0.00001
if (F)
{
  rho_1<--eigen(M)$values[1]+0.00001
  rho_1 <- 0.500       # Target lag-1 autocorrelation
}
# Construct A and Q
A <- Sigma %*% Xi
Q <- t(Xi) %*% (M - rho_1 * diag(L)) %*% Xi

# Find region of invertibility where is Hessian strictly positive
eigenval<-eigen(solve(t(A)%*%A)%*%Q)$values
V<-eigen(solve(t(A)%*%A)%*%Q)$vector
range<-c(-1/max(eigenval),-1/min(eigenval))

# Target vector gamma (e.g., a simple linear trend)
a1<-0.5
# AR(1) target
set.seed(12)
gamma<-rnorm(L)
gamma<-a1^(1:L)/sum(a1^(1:L))



if (F)
{
  gamma <- seq(0, 1, length.out = L)
}


Gamma_0 <- sum(gamma)       # Sum constraint

# Compute effective target in MSE
set.seed(1)
gamma_target<-rnorm(L)
gamma_target<-Sigma%*%Xi%*%gamma

# Select a lambda close to -1/nu_max
lambda<-min(range)-.00000001

H <-  lambda * Q
H <- t(A) %*% A + lambda * Q
det(H)

t(V)%*%H%*%V

t(V)%*%V

H_inv <- solve(H)
idinv<-solve(V)%*%H_inv%*%solve(t(V))
idinv

  
# Calculate mu
num <- sum(H_inv %*% t(A) %*% gamma_target) - Gamma_0
den <- sum(H_inv)
mu <- num / den
  
# Calculate g
g<-t(A) %*% gamma_target - mu * rep(1, L)
# Find largest eigenvalue of H^-1
k_max<-which(abs(diag(idinv))==max(abs(diag(idinv))))
t(g)%*%V[,k_max]
ts.plot(V[,k_max])

t(g)%*%solve(Xi)%*%VM[,1]

solve(Xi)%*%VM[,k_max]/V[,k_max]

idinv%*%t(V)%*%g

# More Penrose inverse
mp_indinv<-idinv
mp_indinv[k_max,k_max]<-0
mp_H_inv<-V%*%mp_indinv%*%t(V)
# Check cointegration constraint
sum(mp_H_inv%*%g)
sum(H_inv%*%g)

ts.plot(cbind(mp_H_inv%*%g,H_inv%*%g))
ts.plot(scale(cbind(mp_H_inv%*%g,H_inv%*%g)))

b_mp<-mp_H_inv%*%g
b<-H_inv%*%g

# b is linear combination of b_mp and V[,k_max]
summary(lm(b~b_mp+V[,k_max]))

b_eps<-Xi%*%H_inv%*%g
ts.plot(b_eps)

# Note: at lower boundary the Rayleigh coefficient is typically larger than rho1
t(b_eps)%*%M%*%b_eps/t(b_eps)%*%b_eps



# Calculate optimal x and filtered y
b_x <- H_inv %*% g
b_eps <- Xi %*% b_x
ts.plot(b_eps) 
t(b_eps)%*%M%*%b_eps/t(b_eps)%*%b_eps


ts.plot(b_x)
ts.plot(V[,L])
b_x/V[,L]

# Xi%*%V[,L] converges to VM[,L]
ts.plot(Xi%*%V[,L])
Xi%*%V[,L]/VM[,1]
V[,L]/solve(Xi)%*%VM[,1]


t(g)%*%V[,L]
t(g)%*%VM[,L]


########################################################################################
########################################################################################
# Example 1
# This leads to the generalized eigenvector V[,9]
rho_1<-0.8
# Construct A and Q
A <- Sigma %*% Xi
Q <- t(Xi) %*% (M - rho_1 * diag(L)) %*% Xi

# Find region of invertibility where is Hessian strictly positive
eigenval<-eigen(solve(t(A)%*%A)%*%Q)$values
V<-eigen(solve(t(A)%*%A)%*%Q)$vector
range<-c(-1/max(eigenval),-1/min(eigenval))

# Select a lambda close to -1/nu_max
lambda<-min(range)-.00000001
# abs(lambda) is not excessively large yet
abs(lambda)

# If lambda is large in absolute value we can approximate t(A) %*% A + lambda * Q by lambda * Q
# In this example abs(lambda) is not very large
H <-  lambda * Q
H <- t(A) %*% A + lambda * Q
H_inv <- solve(H)
idinv<-solve(V)%*%H_inv%*%solve(t(V))
# For rho1=0.8 we see that the last but one (the 9.th) g-eigenvalue  (diagonal element of idinv) is large
# This is because rho1=0.8 is not excessively smooth, so emphasizing the 9-th (instead of 10th) eigenvector is OK.  
# Note: the generalized eigenvalues of the pencil are independent of lambda (they depend on rho1)
# Using the generalized eigenvectors for the spectral decomposition leads to g-eigenvalues of H
# g-eigenvalues depend on lambda
# g-eigenvalues differ from the classic eigenevalues of H based on classic orthogonal spectral decomposition
idinv
# Note: if rho_1 < rho_max (the largest eigenvalue of M), then min(range) is finite (bounded).
# Therefore, lambda in range (range is J_{++}, the region of strict monotonicity of the Hessian) is bounded.
# Consequently, when lambda approaches the lower boundary min(range), the corresponding g_eigenvalue in the inverse Hessian is unbounded
# This result is a bit more tricky when rho_1 to rho_max since then min(range) and thus lambda diverge.
# But the limiting case shows that this also holds at the boundary, i.e., the largest g-eigenvalue diverges.
# This can be tweaked in our code by selecting, e.g., lambda<-min(range)-10^(-10): lambda is 
# closer to min(range) than rho_1 to rho_max.

# Next step: Calculate mu
# This is a crucial step because max g-eigenvalue (here 9th) of H diverges
# Therefore mu `acts' a bit like a scaling, so that the I-SSA solution  becomes MSE optimal
# However, this scaling applies different weights to the generalized eigenvectors. These weights 
# correspond to the spectral weights of the target. In general the spectral weight 
num <- sum(H_inv %*% t(A) %*% gamma_target) - Gamma_0
den <- sum(H_inv)
mu <- num / den

# Calculate g
g<-t(A) %*% gamma_target - mu * rep(1, L)
# Find largest eigenvalue of H^-1
# For rho1=0.8 this is 9
k_max<-which(abs(diag(idinv))==max(abs(diag(idinv))))
k_max
# Recall that 9th eigenvalue was very large: diverges as lambda converges to min(range)
# Accordingly, the spectral weight t(g)%*%V[,k_max] converges to zero since otherwise b would grow unboundedly
t(g)%*%V[,k_max]
# Here's the generalized eigenvector: it is `less smooth' than the last eigenvector
ts.plot(V[,k_max])


# More Penrose inverse
mp_indinv<-idinv
# Replace largest eigenavlue by zero
mp_indinv[k_max,k_max]<-0
mp_H_inv<-V%*%mp_indinv%*%t(V)
# Check cointegration constraint: the cointegration constraint is not satisfied, i.e.,
# V[,k_max] is necessary. The MSE optimal estimate assigns weight to all generalized eigenvectors.
sum(mp_H_inv%*%g)
# Cointegration constraint is met for full inverse
sum(H_inv%*%g)
# Here we see the difference: with and without V[,k_max]
ts.plot(cbind(mp_H_inv%*%g,H_inv%*%g),col=c("blue","black"),main="More Penrose (blue) vs. Full Inverse")
ts.plot(scale(cbind(mp_H_inv%*%g,H_inv%*%g)),col=c("blue","black"),main="Same as above but scaled")

# Compute solution
# Note: if lambda is close to min(range), then b_eps is typically smoother than imposed by rho_1 
b_eps<-Xi%*%H_inv%*%g
ts.plot(b_eps)
# Note: if lambda is close to min(range) then the Rayleigh coefficient is typically larger than rho1
rho_1
t(b_eps)%*%M%*%b_eps/t(b_eps)%*%b_eps





########################################################################################
########################################################################################
# Example 2
# This leads to the generalized eigenvector V[,10]
rho_1<-0.9
# Construct A and Q
A <- Sigma %*% Xi
Q <- t(Xi) %*% (M - rho_1 * diag(L)) %*% Xi

# Find region of invertibility where is Hessian strictly positive
eigenval<-eigen(solve(t(A)%*%A)%*%Q)$values
V<-eigen(solve(t(A)%*%A)%*%Q)$vector
range<-c(-1/max(eigenval),-1/min(eigenval))

# Select a lambda close to -1/nu_max
lambda<-min(range)-.00000001
# abs(lambda) is not excessively large yet
abs(lambda)


# If lambda is large in absolute value we can approximate t(A) %*% A + lambda * Q by lambda * Q
# In this example abs(lambda) is not very large
H <-  lambda * Q
H <- t(A) %*% A + lambda * Q
H_inv <- solve(H)
idinv<-solve(V)%*%H_inv%*%solve(t(V))
# For rho1=0.9 we see that the last  g-eigenvalue (diagonal element of idinv) is large
# This is because rho1=0.9 is already large and requires emphasizing the 10-th eigenvector.  
# Note: the generalized eigenvalues of the pencil are independent of lambda (they depend on rho1)
# Using the generalized eigenvectors for the spectral decomposition leads to g-eigenvalues of H
# g-eigenvalues depend on lambda
# g-eigenvalues differ from the classic eigenevalues of H based on classic orthogonal spectral decomposition
idinv
# Note: if rho_1 < rho_max (the largest eigenvalue of M), then min(range) is finite (bounded).
# Therefore, lambda in range (range is J_{++}, the region of strict monotonicity of the Hessian) is bounded.
# Consequently, when lambda approaches the lower boundary min(range), the corresponding g_eigenvalue in the inverse Hessian is unbounded
# This result is a bit more tricky when rho_1 to rho_max since then min(range) and thus lambda diverge.
# But the limiting case shows that this also holds at the boundary, i.e., the largest g-eigenvalue diverges.
# This can be tweaked in our code by selecting, e.g., lambda<-min(range)-10^(-10): lambda is 
# closer to min(range) than rho_1 to rho_max.

# Calculate mu
# This is a crucial step because max g-eigenvalue (here 10th) of H diverges
# Therefore mu `acts' a bit like a scaling, so that the I-SSA solution  becomes MSE optimal
# However, this scaling applies different weights to the generalized eigenvectors. These weights 
# correspond to the spectral weights of the target. In general the spectral weight 
num <- sum(H_inv %*% t(A) %*% gamma_target) - Gamma_0
den <- sum(H_inv)
mu <- num / den

# Calculate g
g<-t(A) %*% gamma_target - mu * rep(1, L)
# Find largest eigenvalue of H^-1
# For rho1=0.9 this is 10
k_max<-which(abs(diag(idinv))==max(abs(diag(idinv))))
k_max
# Recall that 10th eigenvalue was very large: diverges as lambda converges to min(range)
# Accordingly, the spectral weight t(g)%*%V[,k_max] converges to zero since otherwise b would grow unboundedly
# This is the so-called Hard GTRS Case (Generalized Trust Region Subproblem)
t(g)%*%V[,k_max]
# Here's the generalized eigenvector: it is very smooth (though less so than v_max of M)
ts.plot(V[,k_max])


# More Penrose inverse
mp_indinv<-idinv
# Replace largest eigenavlue by zero
mp_indinv[k_max,k_max]<-0
mp_H_inv<-V%*%mp_indinv%*%t(V)
# Check cointegration constraint: the cointegration constraint is not satisfied, i.e.,
# V[,k_max] is necessary. The MSE optimal estimate assigns weight to all generalized eigenvectors.
# here we see that the sum of filter coefficients of the More Penrose solution massively differs from the true value one
sum(mp_H_inv%*%g)
# Cointegration constraint is met for full inverse
sum(H_inv%*%g)
# Here we see the difference: with and without V[,k_max]
# Note: the importance of V[,k_max] is clearly evident since the More Penrose without V[,k_max] looks quite different
ts.plot(cbind(mp_H_inv%*%g,H_inv%*%g),col=c("blue","black"),main="More Penrose (blue) vs. Full Inverse")
ts.plot(scale(cbind(mp_H_inv%*%g,H_inv%*%g)),col=c("blue","black"),main="Same as above but scaled")

# Compute solution
# Note: if lambda is close to min(range), then b_eps is typically smoother than imposed by rho_1 
b_eps<-Xi%*%H_inv%*%g
ts.plot(b_eps)
# Note: if lambda is close to min(range) then the Rayleigh coefficient is typically larger than rho1
rho_1
t(b_eps)%*%M%*%b_eps/t(b_eps)%*%b_eps


########################################################################################
########################################################################################
# Example 3: here we come close to rho_max
# This leads to eigenvector 10 (i.e. larger smoothness shifts the principal generalized eigenvector)
rho_1<-0.951
# Construct A and Q
A <- Sigma %*% Xi
Q <- t(Xi) %*% (M - rho_1 * diag(L)) %*% Xi

# Find region of invertibility where is Hessian strictly positive
eigenval<-eigen(solve(t(A)%*%A)%*%Q)$values
V<-eigen(solve(t(A)%*%A)%*%Q)$vector
range<-c(-1/max(eigenval),-1/min(eigenval))

# Select a lambda close to -1/nu_max
# lambda is now markedly larger in absolute value 
lambda<-min(range)-.00000001
abs(lambda)

# If lambda is large in absolute value we can approximate t(A) %*% A + lambda * Q by lambda * Q
# In this example abs(lambda) is not very large
H <-  lambda * Q
H <- t(A) %*% A + lambda * Q
H_inv <- solve(H)
idinv<-solve(V)%*%H_inv%*%solve(t(V))
# We see that the last  g-eigenvalue (diagonal element of idinv) is large
# This is because rho1=0.951 is close to rho_max and requires emphasizing the 10-th eigenvector.  
# Note: the generalized eigenvalues of the pencil are independent of lambda (they depend on rho1)
# Using the generalized eigenvectors for the spectral decomposition leads to g-eigenvalues of H
# g-eigenvalues depend on lambda
# g-eigenvalues differ from the classic eigenevalues of H based on classic orthogonal spectral decomposition
idinv
# Note: if rho_1 < rho_max (the largest eigenvalue of M), then min(range) is finite (bounded).
# Therefore, lambda in range (range is J_{++}, the region of strict monotonicity of the Hessian) is bounded.
# Consequently, when lambda approaches the lower boundary min(range), the corresponding g_eigenvalue in the inverse Hessian is unbounded
# This result is a bit more tricky when rho_1 to rho_max since then min(range) and thus lambda diverge.
# But the limiting case shows that this also holds at the boundary, i.e., the largest g-eigenvalue diverges 
# For any rho_1<rho_max. This can be tweaked in our code by selecting, e.g., lambda<-min(range)-10^(-10): lambda is 
# closer to min(range) than rho_1 to rho_max.


# Calculate mu
# This is a crucial step because max g-eigenvalue (here 10th) of H diverges
# Therefore mu `acts' a bit like a scaling, so that the I-SSA solution  becomes MSE optimal
# However, this scaling applies different weights to the generalized eigenvectors. These weights 
# correspond to the spectral weights of the target. In general the spectral weight 
num <- sum(H_inv %*% t(A) %*% gamma_target) - Gamma_0
den <- sum(H_inv)
mu <- num / den

# Calculate g
g<-t(A) %*% gamma_target - mu * rep(1, L)
# Find largest eigenvalue of H^-1
# For rho1=0.9 this is 10
k_max<-which(abs(diag(idinv))==max(abs(diag(idinv))))
k_max
# Recall that 10th eigenvalue was very large: diverges as lambda converges to min(range)
# Accordingly, the spectral weight t(g)%*%V[,k_max] converges to zero since otherwise b would grow unboundedly
# This is the so-called Hard GTRS Case (Generalized Trust Region Subproblem)
t(g)%*%V[,k_max]
# Here's the generalized eigenvector: it is smoother than for rho_1=0.9 in exercise 2 but still less smooth than v_max of M
# Note: generalized eigenvalues and eigenvectors depend on rho_1 (unlike stationary SSA, where A'A is missing in H)
ts.plot(V[,k_max])


# More Penrose inverse
mp_indinv<-idinv
# Replace largest eigenavlue by zero
mp_indinv[k_max,k_max]<-0
mp_H_inv<-V%*%mp_indinv%*%t(V)
# Check cointegration constraint: the cointegration constraint is not satisfied, i.e.,
# V[,k_max] is necessary. The MSE optimal estimate assigns weight to all generalized eigenvectors.
# here we see that the sum of filter coefficients of the More Penrose solution massively differs from the true value one
sum(mp_H_inv%*%g)
# Cointegration constraint is met for full inverse
sum(H_inv%*%g)
# Here we see the difference: with and without V[,k_max]
# Note: the importance of V[,k_max] is clearly evident since the More Penrose without V[,k_max] looks quite different
ts.plot(cbind(mp_H_inv%*%g,H_inv%*%g),col=c("blue","black"),main="More Penrose (blue) vs. Full Inverse")
ts.plot(scale(cbind(mp_H_inv%*%g,H_inv%*%g)),col=c("blue","black"),main="Same as above but scaled")

# Compute solution
# Note: if lambda is close to min(range), then b_eps is typically smoother than imposed by rho_1 
b_eps<-Xi%*%H_inv%*%g
ts.plot(b_eps)
# Note: if lambda is close to min(range) then the Rayleigh coefficient is typically larger than rho1
rho_1
t(b_eps)%*%M%*%b_eps/t(b_eps)%*%b_eps


########################################################################################
########################################################################################
# Example 4: here we are very close to rho_max
# This leads to eigenvector 10 (i.e. larger smoothness shifts the principal generalized eigenvector)
rho_1<-eigen(M)$values[1]-0.0000001
# Construct A and Q
A <- Sigma %*% Xi
Q <- t(Xi) %*% (M - rho_1 * diag(L)) %*% Xi

# Find region of invertibility where is Hessian strictly positive
eigenval<-eigen(solve(t(A)%*%A)%*%Q)$values
V<-eigen(solve(t(A)%*%A)%*%Q)$vector
range<-c(-1/max(eigenval),-1/min(eigenval))

# Select a lambda close to -1/nu_max
# lambda is now very large
lambda<-min(range)-.0000000001
abs(lambda)

# If lambda is large in absolute value we can approximate t(A) %*% A + lambda * Q by lambda * Q
# Let us first use the exact formula
H <-  lambda * Q
H <- t(A) %*% A + lambda * Q
H_inv <- solve(H)
idinv<-solve(V)%*%H_inv%*%solve(t(V))
# We see that the last  g-eigenvalue (diagonal element of idinv) is large
# This is because rho1=0.951 is close to rho_max and requires emphasizing the 10-th eigenvector.  
# Note: the generalized eigenvalues of the pencil are independent of lambda (they depend on rho1)
# Using the generalized eigenvectors for the spectral decomposition leads to g-eigenvalues of H
# g-eigenvalues depend on lambda
# g-eigenvalues differ from the classic eigenevalues of H based on classic orthogonal spectral decomposition
idinv
# Note: if rho_1 < rho_max (the largest eigenvalue of M), then min(range) is finite (bounded).
# Therefore, lambda in range (range is J_{++}, the region of strict monotonicity of the Hessian) is bounded.
# Consequently, when lambda approaches the lower boundary min(range), the corresponding g_eigenvalue in the inverse Hessian is unbounded
# This result is a bit more tricky when rho_1 to rho_max since then min(range) and thus lambda diverge.
# But the limiting case shows that this also holds at the boundary, i.e., the largest g-eigenvalue diverges 
# for any rho_1<rho_max. This can be tweaked in our code by selecting, e.g., lambda<-min(range)-10^(-10): lambda is 
# closer to min(range) than rho_1 to rho_max.


# Calculate mu
# This is a crucial step because max g-eigenvalue (here 10th) of H diverges
# Therefore mu `acts' a bit like a scaling, so that the I-SSA solution  becomes MSE optimal
# However, this scaling applies different weights to the generalized eigenvectors. These weights 
# correspond to the spectral weights of the target. In general the spectral weight 
num <- sum(H_inv %*% t(A) %*% gamma_target) - Gamma_0
den <- sum(H_inv)
mu <- num / den

# Calculate g
g<-t(A) %*% gamma_target - mu * rep(1, L)
# Find largest eigenvalue of H^-1
# For rho1=0.9 this is 10
k_max<-which(abs(diag(idinv))==max(abs(diag(idinv))))
k_max
# Recall that 10th eigenvalue was very large: diverges as lambda converges to min(range)
# Accordingly, the spectral weight t(g)%*%V[,k_max] converges to zero since otherwise b would grow unboundedly
# This is the so-called Hard GTRS Case (Generalized Trust Region Subproblem)
t(g)%*%V[,k_max]
# Here's the generalized eigenvector: it is smoother than for rho_1=0.9 in exercise 2 but still less smooth than v_max of M
# Note: generalized eigenvalues and eigenvectors depend on rho_1 (unlike stationary SSA, where A'A is missing in H)
ts.plot(V[,k_max])


# More Penrose inverse
mp_indinv<-idinv
# Replace largest eigenavlue by zero
mp_indinv[k_max,k_max]<-0
mp_H_inv<-V%*%mp_indinv%*%t(V)
# Check cointegration constraint: the cointegration constraint is not satisfied, i.e.,
# V[,k_max] is necessary. The MSE optimal estimate assigns weight to all generalized eigenvectors.
# here we see that the sum of filter coefficients of the More Penrose solution massively differs from the true value one
sum(mp_H_inv%*%g)
# Cointegration constraint is met for full inverse
sum(H_inv%*%g)
# Here we see the difference: with and without V[,k_max]
# Note: the importance of V[,k_max] is clearly evident since the More Penrose without V[,k_max] looks quite different
ts.plot(cbind(mp_H_inv%*%g,H_inv%*%g),col=c("blue","black"),main="More Penrose (blue) vs. Full Inverse")
ts.plot(scale(cbind(mp_H_inv%*%g,H_inv%*%g)),col=c("blue","black"),main="Same as above but scaled")

# Compute solution
# Note: if lambda is close to min(range), then b_eps is typically smoother than imposed by rho_1 
b_eps<-Xi%*%H_inv%*%g
ts.plot(b_eps)
# Note: if lambda is close to min(range) then the Rayleigh coefficient is typically larger than rho1
# But here rho_1 is very close to rho_max, so both numbers are virtually identical
rho_1
t(b_eps)%*%M%*%b_eps/t(b_eps)%*%b_eps


########################################################################################
########################################################################################
# Example 5: as example 4 but we use approximate expression for Hessian
# Let us now use this approximate expression
H <-  lambda * Q
H_inv <- solve(H)
idinv<-solve(V)%*%H_inv%*%solve(t(V))
# The main difference is that the last eigenvalue is no more very large!!!!
# This will be accounted for by mu below!!!
idinv


# Calculate mu
# This is a crucial step because max g-eigenvalue (here 10th) of H diverges
# Therefore mu `acts' a bit like a scaling, so that the I-SSA solution  becomes MSE optimal
# However, this scaling applies different weights to the generalized eigenvectors. These weights 
# correspond to the spectral weights of the target. In general the spectral weight 
num <- sum(H_inv %*% t(A) %*% gamma_target) - Gamma_0
den <- sum(H_inv)
mu <- num / den

# Calculate g
g<-t(A) %*% gamma_target - mu * rep(1, L)
# Find largest eigenvalue of H^-1
# For rho1=0.9 this is 10
k_max<-which(abs(diag(idinv))==max(abs(diag(idinv))))
k_max
# The main difference to exercise 4 is that t(g)%*%V[,k_max] is not vanishing but gets larger:
# The modified spectral weight (due to modified mu) assigns increasing weight to V[,k_max]. This is necessary 
# because the corresponding eigenvalue is not diverging anymore because A'A is missing in Hessian H. 
t(g)%*%V[,k_max]
# Here's the generalized eigenvector: it is smoother than for rho_1=0.9 in exercise 2 but still less smooth than v_max of M
# Note: generalized eigenvalues and eigenvectors depend on rho_1 (unlike stationary SSA, where A'A is missing in H)
ts.plot(V[,k_max])


# More Penrose inverse
mp_indinv<-idinv
# Replace largest eigenavlue by zero
mp_indinv[k_max,k_max]<-0
mp_H_inv<-V%*%mp_indinv%*%t(V)
# Check cointegration constraint: the cointegration constraint is not satisfied, i.e.,
# V[,k_max] is necessary. The MSE optimal estimate assigns weight to all generalized eigenvectors.
# here we see that the sum of filter coefficients of the More Penrose solution massively differs from the true value one
sum(mp_H_inv%*%g)
# Cointegration constraint is met for full inverse
sum(H_inv%*%g)
# Here we see the difference: with and without V[,k_max]
# Note: the importance of V[,k_max] is clearly evident since the More Penrose without V[,k_max] looks quite different
ts.plot(cbind(mp_H_inv%*%g,H_inv%*%g),col=c("blue","black"),main="More Penrose (blue) vs. Full Inverse")
ts.plot(scale(cbind(mp_H_inv%*%g,H_inv%*%g)),col=c("blue","black"),main="Same as above but scaled")

# Compute solution
# Note: if lambda is close to min(range), then b_eps is typically smoother than imposed by rho_1 
b_eps<-Xi%*%H_inv%*%g
ts.plot(b_eps)
# Note: if lambda is close to min(range) then the Rayleigh coefficient is typically larger than rho1
rho_1
t(b_eps)%*%M%*%b_eps/t(b_eps)%*%b_eps

#########################################################################################################
#########################################################################################################
# Main result: even if A'A is missing in Hessian H, the solution conforms to HT constraint.
# This shows that approximation H \approx lambda*Q is correct as rho_1 to rho_max, though the mechanism 
# of adjustment through mu and modified spectral weight t(g)%*%V[,k_max] of target is far from trivial.
#########################################################################################################
#########################################################################################################




