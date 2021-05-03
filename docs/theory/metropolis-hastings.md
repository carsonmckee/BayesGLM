# Bayesglm Sampling Algorithm

The bayesglm package uses the Metropolis-Hastings algorithm to construct Markov chains. Recall that in the Bayesian Framework, we aim to explore the posterior distribution:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\pi(\theta|x)=\frac{f(x|\theta)p(\theta)}{f(x)}"/>
</p>

where 'f' is the likelihood, 'p' is the prior distribution and 'f(x)' is a normalizing constant based soley on the data, x. In order to explore the posterior distribution we construct a markov chain with a transition kernel that 'preserves' the posterior density, meaning that over sufficient time steps the chain will converge to the posterior distribution. After the chain has converged we can discard the first N iterations as burn-in and use the converged chain iterations as samples from the posterior distibution.

The Metropolis-Hastings algorithm constructs such a Markov chain using two main steps:
1. Given our current position, <img src="https://latex.codecogs.com/svg.latex?&space;\theta"/>, generate a 'proposal' sample, <img src="https://latex.codecogs.com/svg.latex?&space;\phi"/>, from a proposal probability distribution, <img src="https://latex.codecogs.com/svg.latex?&space;q(\phi|\theta)"/>.
2. Correct the proposal sample such that any proposals that stray too far from the target density, <img src="https://latex.codecogs.com/svg.latex?&space;\pi(\theta|x)"/>, are rejected.

If we say that <img src="https://latex.codecogs.com/svg.latex?&space;\phi"/> is our proposal step, generated from our proposal distribution <img src="https://latex.codecogs.com/svg.latex?&space;q(\phi|\theta)"/>, then we accept/reject <img src="https://latex.codecogs.com/svg.latex?&space;\phi"/> with probability
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\alpha(\theta,\phi)=min\left(1,\frac{\pi(\phi)q(\theta|\phi)}{\pi(\theta)q(\phi|\theta)}\right)"/>
</p>

## The Proposal Distribution

The Metropolis-Hastings algorithm requires that we specify a proposal distribution, q, along with any tuning paramters e.g. the proposal distribution variance. The performance of the algorithm in successfully converging to and exploring the target density will depend on our choice of proposal distribution and will normally require some degree of tuning. Note that if the proposal distribution is a symmetric distribution, that is to say
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;q(\phi|\theta)=q(\theta|\phi)"/>
</p>
then the acceptance probability simplifies to,
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\alpha(\theta,\phi)=min\left(1,\frac{\pi(\phi)}{\pi(\theta)}\right)"/>
</p>
This special case is known as the 'Random Walk Metropolis-Hastings Algorithm'. An example of a symmetric proposal distribution would be the Normal distribution.

Within the Bayesglm package, I have opted to use the Normal distribution as the proposal distribution. This, however, still leaves the proposal distribution variance. If the variance of the proposal distribution is fixed high then the proposal jumps may be too large and cause the chain to diverge or prevent it from exploring intricate neighbourhoods of the target distribution. If the proposal variance is too low, then the jumps will be small and convergence to the target distribution from the initial position may be very slow, while also causing the chain to explore the target poorly. Research has shown that tuning the proposal distribution such that the acceptance rate of proposals is between 20-40% is optimal for chain mixing in the target density. 

Clearly this tuning of proposal variances is going to be a nuisance for model fitting as there is going to be a variance paramter per chain and it will require multiple, potentially costly, tuning runs. Therefore, instead of requiring the user to perform this tuning manually, bayesglm will 'autotune' the proposal variance for each chain over the first 5000 chain iterations to get the chain acceptance rates between the 20-40% range. The model will also allow the user to access these acceptance rates post fitting in order to assess the chain mixing for themselves.
