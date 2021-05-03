# Bayesglm Sampling Algorithm

The bayesglm package uses the Metropolis-Hastings algorithm to construct Markov chains. Recall that in the Bayesian Framework, we aim to explore the posterior distribution:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\pi(\theta|x)=\frac{f(x|\theta)p(\theta)}{f(x)}"/>
</p>

where 'f' is the likelihood, 'p' is the prior distribution and 'f(x)' is a normalizing constant based soley on the data, x. In order to explore the posterior distribution we construct a markov chain with a transition kernel that 'preserves' the posterior density, meaning that over sufficient time steps the chain will converge to the posterior distribution. After the chain has converged we can discard the first N iterations as burn-in and use the converged chain iterations as samples from the posterior distibution.

The Metropolis-Hastings algorithm constructs such a Markov chain using two main steps:
1. Given our current position, <img src="https://latex.codecogs.com/svg.latex?&space;\theta"/>, generate a 'proposal' sample, <img src="https://latex.codecogs.com/svg.latex?&space;\phi"/>, from a proposal probability distribution, <img src="https://latex.codecogs.com/svg.latex?&space;q(\phi|\theta)"/>.
2. Correct the proposal sample such that any proposals that stray too far from the target density are rejected.

If we say that <img src="https://latex.codecogs.com/svg.latex?&space;\phi"/> is our proposal step, generated from our proposal distribution <img src="https://latex.codecogs.com/svg.latex?&space;q(\phi|\theta)"/>, then we accept/reject <img src="https://latex.codecogs.com/svg.latex?&space;\phi"/> with probability
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\alpha(\theta,\phi)=min\left(1,\frac{\pi(\phi)q(\theta|\phi)}{\pi(\theta)q(\phi|\theta)}\right)"/>
</p>
