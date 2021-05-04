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

## Why not use Gibbs Sampling?
There are, of course, other sampling algorithms available, such as the Gibbs Sampler. The Gibbs Sampler requires no fine tuning, however, it does require that the full set of posterior conditional distributions is known. The Metropolis-Hastings Algorithm, however, only requires the posterior conditionals up to proportionality. This means that we can easily add in complicated prior distributions and layers of hierarchical priors without needing to explicitly know the full posterior conditionals. This does come at the cost of having to fine tune a proposal distribution, however, this is taken care of within bayesglm. 

## Downfalls of Metropolis-Hastings & Gibbs Samplers
When comparing to the Gibbs Sampler, the Metropolis-Hastings algorithm may be seen as inefficient in that not every jump in the Markov chain may be accepted (every jump is accepted with the Gibbs Sampler). This again means that extra computational resources must be spent to ensure that the chain is adequately exploring the posterior distribution. 

The Gibbs Sampler, however, requires that you generate each sample from the full conditional posterior distribution. In all but trivial cases, this may mean sampling from rather complicated distributions. The Metropolis-Hastings algorithm on the other hand, only has to sample from a proposal distribution (generally a simple distribution) and a uniform (0,1) distribution to accept reject the proposal.

While each of these samplers has their pros/cons, there is a downfall which plagues all standard MCMC methods (including Gibbs,M-H), that is the curse if multidimensional datasets. It is well known that the performance of standard random walk algorithms such as Gibbs sampling, deteriorates as dimensionality of the posterior distribution increases. The Metropolis-Hastings algorithm in particluar struggles with this. 

To understand this consider a scenario where we have only a 1 dimensional posterior distribution. When the M-H algorithm proposes a new chain sample, it can be in three states: inside the tagret density or above/below the target density. Therefore, there is one out of three regions it can go to stay within the target density. If we instead have a two dimensional posterior, then there is only one out of nine regions that result in the chain remaining inside the target density. In general, if D is the dimensionality, there are 3<sup>D</sup>-1 regions that result in a sample being outside of the target density, while there is always only 1 region that contains it. This means that as the number of dimensions increases, the region that contains the target density becomes exponentially smaller compared to the neigbouring volumes, making it much harder for the random walk to go in the 'right' direction. It is with this in mind that 'Hamiltonian Monte Carlo' methods have been developed which enrich the target density with a direction field that guides the random walk in the 'right' direction. It is my intention to include Hamiltonian Monte Carlo in future iterations of bayesglm.