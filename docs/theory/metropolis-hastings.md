# Bayesglm Sampling Algorithm

The bayesglm package uses the Metropolis-Hastings algorithm to constructing Markov chains that explore the posterior distributions. Recall that in the Bayesian Framework, we aim to explore the posterior distribution:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\pi(\theta|x)= \frac{f(x|\theta)p(\theta)}{f(x)}"/>
</p>

where 'f' is the likelihood, 'p' is the prior distribution and 'f(x)' is a normalizing constant based soley on the data. In order to explore the posterior distribution we construct a markov chain for each unknown paramter in the model and run it until it has converged to and sufficiently explored the target distribution (the poseterior). 

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\alpha(\theta,\phi)=min\left(1,\frac{\pi(\phi)q(\theta|\phi)}{\pi(\theta)q(\phi|\theta)}\right)"/>
</p>
