# Bayesglm Sampling Algorithm

The bayesglm package uses the Metropolis-Hastings algorithm to constructing Markov chains that explore the posterior distributions. Recall that in the Bayesian Framework, we aim to explore the posterior distribution:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\pi(\theta|x)= \frac{f(x|\theta)p(\theta)}{f(x)}"/>
</p>

where 'f' is the likelihood, 'p' is the prior distribution and 'f(x)' is a normalizing constant based soley on the data. 

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\alpha(\theta,\phi)=min\left(1,\frac{\pi(\phi)q(\theta|\phi)}{\pi(\theta)q(\phi|\theta)}\right)"/>
</p>
