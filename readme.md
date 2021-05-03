BayesGLM
========

A python API for fitting Generalised Linear Models via Metropolis-Hastings Sampling.

## Overview

BayesGLM offers flexible GLM spceification within a Bayesian Framework, supporting a wide range of Prior distributions while allowing for hierarchical Prior specification. Models are fitted by implementing the Metropolis Hastings Algorithm in order to induce stationary Markov Chains of which posterior samples can be obtained. The underlying mathematics is written in C++ for speed and interfaces with Python using Cython. The model fitting is done via a central 'glm' object, similar in functionality to that found in the R language.

The package utilises pandas, matplotlib and seaborn for plotting and data manipulation. 

## Features

BayesGLM currently supports the following features:
1. Linear and interaction terms of predictors. 
2. Link function specification.
3. Wide range of Exponential Family Distributions.
4. Wide range of Prior disributions.
5. Running multiple Markov Chains for each variable.
6. Specification of burn-in, chain length and chain starting positions.
7. Specification of proposal distribution.
9. Accomodates both single and multi-update MH Algorithm.
9. Range of posterior estimates including mean, standard error and credible intervals.
10. An estimate of the Deviance Information Criterion.
11. Chain mixing/convergence diagnostics, including acceptance rates and ACF plots.
12. Posterior chain summary plots. 

## Documentation