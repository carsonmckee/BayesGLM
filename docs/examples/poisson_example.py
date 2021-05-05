Binary Response Data Example
========

In this example we will look at a small simulated data set that consists of three predictor variables and a binary response variable. Here we will use a Bernoulli GLM with the Logit link function.

First import the required bayesglm modules along with pandas and read the data.
```python
from bayesglm.bglm import glm, Prior
import bayesglm.distributions as dist
import pandas

data = pandas.read_excel("data/bernoulli_data.xls", header=None)
data.columns = ["y", "x1", "x2", "x3"]
```

Now we are going to fit a simple linear model with no interaction terms but first, lets specify some priors for our model parameters and store them in a list.

```python
priors = [Prior("Intercept", dist.Normal(0, 1)),
          Prior("x1", dist.Normal(1, 5)),
          Prior("x2", dist.Normal(0.5, 1.5)),
          Prior("x3", dist.Cauchy(0, 1))]
```

Now lets specify some initial starting positions for out chains. We'll run two chains with different start positions in order to improve our chances of good mixing/convergence.
We will create a dictionary of starting positions for each chain and store them in a list for later.

```python
initial_pos = [{"Intercept" : 5, "x1" : 6, "x2" : -3, "x3" : 1},
               {"Intercept" : 0, "x1" : 2, "x2" : -1, "x3" : -3}]
```

We are now ready to specify our 'glm' model object. For our proposal distribution we will use a Normal distribution with the mean as the current chain position and variance 1.

```python
model = glm(formula="y ~ x1 + x2 + x3",
            priors=priors,
            family=dist.Bernoulli(""),
            data=data,
            link="logit",
            proposal=dist.Normal("", 1))
```

We are now ready to run the Metropolis-Hastings Algorithm for our model. This is done via the "sample" method on our model object. Here we will run for 20,000 samples with the first 5,000 as burn-in and use single parameter updates, as shown below.

```python
model.sample(chain_length=20000,
             initial_pos=initial_pos,
             single_updates=True,
             burn_in=5000,
             num_chains=2)
```
This will produce the following output as the chains update.
```
  [======================================================================>] 100 %
  [======================================================================>] 100 %
```

Now that we have produced some posterior samples, lets look at the model summary using the "summary" method.

```python
model.summary()
```
```
================================================================================================================
Family : bernoulli(formula, link = logit)
Chain Length (burn-in): 20000 (5000)
DIC : 71.858
================================================================================================================
node         Prior                         mean          std err      mc err      [0.025      0.975]   acc. rate
----------------------------------------------------------------------------------------------------------------
Intercept    normal(0, 1)                  1.5511        0.3172       0.0020      0.9683      2.1966      0.3413
x1           normal(1, 5)                  0.7200        0.4516       0.0029     -0.1221      1.6467      0.3458
x2           normal(0.5, 1.5)              0.5403        0.3924       0.0025     -0.2124      1.3349      0.3370
x3           cauchy(0, 1)                  -0.3193       0.3384       0.0021     -1.0005      0.3066      0.3107
================================================================================================================
```

As we can see the summary shows a range of information about our fitted model. In particular we see that the sample acceptance rate is around 30%, within the optimal 20-40% range for good chain mixing. 

We can now plot the trace plots for our posterior chains including the burn-in as shown below.
```python
model.plot_chains(include_burn_in=True)
```

![Posterior MCMC Samples](images/bernoulli_example_trace.png)

As we can see, for each variable in the model, both chains appear to have converged to a stationary distribution and mixing looks fine.

We can check the quality of chain mixing visually by assessing the Autocorrelation plots as shown below.

```python
model.plot_chain_acf()
```

![Posterior MCMC ACF Plots](images/bernoulli_example_acf.png)

We can see that Autocorrelation decays fast and alternates around zero for all variables, mixing appears to be optimal.

We may also want to plot the marginal posterior densities of the chains along with their 95% credible intervals, as shown below.

```python
model.plot_densities(mark_CI=True)
```

![Posterior MCMC Densities](images/bernoulli_example_densities.png)

import pandas
import matplotlib.pyplot as plt

from bayesglm.bglm import glm
from distributions import Normal, Poisson, Uninformed, Laplace, HalfCauchy

if __name__ == "__main__":
    
    data = pandas.read_csv("/Users/carsonmckee/dev/bayesglm_2/bayesglm/pois_data.csv")

    priors = {
        "Intercept" : Uninformed(),
        "x1"        : Normal(mu=0, sigma=3),
        "x2"        : Normal(mu=0, sigma=3),
        "x3"        : Normal(mu=0, sigma=3)
    }

    model = glm(formula="y ~ x1 + x2 + x3",
                priors=priors,
                family=Poisson(link='log'),
                data=data)

    model.fit(chain_length=20000, burn_in=15000, initial_pos={"Intercept":2, "x1":1, "x2":0, "x3":0})

    model.summary()

    model.plot_residuals(type_residual="response", type_fitted="response")
    plt.show()
    plt.close()

    model.qq_plot()
    plt.show()
    plt.close()

    model.plot_chain_trace()
    plt.show()
    plt.close()