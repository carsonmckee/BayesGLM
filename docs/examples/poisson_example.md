Poisson Response Data Example
========

In this example we will look at a small simulated data set that consists of three predictor variables and a count response variable. Here we will use a Poisson GLM with the log link function. The model specification is as follows:

<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;Y_i\sim\mathcal{Pois}\left(\lambda_i\right)"/></p>
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;lambda_i=e^{\beta_0+\beta_{1}x_{i,1}+\beta_{2}x_{i,2}+\beta_{3}x_{i,3}}"/></p>

For our priors we will specify:
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\beta_{0}\sim Uninformed"/></p>
<p align="center">
<img src="https://latex.codecogs.com/svg.latex?&space;\beta_{k}\sim \mathcal{N}(0,3), k=1,2,3"/></p>

Now we will go through how to fit this model in bayesglm. 
First import the required bayesglm modules along with pandas and read the data.
```python
import pandas
from bayesglm.bglm import glm
from bayesglm.distributions import Poisson, Normal, Uninformed

data = pandas.read_csv("data/pois_data.csv")
```

Now we are going to fit a simple linear model with no interaction terms but first, lets specify some priors for our model parameters and store them in a list.

```python
priors = {
    "Intercept" : Uninformed(),
    "x1"        : Normal(mu=0, sigma=3),
    "x2"        : Normal(mu=0, sigma=3),
    "x3"        : Normal(mu=0, sigma=3)
}
```

Now lets specify some initial starting positions for out chain.
We will create a dictionary of starting positions.
```python
initial_pos={"Intercept":2, "x1":1, "x2":0, "x3":0}
```

We are now ready to specify our 'glm' model object.

```python
model = glm(formula="y ~ x1 + x2 + x3",
            priors=priors,
            family=Poisson(link='log'),
            data=data)
```

We are now ready to run the Metropolis-Hastings Algorithm for our model. This is done via the "fit" method on our model object. Here we will run for 30,000 samples with the first 5,000 as burn-in.

```python
model.fit(chain_length=30000, burn_in=5000, initial_pos=initial_pos)
```
This will produce the following output as the chains update.
```
[=====================================================================>] 30000/30000 (5 seconds)
```

Now that we have produced some posterior samples, lets look at the model summary using the "summary" method.

```python
model.summary()
```
```
                                Bayesian GLM Summary
------------------------------------------------------------------------------------
DIC: 744.559        | Chain iters : 30000  | Formula :   y ~ .          
N  : 150            | Burn-in     : 5000   | Family  :   poisson(link=log)
------------------------------------------------------------------------------------
Node       Prior        Mean        s.d.      [2.5%        97.5%]       Acc. Rate
------------------------------------------------------------------------------------
Intercept  uninformed   2.006       0.035     1.937        2.074        0.299       
x1         normal       1.004       0.025     0.957        1.054        0.314       
x2         uninformed   0.180       0.020     0.141        0.218        0.265       
x3         normal       0.088       0.023     0.043        0.132        0.293       
------------------------------------------------------------------------------------
```

As we can see the summary shows a range of information about our fitted model. In particular we see that bayesglm has tuned the sample acceptance rate to around 30%, within the optimal 20-40% range for good chain mixing. 

We can now plot diagnostic plots such as the Quantile-Quantile plot.
```python
model.qq_plot()
plt.show()
plt.close()
```

![QQ Plot](images/pois_qq.png)

The plot above shows our Poisson error assumption appears to be vaild.

We can also plot the model residuals.

```python
model.plot_residuals()
plt.show()
plt.close()
```

![Residuals](images/pois_resiual.png)

We may also plot the Markov chain traces as shown below. Here we do not include the burn-in iterations.

```python
model.plot_chain_trace(burn_in=False)
plt.show()
plt.close()
```

![Posterior MCMC Chains](images/pois_chains.png)
    
    