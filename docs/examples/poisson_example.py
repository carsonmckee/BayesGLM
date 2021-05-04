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