import pandas
import logging
import random

import utils as utils
import py_model as py_model
import matplotlib.pyplot as plt

from distributions import Distribution, Normal, Binomial, Poisson, Gamma, Uninformed, Laplace
from pandas.api.types import is_numeric_dtype
from typing import Dict, List


logger = logging.getLogger(__name__)


class glm(object):
    """
    main glm object for fitting models
    """

    def __init__(self,
                 formula: str,
                 family: Distribution, 
                 data: pandas.DataFrame,
                 priors: Dict[str, Distribution],
                 normalize_data: bool=False,
                 offset: pandas.DataFrame=None):
        
        utils.check_formula(formula) 
        utils.check_distribution(family)
        utils.resolve_priors(priors)
        self.__family_prior_names = [p for p in family.ARG_VALUES[1:] if isinstance(p, str)] 
        self.__formula = formula
        self.__scale_means = {}
        self.__scale_sd = {}
        self.__data = data
        if normalize_data:
            self.__scale_data(self.__data)
        self.__link = family.link
        self.__response, self.__model_matrix = utils.model_matrix(self.__formula, self.__data)
        self.__response = self.__response*self.__scale_sd.get(self.__response.name,1) + self.__scale_means.get(self.__response.name,0)
        self.__prior_names_ordered = utils.remove_and_sort_priors(priors, self.__family_prior_names, self.__model_matrix.columns)
        self.__family = family
        self.__priors = priors
        self.__offset = offset
        self.__chains = None
        self.__model_fitted = False
        self.__burn_in = None
        
        prior_spec, prior_args = utils.get_prior_specified_and_params(self.__prior_names_ordered, priors)
        family_spec = [True if not isinstance(arg, str) else False for arg in family.ARG_VALUES[1:]]
        family_args = [arg if isinstance(arg, (int, float)) else self.__prior_names_ordered.index(arg) for arg in family.ARG_VALUES[1:]]
        beta_start_ind = len(self.__prior_names_ordered) - len(self.__model_matrix.columns)

        if not isinstance(self.__offset, (pandas.Series)):
            offset = [0]*len(data)
        else:
            offset = self.__offset.tolist()

        self.__c_model = py_model.PyModel(family.NAME.encode('utf-8'),
                                        formula.encode('utf-8'),
                                        family_args, 
                                        family_spec,
                                        [priors.get(prior).NAME.encode('utf-8') for prior in self.__prior_names_ordered], 
                                        [n.encode('utf-8') for n in self.__prior_names_ordered],
                                        prior_spec,
                                        prior_args, 
                                        self.__link.encode('utf-8'), 
                                        self.__model_matrix.values.tolist(),
                                        self.__response.values.tolist(), 
                                        offset,
                                        beta_start_ind)
    
    @property
    def family(self):
        """
        returns the fammily specified in the model
        """
        return self.__family

    @property
    def data(self):
        """
        returns the data passed to the model
        """
        return self.__data

    @property
    def formula(self):
        """
        returns the formula passed to the model
        """
        return self.__formula

    @property
    def model_matrix(self):
        """
        returns the constructed model matrix based on the formula and data passed to the model
        """
        return self.__model_matrix

    @property
    def y(self):
        """
        returns response data passed to the model
        """
        return self.__response

    @property
    def coefficients(self):
        """
        returns the fitted model coefficients 
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        out = pandas.DataFrame(self.__c_model.coefficients(),
                               self.__prior_names_ordered)
        return out

    def fit(self, chain_length: int=10000, 
                  burn_in: int=4000, 
                  initial_pos: Dict[str, float]=None):
        """
        fits the model using the Metropolis-Hastings algorithm. 
        """

        if initial_pos:
            initial_pos = [initial_pos.get(prior, random.uniform(0.5, 8)) for prior in self.__prior_names_ordered]
        else:
            initial_pos = [random.uniform(0.5, 8) for i in range(0, len(self.__prior_names_ordered))]
        self.__burn_in = burn_in
        self.__c_model.fit(chain_length, burn_in, initial_pos)
        self.__model_fitted = True
        acc_rates = self.acceptance_rates()
        self.__chains = self.get_chains()
        if (acc_rates < 0.15).any()[0] or (acc_rates > 0.5).any()[0]:
            logger.warning("Convergence Warning: Some chains have high/low acceptance rates, chain mixing may be poor.")
        
    def plot_prior_hierarchy(self):
        pass

    def summary(self):
        """
        prints a summary of the fitted model
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        self.__c_model.summary()
       
    def fitted(self, scale='response'):
        """
        returns the fitted values on either the link or response scale 
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        return pandas.Series(self.__c_model.fitted(scale.encode('utf-8')))

    def predict(self, data: pandas.DataFrame, type: str = 'response', offset=None):
        """
        returns predicted model values on either the link or response scale based on new data passed in.
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        if self.__response.name not in data.columns:
            data[self.__response.name] = 0
        if not offset:
            offset = [0]*len(data)
        else:
            offset = offset.tolist()
        #TODO deal with when data normalization is turned on
        res, model_mat = utils.model_matrix(self.__formula, data)
        out = self.__c_model.predict(model_mat.values.tolist(), offset, type.encode('utf-8'))
        return pandas.Series(out)

    def qq_plot(self):
        """
        plots the theoretical vs observed quantiles as a scatter plot on the current axis
        """

        theory_quant, sample_quant = self.__c_model.qq_vectors()
        plt.scatter(theory_quant, sample_quant, marker = '+', c = 'black', linewidth=0.75)
        x = y = [-4, 4]
        plt.plot(x, y, '--', c="grey", linewidth=0.65)
        plt.xlim([0, 1.1])
        plt.ylim([0, 1.1])
        plt.xlabel('Theoretical Quantiles')
        plt.ylabel('Sample Quantiles')
    
    def get_chains(self):
        """
        returns a pandas dataframe with each column containing the markov chains
        """

        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        raw_chains = self.__c_model.get_chains()
        df = pandas.DataFrame.from_records(raw_chains)
        df=df.transpose()
        df.columns = self.__prior_names_ordered
        return df

    def residuals(self, type = "pearson"):
        """
        returns the residuals of the fitted model, supports pearson, response or link residuals
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        return self.__c_model.residuals(type.encode('utf-8'))

    def plot_residuals(self, type_residual="pearson", type_fitted="response", h_line=True):
        """
        plots residuals vs fitted values
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        r = self.residuals(type_residual)
        f = self.fitted(type_fitted)

        plt.scatter(f, r, facecolors='none', edgecolors='k', linewidth=0.4)
        x = [0, max(f)]
        y = [0,0]
        plt.plot(x, y, '--', c="grey", linewidth=0.65)
        plt.xlabel(type_fitted)
        plt.ylabel(type_residual + " residuals")

    def credible_interval(self, variable: str, alpha: float=0.05, burn_in: bool=False):
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        raise NotImplementedError("Not yet implemented.")
    
    @property
    def aic(self):
        """
        returns the aic of the fitted model
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        return self.__c_model.AIC()

    @property
    def dic(self):
        """
        returns the dic of the fitted model
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        return self.__c_model.DIC()

    def acceptance_rates(self):
        """
        returns acceptance rates of the markov chains
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        out = pandas.DataFrame(self.__c_model.acceptance_rates(),
                               self.__prior_names_ordered)
        return out

    def plot_prior_densities(self, variables: List[str]=None):
        """
        plots the marginal posterior densities of given parameters using the fitted markov chains
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        if not variables:
            variables = self.__prior_names_ordered
        fig, axs = plt.subplots(len(variables), sharex=False, sharey=False)
        for i, variable in enumerate(variables):
            self.__chains[variable][self.__burn_in:].plot(kind="density", ax=axs[i])
            axs[i].set_xlabel(variable)
            axs[i].set_ylabel("")

        plt.tight_layout()
        fig.text(0, 0.5, 'Density', va='center', rotation='vertical')
        return axs

    def plot_chain_trace(self, variables: List[str]=None):
        """
        plots the markov chains of a given list of parameters
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        if not variables:
            variables = self.__prior_names_ordered
        fig, axs = plt.subplots(len(variables), sharex=True, sharey=False)
        for i, variable in enumerate(variables):
            axs[i].plot(self.__chains.index, self.__chains[variable], linewidth=0.6)
            axs[i].set_ylabel(variable)

        return axs
        
    def plot_chain_acfs(self, variables: List[str], use_burn_in: bool=False):
        """
        plots the chain autocorrelation for a given list of parameters
        """
        if not self.__model_fitted:
            raise RuntimeError("Model has not been fitted, call glm.fit.")
        raise NotImplementedError("Not yet implemented.")
    
    def __scale_data(self, data: pandas.DataFrame):
        for col in data.columns:
            if is_numeric_dtype(data[col]):
                m, sd = data[col].mean(), data[col].var()**(0.5)
                self.__scale_means[col], self.__scale_sd[col] = m, sd
                data[col] = (data[col] - m)/sd


if __name__ == "__main__":

    data = pandas.read_excel("/Users/carsonmckee/Dev/bayesglm_2/bayesglm/bernoulli_data.xls", header=None)
    data.columns = ["y", "x1", "x2", "x3"]

    data2 = pandas.read_csv("/Users/carsonmckee/dev/bayesglm_2/bayesglm/pois_data.csv")

    data3 = pandas.read_csv("/Users/carsonmckee/Dev/bayesglm_2/bayesglm/normal_data.csv")

    data4 = pandas.read_csv("/Users/carsonmckee/Dev/bayesglm_2/bayesglm/exp_data.csv")
    
    priors = {
        "Intercept" : Uninformed(),
        "x1"        : Normal(mu=0, sigma=3),
        #"x2"        : Uninformed(),
        "x3"        : Normal(mu=0, sigma=3),
        "x4"        : Uninformed(),
        "x5"        : Uninformed(),
        "x6"        : Uninformed(),
        "dispersion": Gamma(rate=1, shape="shape"),
        "shape"     : Gamma(1)
    }

    model = glm(formula="y ~ .",
                priors=priors,
                family=Binomial(),
                data=data)

    init = {"Intercept":1, "x1": 0, "x2":0, "x3":0}            

    model.fit(chain_length=20000, burn_in=15000, initial_pos=init)

    model.summary()

    model.qq_plot()
    plt.show()
    plt.close()

    model.plot_residuals()
    plt.show()
    plt.close()

    model.plot_chain_trace()
    plt.show()
    plt.close()

    plots = model.plot_prior_densities()
    plt.show()
    plt.close()

    print(0)