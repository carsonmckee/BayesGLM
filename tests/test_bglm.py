import unittest
import sys
sys.path.append('/Users/carsonmckee/Dev/bayesglm_2/bayesglm')
import bglm
import distributions


class TestGLMInit(unittest.TestCase):

    def test_init(self):
        t = bglm.Prior('a', distributions.Normal(0,1))

class TestGLMFit(unittest.TestCase):

    def test_fit_poisson(self):
        pass

    def test_fit_normal(self):
        pass

    def test_fit_gamma(self):
        pass

    def test_fit_binomial(self):
        pass

    def test_fit_laplace(self):
        pass

class TestGLMFitted(unittest.TestCase):

    def test_fitted_poisson(self):
        pass

    def test_fitted_normal(self):
        pass

    def test_fitted_gamma(self):
        pass

    def test_fitted_binomial(self):
        pass

    def test_fitted_laplace(self):
        pass

class TestGLMPredict(unittest.TestCase):

    def test_predict_poisson(self):
        pass

    def test_predict_normal(self):
        pass

    def test_predict_gamma(self):
        pass

    def test_predict_binomial(self):
        pass

    def test_predict_laplace(self):
        pass

class TestGLMChains(unittest.TestCase):

    def test_get_chains(self):
        pass

    def test_acceptance_rates(self):
        pass


class TestGLMResiduals(unittest.TestCase):

    def test_residuals_poisson(self):
        pass

    def test_residuals_normal(self):
        pass

    def test_residuals_gamma(self):
        pass

    def test_residuals_binomial(self):
        pass

    def test_residuals_laplace(self):
        pass


class TestGLMAICDIC(unittest.TestCase):

    def test_aic_poisson(self):
        pass

    def test_aic_normal(self):
        pass

    def test_aic_gamma(self):
        pass

    def test_aic_binomial(self):
        pass

    def test_aic_laplace(self):
        pass

    def test_dic_poisson(self):
        pass

    def test_dic_normal(self):
        pass

    def test_dic_gamma(self):
        pass

    def test_dic_binomial(self):
        pass

    def test_dic_laplace(self):
        pass


    

class TestPrior(unittest.TestCase):

    def test_init(self):
        pass

if __name__ == '__main__':
    unittest.main()