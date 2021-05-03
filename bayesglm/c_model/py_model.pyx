# distutils: language = c++

from Model cimport Model
from libcpp.vector cimport vector
import numpy as np

cdef class PyModel(object):
    cdef Model*c_model 
      
    def __cinit__(self, family_name,
                        formula_str,
                        extra_family_params, 
                        extra_family_specified,
                        prior_names, 
                        prior_var_names,
                        prior_params_specified,
                        prior_params, 
                        link_name, 
                        design_matrix,
                        response_vector, 
                        offset, 
                        beta_vec_start_ind):
        
        self.c_model = new Model(family_name,
                                formula_str,
                                extra_family_params, 
                                extra_family_specified,
                                prior_names, 
                                prior_var_names,
                                prior_params_specified,
                                prior_params, 
                                link_name,
                                design_matrix,
                                response_vector, 
                                offset, 
                                beta_vec_start_ind)
 
    def __dealloc__(self):
        del self.c_model 
     
    def fit(self, iters, burn_in, initial_pos): 
        self.c_model.fit(iters, burn_in, initial_pos)
    
    def get_chains(self):
        return self.c_model.get_chains() 
    
    def get_design(self): 
        return self.c_model.get_design() 
     
    def get_response(self):
        return self.c_model.get_response()
     
    def DIC(self): 
        return self.c_model.DIC()
 
    def AIC(self, burn_in=0): 
        return self.c_model.AIC()
     
    def acceptance_rates(self):
        return self.c_model.get_acceptance_rates() 
    
    def coefficients(self):
        return self.c_model.get_coefficients()
     
    def summary(self):
        return self.c_model.summary()
    
    def fitted(self, scale): 
        return self.c_model.fitted(scale)
    
    def residuals(self, type): 
        return self.c_model.residuals(type)
    
    def predict(self, data, offset, type):
        return self.c_model.predict(data, offset, type) 
     
    def qq_vectors(self):
        return self.c_model.qq_vectors()