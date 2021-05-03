from libcpp cimport bool

cdef extern from "Model.cpp":
    pass

cdef extern from "<vector>" namespace "std":
    cdef cppclass vector[T]:
        cppclass iterator:
            T operator*()
            iterator operator++()
            bint operator==(iterator)
            bint operator!=(iterator)
        vector()
        void push_back(T&)
        T& operator[](int)
        T& at(int)
        iterator begin()
        iterator end()

cdef extern from "string" namespace "std":
    cdef cppclass string:
        char* c_str()

# Declare the class with cdef
cdef extern from "Model.h":
    cdef cppclass Model:
        Model(string family_name,
              string formula_str,
              vector[double] extra_family_params, 
              vector[bool] extra_family_specified,
              vector[string] prior_names, 
              vector[string] prior_var_names,
              vector[vector[bool]] prior_params_specified,
              vector[vector[double]] prior_params, 
              string link_name, 
              vector[vector[double]] design_matrix,
              vector[double] response_vector, 
              vector[double] offset,
              int beta_vec_start_ind) except +

        void fit(int, int, vector[double])
        vector[vector[double]] get_chains()
        vector[double] get_response()
        vector[vector[double]] get_design()
        double DIC()
        double AIC()
        vector[double] get_acceptance_rates()
        vector[double] get_coefficients()
        void summary()
        vector[double] fitted(string)
        vector[double] residuals(string)
        vector[double] predict(vector[vector[double]], vector[double], string)
        vector[vector[double]] qq_vectors()
