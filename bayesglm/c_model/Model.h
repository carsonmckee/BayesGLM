#ifndef MODEL_H
#define MODEL_H

#include "Eigen/Dense"
#include <iostream>
#include <string>
#include <vector>
#include <map>

using Eigen::MatrixXd;
using Eigen::ArrayXd;

typedef double (*pfunc)(const double &, const std::vector<double> &);
typedef double (*sfunc)(const std::vector<double> &);
typedef ArrayXd (*link_function)(const ArrayXd &);
typedef ArrayXd (*family_prob_func)(const ArrayXd &, const std::vector<ArrayXd> &);
typedef double (*var_function)(const double &);

void line();

class Model{
    public:
        Model(std::string family_name,
              std::string formula_str,
              std::vector<double> family_params, 
              std::vector<bool> family_specified,
              std::vector<std::string> prior_names, 
              std::vector<std::string> prior_var_names,
              std::vector<std::vector<bool> > prior_params_specified,
              std::vector<std::vector<double> > prior_params,
              std::string link_name, 
              std::vector<std::vector<double> > design_matrix,
              std::vector<double> response_vector, 
              std::vector<double> offset,
              int beta_vec_start_ind);

        ~Model();

        void fit(int iters, int burn_in, std::vector<double> initial_pos);

        void set_chain_initial_pos(std::vector<double> x);
        std::vector<std::vector<double> > get_chains();

        std::vector<std::vector<double> > get_design();
        std::vector<double> get_response();

        double DIC();
        double AIC();
        std::vector<double> fitted(std::string scale = "response"); //get fitted values 
        std::vector<double> residuals(std::string type = "raw");
        std::vector<double> get_acceptance_rates(); //get acceptance rates for each chain
        std::vector<double> get_coefficients();
        std::vector<double> predict(const std::vector<std::vector<double> > &data, const std::vector<double> &offset, const std::string &type);
        void summary();
        std::vector<std::vector<double> > qq_vectors();


    private:
        MatrixXd chains;
        MatrixXd X;
        MatrixXd y;
        MatrixXd beta;
        MatrixXd offset;
        MatrixXd accept_reject_mat;
        int num_priors;
        int beta_vec_start_ind;
        int beta_size;
        int n_samples;
        int burn_in;

        std::vector<double> lower_ci;
        std::vector<double> upper_ci;
        std::vector<double> chain_sd;

        std::vector<std::vector<bool> > prior_params_specified;
        std::vector<std::vector<double> > prior_params;
        std::vector<double> family_params;
        std::vector<ArrayXd> family_args_temp;
        std::vector<bool> family_specified;
        std::vector<double> proposal_variance;

        std::string link_name;
        std::string proposal_name;
        std::string family_name;
        std::string formula_str;
        std::vector<std::string> prior_names;
        std::vector<std::string> prior_var_names;

        std::vector<int> beta_ind_to_chain_ind;

        std::vector<pfunc> prior_prob_funcs; //vector holding pointers to probability functions for each prior
        pfunc proposal_p_func; //pointer to sample func for proposal dist
        sfunc proposal_s_func; //pointer to probability func for proposal dist
        family_prob_func family_p_func; //pointer to family probability function
        family_prob_func family_cdf; //pointer to family cumulative density/mass function
        link_function link_func; //pointer to link function
        var_function var_func; //pointer to link function

        //methods for running mcmc chains
        Eigen::ArrayXd likelihood(const int &iter, const bool &log=false); //calculate likelihood for given chain iteration
        double prior_prob(const double &x, const int &prior_row_number, const int &iter); //calculate p(theta)
        void set_beta_vec(MatrixXd &vec, const int &iter);

        MatrixXd set_design_from_vector(std::vector<std::vector<double> > data);
        MatrixXd set_response_from_vector(std::vector<double> data);
        MatrixXd set_offset_from_vector(std::vector<double> data);
        std::vector<double> accpetance_rates;
        MatrixXd coefficients;
        void set_chain_ci_and_sd();
        void autotune_variance(const int &iter, const int &lag);

};

class PF_Lookup{
    public:
        std::map<std::string, pfunc> FUNC_MAP;
        PF_Lookup();
};

class CDF_Lookup{
    public:
        std::map<std::string, family_prob_func> FUNC_MAP;
        CDF_Lookup();
};

class FAM_PF_Lookup{
    public:
        std::map<std::string, family_prob_func> FUNC_MAP;
        FAM_PF_Lookup();
};

class SF_Lookup{
    public:
        std::map<std::string, sfunc> FUNC_MAP;
        SF_Lookup();
};

class Link_Lookup{
    public:
        std::map<std::string, link_function> FUNC_MAP;
        Link_Lookup();
};

class VF_Lookup{
    public:
        std::map<std::string, var_function> FUNC_MAP;
        VF_Lookup();
};


#endif