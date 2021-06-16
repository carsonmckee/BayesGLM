#include "Model.h"
#include "Eigen/Dense"
#include "statistics.cpp"

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <chrono>

using Eigen::MatrixXd;
using namespace std::chrono;

void progress(const float &prog, const float &total){

    float progress = prog/total;

    int barWidth = 70;
    std::cout << "  [";
    int pos = barWidth * progress;

    for (int i = 0; i <= barWidth; ++i) {
        if (i < pos) std::cout << "=";
        else if (i == pos) std::cout << ">";
        else std::cout << " ";
    }

    std::cout << "] " << int(prog) << "/" << int(total) << "\r";
    std::cout.flush();
}


Model::Model(std::string family_name,
             std::string formula_str,
             std::vector<double> extra_family_params, 
             std::vector<bool> extra_family_specified,
             std::vector<std::string> prior_names, 
             std::vector<std::string> prior_var_names,
             std::vector<std::vector<bool> > prior_params_specified,
             std::vector<std::vector<double> > prior_params, 
             std::string link_name, 
             std::vector<std::vector<double> > design_matrix,
             std::vector<double> response_vector, 
             std::vector<double> offset,
             int beta_vec_start_ind){

    this->family_params = extra_family_params;
    this->formula_str = formula_str;
    this->family_specified = extra_family_specified;
    this->prior_params_specified = prior_params_specified;
    this->prior_params = prior_params;
    this->proposal_variance = std::vector<double>(num_priors, 1);
    this->family_name = family_name;
    this->link_name = link_name;
    this->prior_names = prior_names;
    this->prior_var_names = prior_var_names;
    this->num_priors = prior_names.size();
    this->beta_vec_start_ind = beta_vec_start_ind;
    this->beta_size = num_priors-beta_vec_start_ind;

    PF_Lookup probability_func_map;
    FAM_PF_Lookup fam_func_map;
    SF_Lookup sample_func_map;
    Link_Lookup link_func_map;
    VF_Lookup var_func_map;
    CDF_Lookup family_cdf_map;

    this->link_func = link_func_map.FUNC_MAP[link_name];
    this->var_func = var_func_map.FUNC_MAP[family_name];
    this->family_p_func = fam_func_map.FUNC_MAP[family_name];
    this->family_cdf = family_cdf_map.FUNC_MAP[family_name];
    for(int i=0; i<prior_names.size(); i++){
        this->prior_prob_funcs.push_back(probability_func_map.FUNC_MAP[prior_names[i]]);
    }

    this->X = set_design_from_vector(design_matrix);
    this->y = set_response_from_vector(response_vector);
    this->offset = set_offset_from_vector(offset);
    this->n_samples = y.rows();
    this->family_args_temp = std::vector<ArrayXd>(1+extra_family_params.size());
    this->beta = Eigen::MatrixXd::Constant(beta_size, 1, 0);
    this->coefficients = Eigen::MatrixXd::Constant(num_priors, 1, 0);
    this->accept_reject_mat = Eigen::MatrixXd::Constant(num_priors, 5000, 0);
    this->proposal_variance = std::vector<double>(num_priors, 1);

}

Model::~Model(){

}

void Model::set_chain_initial_pos(std::vector<double> x){

}

MatrixXd Model::set_offset_from_vector(std::vector<double> data){
    MatrixXd out(data.size(), 1);
    for(int i=0; i<data.size(); i++){
        out(i,0) = data[i];
    }
    return out;
}

std::vector<double> Model::get_coefficients(){
    std::vector<double> out(num_priors);
    for(int i=0; i<num_priors; i++){
        out[i] = coefficients(i,0);
    }
    return out;
}

void Model::fit(int iters, int burn_in, std::vector<double> initial_pos){

    PF_Lookup probability_func_map;
    SF_Lookup sample_func_map;

    this->proposal_s_func = sample_func_map.FUNC_MAP["normal"];
    this->proposal_p_func = probability_func_map.FUNC_MAP["normal"];

    auto start = high_resolution_clock::now();

    this->burn_in = burn_in;

    //initialise chains
    chains = Eigen::MatrixXd::Constant(num_priors, iters+1, 0);
    for(int i=0; i<num_priors; i++){
        chains(i, 0) = initial_pos[i];
    }

    //set up acceptance rate vector
    accpetance_rates = std::vector<double>(num_priors);

    Eigen::ArrayXd likelihood_curr(n_samples, 1);
    Eigen::ArrayXd likelihood_prop(n_samples, 1);

    for(int iter=1; iter<iters; iter++){
        
        for(int p=0; p<num_priors; p++){
            
            //get proposal sample
            double current = chains(p, iter);
            double proposal = this->proposal_s_func({current, proposal_variance[p]});

            double prior_ratio = prior_prob(proposal, p, iter) / prior_prob(current, p, iter);
            likelihood_curr = likelihood(iter);

            //set proposal onto chain
            chains(p, iter) = proposal;
            chains(p, iter+1) = proposal;
            likelihood_prop = likelihood(iter);

            double like_ratio = (likelihood_prop*likelihood_curr.inverse()).unaryExpr([](double v) { return std::isfinite(v)? v : 1.0; }).prod();

            double u = distributions::runiform(0, 1);

            //calculate A
            double A = like_ratio * prior_ratio;

            if(u > A){
                //reject, set current back into chains
                chains(p, iter) = current;
                chains(p, iter+1) = current;
            } 
            else{
                accpetance_rates[p] += 1;
                if(iter < 5000){
                    accept_reject_mat(p, iter) = 1.0;
                }
            }
        }
        
        if(iter < 5000 && iter > 0){
            autotune_variance(iter, 150);
        }
        
        progress(float(iter+1), float(iters));
    }
    
    for(int i=0; i<num_priors; i++){
        accpetance_rates[i] = accpetance_rates[i]/iters;
    }
    
    for(int i=0; i<num_priors; i++){
        double sum = 0;
        for(int j=burn_in; j<iters; j++){
            sum = sum+chains(i,j);
        }
        coefficients(i,0) = sum/(iters-burn_in);
    }

    set_chain_ci_and_sd();

    // Get ending timepoint
    auto stop = high_resolution_clock::now();
  
    auto duration = duration_cast<seconds>(stop - start);
    std::cout << "[";
    for(int i=0; i<69; i++){
        std::cout << "=";
    }
    std::cout << ">]" << " " << iters << "/" << iters << " (" << duration.count() << " seconds)\n";
}

void Model::autotune_variance(const int &iter, const int &lag){

    if( iter % lag == 0){

        for(int p=0; p<num_priors; p++){
            double sum = 0;
            for(int i=(iter-lag); i<iter; i++){
                sum += accept_reject_mat(p, i);
            }
            double rate = sum/lag;

            if(rate <= 0.1){
                proposal_variance[p] *= 0.4;
            } else if (rate <= 0.2 && rate > 0.1){
                proposal_variance[p] *= 0.6;
            } else if (rate >= 0.4 && rate < 0.5){
                proposal_variance[p] *= 1.5;
            } else if (rate >= 0.5 && rate < 0.65){
                proposal_variance[p] *= 2;
            } else if (rate >= 0.65 && rate < 0.85){
                proposal_variance[p] *= 2.5;
            } else if (rate >= 0.85){
                proposal_variance[p] *= 3;
            }
        }
    }

}

void Model::set_chain_ci_and_sd(){

    int chain_length = chains.cols();

    std::vector<double> row_copy(chain_length-burn_in);
    lower_ci = std::vector<double>(num_priors);
    upper_ci = std::vector<double>(num_priors);
    chain_sd = std::vector<double>(num_priors);

    int lower_ind = int(0.025*(chain_length-burn_in));
    int upper_ind = int(0.975*(chain_length-burn_in));

    for(int r=0; r<num_priors; r++){
        for(int i=burn_in; i<chain_length; i++){
            row_copy[i-burn_in] = chains(r, i);
        }
        
        std::sort(row_copy.begin(), row_copy.end());
        lower_ci[r] = row_copy[lower_ind];
        upper_ci[r] = row_copy[upper_ind];
        chain_sd[r] = std::sqrt(variance(row_copy));
    }
}

void Model::summary(){
    
    double dic = DIC();

    std::cout<<std::setw(53)<<"Bayesian GLM Summary\n";
    line();
    std::cout<<std::setw(5)<< "DIC: " << std::setw(15) << std::left << dic << std::setw(11) << "| Chain iters : " << std::setw(6) << (this->chains.cols() - 1)
    << std::setw(15) << " | Formula : " << std::setw(15) << this->formula_str <<"\n";
    std::cout<<std::setw(5)<< "N  : " << std::setw(15) << std::left << this->n_samples << std::setw(11) << "| Burn-in     : " << std::setw(6) << this->burn_in
    << std::setw(15) << " | Family  : " << this->family_name << "(link=" << this->link_name << ")\n";
    line();
    std::cout<<std::setw(11)<< "Node" << std::setw(13)<< "Prior" <<std::setw(12)<< "Mean"<<std::setw(10)<<"s.d."
    <<std::setw(13)<<"[2.5%"<<std::setw(13)<<"97.5%]"<<std::setw(10)<<"Acc. Rate\n";
    line();
    for(int i=0; i<num_priors; i++){
        std::cout << std::fixed << std::showpoint;
        std::cout << std::setprecision(3);
        std::cout<<std::setw(11) << prior_var_names[i]
                 <<std::setw(13) << prior_names[i]
                 <<std::setw(12) << coefficients(i,0)
                 <<std::setw(10) << chain_sd[i]
                 <<std::setw(13) << lower_ci[i]
                 <<std::setw(13) << upper_ci[i]
                 <<std::setw(12)<< accpetance_rates[i] << "\n";
    }
    line();

}

void Model::set_beta_vec(MatrixXd &vec, const int &iter){
    for(int i=0; i<beta_size; i++){
        vec(i,0) = chains(i+beta_vec_start_ind, iter);
    }
}


Eigen::ArrayXd Model::likelihood(const int &iter, const bool &log){

    //first get current beta column 
    int beta_len = num_priors-beta_vec_start_ind;
    set_beta_vec(beta, iter);

    ArrayXd X_beta = (X*beta + offset).array();
    Eigen::ArrayXd I = Eigen::ArrayXd::Constant(n_samples, 1, 1.0);

    //get family extra args 
    for(int f=1; f<family_params.size()+1; f++){
        if(!family_specified[f-1]){
            family_args_temp[f] = chains(int(family_params[f-1]), iter)*I; 
        }else{
            family_args_temp[f] = family_params[f-1]*I;
        }
    }
    
    Eigen::ArrayXd l(n_samples,1);
    ArrayXd scaled(n_samples, 1);

    if(family_name == "gamma"){
        scaled = family_args_temp[1];
    } else {
        scaled = I;
    }
    
    if(log){
        family_args_temp[0] = scaled*link_func(X_beta);
        l = family_p_func(y.array(), family_args_temp).log();
    } else {
        family_args_temp[0] = scaled*link_func(X_beta);
        l = family_p_func(y.array(), family_args_temp);
    }
    
    return l;
}

double Model::prior_prob(const double &x, const int &prior_row_number, const int &iter){

    //get prior dist params
    std::vector<double> prior_args = prior_params[prior_row_number]; 

    for(int i=0; i<prior_params[prior_row_number].size(); i++){
        if(!prior_params_specified[prior_row_number][i]){
            prior_args[i] = chains(int(prior_params[prior_row_number][i]), iter); 
        }else{
            prior_args[i] = prior_params[prior_row_number][i]; 
        }
    }
    return prior_prob_funcs[prior_row_number](x, prior_args);
}

double Model::DIC(){
    //function to get the DIC of the model after sampling has been carried out
    int chain_length = chains.cols();
    std::vector<double> posterior_likelihoods;
    for(int i=burn_in; i<chain_length; i++){
        //calculate log posterior likelihood for the chain iteration
        double posterior = likelihood(i, true).sum();
        posterior_likelihoods.push_back(posterior);
    }
    double post_mean = mean(posterior_likelihoods);
    //calculate point est likelihood
    MatrixXd copy = chains.col(0);
    chains.col(0) = coefficients;
    double point_est_like = likelihood(0, true).sum();
    //put the chain back in place
    chains.col(0) = copy;

    return -4.0*post_mean + 2.0*point_est_like;
}

double Model::AIC(){ 
    //function to get the DIC of the model after sampling has been carried out
    
    MatrixXd copy = chains.col(0);
    chains.col(0) = coefficients;
    double point_est_like = likelihood(0, true).sum();
    //put the chain back in place
    chains.col(0) = copy;

    double num_param = double(this->num_priors);

    return -2.0*point_est_like + 2*num_param;

}


std::vector<double> Model::get_acceptance_rates(){
    return accpetance_rates;
}

MatrixXd Model::set_design_from_vector(std::vector<std::vector<double> > data){
    int nrows = data.size();
    int ncols = data[0].size();
    MatrixXd out(nrows, ncols);

    for(int r=0; r<nrows; r++){
        for(int c=0; c<ncols; c++){
            out(r,c) = data[r][c];
        }
    }
    return out;
}

std::vector<std::vector<double> > Model::get_chains(){
    std::vector<std::vector<double> > out(chains.rows(), std::vector<double>(chains.cols()));

    for(int r=0; r<chains.rows(); r++){
        for(int c=0; c<chains.cols(); c++){
            out[r][c] = chains(r, c);
        }
    }

    return out;
}

MatrixXd Model::set_response_from_vector(std::vector<double> data){
    int nrows = data.size();
    MatrixXd out(nrows, 1);
    for(int r=0; r<nrows; r++){
        out(r,0) = data[r];
    }
    return out;
}

std::vector<std::vector<double> >  Model::get_design(){
    std::vector<std::vector<double> > out(X.rows(), std::vector<double>(X.cols()));
    for(int r=0; r<X.rows(); r++){
        for(int c=0; c<X.cols(); c++){
            out[r][c] = X(r,c);
        }
    }
    return out;
}

std::vector<double>  Model::get_response(){
    std::vector<double> out(y.rows());
    for(int r=0; r<y.rows(); r++){
        out[r] = y(r,0);
    }
    return out;
}

 
void line(){
    for(int i=1;i<43;i++){
          std::cout<<"--";
    }
     std::cout<<"\n";
}

std::vector<double> Model::fitted(std::string scale){
    std::vector<double> out(n_samples);
    MatrixXd beta = coefficients.block(beta_vec_start_ind, 0, beta_size, 1);
    ArrayXd XB = (X*beta + offset).array();
    ArrayXd scaled = ArrayXd::Constant(n_samples, 1, 1.0);

    Eigen::ArrayXd I = Eigen::ArrayXd::Constant(n_samples, 1, 1.0);

    //get family extra args 
    for(int f=1; f<family_params.size()+1; f++){
        if(!family_specified[f-1]){
            family_args_temp[f] = coefficients(int(family_params[f-1]))*I; 
        }else{
            family_args_temp[f] = family_params[f-1]*I;
        }
    }

    if(family_name == "gamma"){
        scaled = family_args_temp[1];
    }

    if(scale == "response"){
        ArrayXd linked = scaled*link_func(XB);
        for(int i=0; i< n_samples; i++){
            out[i] = linked(i,0);
        }
    } else if (scale == "link") {
        for(int i=0; i< n_samples; i++){
            out[i] = XB(i,0);
        }
    }
    
    return out;

}


std::vector<double> Model::residuals(std::string type){
    std::vector<double> out(n_samples);

    std::vector<double> fit_ = fitted("response");
    for(int i=0; i<n_samples; i++){
        out[i] = y(i,0) - fit_[i];
    }

    if(type == "pearson"){
        for(int i=0; i<n_samples; i++){
            out[i] = out[i]/std::sqrt(var_func(fit_[i]));
        }
    }
    
    return out;
}

std::vector<double> Model::predict(const std::vector<std::vector<double> > &data, const std::vector<double> &offset, const std::string &type){
    int nrows = data.size();
    int ncols = data[0].size();
    MatrixXd new_dat = set_design_from_vector(data);
    MatrixXd offset_(offset.size(), 1);
    std::vector<double> out(nrows);

    for(int i=0; i<offset.size(); i++){
        offset_(i, 0) = offset[i];
    }

    MatrixXd beta = coefficients.block(beta_vec_start_ind, 0, beta_size, 1);
    ArrayXd XB = (new_dat*beta + offset_).array();

    if(type == "response"){
        XB = link_func(XB);
    }

    for(int i=0; i<nrows; i++){
        out[i] = XB(i,0);
    }

    return out;
}

std::vector<std::vector<double> > Model::qq_vectors(){
    std::vector<std::vector<double> > out(2);
    std::vector<double> sample_quantiles(n_samples);
    std::vector<double> theory_quantiles(n_samples);

    MatrixXd beta = coefficients.block(beta_vec_start_ind, 0, beta_size, 1);
    ArrayXd XB = (X*beta + offset).array();
    ArrayXd scaled = ArrayXd::Constant(n_samples, 1, 1.0);

    Eigen::ArrayXd I = Eigen::ArrayXd::Constant(n_samples, 1, 1.0);

    //get family extra args 
    for(int f=1; f<family_params.size()+1; f++){
        if(!family_specified[f-1]){
            family_args_temp[f] = coefficients(int(family_params[f-1]))*I; 
        }else{
            family_args_temp[f] = family_params[f-1]*I;
        }
    }

    if(family_name == "gamma"){
        scaled = family_args_temp[1];
    }

    ArrayXd linked = scaled*link_func(XB);
    family_args_temp[0] = linked;

    ArrayXd cdf_ = family_cdf(y.array(), family_args_temp);
    for(int i=0; i<n_samples; i++){
        sample_quantiles[i] = distributions::punif(cdf_(i,0), {0, 1});
        theory_quantiles[i] = (double(i)/double(n_samples));
    }
    std::sort(sample_quantiles.begin(), sample_quantiles.end());
    out[0] = theory_quantiles;
    out[1] = sample_quantiles;
    
    return out;
}


PF_Lookup::PF_Lookup(){
    FUNC_MAP["normal"] = &distributions::dnorm;
    FUNC_MAP["horseshoe"] = &distributions::dhorseshoe;
    FUNC_MAP["halfcauchy"] = &distributions::dhalfcauchy;
    FUNC_MAP["binomial"] = &distributions::dbern;
    FUNC_MAP["poisson"] = &distributions::dpois;
    FUNC_MAP["gamma"] = &distributions::dgamma;
    FUNC_MAP["laplace"] = &distributions::dlaplace;
    FUNC_MAP["uninformed"] = &distributions::uninformed;

};

CDF_Lookup::CDF_Lookup(){
    FUNC_MAP["normal"] = &matrix_distributions::pnorm;
    FUNC_MAP["binomial"] = &matrix_distributions::pbern;
    FUNC_MAP["poisson"] = &matrix_distributions::ppois;
    FUNC_MAP["gamma"] = &matrix_distributions::pgamma;
    FUNC_MAP["laplace"] = &matrix_distributions::plaplace;
}

FAM_PF_Lookup::FAM_PF_Lookup(){
    FUNC_MAP["normal"] = &matrix_distributions::dnorm;
    FUNC_MAP["binomial"] = &matrix_distributions::dbern;
    FUNC_MAP["poisson"] = &matrix_distributions::dpois;
    FUNC_MAP["gamma"] = &matrix_distributions::dgamma;
    FUNC_MAP["laplace"] = &matrix_distributions::dlaplace;
}

SF_Lookup::SF_Lookup(){
    FUNC_MAP["normal"] = &distributions::rnorm;
    //FUNC_MAP["uniform"] = 
    //FUNC_MAP["laplace"] = 
    //FUNC_MAP[""]  
};

Link_Lookup::Link_Lookup(){
    FUNC_MAP["identity"] = &links::identity;
    FUNC_MAP["logit"] = &links::invlogit;
    FUNC_MAP["log"] = &links::invlog;
    FUNC_MAP["inverse"] = &links::inverse;
    FUNC_MAP["probit"] = &links::probit;
};

VF_Lookup::VF_Lookup(){
    FUNC_MAP["normal"] = &variance_functions::normal;
    FUNC_MAP["laplace"] = &variance_functions::normal;
    FUNC_MAP["gamma"] = &variance_functions::gamma;
    FUNC_MAP["poisson"] = &variance_functions::poisson;
    FUNC_MAP["binomial"] = &variance_functions::binomial;
};