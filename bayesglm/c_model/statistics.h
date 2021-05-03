#ifndef STATISTICS_H
#define STATISTICS_H
#include <iostream>
#include <vector>
#include <string>
#include "Eigen/Dense"

using Eigen::ArrayXd;

namespace distributions{

    //continuous
    double runiform(const std::vector<double> &params);
    //double runiform(double a=0, double b=1);
    std::vector<double> runif(const int &n, const double &a=0, const double &b=1);
    double dunif(const double &x, const std::vector<double> &params);
    double punif(const double &q, const double &a=0, const double &b=1);
    double punif(const double &p, const std::vector<double> &params);
    double qunif(const double &p, const double &a=0, const double &b=1);

    double rnorm(const std::vector<double> &params);
    std::vector<double> rnorm(const int &n, const double &mu, const double &sigma);
    double dnorm(const double &x, const std::vector<double> &params);
    double pnorm(const double &q, const std::vector<double> &params);
    double qnorm(const double &p, const std::vector<double> &params);

    double dbern(const double &x, const std::vector<double> &params);
    double pbern(const double &q, const std::vector<double> &params);

    double dpois(const double &x, const std::vector<double> &params);
    double ppois(const double &q, const std::vector<double> &params);

    double dgamma(const double &x, const std::vector<double> &params);
    double pgamma(const double &x, const std::vector<double> &params);

    double dbeta(const double &x, const std::vector<double> &params);
    double pbeta(const double &x, const std::vector<double> &params);

    double dlaplace(const double &x, const std::vector<double> &param);

    double uninformed(const double &x, const std::vector<double> &params);

    double dhalfcauchy(const double &x, const std::vector<double> &params);

    double dhorseshoe(const double &x, const std::vector<double> &params);

}


namespace variance_functions{

    double normal(const double &mean);

    double poisson(const double &mean);

    double binomial(const double &mean);

    double gamma(const double &mean);

}

namespace matrix_distributions{
    ArrayXd dunif(const ArrayXd &x, const std::vector<ArrayXd> &param);
    ArrayXd punif(const ArrayXd &q, const std::vector<ArrayXd> &param);

    ArrayXd dnorm(const ArrayXd &x, const std::vector<ArrayXd> &params);
    ArrayXd pnorm(const ArrayXd &q, const std::vector<ArrayXd> &params);

    ArrayXd dbern(const ArrayXd &x, const std::vector<ArrayXd> &params);
    ArrayXd pbern(const ArrayXd &q, const std::vector<ArrayXd> &params);

    ArrayXd dpois(const ArrayXd &x, const std::vector<ArrayXd> &params);
    ArrayXd ppois(const ArrayXd &q, const std::vector<ArrayXd> &params);

    ArrayXd dgamma(const ArrayXd &x, const std::vector<ArrayXd> &params);
    ArrayXd pgamma(const ArrayXd &q, const std::vector<ArrayXd> &params);

    ArrayXd dlaplace(const ArrayXd &x, const std::vector<ArrayXd> &param);
    ArrayXd plaplace(const ArrayXd &q, const std::vector<ArrayXd> &param);
}

namespace links{
    
    ArrayXd identity(const ArrayXd &x);
    
    ArrayXd invlogit(const ArrayXd &x);

    ArrayXd invlog(const ArrayXd &x);

    ArrayXd inverse(const ArrayXd &x);

    ArrayXd probit(const ArrayXd &x);
}

//utils
double erf_(const double &x);
double erf_inv(const double &x);
double gamma_func(double z);

double mean(const std::vector<double> &vector);

double variance(const std::vector<double> &vector);

double kurtosis(const std::vector<double> &vector);

double skewness(const std::vector<double> &vector);

double sum(const std::vector<double> &vector);

double sign(const double &x);

template<typename T>
T abs(T x);

template<typename T>
T min(T x, T y);

double big_factorial(int x);

double factorial(const double &x);

double factorial_2(long x);

double pochhammer(double x, int k);

double lower_gamma_func(const double &s, const double &z);

double ecdf(const double &x, const std::vector<double> &xi);

#endif