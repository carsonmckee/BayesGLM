#include <iostream>
#include "statistics.h"
#include <random>
#include <vector>
#include <string>
#include <fstream>
#include "Eigen/Dense"

using Eigen::ArrayXd;
using Eigen::pow;
using Eigen::exp;

double PI = atan(1)*4;
double e = std::exp(1);

namespace variance_functions{

    double normal(const double &mean){
        return 1;
    }

    double poisson(const double &mean){
        return mean;
    }

    double binomial(const double &mean){
        return mean*(1-mean);
    }

    double gamma(const double &mean){
        return mean*mean;
    }

}


namespace links{

    ArrayXd identity(const ArrayXd &x){
        return x;
    }

    ArrayXd invlogit(const ArrayXd &x){
        return 1.0/(1.0+exp(-1.0*x));
    }

    ArrayXd invlog(const ArrayXd &x){
        return exp(x);
    }

    ArrayXd inverse(const ArrayXd &x){
        return 1/x;
    }

    ArrayXd probit(const ArrayXd &x){
        ArrayXd I = ArrayXd::Constant(x.rows(), 1, 1);
        return matrix_distributions::pnorm(x, {0*I, I});
    }
}

namespace distributions{

    //set up random seed generator
    std::random_device rand_dev;
    std::mt19937 generator(rand_dev());

    //----------------------------------
    //uniform 
    std::vector<double> runif(const int &n, const double &a, const double &b) {
        //n random samples from uniform(a,b) dist
        std::vector<double> out(n);
        std::uniform_real_distribution<double>  distr(a, b);

        for (int i = 0; i < n; i++){
            double rand = distr(generator);
            out[i] = rand;
        }
        return out;
    }

    double runiform(const std::vector<double> &params){
        std::uniform_real_distribution<double>  distr(params[0], params[1]);
        return distr(generator);
    }

    double runiform(const double &a, const double &b){
        std::uniform_real_distribution<double>  distr(a, b);
        return distr(generator);
    }

    double dunif(const double &x, const std::vector<double> &params) {
        double a = params[0];
        double b = params[1];
        //PDF for uniform(a,b) dist
        if(x >= a && x <= b){
            return 1/(b-a);
        } else{
            return 0.0;
        }
    }

    double punif(const double &q, const double &a, const double &b) {
        //CDF for uniform(a,b) dist

        if(q < a){
            return 0.0;
        } else if(q > b){
            return 1.0;
        } else{
            return (q-a)/(b-a);
        }
    }

    double punif(const double &q, const std::vector<double> &params) {
        //CDF for uniform(a,b) dist
        double a = params[0];
        double b = params[1];

        if(q < a){
            return 0.0;
        } else if(q > b){
            return 1.0;
        } else{
            return (q-a)/(b-a);
        }
    }

    double qunif(const double &p, const double &a, const double &b){
        //inverse CDF for uniform(a,b) dist

        if(p < 0 || p > 1){
            std::cout << "qunif -> ERROR: p should be in range [0,1] \n";
            return 0;
        }
        return p*(b-a) + a;
    }

    //normal ---------
    double pnorm(const double &q, const std::vector<double> &params){
        double mu = params[0];
        double sigma = params[1];
        return 0.5*(1+std::erf( (q-mu) / (sigma*sqrt(2)) ) );
    }

    double qnorm(const double &p, const std::vector<double> &params){
        double mu = params[0];
        double sigma = params[1];
        double inv = erf_inv(2*p-1);
        return (inv*sqrt(2)*sigma) + mu;
    }

    std::vector<double> rnorm(const int &n, const double &mu, const double &sigma){
        //first genrate uniform 0,1 then transform to normal using inv cdf
        std::vector<double> uniform = runif(n, 0, 1);
        std::vector<double> out(n);

        for(int i = 0; i<n; i++){
            out[i] = qnorm(uniform[i], {mu, sigma});
        }

        return out;
    }

    double rnorm(const std::vector<double> &params){

        return qnorm(runiform(0,1), params);
    }

    double dnorm(const double &x, const std::vector<double> &params){
        double mu = params[0];
        double sigma = params[1];
        double con = 1/(sigma*sqrt(2*PI));
        double expon = -0.5*std::pow((x-mu)/sigma, 2);

        return con*std::exp(expon);
    }

    double dbern(const double &x, const std::vector<double> &params){
        double p = params[0];
        if(x == 0){
            return 1-p;
        }else{
            return p;
        }
    }

    double pbern(const double &q, const std::vector<double> &params){
        double p = params[0];
        if(q < 0.0){
            return 0;
        } else if (q>=0.0 && q<1){
            return (1-p);
        } else {
            return 1;
        }
    }

    double dpois(const double &x, const std::vector<double> &params){
        if(params[0] <= 0.0 || x < 0.0){
            return 0;
        }else{
            return (std::pow(params[0],x)*std::exp(-params[0]))/factorial(x);
        }
    }

    double ppois(const double &q, const std::vector<double> &params){
        double lamb = params[0];
        int k = std::floor(q);
        double sum = 0;
        for(int i=0; i<k; i++){
            sum += std::pow(lamb, i)/std::tgammal(i+1);
        }
        return std::exp(-lamb)*sum;
    }

    double dgamma(const double &x, const std::vector<double> &params){
        double alpha = params[1];
        double beta = 1/params[0];
        if(alpha <= 0.0 || beta <= 0.0 || x <= 0.0){
            return 0;
        }else{
            return (std::pow(beta, alpha)/std::tgamma(alpha)) * std::pow(x, alpha-1) * std::exp(-beta*x);
        }
    }

    double pgamma(const double &x, const std::vector<double> &params){
        double alpha = params[1];
        double beta = 1/params[0];
        return (1/std::tgammal(alpha)) * lower_gamma_func(alpha, beta*x);
    }

    double dlaplace(const double &x, const std::vector<double> &param){
        double loc = param[0];
        double scale = param[1];
        if(scale <= 0){
            return 0;
        } else {
            return (1/(2*scale)) * exp(-(abs(x-loc)/scale));
        }
    }

    double uninformed(const double &x, const std::vector<double> &params){
        return 1;
    }

    double dhalfcauchy(const double &x, const std::vector<double> &params){
        double mu = params[0];
        double sigma = params[1];
        if(x >= mu){
            return (2/(PI*sigma)) * (1 / (1 + std::pow(x-mu,2)/std::pow(sigma,2)));
        }else{
            return 0;
        }
    }

    double dhorseshoe(const double &x, const std::vector<double> &params){
        double global_shrink = params[0];
        double local_shrink = params[1];
        return distributions::dnorm(x, {0, global_shrink*local_shrink});
    }
}

namespace matrix_distributions{

    ArrayXd dlaplace(const ArrayXd &x, const std::vector<ArrayXd> &param){
        ArrayXd loc = param[0];
        ArrayXd scale = param[1];
        ArrayXd b = (scale > 0).cast <double> ();

        return b * (1/(2*scale)) * exp(-((x-loc).abs()/scale));
    }

    ArrayXd plaplace(const ArrayXd &q, const std::vector<ArrayXd> &param){
        ArrayXd loc = param[0];
        ArrayXd scale = param[1];
        ArrayXd b = (scale > 0).cast <double> ();

        return ((q < loc).cast <double> ()) * 0.5 * ((q-loc)/scale).exp() + (((q >= loc).cast <double> ()) * (1 - 0.5*(-(q-loc)/scale).exp()) );
    }
    
    ArrayXd dunif(const ArrayXd &x, const std::vector<ArrayXd> &param) {
        ArrayXd a = param[0];
        ArrayXd b = param[1];
        //PDF for uniform(a,b) dist
        return ((x >= a && x <= b).cast <double> ()) * (1/(b-a));
    }

    ArrayXd dnorm(const ArrayXd &x, const std::vector<ArrayXd> &params){
        ArrayXd mu = params[0];
        ArrayXd sigma = params[1];
        ArrayXd con = 1/(sigma*sqrt(2*PI));

        return con*(-0.5*((x-mu)/sigma).pow(2)).exp();
    }

    ArrayXd pnorm(const ArrayXd &q, const std::vector<ArrayXd> &params){
        ArrayXd mu = params[0];
        ArrayXd sigma = params[1];
        ArrayXd a = (q-mu) / (sigma*sqrt(2));
        for(int i=0; i<q.rows(); i++){
            a(i,0) = std::erf(a(i,0));
        }
        return 0.5*(1 + a);
    }

    ArrayXd dbern(const ArrayXd &x, const std::vector<ArrayXd> &params){
        ArrayXd p = params[0];
        return (((x == 0).cast <double> ())*(1-p) + ((x == 1).cast <double> ())*p);
    }

    ArrayXd pbern(const ArrayXd &q, const std::vector<ArrayXd> &params){
        ArrayXd p = params[0];
        return (((q >= 0  && q < 1).cast <double> ())*(1-p) + ((q >= 1).cast <double> ()));
    }

    ArrayXd dpois(const ArrayXd &x, const std::vector<ArrayXd> &params){
        ArrayXd out(x.rows(), 1);
        ArrayXd lamb = params[0];
        /*
        ArrayXd b = (((params[0] > 0.0 && x >= 0.0).cast <double> ()));
        ArrayXd cond_x = b*x;
        return b*((pow(params[0], cond_x)*exp(-params[0]) / cond_x.unaryExpr(&factorial)));
        */
        for(int i=0; i<x.rows(); i++){
            if(lamb(i,0) > 50){ //use normal approximation for large lambda - save computation and retain precision
                out(i,0) = distributions::dnorm(x(i,0), {lamb(i,0), std::sqrt(lamb(i,0))});
            } else {
                out(i,0) = distributions::dpois(x(i,0), {lamb(i,0)});
            }
       }
       return out;
    }

    ArrayXd ppois(const ArrayXd &q, const std::vector<ArrayXd> &params){
        ArrayXd out(q.rows(), 1);
        ArrayXd lamb = params[0];
        for(int i=0; i<q.rows(); i++){
            if(lamb(i,0) > 50){ //use normal approximation for large lambda - save computation and retain precision
                out(i,0) = distributions::pnorm(q(i,0), {lamb(i,0), std::sqrt(lamb(i,0))});
            } else {
                out(i,0) = distributions::ppois(q(i,0), {lamb(i,0)});
            }
       }
       return out;
    }

    ArrayXd dgamma(const ArrayXd &x, const std::vector<ArrayXd> &params){
        ArrayXd alpha = params[1];
        ArrayXd beta = 1/params[0];
        ArrayXd cond_x = ((alpha > 0.0 && beta > 0.0 && x > 0.0).cast <double> ())*x;
        
        return cond_x*((pow(beta, alpha) / alpha.unaryExpr(&gamma_func)) * pow(cond_x, alpha-1) * exp(-beta*cond_x));
    }

    ArrayXd pgamma(const ArrayXd &q, const std::vector<ArrayXd> &params){
        ArrayXd alpha = params[1];
        ArrayXd beta = params[0];
        ArrayXd out(q.rows(), 1);

        for(int i=0; i<q.rows(); i++){
            out(i,0) = distributions::pgamma(q(i, 0), {alpha(i,0), beta(i,0)});
        }
        
        return out;
    }

}

double sign(const double &x){
    if(x < 0.0){
        return -1.0;
    } else{
        return 1.0;
    }
}

//error function (approximation)
double erf_(const double &x){

    double sign;
    if(x < 0){
        sign = -1.0;
    } else {
        sign = 1.0;
    }

    double a = 0.147;
    double pow_2 = std::pow(x, 2);
    double d = ((4/PI) + a*pow_2) / (1 + a*pow_2);

    double inside = 1 - exp(-1*pow_2*d);
    return sign*sqrt(inside);
}

//inverse error function (approximation)
double erf_inv(const double &x){
    //approximation
    double sign;
    if(x < 0){
        sign = -1.0;
    } else {
        sign = 1.0;
    }

    double a = 0.147;

    double calc = log(1 - pow(x,2)) / 2;
    double p1 = pow((2/(a*PI)) + calc, 2);
    double p2 = calc*2/a;
    double p3 = (2/(a*PI)) + calc;
    double unscaled = sign*sqrt(sqrt(p1-p2) - p3);
    return unscaled;
}

double gamma_func(double z){
    return std::tgammal(z);
}

double sum(const std::vector<double> &vector){
    double out = 0.0;
    int n = vector.size();
    for(int i=0; i<n; i++){
        out = out + vector[i];
    }
    return out;
}

double mean(const std::vector<double> &vector){
    int n = vector.size();
    return sum(vector)/n;
}

double variance(const std::vector<double> &vector){
    double m = mean(vector);
    int n = vector.size();
    double sum = 0;
    for(int i=0; i<n; i++){
        sum = sum + std::pow((vector[i]-m), 2);
    }
    return sum/(n-1);
}

double kurtosis(const std::vector<double> &vector){
    double mean_ = mean(vector);
    double n = vector.size();
    std::vector<double> centered_2(n);
    std::vector<double> centered_4(n);

    for(int i = 0; i<n; i++){
        double centered = vector[i] - mean_;
        centered_2[i] = centered*centered;
        centered_4[i] = centered_2[i]*centered_2[i];
    }

    double numerator = sum(centered_4)/n;
    double denomenator = sum(centered_2)/n;
    
    return (numerator)/(denomenator*denomenator);
}

double skewness(const std::vector<double> &vector){
    double mean_ = mean(vector);
    int n = vector.size();
    std::vector<double> centered_2(n);
    std::vector<double> centered_3(n);

    for(int i = 0; i<n; i++){
        double centered = vector[i] - mean_;
        centered_2[i] = centered*centered;
        centered_3[i] = centered_2[i]*centered;
    }

    double numerator = sum(centered_3)/n;
    double denomenator = std::pow(sum(centered_2)/(n-1), 1.5);

    return numerator/denomenator;
}

template<typename T>
T abs(T x){
    if(x < 0){
        return -1*x;
    }else{
        return x;
    }
}

template<typename T>
T min(T x, T y){
    if(x < y){
        return x;
    }
    else{
        return y;
    }
}

double factorial_cache[31] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320,
                                362880, 3628800, 39916800, 479001600,
                                6227020800, 87178291200, 1307674368000,
                                20922789888000, 355687428096000, 6402373705728000,
                                121645100408832000, 2432902008176640000, 
                                51090942171709440000.0, 1124000727777607680000.0, 
                                25852016738884976640000.0, 620448401733239439360000.0,
                                15511210043330985984000000.0, 403291461126605635584000000.0, 
                                10888869450418352160768000000.0, 304888344611713860501504000000.0,
                                8841761993739701954543616000000.0, 265252859812191058636308480000000.0};

double factorial(const double &x){
    if(x <= 30){
        return factorial_cache[int(x)];
    } else{
        return std::tgammal(x+1);
    }
}

double factorial_2(long x){
    if(x == 0){
        return std::log(1);
    }else{
        return std::log(x) + factorial_2(x-1);
    }
}

double pochhammer(double x, int k){

    if(k == 0){
        return 1;
    } else {
        return std::pow(x, k-1) * pochhammer(x, k-1);
    }

}

double lower_gamma_func(const double &s, const double &z){
    double z_pow[50];
    double s_fact[50];
    z_pow[0] = 1;
    s_fact[0] = s;

    double sum = 1.0/s;

    for(int k=1; k<50; k++){
        z_pow[k] = z_pow[k-1]*z;
        s_fact[k] = s_fact[k-1]*(s+k);

        sum += z_pow[k]/s_fact[k];
    }
    return std::pow(z,s)*std::exp(-z)*sum;
}

