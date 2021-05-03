#include "Eigen/Dense"
#include <iostream>
#include <vector>

using Eigen::ArrayXd;
using Eigen::pow;
using Eigen::exp;

double PI = atan(1)*4;

long double factorial_cache[31] = {1, 1, 2, 6, 24, 120, 720, 5040, 40320,
                                362880, 3628800, 39916800, 479001600,
                                6227020800, 87178291200, 1307674368000,
                                20922789888000, 355687428096000, 6402373705728000,
                                121645100408832000, 2432902008176640000, 
                                51090942171709440000.0, 1124000727777607680000.0, 
                                25852016738884976640000.0, 620448401733239439360000.0,
                                15511210043330985984000000.0, 403291461126605635584000000.0, 
                                10888869450418352160768000000.0, 304888344611713860501504000000.0,
                                8841761993739701954543616000000.0, 265252859812191058636308480000000.0};

double factorial(const double &x);

double factorial(const double &x){
    if(x <= 30){
        return factorial_cache[int(x)];
    } else{
        return std::tgammal(x+1);
    }
}

ArrayXd dunif(const ArrayXd &x, const std::vector<ArrayXd> &param);
ArrayXd dnorm(const ArrayXd &x, const std::vector<ArrayXd> &params);
ArrayXd dbern(const ArrayXd &x, const std::vector<ArrayXd> &params);
ArrayXd dpois(const ArrayXd &x, const std::vector<ArrayXd> &params);
ArrayXd dgamma(const ArrayXd &x, const std::vector<ArrayXd> &params);

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

ArrayXd dbern(const ArrayXd &x, const std::vector<ArrayXd> &params){
    ArrayXd p = params[0];
    return (((x == 0).cast <double> ())*(1-p) + ((x == 1).cast <double> ())*p);
}

ArrayXd dpois(const ArrayXd &x, const std::vector<ArrayXd> &params){
    ArrayXd b = (((params[0] > 0.0 && x >= 0.0).cast <double> ()));
    ArrayXd cond_x = b*x;
    return b*((pow(params[0], cond_x)*exp(-params[0]) / cond_x.unaryExpr(&factorial)));
}

ArrayXd dgamma(const ArrayXd &x, const std::vector<ArrayXd> &params){
        ArrayXd alpha = params[1];
        ArrayXd beta = 1/params[0];
        ArrayXd cond_x = ((alpha > 0.0 && beta > 0.0 && x > 0.0).cast <double> ())*x;
        
        return cond_x*((pow(beta, alpha) / alpha.unaryExpr(&std::tgammal)) * pow(cond_x, alpha-1) * exp(-beta*cond_x));
}

double pochhammer(double x, int k){

    if(k == 0){
        return 1;
    } else {
        return (x+k-1) * pochhammer(x, k-1);
    }

}


double lower_gamma_func(double s, double z){
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

double pgamma(const double &x, const std::vector<double> &params){
        double alpha = params[1];
        double beta = 1/params[0];
        return (1/std::tgammal(alpha)) * lower_gamma_func(alpha, beta*x);
}

template<typename T>
T abs(T x){
    if(x < 0){
        return -1*x;
    }else{
        return x;
    }
}

double my_erf(const double &x){

    double t = 1/(1+0.5*abs(x));
    double t_pow[10];

    t_pow[0] = 1.0;
    for(int i=1; i<10; i++){
        t_pow[i] = t*t_pow[i-1];
    }

    double s;
    s = (-x*x)-1.26551223+1.00002368*t_pow[1]+0.37409196*t_pow[2]+0.09678418*t_pow[3] 
        -0.18628806*t_pow[4]+0.27886807*t_pow[5]-1.13520398*t_pow[6]+1.48851587*t_pow[7]
        -0.82215223*t_pow[8]+0.17087277*t_pow[9];
    double tor = t*std::exp(s);
    if(x>=0.0){
        return 1-tor;
    }else{
        return tor-1;
    }
}


ArrayXd dlaplace(const ArrayXd &x, const std::vector<ArrayXd> &param){
    ArrayXd loc = param[0];
    ArrayXd scale = param[1];
    ArrayXd b = (scale > 0).cast <double> ();

    return b * (1/(2*scale)) * exp(-((x-loc).abs()/scale));
}

int main(){

    ArrayXd a = Eigen::ArrayXd::Constant(10, 1, -3);
    a(1,0) = 0;
    ArrayXd I = Eigen::ArrayXd::Constant(10, 1, 1);

    std::vector<ArrayXd> params(2);
    params[0] = 2*I, 
    params[1] = 2*I;
    std::cout << dlaplace(a, params) << "\n";

    return  0;
}