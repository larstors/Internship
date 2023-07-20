#include<stdio.h>
#include<math.h>
#include<string.h>
#include<stdlib.h>



double func(double k, double a, double lambda, double D){
    return D / 2.0 * pow(k, 2.0) + lambda * (cosh(a * k) - 1.0);
}

double dfunc(double k, double a, double lambda, double D){
    return D * k + lambda * a * sinh(a * k);
}

double newton(double m, double initial_guess, double a, double lambda, double D){

    double xn = initial_guess;
    double xn1 = 0.0;

    double threshold = 1e-6;


    while (fabs(func(xn, a, lambda, D)) < threshold){
        xn1 = xn - func(xn, a, lambda, D) / dfunc(xn, a, lambda, D);
        xn = xn1;
    }

    return 0.0;
}






