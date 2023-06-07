# Internship
Repo for Internship in RS Sollich Working on non-Gaussian noise problems with non-trivial correlations.

Note that the code is not optimized very well. I will work on commenting it a bit more in the coming days....

All source code of relevance is in the folder _numerics_. There, you will find two files that are of particular interest: *legendre.py* and *output.ipynb*. The former is the file which acts as a class library. There are several classes there, each for specific cases. The one of interest, as it is our case, is the class *transform*.

There are several arguments to be given to it when initialising (I will omit explanation where obvious from context):
* lambda_: float 
* a: float
* tau: float
* D1: float. Set this to 0 for our case. I am thinking of removing it in total, as it doesn't contribute anything to our case
* D2: float. To reproduce Bray et al. use D2=2, as they have slightly different definition from us
* N: int. Number of points in each dimension, e.g. q has N values
* noise: str. Technically to change noise for PSN. Not relevant now, set to 'd'.
* pot: str. Technically to change potential. Not relevant now, set to 'm'
tmax: float. Maximal time
* sigma: float. Needed for different noise, not relevant, can be ignored
* b: float. same as above
* const_i: float. Initial condition for position.
* const_f: float. Final position.
* scaling: float. Not relevant.


After initialising the class, all that remains is to call the minimize function of the class. This has three optional arguments, that technically would allow to adjust initial guess (although that is not active as I am using a straight line atm) and for the maximal number of iterations (maxiter) of the minimization. 

The structure is as follows:
Calling minimize starts the minimization of the MSR_action function. This function first calculates the derivatives of q and y as needed. From this, it calculates the k_1,k_2 by calling the Legendre_transform function. There, the different cases are distinguished. Generally, it will calculate k_2 by solving the Legendre transform, i.e. solving for the 0s of the function Opt_Func. 




