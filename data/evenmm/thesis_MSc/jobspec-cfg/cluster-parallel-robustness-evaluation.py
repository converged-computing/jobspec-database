from scipy import *
import scipy.io
import scipy.ndimage
import numpy as np
import scipy.optimize as spoptim
import numpy.random
import matplotlib
#matplotlib.use('Agg') # When running on cluster, plots cannot be shown and this must be used
import matplotlib.pyplot as plt
import time
import sys
plt.rc('image', cmap='viridis')
from scipy import optimize
numpy.random.seed(13)
from multiprocessing import Pool
from sklearn.decomposition import PCA
#from parameter_file_robustness import * # where all the parameters are set (Not needed because importing in function library)
from function_library import * # loglikelihoods, gradients, covariance functions, tuning curve definitions, posterior tuning curve inference

###########################################
##### Cluster - Robustness evaluation #####
###########################################

## Set T and background noise level
## Array of 21 lambda peak strengths is done in parallel using job-array
## For each lambda peak strength: Run 20 seeds sequentially
## For each seed, the best RMSE is taken from an ensemble of 3-5 initializations with different wmoothingwindow in the PCA (run sequentially)

## History:
## Branched off from em-algorithm on 11.05.2020
## and from robust-sim-data on 28.05.2020
## then from robust-efficient-script on 30.05.2020
## then from parallel-robustness-evaluation.py on 18.06.2020

######################################
## Data generation                  ##
######################################
K_t_generate = exponential_covariance(np.linspace(1,T,T).reshape((T,1)),np.linspace(1,T,T).reshape((T,1)), sigma_x_generate_path, delta_x_generate_path)

############################
# Tuning curve definitions #
############################

if UNIFORM_BUMPS:
    # Uniform positioning and width:'
    bumplocations = [min_neural_tuning_X + (i+0.5)/N*(max_neural_tuning_X - min_neural_tuning_X) for i in range(N)]
    bump_delta_distances = tuning_width_delta * np.ones(N)
else:
    # Random placement and width:
    bumplocations = min_neural_tuning_X + (max_neural_tuning_X - min_neural_tuning_X) * np.random.random(N)
    bump_delta_distances = tuning_width_delta + tuning_width_delta/4*np.random.random(N)

def bumptuningfunction(x, i, peak_f_offset): 
    x1 = x
    x2 = bumplocations[i]
    delta_bumptuning = bump_delta_distances[i]
    if COVARIANCE_KERNEL_KX == "periodic":
        distancesquared = min([(x1-x2)**2, (x1+2*pi-x2)**2, (x1-2*pi-x2)**2])
    elif COVARIANCE_KERNEL_KX == "nonperiodic":
        distancesquared = (x1-x2)**2
    return baseline_f_value + peak_f_offset * exp(-distancesquared/(2*delta_bumptuning))

def offset_function(offset_for_estimate, X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce):
    offset_estimate = X_estimate + offset_for_estimate
    return x_posterior_no_la(offset_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)

def scaling_function(scaling_factor, X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce):
    scaled_estimate = scaling_factor*X_estimate
    return x_posterior_no_la(scaled_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)

def scale_and_offset_function(scale_offset, X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce):
    scaled_estimate = scale_offset[0] * X_estimate + scale_offset[1]
    return x_posterior_no_la(scaled_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)
    #return just_fprior_term(scaled_estimate)

######################################
## RMSE function                    ##
######################################
def find_rmse_for_this_lambda_this_seed(seedindex):
    global lower_domain_limit
    global upper_domain_limit
    starttime = time.time()
    #print("Seed", seeds[seedindex], "started.")
    peak_f_offset = np.log(peak_lambda_global) - baseline_f_value
    np.random.seed(seeds[seedindex])
    # Generate path
    path = (upper_domain_limit-lower_domain_limit)/2 + numpy.random.multivariate_normal(np.zeros(T), K_t_generate)
    #path = np.linspace(lower_domain_limit, upper_domain_limit, T)
    if KEEP_PATH_INSIDE_DOMAIN_BY_FOLDING:
        # Use boolean masks to keep X within min and max of tuning 
        path -= lower_domain_limit # bring path to 0
        modulo_two_pi_values = path // (upper_domain_limit)
        oddmodulos = (modulo_two_pi_values % 2).astype(bool)
        evenmodulos = np.invert(oddmodulos)
        # Even modulos: Adjust for being outside
        path[evenmodulos] -= upper_domain_limit*modulo_two_pi_values[evenmodulos]
        # Odd modulos: Adjust for being outside and flip for continuity
        path[oddmodulos] -= upper_domain_limit*(modulo_two_pi_values[oddmodulos])
        differences = upper_domain_limit - path[oddmodulos]
        path[oddmodulos] = differences
        path += lower_domain_limit # bring path back to min value for tuning
    if SCALE_UP_PATH_TO_COVER_DOMAIN:
        # scale to cover the domain:
        path -= min(path)
        path /= max(path)
        path *= (upper_domain_limit-lower_domain_limit)
        path += lower_domain_limit
    if PLOTTING:
        ## plot path 
        if T > 100:
            plt.figure(figsize=(10,3))
        else:
            plt.figure()
        plt.plot(path, color="black", label='True X') 
        #plt.plot(path, '.', color='black', markersize=1.) # trackingtimes as x optional
        #plt.plot(trackingtimes-trackingtimes[0], path, '.', color='black', markersize=1.) # trackingtimes as x optional
        plt.xlabel("Time bin")
        plt.ylabel("x")
        plt.title("True path of X")
        plt.ylim((lower_domain_limit, upper_domain_limit))
        #plt.title("Simulated path of X")
        #plt.yticks([-15,-10,-5,0,5,10,15])
        plt.tight_layout()
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robustness-eval-T-" + str(T)  + "-seed-" + str(seeds[seedindex]) + "-path.png")
    ## Generate spike data. True tuning curves are defined here
    if TUNINGCURVE_DEFINITION == "triangles":
        tuningwidth = 1 # width of tuning (in radians)
        biasterm = -2 # Average H outside tuningwidth -4
        tuningcovariatestrength = np.linspace(0.5*tuningwidth,10.*tuningwidth, N) # H value at centre of tuningwidth 6*tuningwidth
        neuronpeak = [min_neural_tuning_X + (i+0.5)/N*(max_neural_tuning_X - min_neural_tuning_X) for i in range(N)]
        true_f = np.zeros((N, T))
        y_spikes = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                if COVARIANCE_KERNEL_KX == "periodic":
                    distancefrompeaktopathpoint = min([ abs(neuronpeak[i]+2.*pi-path[t]),  abs(neuronpeak[i]-path[t]),  abs(neuronpeak[i]-2.*pi-path[t]) ])
                elif COVARIANCE_KERNEL_KX == "nonperiodic":
                    distancefrompeaktopathpoint = abs(neuronpeak[i]-path[t])
                Ht = biasterm
                if(distancefrompeaktopathpoint < tuningwidth):
                    Ht = biasterm + tuningcovariatestrength[i] * (1-distancefrompeaktopathpoint/tuningwidth)
                true_f[i,t] = Ht
                # Spiking
                if LIKELIHOOD_MODEL == "bernoulli":
                    spike_probability = exp(Ht)/(1.+exp(Ht))
                    y_spikes[i,t] = 1.0*(rand()<spike_probability)
                    # If you want to remove randomness: y_spikes[i,t] = spike_probability
                elif LIKELIHOOD_MODEL == "poisson":
                    spike_rate = exp(Ht)
                    y_spikes[i,t] = np.random.poisson(spike_rate)
                    # If you want to remove randomness: y_spikes[i,t] = spike_rate
    elif TUNINGCURVE_DEFINITION == "bumps":
        true_f = np.zeros((N, T))
        y_spikes = np.zeros((N, T))
        for i in range(N):
            for t in range(T):
                true_f[i,t] = bumptuningfunction(path[t], i, peak_f_offset)
                if LIKELIHOOD_MODEL == "bernoulli":
                    spike_probability = exp(true_f[i,t])/(1.+exp(true_f[i,t]))
                    y_spikes[i,t] = 1.0*(rand()<spike_probability)
                elif LIKELIHOOD_MODEL == "poisson":
                    spike_rate = exp(true_f[i,t])
                    y_spikes[i,t] = np.random.poisson(spike_rate)
    if PLOTTING:
        ## Plot true f in time
        plt.figure()
        color_idx = np.linspace(0, 1, N)
        plt.title("True log tuning curves f")
        plt.xlabel("x")
        plt.ylabel("f value")
        x_space_grid = np.linspace(lower_domain_limit, upper_domain_limit, T)
        for i in range(N):
            plt.plot(x_space_grid, true_f[i], linestyle='-', color=plt.cm.viridis(color_idx[i]))
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robustness-eval-true-f.png")
    if PLOTTING:
        ## Plot firing rate h in time
        plt.figure()
        color_idx = np.linspace(0, 1, N)
        plt.title("True firing rate h")
        plt.xlabel("x")
        plt.ylabel("Firing rate")
        x_space_grid = np.linspace(lower_domain_limit, upper_domain_limit, T)
        for i in range(N):
            plt.plot(x_space_grid, np.exp(true_f[i]), linestyle='-', color=plt.cm.viridis(color_idx[i]))
        plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robustness-eval-true-h.png")
    ###############################
    # Covariance matrix Kgg_plain #
    ###############################
    # Inducing points based on a predetermined range
    x_grid_induce = np.linspace(min_inducing_point, max_inducing_point, N_inducing_points) #np.linspace(min(path), max(path), N_inducing_points)
    #print("Min and max of path:", min(path), max(path))
    #print("Min and max of grid:", min(x_grid_induce), max(x_grid_induce))
    K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
    ######################
    # Initialize X and F #
    ######################
    # Here the PCA ensemble comes into play:
    ensemble_array_L_value = np.zeros(len(ensemble_smoothingwidths))
    ensemble_array_X_rmse = np.zeros(len(ensemble_smoothingwidths))
    ensemble_array_X_estimate = np.zeros((len(ensemble_smoothingwidths), T))
    ensemble_array_F_estimate = np.zeros((len(ensemble_smoothingwidths), N, T))
    ensemble_array_y_spikes = np.zeros((len(ensemble_smoothingwidths), N, T))
    ensemble_array_path = np.zeros((len(ensemble_smoothingwidths), T))
    for smoothingwindow_index in range(len(ensemble_smoothingwidths)):
        smoothingwindow_for_PCA = ensemble_smoothingwidths[smoothingwindow_index]
        # PCA initialization: 
        celldata = zeros(shape(y_spikes))
        for i in range(N):
            celldata[i,:] = scipy.ndimage.filters.gaussian_filter1d(y_spikes[i,:], smoothingwindow_for_PCA) # smooth
            #celldata[i,:] = (celldata[i,:]-mean(celldata[i,:]))/std(celldata[i,:])                 # standardization requires at least one spike
        X_pca_result = PCA(n_components=1, svd_solver='full').fit_transform(transpose(celldata))
        X_pca_initial = np.zeros(T)
        for i in range(T):
            X_pca_initial[i] = X_pca_result[i]
        # Scale PCA initialization to fit domain:
        X_pca_initial -= min(X_pca_initial)
        X_pca_initial /= max(X_pca_initial)
        X_pca_initial *= (upper_domain_limit-lower_domain_limit)
        X_pca_initial += lower_domain_limit
        # Flip PCA initialization correctly by comparing to true path
        X_pca_initial_flipped = 2*mean(X_pca_initial) - X_pca_initial
        X_pca_initial_rmse = np.sqrt(sum((X_pca_initial-path)**2) / T)
        X_pca_initial_flipped_rmse = np.sqrt(sum((X_pca_initial_flipped-path)**2) / T)
        if X_pca_initial_flipped_rmse < X_pca_initial_rmse:
            X_pca_initial = X_pca_initial_flipped
            # Scale PCA initialization to fit domain:
            X_pca_initial -= min(X_pca_initial)
            X_pca_initial /= max(X_pca_initial)
            X_pca_initial *= (upper_domain_limit-lower_domain_limit)
            X_pca_initial += lower_domain_limit
        if PLOTTING:
            # Plot PCA initialization
            if T > 100:
                plt.figure(figsize=(10,3))
            else:
                plt.figure()
            plt.xlabel("Time bin")
            plt.ylabel("x")
            plt.title("PCA initial of X")
            plt.plot(path, color="black", label='True X')
            plt.plot(X_pca_initial, label="Initial")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robustness-eval-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + "-PCA-initial.png")
        # Initialize X
        np.random.seed(0)
        if X_initialization == "true":
            X_initial = np.copy(path)
        if X_initialization == "true_noisy":
            X_initial = np.copy(path) + np.pi/4*np.sin(np.linspace(0,10*np.pi,T))
            upper_domain_limit = 2*np.pi
            lower_domain_limit = 0
            #X_initial = np.copy(path) + 1*np.random.multivariate_normal(np.zeros(T), K_t) #2*np.random.random(T) - 1
            X_initial -= lower_domain_limit # bring X_initial to 0
            modulo_two_pi_values = X_initial // (upper_domain_limit)
            oddmodulos = (modulo_two_pi_values % 2).astype(bool)
            evenmodulos = np.invert(oddmodulos)
            # Even modulos: Adjust for being outside
            X_initial[evenmodulos] -= upper_domain_limit*modulo_two_pi_values[evenmodulos]
            # Odd modulos: Adjust for being outside and flip for continuity
            X_initial[oddmodulos] -= upper_domain_limit*(modulo_two_pi_values[oddmodulos])
            differences = upper_domain_limit - X_initial[oddmodulos]
            X_initial[oddmodulos] = differences
            X_initial += lower_domain_limit # bring X_initial back to min value for tuning
        if X_initialization == "ones":
            X_initial = np.ones(T)
        if X_initialization == "pca":
            X_initial = X_pca_initial
        if X_initialization == "randomrandom":
            X_initial = (max_inducing_point - min_inducing_point)*np.random.random(T)
        if X_initialization == "randomprior":
            X_initial = (max_inducing_point - min_inducing_point)*np.random.multivariate_normal(np.zeros(T), K_t)
        if X_initialization == "linspace":
            X_initial = np.linspace(min_inducing_point, max_inducing_point, T) 
        if X_initialization == "supreme":
            X_initial = np.load("X_estimate_supreme.npy")
        if X_initialization == "flatrandom":
            X_initial = 1.5*np.ones(T) + 0.2*np.random.random(T)
        if X_initialization == "flat":
            X_initial = 1.5*np.ones(T)
        initial_rmse = np.sqrt(sum((X_initial-path)**2) / T)
        print("Initial RMSE", initial_rmse)
        X_estimate = np.copy(X_initial)
        # Initialize F
        F_initial = np.sqrt(y_spikes) - np.amax(np.sqrt(y_spikes))/2 #np.log(y_spikes + 0.0008)
        F_estimate = np.copy(F_initial)
        if GIVEN_TRUE_F:
            F_estimate = true_f
        if PLOTTING:
            if T > 100:
                plt.figure(figsize=(10,3))
            else:
                plt.figure()
            #plt.title("Path of X")
            plt.title("X estimate")
            plt.xlabel("Time bin")
            plt.ylabel("x")
            plt.plot(path, color="black", label='True X')
            plt.plot(X_initial, label='Initial')
            #plt.legend(loc="upper right")
            #plt.ylim((lower_domain_limit, upper_domain_limit))
            plt.tight_layout()
            plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robustness-eval-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + ".png")
        if PLOT_GRADIENT_CHECK:
            sigma_n = np.copy(global_initial_sigma_n)
            # Adding tiny jitter term to diagonal of K_gg (not the same as sigma_n that we're adding to the diagonal of K_xgK_gg^-1K_gx later on)
            K_gg = K_gg_plain + jitter_term*np.identity(N_inducing_points) ##K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
            X_gradient = x_jacobian_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)
            if T > 100:
                plt.figure(figsize=(10,3))
            else:
                plt.figure()
            plt.xlabel("Time bin")
            plt.ylabel("x")
            plt.title("Gradient at initial X")
            plt.plot(path, color="black", label='True X')
            plt.plot(X_initial, label="Initial")
            #plt.plot(X_gradient, label="Gradient")
            plt.plot(X_estimate + 2*X_gradient/max(X_gradient), label="Gradient plus offset")
            plt.legend(loc="upper right")
            plt.tight_layout()
            plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robustness-eval-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + "-Gradient.png")
            exit()
            """
            print("Testing gradient...")
            #X_estimate = np.copy(path)
            #F_estimate = true_f
            print("Gradient difference using check_grad:",scipy.optimize.check_grad(func=x_posterior_no_la, grad=x_jacobian_no_la, x0=path, args=(sigma_n, F_estimate, K_gg, x_grid_induce)))

            #optim_gradient = optimization_result.jac
            print("Epsilon:", np.sqrt(np.finfo(float).eps))
            optim_gradient1 = scipy.optimize.approx_fprime(xk=X_estimate, f=x_posterior_no_la, epsilon=1*np.sqrt(np.finfo(float).eps), args=(sigma_n, F_estimate, K_gg, x_grid_induce))
            optim_gradient2 = scipy.optimize.approx_fprime(xk=X_estimate, f=x_posterior_no_la, epsilon=x_posterior_no_la, 1e-4, args=(sigma_n, F_estimate, K_gg, x_grid_induce))
            optim_gradient3 = scipy.optimize.approx_fprime(xk=X_estimate, f=x_posterior_no_la, epsilon=x_posterior_no_la, 1e-2, args=(sigma_n, F_estimate, K_gg, x_grid_induce))
            optim_gradient4 = scipy.optimize.approx_fprime(xk=X_estimate, f=x_posterior_no_la, epsilon=x_posterior_no_la, 1e-2, args=(sigma_n, F_estimate, K_gg, x_grid_induce))
            calculated_gradient = x_jacobian_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)
            difference_approx_fprime_1 = optim_gradient1 - calculated_gradient
            difference_approx_fprime_2 = optim_gradient2 - calculated_gradient
            difference_approx_fprime_3 = optim_gradient3 - calculated_gradient
            difference_approx_fprime_4 = optim_gradient4 - calculated_gradient
            difference_norm1 = np.linalg.norm(difference_approx_fprime_1)
            difference_norm2 = np.linalg.norm(difference_approx_fprime_2)
            difference_norm3 = np.linalg.norm(difference_approx_fprime_3)
            difference_norm4 = np.linalg.norm(difference_approx_fprime_4)
            print("Gradient difference using approx f prime, epsilon 1e-8:", difference_norm1)
            print("Gradient difference using approx f prime, epsilon 1e-4:", difference_norm2)
            print("Gradient difference using approx f prime, epsilon 1e-2:", difference_norm3)
            print("Gradient difference using approx f prime, epsilon 1e-2:", difference_norm4)
            plt.figure()
            plt.title("Gradient compared to numerical gradient")
            plt.plot(calculated_gradient, label="Analytic")
            #plt.plot(optim_gradient1, label="Numerical 1")
            plt.plot(optim_gradient2, label="Numerical 2")
            plt.plot(optim_gradient3, label="Numerical 3")
            plt.plot(optim_gradient4, label="Numerical 4")
            plt.legend()
            plt.figure()
            #plt.plot(difference_approx_fprime_1, label="difference 1")
            plt.plot(difference_approx_fprime_2, label="difference 2")
            plt.plot(difference_approx_fprime_3, label="difference 3")
            plt.plot(difference_approx_fprime_4, label="difference 4")
            plt.legend()
            plt.show()
            exit()
            """
        #############################
        # Iterate with EM algorithm #
        #############################
        prev_X_estimate = np.Inf
        sigma_n = np.copy(global_initial_sigma_n)
        startalgorithmtime = time.time()
        for iteration in range(N_iterations):
            if iteration > 0:
                sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
                if LET_INDUCING_POINTS_CHANGE_PLACE_WITH_X_ESTIMATE:
                    x_grid_induce = np.linspace(min(X_estimate), max(X_estimate), N_inducing_points) # Change position of grid to position of estimate
            # Adding tiny jitter term to diagonal of K_gg (not the same as sigma_n that we're adding to the diagonal of K_xgK_gg^-1K_gx later on)
            K_gg = K_gg_plain + jitter_term*np.identity(N_inducing_points) ##K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
            K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
            # Find F estimate only if we're not at the first iteration
            if iteration == 0:
                print("L value of initial estimate", x_posterior_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce))
            if iteration > 0:
                if LIKELIHOOD_MODEL == "bernoulli":
                    for i in range(N):
                        y_i = y_spikes[i]
                        optimization_result = optimize.minimize(fun=f_loglikelihood_bernoulli, x0=F_estimate[i], jac=f_jacobian_bernoulli, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
                        F_estimate[i] = optimization_result.x
                elif LIKELIHOOD_MODEL == "poisson":
                    for i in range(N):
                        y_i = y_spikes[i]
                        optimization_result = optimize.minimize(fun=f_loglikelihood_poisson, x0=F_estimate[i], jac=f_jacobian_poisson, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
                        F_estimate[i] = optimization_result.x 
            # Find next X estimate, that can be outside (0,2pi)
            if NOISE_REGULARIZATION:
                X_estimate += 2*np.random.multivariate_normal(np.zeros(T), K_t_generate) - 1
            if SMOOTHING_REGULARIZATION and iteration < (N_iterations-1) :
                X_estimate = scipy.ndimage.filters.gaussian_filter1d(X_estimate, 4)
            if GRADIENT_FLAG: 
                optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", jac=x_jacobian_no_la, options = {'disp':False})
            else:
                optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", options = {'disp':False})
            X_estimate = optimization_result.x
            if (iteration == (FLIP_AFTER_HOW_MANY - 1)) and FLIP_AFTER_SOME_ITERATION:
                # Flipping estimate after iteration 1 has been plotted
                X_estimate = 2*mean(X_estimate) - X_estimate
            if USE_OFFSET_AND_SCALING_AT_EVERY_ITERATION:
                X_estimate -= min(X_estimate) #set offset of min to 0
                X_estimate /= max(X_estimate) #scale length to 1
                X_estimate *= (max(path)-min(path)) #scale length to length of path
                X_estimate += min(path) #set offset to offset of path
            if PLOTTING:
                plt.plot(X_estimate, label='Estimate')
                #plt.ylim((lower_domain_limit, upper_domain_limit))
                plt.tight_layout()
                plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robustness-eval-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + ".png")
            if np.linalg.norm(X_estimate - prev_X_estimate) < TOLERANCE:
                #print("Seed", seeds[seedindex], "Iterations:", iteration+1, "Change in X smaller than TOL")
                break
            #if iteration == N_iterations-1:
            #    print("Seed", seeds[seedindex], "Iterations:", iteration+1, "N_iterations reached")
            prev_X_estimate = X_estimate
        if USE_OFFSET_AND_SCALING_AFTER_CONVERGENCE:
            X_estimate -= min(X_estimate) #set offset of min to 0
            X_estimate /= max(X_estimate) #scale length to 1
            X_estimate *= (max(path)-min(path)) #scale length to length of path
            X_estimate += min(path) #set offset to offset of path
        # Flipped 
        X_flipped = - X_estimate + 2*mean(X_estimate)
        # Rootmeansquarederror for X
        X_rmse = np.sqrt(sum((X_estimate-path)**2) / T)
        X_flipped_rmse = np.sqrt(sum((X_flipped-path)**2) / T)
        ##### Check if flipped and maybe iterate again with flipped estimate
        if X_flipped_rmse < X_rmse and RECONVERGE_IF_FLIPPED:
            #print("RMSE for X:", X_rmse)
            #print("RMSE for X flipped:", X_flipped_rmse)
            #print("Re-iterating because of flip")
            x_grid_induce = np.linspace(min_inducing_point, max_inducing_point, N_inducing_points) #np.linspace(min(path), max(path), N_inducing_points)
            K_gg_plain = squared_exponential_covariance(x_grid_induce.reshape((N_inducing_points,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
            X_initial_2 = np.copy(X_flipped)
            X_estimate = np.copy(X_flipped)
            F_estimate = np.copy(F_initial)
            if GIVEN_TRUE_F:
                F_estimate = true_f
            if PLOTTING:
                if T > 100:
                    plt.figure(figsize=(10,3))
                else:
                    plt.figure()
                #plt.title("After flipping") # as we go
                plt.xlabel("Time bin")
                plt.ylabel("x")
                plt.plot(path, color="black", label='True X')
                plt.plot(X_initial_2, label='Initial')
                #plt.ylim((lower_domain_limit, upper_domain_limit))
                plt.tight_layout()
                plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robustness-eval-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + "-flipped.png")
            #############################
            # EM after flipped          #
            #############################
            prev_X_estimate = np.Inf
            sigma_n = np.copy(global_initial_sigma_n)
            for iteration in range(N_iterations):
                if iteration > 0:
                    sigma_n = sigma_n * lr  # decrease the noise variance with a learning rate
                    if LET_INDUCING_POINTS_CHANGE_PLACE_WITH_X_ESTIMATE:
                        x_grid_induce = np.linspace(min(X_estimate), max(X_estimate), N_inducing_points) # Change position of grid to position of estimate
                # Adding tiny jitter term to diagonal of K_gg (not the same as sigma_n that we're adding to the diagonal of K_xgK_gg^-1K_gx later on)
                K_gg = K_gg_plain + jitter_term*np.identity(N_inducing_points) ##K_gg = K_gg_plain + sigma_n*np.identity(N_inducing_points)
                K_xg_prev = squared_exponential_covariance(X_estimate.reshape((T,1)),x_grid_induce.reshape((N_inducing_points,1)), sigma_f_fit, delta_f_fit)
                # Find F estimate only if we're not at the first iteration
                if iteration > 0:
                    if LIKELIHOOD_MODEL == "bernoulli":
                        for i in range(N):
                            y_i = y_spikes[i]
                            optimization_result = optimize.minimize(fun=f_loglikelihood_bernoulli, x0=F_estimate[i], jac=f_jacobian_bernoulli, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_bernoulli, 
                            F_estimate[i] = optimization_result.x
                    elif LIKELIHOOD_MODEL == "poisson":
                        for i in range(N):
                            y_i = y_spikes[i]
                            optimization_result = optimize.minimize(fun=f_loglikelihood_poisson, x0=F_estimate[i], jac=f_jacobian_poisson, args=(sigma_n, y_i, K_xg_prev, K_gg), method = 'L-BFGS-B', options={'disp':False}) #hess=f_hessian_poisson, 
                            F_estimate[i] = optimization_result.x 
                # Find next X estimate, that can be outside (0,2pi)
                if NOISE_REGULARIZATION:
                    X_estimate += 2*np.random.multivariate_normal(np.zeros(T), K_t_generate) - 1
                if SMOOTHING_REGULARIZATION and iteration < (N_iterations-1) :
                    X_estimate = scipy.ndimage.filters.gaussian_filter1d(X_estimate, 4)
                if GRADIENT_FLAG: 
                    optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", jac=x_jacobian_no_la, options = {'disp':False})
                else:
                    optimization_result = optimize.minimize(fun=x_posterior_no_la, x0=X_estimate, args=(sigma_n, F_estimate, K_gg, x_grid_induce), method = "L-BFGS-B", options = {'disp':False})
                X_estimate = optimization_result.x
                if (iteration == (FLIP_AFTER_HOW_MANY - 1)) and FLIP_AFTER_SOME_ITERATION:
                    # Flipping estimate after iteration 1 has been plotted
                    X_estimate = 2*mean(X_estimate) - X_estimate
                if USE_OFFSET_AND_SCALING_AT_EVERY_ITERATION:
                    X_estimate -= min(X_estimate) #set offset of min to 0
                    X_estimate /= max(X_estimate) #scale length to 1
                    X_estimate *= (max(path)-min(path)) #scale length to length of path
                    X_estimate += min(path) #set offset to offset of path
                if PLOTTING:
                    plt.plot(X_estimate, label='Estimate (after flip)')
                    #plt.ylim((lower_domain_limit, upper_domain_limit))
                    plt.tight_layout()
                    plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robustness-eval-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + "-flipped.png")
                if np.linalg.norm(X_estimate - prev_X_estimate) < TOLERANCE:
                    #print("Seed", seeds[seedindex], "Iterations after flip:", iteration+1, "Change in X smaller than TOL")
                    break
                #if iteration == N_iterations-1:
                #    print("Seed", seeds[seedindex], "Iterations after flip:", iteration+1, "N_iterations reached")
                prev_X_estimate = X_estimate
            if USE_OFFSET_AND_SCALING_AFTER_CONVERGENCE:
                X_estimate -= min(X_estimate) #set offset of min to 0
                X_estimate /= max(X_estimate) #scale length to 1
                X_estimate *= (max(path)-min(path)) #scale length to length of path
                X_estimate += min(path) #set offset to offset of path
                # Check if flipped is better even after flipped convergence:
                X_flipped = - X_estimate + 2*mean(X_estimate)
                # Rootmeansquarederror for X
                X_rmse = np.sqrt(sum((X_estimate-path)**2) / T)
                X_flipped_rmse = np.sqrt(sum((X_flipped-path)**2) / T)
                ##### Check if flipped and maybe iterate again with flipped estimate
                if X_flipped_rmse < X_rmse:
                    X_estimate = X_flipped
            # Rootmeansquarederror for X
            X_rmse = np.sqrt(sum((X_estimate-path)**2) / T)
        #print("Seed", seeds[seedindex], "smoothingwindow", smoothingwindow_for_PCA, "finished. RMSE for X:", X_rmse)
        #SStot = sum((path - mean(path))**2)
        #SSdev = sum((X_estimate-path)**2)
        #Rsquared = 1 - SSdev / SStot
        #Rsquared_values[seed] = Rsquared
        #print("R squared value of X estimate:", Rsquared, "\n")
        #####
        # Rootmeansquarederror for F
        #if LIKELIHOOD_MODEL == "bernoulli":
        #    h_estimate = np.divide( np.exp(F_estimate), (1 + np.exp(F_estimate)))
        #if LIKELIHOOD_MODEL == "poisson":
        #    h_estimate = np.exp(F_estimate)
        #F_rmse = np.sqrt(sum((h_estimate-true_f)**2) / (T*N))
        if PLOTTING:
            if T > 100:
                plt.figure(figsize=(10,3))
            else:
                plt.figure()
            plt.title("Final estimate") # as we go
            plt.xlabel("Time bin")
            plt.ylabel("x")
            plt.plot(path, color="black", label='True X')
            plt.plot(X_initial, label='Initial')
            plt.plot(X_estimate, label='Estimate')
            plt.legend(loc="upper right")
            #plt.ylim((lower_domain_limit, upper_domain_limit))
            plt.tight_layout()
            plt.savefig(time.strftime("./plots/%Y-%m-%d")+"-robustness-eval-T-" + str(T) + "-lambda-" + str(peak_lambda_global) + "-background-" + str(baseline_lambda_value) + "-seed-" + str(seeds[seedindex]) + "-final-L-" + str(x_posterior_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)) + ".png")
        ensemble_array_X_rmse[smoothingwindow_index] = X_rmse
        ensemble_array_L_value[smoothingwindow_index] = x_posterior_no_la(X_estimate, sigma_n, F_estimate, K_gg, x_grid_induce)
        ensemble_array_X_estimate[smoothingwindow_index] = X_estimate
        ensemble_array_F_estimate[smoothingwindow_index] = F_estimate
        ensemble_array_y_spikes[smoothingwindow_index] = y_spikes
        ensemble_array_path[smoothingwindow_index] = np.copy(path)
        # End of loop for one smoothingwidth
    # Three smoothingwidths done: Find best X estimate based on L value or RMSE score across
    final_rmse = ensemble_array_X_rmse[0] # when only one window 
    print("Final RMSE for tuning width 5", final_rmse)
    index_of_smoothing_with_best_RMSE = np.argmin(ensemble_array_X_rmse)
    best_X_rmse_based_on_RMSE = ensemble_array_X_rmse[index_of_smoothing_with_best_RMSE]
    index_of_smoothing_with_best_L = np.argmin(ensemble_array_L_value)
    best_X_rmse_based_on_L = ensemble_array_X_rmse[index_of_smoothing_with_best_L]
    rmse_for_smoothingwidth_3 = ensemble_array_X_rmse[0]
    rmse_for_smoothingwidth_5 = ensemble_array_X_rmse[1]
    rmse_for_smoothingwidth_10 = ensemble_array_X_rmse[2]
    X_estimate = ensemble_array_X_estimate[index_of_smoothing_with_best_L]
    F_estimate = ensemble_array_F_estimate[index_of_smoothing_with_best_L]
    y_spikes = ensemble_array_y_spikes[index_of_smoothing_with_best_L]
    path = ensemble_array_path[index_of_smoothing_with_best_L]
    endtime = time.time()
    print("\nSeed", seeds[seedindex])
    print("Time use:", endtime - starttime)
    print("Time use without overhead", time.time()-startalgorithmtime)
    print("RMSEs   :", ensemble_array_X_rmse, "Best smoothing window:         ", ensemble_smoothingwidths[index_of_smoothing_with_best_RMSE], "Best RMSE:", best_X_rmse_based_on_RMSE)
    print("L values:", ensemble_array_L_value, "Best smoothing window:", ensemble_smoothingwidths[index_of_smoothing_with_best_L], "Best RMSE:", best_X_rmse_based_on_L)
    print("                                                                  Smoothingwidth 3  RMSE:", rmse_for_smoothingwidth_3)
    print("                                                                  Smoothingwidth 5  RMSE:", rmse_for_smoothingwidth_5)
    print("                                                                  Smoothingwidth 10 RMSE:", rmse_for_smoothingwidth_10)
    return [best_X_rmse_based_on_RMSE, best_X_rmse_based_on_L, rmse_for_smoothingwidth_3, rmse_for_smoothingwidth_5, rmse_for_smoothingwidth_10, X_estimate, F_estimate, y_spikes, path] # Returning X, F estimates based on L value since that is the best we can do unsupervised

if __name__ == "__main__": 
    # The job index is the lambda index
    # Seeds are done sequentially and hope we don't choke on them. Then one job requires 4 OMP_THREADS
    # For each seed we do the pca ensemble sequantially too, and we let the numpy do its parallellization thing 
    lambda_index = int(sys.argv[1]) 
    # The other version:
    # The index in the job array is interpreted as a two-dimensional list with Cols equal to the number of seeds and Rows equal to the number of lambdas
    #n_cols = len(seeds)
    #n_rows = len(peak_lambda_array) 
    #lambda_index = int( int(sys.argv[1]) // n_cols )
    #seedindex = int( int(sys.argv[1]) % n_cols )

    print("Likelihood model:",LIKELIHOOD_MODEL)
    print("Covariance kernel for Kx:", COVARIANCE_KERNEL_KX)
    print("Using gradient?", GRADIENT_FLAG)
    print("Noise regulation:",NOISE_REGULARIZATION)
    print("Tuning curve definition:", TUNINGCURVE_DEFINITION)
    print("Uniform bumps:", UNIFORM_BUMPS)
    print("Plotting:", PLOTTING)
    print("Infer F posteriors:", INFER_F_POSTERIORS)
    print("Initial sigma_n:", global_initial_sigma_n)
    print("Learning rate:", lr)
    print("T:", T)
    print("N:", N)
    print("Smoothingwidths:", ensemble_smoothingwidths)
    print("Number of seeds we average over:", NUMBER_OF_SEEDS)
    if FLIP_AFTER_SOME_ITERATION:
        print("NBBBB!!! We're flipping the estimate in line 600.")
    print("\n")

    global peak_lambda_global
    peak_lambda_global = peak_lambda_array[lambda_index]

    print("Lambda", peak_lambda_global, "started!")
    seed_rmse_array_based_on_RMSE = np.zeros(len(seeds))
    seed_rmse_array_based_on_L = np.zeros(len(seeds))
    seed_rmse_array_for_smoothingwidth_3 = np.zeros(len(seeds))
    seed_rmse_array_for_smoothingwidth_5 = np.zeros(len(seeds))
    seed_rmse_array_for_smoothingwidth_10 = np.zeros(len(seeds))
    X_array = np.zeros((len(seeds), T))
    F_array = np.zeros((len(seeds), N, T))
    Y_array = np.zeros((len(seeds), N, T))
    path_array = np.zeros((len(seeds), T))

    for i in range(len(seeds)):
        result_array = find_rmse_for_this_lambda_this_seed(i) # i = seedindex
        seed_rmse_array_based_on_RMSE[i] = result_array[0]
        seed_rmse_array_based_on_L[i] = result_array[1]
        seed_rmse_array_for_smoothingwidth_3[i] = result_array[2]
        seed_rmse_array_for_smoothingwidth_5[i] = result_array[3]
        seed_rmse_array_for_smoothingwidth_10[i] = result_array[4]
        X_array[i] = result_array[5]
        F_array[i] = result_array[6]
        Y_array[i] = result_array[7]
        path_array[i] = result_array[8]
    
    # Using RMSE to choose best final X:
    np.save("m_s_arrays/RMSE-m-base-" + str(baseline_lambda_value) + "-T-" + str(T) + "-lambda-index-" + str(lambda_index), np.mean(seed_rmse_array_based_on_RMSE))
    np.save("m_s_arrays/RMSE-s-base-" + str(baseline_lambda_value) + "-T-" + str(T) + "-lambda-index-" + str(lambda_index), sum((seed_rmse_array_based_on_RMSE - np.mean(seed_rmse_array_based_on_RMSE))**2))
    # Using L to choose best final X:
    np.save("m_s_arrays/L-m-base-" + str(baseline_lambda_value) + "-T-" + str(T) + "-lambda-index-" + str(lambda_index), np.mean(seed_rmse_array_based_on_L))
    np.save("m_s_arrays/L-s-base-" + str(baseline_lambda_value) + "-T-" + str(T) + "-lambda-index-" + str(lambda_index), sum((seed_rmse_array_based_on_L - np.mean(seed_rmse_array_based_on_L))**2))
    # Sticking with smoothingwidth 3:
    np.save("m_s_arrays/3-m-base-" + str(baseline_lambda_value) + "-T-" + str(T) + "-lambda-index-" + str(lambda_index), np.mean(seed_rmse_array_for_smoothingwidth_3))
    np.save("m_s_arrays/3-s-base-" + str(baseline_lambda_value) + "-T-" + str(T) + "-lambda-index-" + str(lambda_index), sum((seed_rmse_array_for_smoothingwidth_3 - np.mean(seed_rmse_array_for_smoothingwidth_3))**2))
    # Sticking with smoothingwidth 5:
    np.save("m_s_arrays/5-m-base-" + str(baseline_lambda_value) + "-T-" + str(T) + "-lambda-index-" + str(lambda_index), np.mean(seed_rmse_array_for_smoothingwidth_5))
    np.save("m_s_arrays/5-s-base-" + str(baseline_lambda_value) + "-T-" + str(T) + "-lambda-index-" + str(lambda_index), sum((seed_rmse_array_for_smoothingwidth_5 - np.mean(seed_rmse_array_for_smoothingwidth_5))**2))
    # Sticking with smoothingwidth 10:
    np.save("m_s_arrays/10-m-base-" + str(baseline_lambda_value) + "-T-" + str(T) + "-lambda-index-" + str(lambda_index), np.mean(seed_rmse_array_for_smoothingwidth_10))
    np.save("m_s_arrays/10-s-base-" + str(baseline_lambda_value) + "-T-" + str(T) + "-lambda-index-" + str(lambda_index), sum((seed_rmse_array_for_smoothingwidth_10 - np.mean(seed_rmse_array_for_smoothingwidth_10))**2))

    print("\n")
    print("Lambda strength:", peak_lambda_global)
    print("RMSE for X (chosen by RMSE   ) averaged across seeds:", np.mean(seed_rmse_array_based_on_RMSE))
    print("Sum of squared errors in the RMSE:", sum((seed_rmse_array_based_on_RMSE - np.mean(seed_rmse_array_based_on_RMSE))**2))
    print("RMSE for X (chosen by L value) averaged across seeds:", np.mean(seed_rmse_array_based_on_L))
    print("Sum of squared errors in the RMSE:", sum((seed_rmse_array_based_on_L - np.mean(seed_rmse_array_based_on_L))**2))
    print("RMSE for X (smoothing width 3) averaged across seeds:", np.mean(seed_rmse_array_for_smoothingwidth_3))
    print("Sum of squared errors in the RMSE:", sum((seed_rmse_array_for_smoothingwidth_3 - np.mean(seed_rmse_array_for_smoothingwidth_3))**2))
    print("RMSE for X (smoothing width 5) averaged across seeds:", np.mean(seed_rmse_array_for_smoothingwidth_5))
    print("Sum of squared errors in the RMSE:", sum((seed_rmse_array_for_smoothingwidth_5 - np.mean(seed_rmse_array_for_smoothingwidth_5))**2))
    print("RMSE for X (smoothing width 10) averaged across seeds:", np.mean(seed_rmse_array_for_smoothingwidth_10))
    print("Sum of squared errors in the RMSE:", sum((seed_rmse_array_for_smoothingwidth_10 - np.mean(seed_rmse_array_for_smoothingwidth_10))**2))
    print("\n")
    # Finished all seeds for this lambda

    if INFER_F_POSTERIORS:
        # Grid for plotting
        bins_for_plotting = np.linspace(lower_domain_limit, upper_domain_limit, num=N_plotgridpoints + 1)
        x_grid_for_plotting = 0.5*(bins_for_plotting[:(-1)]+bins_for_plotting[1:])
        peak_lambda_global = peak_lambda_array[-1] 
        print("Peak lambda:", peak_lambda_global)
        peak_f_offset = np.log(peak_lambda_global) - baseline_f_value
        #posterior_f_inference(X_estimate, F_estimate, sigma_n, y_spikes, path, x_grid_for_plotting, bins_for_plotting, peak_f_offset, baseline_f_value, binsize)
        posterior_f_inference(X_array[0], F_array[0], 1, Y_array[0], path_array[0], x_grid_for_plotting, bins_for_plotting, peak_f_offset, baseline_f_value, 1000) # Bin size has no physical meaning for synthetic data

